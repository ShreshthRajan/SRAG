"""
Minimal-by-design model wrapper for the variance-gate experiments.

Two model paths:
  - build_tiny_model(): a from-scratch tiny Llama (no download, offline, CPU) for the
    correctness/smoke result. The variance-gate claim is arithmetic, so a tiny model
    proves the mechanism exactly.
  - load_real_model(): Qwen2.5-1.5B (+ tokenizer) for the Colab paper figure.

LoRA is attached with SRAG-V's real hyperparameters (r=64, alpha=128, 7 target modules).
log-probability mirrors SRAG-V grpo_trainer._compute_log_probability (shift + mean over output tokens).
"""
import torch
import torch.nn.functional as F

from peft import LoraConfig, get_peft_model, TaskType


# --------------------------- models ---------------------------------------
def build_tiny_model(seed: int = 0, vocab: int = 256, hidden: int = 64, layers: int = 2, heads: int = 4):
    """Tiny Llama built from config — no network, runs on CPU in milliseconds."""
    from transformers import LlamaConfig, LlamaForCausalLM
    torch.manual_seed(seed)
    cfg = LlamaConfig(
        vocab_size=vocab, hidden_size=hidden, intermediate_size=hidden * 2,
        num_hidden_layers=layers, num_attention_heads=heads, num_key_value_heads=heads,
        max_position_embeddings=512, tie_word_embeddings=True,
    )
    model = LlamaForCausalLM(cfg)
    return model.eval()


def load_real_model(name: str, device: str, dtype: str = "bf16"):
    from transformers import AutoTokenizer, AutoModelForCausalLM
    tok = AutoTokenizer.from_pretrained(name, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    td = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[dtype]
    model = AutoModelForCausalLM.from_pretrained(
        name, torch_dtype=td if device != "cpu" else torch.float32,
        trust_remote_code=True, low_cpu_mem_usage=True,
    ).to(device)
    return tok, model


def attach_lora(model, hparams, target_modules=None):
    """Attach a fresh LoRA adapter (lora_B starts at exactly 0)."""
    tm = target_modules or hparams["target_modules"]
    # Tiny model may not have all 7 modules under the same names; intersect to be safe.
    present = {n.split(".")[-1] for n, _ in model.named_modules()}
    tm = [m for m in tm if m in present] or ["q_proj", "v_proj"]
    lcfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM, r=hparams["rank"], lora_alpha=hparams["alpha"],
        lora_dropout=hparams["dropout"], target_modules=tm, bias="none",
    )
    return get_peft_model(model, lcfg)


# --------------------------- param meters ----------------------------------
def lora_named(model, kind=""):
    """Trainable LoRA params; kind in {'', 'lora_A', 'lora_B'}."""
    out = []
    for n, p in model.named_parameters():
        if p.requires_grad and "lora_" in n and (kind == "" or kind in n):
            out.append((n, p))
    return out


def grad_l2(named):
    s = 0.0
    for _, p in named:
        if p.grad is not None:
            s += float(p.grad.detach().double().pow(2).sum())
    return s ** 0.5


def clone_state(named):
    return {n: p.detach().clone() for n, p in named}


def restore_state(model, state):
    with torch.no_grad():
        for n, p in model.named_parameters():
            if n in state:
                p.copy_(state[n])


def delta_l2(named, prev_state):
    s = 0.0
    for n, p in named:
        s += float((p.detach().double() - prev_state[n].double()).pow(2).sum())
    return s ** 0.5


# --------------------------- log-prob (mirrors SRAG-V) ---------------------
def logprob_from_ids(model, input_ids, prompt_len):
    """Mean log-prob of output tokens (positions >= prompt_len). Differentiable."""
    out = model(input_ids=input_ids)
    logits = out.logits
    tgt = input_ids[0, prompt_len:]
    lg = logits[0, prompt_len - 1:-1, :]
    m = min(lg.shape[0], tgt.shape[0])
    if m == 0:
        return torch.zeros((), device=logits.device, dtype=logits.dtype, requires_grad=True)
    lg, tgt = lg[:m], tgt[:m]
    logp = F.log_softmax(lg.float(), dim=-1)
    sel = logp.gather(1, tgt.unsqueeze(-1)).squeeze(-1)
    return sel.mean()


def logprob_from_text(model, tok, prompt, output, max_len=1024, device="cpu"):
    full = tok(prompt + output, return_tensors="pt", truncation=True, max_length=max_len)
    pl = tok(prompt, return_tensors="pt", truncation=True, max_length=max_len)["input_ids"].shape[1]
    ids = full["input_ids"].to(device)
    return logprob_from_ids(model, ids, pl)


def mean_token_entropy_from_ids(model, input_ids, prompt_len):
    """Mean per-token predictive entropy over the output span (policy entropy proxy). No grad needed."""
    out = model(input_ids=input_ids)
    logits = out.logits[0, prompt_len - 1:-1, :].float()
    if logits.shape[0] == 0:
        return 0.0
    logp = F.log_softmax(logits, dim=-1)
    ent = -(logp.exp() * logp).sum(dim=-1).mean()
    return float(ent.item())


def entropy_from_text(model, tok, prompt, output, max_len=1024, device="cpu"):
    full = tok(prompt + output, return_tensors="pt", truncation=True, max_length=max_len)
    pl = tok(prompt, return_tensors="pt", truncation=True, max_length=max_len)["input_ids"].shape[1]
    ids = full["input_ids"].to(device)
    return mean_token_entropy_from_ids(model, ids, pl)


def batched_seq_stats(model, input_ids, attention_mask, prompt_len, want_hidden=True):
    """ONE batched forward -> per-sequence MEAN output-token logprob (differentiable) + pooled last-hidden (np).
    All sequences share `prompt_len` (same prompt prefix) and are RIGHT-padded. Returns (logp[G] tensor, hid[G,H] np|None)."""
    out = model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=want_hidden)
    logits = out.logits.float()
    G, L, V = logits.shape
    logp_all = F.log_softmax(logits[:, :-1, :], dim=-1)               # predict token t from position t-1
    tgt = input_ids[:, 1:]
    tok_logp = logp_all.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)      # (G, L-1)
    pos = torch.arange(L - 1, device=input_ids.device).unsqueeze(0)
    out_mask = ((pos >= prompt_len - 1) & attention_mask[:, 1:].bool()).float()   # output, non-pad
    denom = out_mask.sum(1).clamp(min=1.0)
    seq_logp = (tok_logp * out_mask).sum(1) / denom                   # (G,)
    hid = None
    if want_hidden:
        hs = out.hidden_states[-1].float()
        fpos = torch.arange(L, device=input_ids.device).unsqueeze(0)
        hmask = ((fpos >= prompt_len) & attention_mask.bool()).float().unsqueeze(-1)
        hid = ((hs * hmask).sum(1) / hmask.sum(1).clamp(min=1.0)).detach().cpu().numpy()
    return seq_logp, hid


def batched_probe(model, tok, prompt, outputs, device, max_len=768, want_hidden=True):
    """Tokenize prompt+'\\n'+output_i (right-padded) and run batched_seq_stats in ONE forward for the whole group."""
    pl = len(tok(prompt, truncation=True, max_length=max_len)["input_ids"])
    fulls = [prompt + "\n" + o for o in outputs]
    old_side = tok.padding_side
    tok.padding_side = "right"
    enc = tok(fulls, return_tensors="pt", padding=True, truncation=True, max_length=max_len)
    tok.padding_side = old_side
    ids = enc["input_ids"].to(device); am = enc["attention_mask"].to(device)
    return batched_seq_stats(model, ids, am, pl, want_hidden=want_hidden)


def probe_from_text(model, tok, prompt, output, max_len=1024, device="cpu"):
    """ONE no-grad forward returning the three monitor inputs for an output span:
       (mean_output_logprob: float, mean_output_entropy: float, pooled_last_hidden: np.ndarray[H]).
    Used to compute confidence (ECE), policy entropy, and the representation matrix for spectral entropy
    without three separate forwards. Pooling = mean of the last hidden layer over the OUTPUT token span."""
    import numpy as np
    full = tok(prompt + output, return_tensors="pt", truncation=True, max_length=max_len)
    pl = tok(prompt, return_tensors="pt", truncation=True, max_length=max_len)["input_ids"].shape[1]
    ids = full["input_ids"].to(device)
    with torch.no_grad():
        out = model(input_ids=ids, output_hidden_states=True)
    hs = out.hidden_states[-1][0]                       # (T, H)
    span = hs[pl - 1:-1] if hs.shape[0] > pl else hs    # output-token hidden states
    pooled = span.float().mean(0).detach().cpu().numpy()
    logits = out.logits[0, pl - 1:-1, :].float()
    if logits.shape[0] == 0:
        return 0.0, 0.0, pooled
    logp = F.log_softmax(logits, dim=-1)
    tgt = ids[0, pl:]
    m = min(logits.shape[0], tgt.shape[0])
    mean_lp = float(logp[:m].gather(1, tgt[:m].unsqueeze(-1)).squeeze(-1).mean().item())
    ent = float((-(logp.exp() * logp).sum(dim=-1)).mean().item())
    return mean_lp, ent, pooled
