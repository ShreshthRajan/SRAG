"""
ML Foundry Integration for SRAG-V Step 2 Training.
Implements 4×L4 GPU deployment for breakthrough-level self-play training.
"""

import os
import requests
import json
import time
import logging
from typing import Dict, List, Optional, Any
from pathlib import Path
import yaml
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class FoundryJobConfig:
    """Configuration for ML Foundry training job."""
    
    # Compute configuration
    gpu_type: str = "L4"
    gpu_count: int = 4
    cpu_cores: int = 16
    memory_gb: int = 64
    
    # Training configuration
    max_runtime_hours: int = 6
    auto_shutdown: bool = True
    
    # Environment
    python_version: str = "3.10"
    cuda_version: str = "12.1"


class MLFoundryClient:
    """Client for ML Foundry API integration."""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        self.config_path = config_path
        self.config = self._load_config()
        
        # ML Foundry configuration
        self.api_key = os.getenv("ML_FOUNDRY_API_KEY")
        self.project_id = os.getenv("ML_FOUNDRY_PROJECT_ID")
        self.base_url = self.config.get('foundry', {}).get('base_url', 'https://api.mlfoundry.com/v2')
        
        if not self.api_key:
            raise ValueError("ML_FOUNDRY_API_KEY environment variable not set")
        
        # If project_id not provided, get it from project name "ARG"
        if not self.project_id:
            self.project_id = self._get_project_id_by_name("ARG")
            if not self.project_id:
                raise ValueError("Could not find project 'ARG' or ML_FOUNDRY_PROJECT_ID not set")
        
        self.headers = {
            'Authorization': f'Bearer {self.api_key}',
            'Content-Type': 'application/json'
        }
        
        logger.info(f"ML Foundry client initialized for project {self.project_id}")
    
    def _get_project_id_by_name(self, project_name: str) -> Optional[str]:
        """Get project ID by project name."""
        try:
            response = requests.get(
                f"{self.base_url}/projects",
                headers={'Authorization': f'Bearer {self.api_key}'}
            )
            
            if response.status_code == 200:
                projects_data = response.json()
                logger.debug(f"API Response: {projects_data}")
                
                # Handle different API response formats
                projects_list = projects_data
                if isinstance(projects_data, dict):
                    # If response is wrapped in a data field
                    projects_list = projects_data.get('data', projects_data.get('projects', [projects_data]))
                
                if not isinstance(projects_list, list):
                    projects_list = [projects_list]
                
                for project in projects_list:
                    # Try different possible field names - ML Foundry uses 'fid' for project ID
                    project_id = project.get('fid') or project.get('id') or project.get('project_id') or project.get('_id')
                    project_name_field = project.get('name') or project.get('project_name') or project.get('title')
                    
                    if project_name_field == project_name and project_id:
                        logger.info(f"Found project '{project_name}' with ID: {project_id}")
                        return project_id
                
                # List available projects for debugging
                available_projects = [p.get('name', 'unnamed') for p in projects_list]
                logger.error(f"Project '{project_name}' not found. Available: {available_projects}")
                return None
            else:
                logger.error(f"Failed to list projects: {response.status_code} - {response.text}")
                return None
        except Exception as e:
            logger.error(f"Error getting project ID: {e}")
            import traceback
            logger.debug(f"Full traceback: {traceback.format_exc()}")
            return None
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        try:
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)
            return config.get('sragv', {})
        except Exception as e:
            logger.error(f"Failed to load config: {e}")
            raise
    
    def create_training_job(self, job_config: FoundryJobConfig) -> str:
        """Create a training job on ML Foundry using spot bids."""
        
        logger.info("Creating ML Foundry training job via spot bid...")
        logger.info(f"Configuration: {job_config.gpu_count}×{job_config.gpu_type} GPUs, {job_config.memory_gb}GB RAM")
        
        # First, check availability to find the right instance type
        instance_type = self._find_gpu_instance_type(job_config.gpu_type, job_config.gpu_count)
        if not instance_type:
            raise Exception(f"No available instance type for {job_config.gpu_count}×{job_config.gpu_type} GPUs")
        
        # Create startup script with all training code
        startup_script = self._create_startup_script()
        
        # Get registered SSH key ID
        ssh_key_id = self._get_ssh_key_id()
        
        # Post-fix priority: Use best available regions (issue fixed)
        region_configs = [
            {"region": "us-central2-a"},  # 8×H100 available, 30 capacity, $8.00
            {"region": "eu-central1-b"},  # 4×A100 available, 17 capacity, $2.00 (FIXED)
            {"region": "us-central1-a"},  # 8×H100 available, 1 capacity, $8.00
            {"region": "eu-central1-a"},  # Backup option
        ]
        
        for config in region_configs:
            # Create spot bid payload with higher bid to prevent preemption
            bid_payload = {
                "project": self.project_id,
                "name": f"sragv-step2-monitored-{int(time.time())}",
                "instance_type": instance_type,
                "limit_price": "$30.00",  # Higher price to prevent preemption
                "instance_quantity": 1,
                "launch_specification": {
                    "startup_script": startup_script,
                    "volumes": [],  # Required field - empty for now
                    "ssh_keys": [ssh_key_id]  # Use SSH key ID
                }
            }
            
            # Add region if specified
            if "region" in config:
                bid_payload["region"] = config["region"]
                logger.info(f"Trying region: {config['region']}")
            else:
                logger.info("Trying without region specification (auto-select)")
            
            response = requests.post(
                f"{self.base_url}/spot/bids",
                headers=self.headers,
                json=bid_payload
            )
            
            if response.status_code in [200, 201]:
                bid_data = response.json()
                bid_id = bid_data.get('id') or bid_data.get('fid')
                logger.info(f"✅ Training spot bid created successfully: {bid_id}")
                region_info = config.get("region", "auto-selected")
                logger.info(f"Instance will be provisioned in region: {region_info}")
                return bid_id
            elif "not available in this region" in response.text or "Region not found" in response.text:
                region_name = config.get("region", "auto-select")
                logger.warning(f"Configuration failed for {region_name}, trying next...")
                continue
            else:
                # Different error, log it and continue trying
                region_name = config.get("region", "auto-select")
                logger.warning(f"Configuration {region_name} failed: {response.status_code} - {response.text}")
                continue
        
        # If we get here, all configurations failed
        error_msg = f"Failed to create spot bid in all regions: {response.status_code} - {response.text}"
        logger.error(error_msg)
        raise Exception(error_msg)
    
    def _prepare_training_files(self) -> Dict[str, str]:
        """Prepare training files for upload."""
        files = {}
        
        # Core source files
        src_files = [
            "src/sragv/__init__.py",
            "src/sragv/orchestrator.py",
            "src/sragv/models/__init__.py",
            "src/sragv/models/base_player.py",
            "src/sragv/models/problem_generator.py",
            "src/sragv/models/solution_generator.py", 
            "src/sragv/models/verification_generator.py",
            "src/sragv/models/meta_verifier.py",
            "src/sragv/training/__init__.py",
            "src/sragv/training/grpo_trainer.py",
            "src/sragv/training/map_elites.py",
            "src/sragv/training/reward_system.py",
            "src/sragv/training/self_play_trainer.py",
            "src/sragv/data/__init__.py",
            "src/sragv/data/simple_data_loader.py"
        ]
        
        # Training script
        training_files = [
            "run_step2_training.py"
        ]
        
        # Configuration files
        config_files = [
            "config/config.yaml"
        ]
        
        # Dataset files
        data_files = [
            "data/expanded_apps.json"
        ]
        
        all_files = src_files + training_files + config_files + data_files
        
        for file_path in all_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    files[file_path] = f.read()
                logger.debug(f"Added file: {file_path}")
            except Exception as e:
                logger.warning(f"Could not read {file_path}: {e}")
        
        logger.info(f"Prepared {len(files)} files for training job")
        return files
    
    def _get_requirements(self) -> List[str]:
        """Get pip requirements for the training environment."""
        requirements = [
            "torch>=2.0.0",
            "transformers>=4.40.0",
            "accelerate>=0.25.0", 
            "peft>=0.16.0",
            "bitsandbytes>=0.41.0",
            "datasets>=2.0.0",
            "numpy>=1.24.0",
            "pyyaml>=6.0",
            "tqdm>=4.66.0",
            "scikit-learn>=1.3.0",
            "matplotlib>=3.7.0",
            "seaborn>=0.12.0",
            "wandb>=0.16.0",
            "omegaconf>=2.3.0"
        ]
        return requirements
    
    def _find_gpu_instance_type(self, gpu_type: str, gpu_count: int) -> Optional[str]:
        """Find the appropriate instance type for GPU requirements."""
        try:
            # Get instance type details
            response = requests.get(
                f"{self.base_url}/instance-types",
                headers=self.headers
            )
            
            if response.status_code == 200:
                instance_types = response.json()
                logger.debug(f"Available instance types: {instance_types}")
                
                # Find best match for GPU requirements
                # Since ML Foundry doesn't have L4s, use A100s which are better
                if gpu_count == 4:
                    # Look for 4x A100 instance type
                    for instance_type in instance_types:
                        if instance_type.get('num_gpus') == 4 and instance_type.get('gpu_type') == 'A100':
                            logger.info(f"Using 4×A100 instance (better than 4×L4): {instance_type['name']}")
                            return instance_type['fid']
                
                # Fallback to 2x A100 if 4x not available
                for instance_type in instance_types:
                    if instance_type.get('num_gpus') == 2 and instance_type.get('gpu_type') == 'A100':
                        logger.info(f"Using 2×A100 instance: {instance_type['name']}")
                        return instance_type['fid']
                
                # Final fallback to 1x A100
                for instance_type in instance_types:
                    if instance_type.get('num_gpus') == 1 and instance_type.get('gpu_type') == 'A100':
                        logger.info(f"Using 1×A100 instance: {instance_type['name']}")
                        return instance_type['fid']
                
                logger.error("No suitable GPU instance type found")
                return None
            else:
                logger.error(f"Failed to get instance types: {response.status_code}")
                return None
        except Exception as e:
            logger.error(f"Error checking instance types: {e}")
            return None
    
    def _get_ssh_key_id(self) -> str:
        """Get registered SSH key ID from ML Foundry."""
        try:
            # Check existing SSH keys for this project
            response = requests.get(
                f"{self.base_url}/ssh-keys?project={self.project_id}",
                headers=self.headers
            )
            
            if response.status_code == 200:
                ssh_keys = response.json()
                if ssh_keys:
                    # Use existing SSH key
                    ssh_key_id = ssh_keys[0].get('fid') or ssh_keys[0].get('id')
                    logger.info(f"Using existing SSH key: {ssh_key_id}")
                    return ssh_key_id
            
            # No SSH key exists, register one
            logger.info("No SSH key found, registering new one...")
            return self._register_ssh_key()
            
        except Exception as e:
            logger.error(f"Error getting SSH key ID: {e}")
            raise Exception(f"Could not get SSH key ID: {e}")
    
    def _register_ssh_key(self) -> str:
        """Register SSH key with ML Foundry."""
        # Read local SSH public key
        ssh_key_paths = [
            os.path.expanduser("~/.ssh/id_ed25519.pub"),
            os.path.expanduser("~/.ssh/id_rsa.pub"),
            os.path.expanduser("~/.ssh/id_ecdsa.pub")
        ]
        
        ssh_public_key = None
        for key_path in ssh_key_paths:
            if os.path.exists(key_path):
                try:
                    with open(key_path, 'r') as f:
                        ssh_public_key = f.read().strip()
                    logger.info(f"Found SSH key: {key_path}")
                    break
                except Exception as e:
                    logger.warning(f"Could not read SSH key {key_path}: {e}")
                    continue
        
        if not ssh_public_key:
            raise Exception("No SSH public key found")
        
        # Register SSH key with ML Foundry
        ssh_key_payload = {
            "project": self.project_id,
            "name": "sragv-training-key",
            "public_key": ssh_public_key
        }
        
        response = requests.post(
            f"{self.base_url}/ssh-keys",
            headers=self.headers,
            json=ssh_key_payload
        )
        
        if response.status_code in [200, 201]:
            key_data = response.json()
            ssh_key_id = key_data.get('fid') or key_data.get('id')
            logger.info(f"Registered SSH key: {ssh_key_id}")
            return ssh_key_id
        else:
            raise Exception(f"Failed to register SSH key: {response.status_code} - {response.text}")
    
    def _create_startup_script(self) -> str:
        """Create startup script that downloads code from GitHub."""
        
        script_parts = [
            "#!/bin/bash",
            "set -e",
            "",
            "# Install dependencies",
            "pip install torch>=2.0.0 transformers>=4.40.0 accelerate>=0.25.0 peft>=0.16.0",
            "pip install bitsandbytes>=0.41.0 datasets>=2.0.0 numpy>=1.24.0 pyyaml>=6.0",
            "pip install tqdm>=4.66.0 scikit-learn>=1.3.0 python-dotenv>=1.0.0",
            "",
            "# Clone the training code",
            "cd /workspace",
            "git clone https://github.com/ShreshthRajan/SRAG.git srag-training",
            "cd srag-training",
            "",
            "# Create logs and checkpoints directories",
            "mkdir -p logs checkpoints",
            "",
            "# Set environment variables",
            "export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True",
            "export TOKENIZERS_PARALLELISM=false",
            "export OMP_NUM_THREADS=8",
            "",
            "# Run training",
            "echo 'Starting SRAG-V Step 2 training...'",
            "python run_step2_training.py 2>&1 | tee logs/training.log",
            "",
            "echo 'Training completed successfully!'"
        ]
        
        return "\n".join(script_parts)
    
    def monitor_job(self, job_id: str) -> Dict[str, Any]:
        """Monitor spot bid/training job status."""
        # Try spot bid endpoint first (for bid_* IDs)
        if job_id.startswith('bid_'):
            response = requests.get(
                f"{self.base_url}/spot/bids/{job_id}",
                headers=self.headers
            )
        else:
            # Regular job endpoint
            response = requests.get(
                f"{self.base_url}/jobs/{job_id}",
                headers=self.headers
            )
        
        if response.status_code == 200:
            job_data = response.json()
            
            # Handle spot bid response format
            if job_id.startswith('bid_'):
                status = job_data.get('status', 'unknown')
                instance_status = job_data.get('instance_status', 'unknown')
                
                logger.info(f"Spot Bid {job_id} status: {status}")
                logger.info(f"Instance status: {instance_status}")
                
                # Check if instance is running
                if instance_status == 'running':
                    logger.info("🚀 Instance is running - training in progress!")
                elif instance_status == 'provisioning':
                    logger.info("⏳ Instance is being provisioned...")
                elif instance_status == 'completed':
                    logger.info("🎉 Training completed successfully!")
                elif instance_status == 'failed':
                    error_info = job_data.get('error', 'Unknown error')
                    logger.error(f"💥 Training failed: {error_info}")
                
                return job_data
            else:
                # Regular job monitoring
                status = job_data.get('status', 'unknown')
                logger.info(f"Job {job_id} status: {status}")
                
                if status == 'completed':
                    logger.info("🎉 Training completed successfully!")
                elif status == 'failed':
                    error_info = job_data.get('error', 'Unknown error')
                    logger.error(f"💥 Training failed: {error_info}")
                elif status == 'running':
                    progress = job_data.get('progress', {})
                    logger.info(f"⏳ Training in progress: {progress}")
                
                return job_data
        else:
            logger.error(f"Failed to get job status: {response.status_code} - {response.text}")
            return {"status": "unknown", "error": f"API error: {response.status_code}"}
    
    def _download_results(self, job_id: str):
        """Download training results."""
        logger.info("Downloading training results...")
        
        # Download logs
        logs_response = requests.get(
            f"{self.base_url}/jobs/{job_id}/logs",
            headers=self.headers
        )
        
        if logs_response.status_code == 200:
            logs_dir = Path("logs/foundry")
            logs_dir.mkdir(parents=True, exist_ok=True)
            
            with open(logs_dir / f"training_logs_{job_id}.txt", 'w') as f:
                f.write(logs_response.text)
            
            logger.info(f"✅ Logs saved to {logs_dir / f'training_logs_{job_id}.txt'}")
        
        # Download model checkpoints
        artifacts_response = requests.get(
            f"{self.base_url}/jobs/{job_id}/artifacts",
            headers=self.headers
        )
        
        if artifacts_response.status_code == 200:
            artifacts = artifacts_response.json()
            checkpoints_dir = Path("checkpoints/foundry")
            checkpoints_dir.mkdir(parents=True, exist_ok=True)
            
            for artifact in artifacts:
                if artifact['type'] == 'model_checkpoint':
                    # Download checkpoint
                    download_url = artifact['download_url']
                    checkpoint_response = requests.get(download_url)
                    
                    if checkpoint_response.status_code == 200:
                        checkpoint_path = checkpoints_dir / artifact['filename']
                        with open(checkpoint_path, 'wb') as f:
                            f.write(checkpoint_response.content)
                        
                        logger.info(f"✅ Downloaded checkpoint: {checkpoint_path}")
            
            logger.info("🎯 All training results downloaded successfully")
    
    def launch_step2_training(self) -> str:
        """Launch Step 2 training on ML Foundry."""
        logger.info("🚀 Launching SRAG-V Step 2 training on ML Foundry...")
        
        # Create optimal job configuration for Step 2
        job_config = FoundryJobConfig(
            gpu_type="L4",
            gpu_count=4,
            cpu_cores=16,
            memory_gb=64,
            max_runtime_hours=4,  # Conservative estimate
            auto_shutdown=True
        )
        
        # Create and launch job
        job_id = self.create_training_job(job_config)
        
        logger.info(f"🎯 Step 2 training launched successfully!")
        logger.info(f"Job ID: {job_id}")
        logger.info(f"Expected completion: ~3 hours")
        logger.info(f"Monitor progress with: monitor_job('{job_id}')")
        
        return job_id


def main():
    """Test ML Foundry integration."""
    try:
        client = MLFoundryClient()
        job_id = client.launch_step2_training()
        
        # Monitor job (example)
        while True:
            job_data = client.monitor_job(job_id)
            status = job_data.get('status', 'unknown')
            
            if status in ['completed', 'failed']:
                break
            
            time.sleep(60)  # Check every minute
            
    except Exception as e:
        logger.error(f"Training launch failed: {e}")
        raise


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()