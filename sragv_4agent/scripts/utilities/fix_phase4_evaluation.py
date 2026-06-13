#!/usr/bin/env python3
"""
Quick fix for Phase 4 evaluation method name issue.
"""

# Read the evaluation file and fix the method name
with open('run_phase4_comprehensive_evaluation.py', 'r') as f:
    content = f.read()

# Replace generate_solutions with generate
content = content.replace('generate_solutions', 'generate')

# Write back the fixed file
with open('run_phase4_comprehensive_evaluation_fixed.py', 'w') as f:
    f.write(content)

print("✅ Fixed method name: generate_solutions → generate")
print("✅ Created: run_phase4_comprehensive_evaluation_fixed.py")