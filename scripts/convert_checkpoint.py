#!/usr/bin/env python3
"""Convert XLA model to CPU checkpoint"""
import torch
import os
os.environ['TPU_CHIPS_PER_PROCESS_BOUNDS'] = '1,1,1'
os.environ['TPU_PROCESS_BOUNDS'] = '1,1,1'

print('Loading XLA checkpoint...')
state_dict = torch.load('/tmp/transformer_v7_best.pt')

print('Converting to CPU...')
cpu_state = {k: v.cpu() for k, v in state_dict.items()}

print('Saving CPU checkpoint...')
torch.save(cpu_state, '/tmp/transformer_v7_cpu.pt')
print('Done! Saved to /tmp/transformer_v7_cpu.pt')

# Also upload to GCS
import subprocess
subprocess.run(['gsutil', 'cp', '/tmp/transformer_v7_cpu.pt', 
                'gs://cift-data-united-option-388113/models/'])
print('Uploaded to GCS')
