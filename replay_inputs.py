#!/usr/bin/env python3
"""
Script to replay saved actor inputs for debugging.
This loads dumped inputs and feeds them to a standalone actor instance.
"""

import os
import sys
import pickle
import torch
import argparse
from datetime import datetime

# Add OpenRLHF to Python path
sys.path.insert(0, '../OpenRLHF')

# Import the actor initialization function
from init_actor_no_ds import init_actor_standalone


def load_input_dumps(dump_file):
    with open(dump_file, 'rb') as f:
        data = pickle.load(f)
        data['source_file'] = dump_file
    return data


def replay_single_input(actor, dump_data, device, verbose=True):
    """Replay a single input dump through the actor"""
    
    if verbose:
        print(f"\n=== Replaying Input ===")
        print(f"Source: {os.path.basename(dump_data['source_file'])}")
        print(f"Timestamp: {dump_data.get('timestamp', 'unknown')}")
        print(f"Rank: {dump_data.get('rank', 'unknown')}")
        print(f"Batch size: {dump_data.get('batch_size', 'unknown')}")
        print(f"Sequence length: {dump_data.get('seq_len', 'unknown')}")
    
    # Extract input data
    sequences = dump_data['sequences']
    action_mask = dump_data['action_mask']
    attention_mask = dump_data['attention_mask']
    return_output = dump_data.get('return_output', False)
    allgather_logits = dump_data.get('allgather_logits', False)
    return_logprobs = dump_data.get('return_logprobs', False)
    packed_seq_lens = dump_data.get('packed_seq_lens', None)
    return_entropy = dump_data.get('return_entropy', False)

    # Move data to device
    if sequences is not None:
        sequences = sequences.to(device)
    if action_mask is not None:
        action_mask = action_mask.to(device)
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)
    
    # Run forward pass
    try:
        start_time = torch.cuda.synchronize() if torch.cuda.is_available() else None
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        import time
        cpu_start_time = time.time()
        
        with torch.no_grad():
            result = actor(
                sequences=sequences,
                action_mask=action_mask,
                attention_mask=attention_mask,
                return_output=return_output,
                allgather_logits=allgather_logits,
                return_logprobs=return_logprobs,
                packed_seq_lens=packed_seq_lens,
                return_entropy=return_entropy,
            )
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        cpu_end_time = time.time()
        forward_time = cpu_end_time - cpu_start_time
        
        return {
            'success': True,
            'result': result,
            'forward_time': forward_time,
            'input_info': {
                'batch_size': dump_data.get('batch_size', 'unknown'),
                'seq_len': dump_data.get('seq_len', 'unknown'),
                'rank': dump_data.get('rank', 'unknown'),
                'timestamp': dump_data.get('timestamp', 'unknown'),
            }
        }
        
    except Exception as e:
        if verbose:
            print(f"Error during forward pass: {e}")
            import traceback
            traceback.print_exc()
        
        return {
            'success': False,
            'error': str(e),
            'input_info': {
                'batch_size': dump_data.get('batch_size', 'unknown'),
                'seq_len': dump_data.get('seq_len', 'unknown'),
                'rank': dump_data.get('rank', 'unknown'),
                'timestamp': dump_data.get('timestamp', 'unknown'),
            }
        }


def replay_inputs(flash_attn=False, compile=False, packing=False):
    """Replay all available input dumps"""
    
   
    print(f"Starting at: {datetime.now()}")
    
    # Initialize actor
    print(f"\n1. Initializing actor ...")
    try:
        components = init_actor_standalone(
            no_flash_attn=not flash_attn,
            compile=compile,
            no_packing=not packing
        )
        actor = components['actor']
        device = torch.cuda.current_device() if torch.cuda.is_available() else 'cpu'
        print(f"Actor initialized successfully on device: {device}")
    except Exception as e:
        print(f"Failed to initialize actor: {e}")
        return False
    
    # Load input dumps with rank-specific shuffling
    print(f"\n2. Loading input dump ...")
    file = "input_dumps/actor_inputs_rank0_20250619_052658_128923.pkl"
    dump_data = load_input_dumps(file)
    
    # Replay each input
    print(f"\n3. Replaying an input ...")
    result = replay_single_input(actor, dump_data, device)
    
    if result['success']:
        print(f"Replay successful!")
    else:
        print(f"Replay failed with error: {result['error']}")
   
    

def main():
    parser = argparse.ArgumentParser(description="Replay actor inputs for debugging")
    parser.add_argument('--flash-attn', action='store_true', help="Enable Flash Attention")
    parser.add_argument('--compile', action='store_true', help="Enable PyTorch compilation (without DeepCompile)")
    parser.add_argument('--packing', action='store_true', help="Enable packing samples")
    
    args = parser.parse_args()
    
    replay_inputs(
        flash_attn=args.flash_attn, 
        compile=args.compile, 
        packing=args.packing,
    )


if __name__ == "__main__":
    import torch
    torch._dynamo.config.accumulated_cache_size_limit = 256
    torch._dynamo.config.cache_size_limit = 128
    torch._dynamo.config.verbose = True
    torch._dynamo.config.enable_compiler_collectives = True
    torch._dynamo.config.capture_scalar_outputs = True
    main()