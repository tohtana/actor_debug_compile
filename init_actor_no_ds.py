#!/usr/bin/env python3
"""
Standalone script to initialize only the actor module for debugging.
This replicates the actor initialization from OpenRLHF without Ray dependencies.
"""

import sys
import torch
import argparse
from transformers.trainer import get_scheduler

# Add OpenRLHF to Python path
sys.path.insert(0, '../OpenRLHF')

from openrlhf.models import Actor
from openrlhf.utils import get_tokenizer

def create_args(no_flash_attn=False, compile=False, no_packing=False, local_rank=-1):
    """Create args similar to the full training script"""
    args = argparse.Namespace()
    
    args.pretrain = "meta-llama/Llama-3.1-8B-Instruct"
    args.flash_attn = torch.cuda.is_available() and not no_flash_attn  # Use flash attention if CUDA available and not disabled
    args.bf16 = True
    args.load_in_4bit = False
    args.lora_rank = 0
    args.lora_alpha = 16
    args.target_modules = None
    args.lora_dropout = 0
    args.packing_samples = not no_packing  # Disable packing if no_packing specified
    args.temperature = 1.0
    args.use_liger_kernel = False
    args.compile = compile  # Enable PyTorch compilation if specified
    
    # Other configurations
    args.enable_ema = False
    args.micro_train_batch_size = 2
    args.eps_clip = 0.2
    args.ema_beta = 0.992
    args.disable_fast_tokenizer = False
    args.local_rank = local_rank
    args.seed = 42
    
    return args

def init_actor_standalone(no_flash_attn=False, compile=False, no_packing=False, local_rank=-1):
    """Initialize the actor model standalone without Ray"""
    
    print("Initializing actor model for debugging...")
    
    # Create args
    args = create_args(no_flash_attn, compile, no_packing, local_rank)
    
    print("Note: Running in standalone mode (no distributed training)")
    print(f"Using model: {args.pretrain}")
    print(f"Flash Attention: {args.flash_attn}")
    print(f"BF16: {args.bf16}")
    print(f"PyTorch Compile: {args.compile}")
    print(f"Packing Samples: {args.packing_samples}")
    
    # Initialize actor
    actor = Actor(
        args.pretrain,
        use_flash_attention_2=args.flash_attn,
        bf16=args.bf16,
        load_in_4bit=args.load_in_4bit,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        target_modules=args.target_modules,
        lora_dropout=args.lora_dropout,
        packing_samples=args.packing_samples,
        temperature=args.temperature,
        use_liger_kernel=args.use_liger_kernel,
    )
    
    print("Actor model initialized successfully!")
    print(f"Actor model parameters: {sum(p.numel() for p in actor.parameters())}")
    
    strategy = None

    # move actor to GPU if available
    if torch.cuda.is_available():
        device = torch.cuda.current_device()
        actor.to(device)
        print(f"Actor model moved to GPU: {device}")

    # Configure tokenizer
    tokenizer = get_tokenizer(
        args.pretrain, 
        actor.model, 
        "left", 
        strategy, 
        use_fast=not args.disable_fast_tokenizer
    )
    
    print(f"Tokenizer configured. Vocab size: {len(tokenizer)}")
    
    if args.compile:
        print("Enabling PyTorch compilation (without DeepSpeed DeepCompile)...")
        try:
            actor.compile()
            print("PyTorch compilation enabled successfully!")
            print("Note: DeepSpeed config has deepcompile: False, but engine.compile() was called")
        except Exception as e:
            print(f"Warning: Failed to enable PyTorch compilation: {e}")
    
    return {
        'actor': actor,
        'actor_optim': None, 
        'actor_scheduler': None,
        'tokenizer': tokenizer,
        'strategy': strategy,
        'args': args
    }
