"""
Ingestion Script for Live Memory (Phase 2)

Goal: Feed episodic data into a frozen model to populate its memory state.
Usage: python -m scripts.ingest --model_path outputs/d8/model_000100.pt --data_path data/synthetic-cairo-episodes.txt
"""

import os
import torch
import argparse
from nanochat.tokenizer import get_tokenizer
from nanochat.checkpoint_manager import load_checkpoint, build_model
from nanochat.memory_manager import DiskTieredMemory
from nanochat.common import get_base_dir, print0

def ingest(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print0(f"Ingesting on {device}...")

    # 1. Load Model (Frozen)
    print0(f"Loading model from {args.model_path}...")
    checkpoint_dir = os.path.dirname(args.model_path)
    step = int(os.path.basename(args.model_path).split('_')[1].split('.')[0])
    
    # We use build_model to handle config patching and initialization
    # Note: Phase='eval' ensures dropout is off, etc.
    model, tokenizer, meta_data = build_model(checkpoint_dir, step, torch.device(device), phase="eval")
    model.eval() # Double check

    # 2. Prepare Data
    print0(f"Reading data from {args.data_path}...")
    with open(args.data_path, 'r', encoding='utf-8') as f:
        text = f.read()
    
    # Tokenize
    tokens = tokenizer.encode(text, prepend="<|bos|>")
    tokens = torch.tensor(tokens, dtype=torch.long, device=device).unsqueeze(0) # (1, T)
    
    # 3. Ingest Loop
    chunk_size = args.chunk_size
    seq_len = tokens.size(1)
    
    # Initialize hardware-native memory state
    output_path = args.output_path or os.path.join(get_base_dir(), "cairo_memory_state.dat")
    state = {
        'memory_manager': DiskTieredMemory(filepath=output_path, d_model=model.config.n_embd, device=device)
    }
    
    print0(f"Ingesting {seq_len} tokens in chunks of {chunk_size}...")
    
    with torch.inference_mode():
        for i in range(0, seq_len, chunk_size):
            chunk = tokens[:, i:i+chunk_size]
            _, state = model(chunk, state=state, return_state=True)
            print0(f"Processed chunk {i}-{min(i+chunk_size, seq_len)} | Stored traces: {state['memory_manager'].head}")

    # 4. Save State
    state['memory_manager'].save()
    
    # We strip the memory manager out of the state dict and just save the leftover buffer
    buffer_state = {'buffer': state.get('buffer', None)}
    buffer_path = output_path.replace(".dat", "_buffer.pt")
    torch.save(buffer_state, buffer_path)
    
    print0(f"Disk memory stream saved to {output_path}")
    print0(f"Leftover buffer state saved to {buffer_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True, help="Path to pretrained model checkpoint")
    parser.add_argument("--data_path", type=str, required=True, help="Path to text file to ingest")
    parser.add_argument("--output_path", type=str, default=None, help="Where to save the memory state")
    parser.add_argument("--chunk_size", type=int, default=512, help="Context window size for ingestion chunks")
    args = parser.parse_args()
    
    ingest(args)
