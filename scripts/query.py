"""
Query Script for Live Memory (Phase 3)

Goal: Query a frozen model using a pre-populated memory state.
Usage: python -m scripts.query --model_path outputs/d8/model_000100.pt --state_path cairo_memory_state.pt
"""

import os
import torch
import argparse
from nanochat.tokenizer import get_tokenizer
from nanochat.checkpoint_manager import build_model
from nanochat.memory_manager import DiskTieredMemory
from nanochat.engine import KVCache
from nanochat.common import print0

def query(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 1. Load Model
    print0(f"Loading model from {args.model_path}...")
    checkpoint_dir = os.path.dirname(args.model_path)
    step = int(os.path.basename(args.model_path).split('_')[1].split('.')[0])
    model, tokenizer, _ = build_model(checkpoint_dir, step, torch.device(device), phase="eval")
    model.eval()

    # 2. Load State
    print0(f"Loading memory stream from {args.state_path}...")
    buffer_path = args.state_path.replace(".dat", "_buffer.pt")
    
    state = {}
    if os.path.exists(buffer_path):
        buffer_state = torch.load(buffer_path, map_location=device)
        state['buffer'] = buffer_state.get('buffer', None)
        
    state['memory_manager'] = DiskTieredMemory(filepath=args.state_path, max_traces=100000, d_model=model.config.n_embd, device=device)
    print0(f"Loaded disk state with {state['memory_manager'].head} traces.")

    # 3. Interactive Loop
    print0("\nReady for queries! (Type 'exit' to quit)")
    print0("-" * 50)
    
    m = model.config
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    kv_model_kwargs = {"num_heads": m.n_kv_head, "head_dim": m.n_embd // m.n_head, "num_layers": m.n_layer}
    
    while True:
        user_input = input("User: ")
        if user_input.lower() in ["exit", "quit"]:
            break
            
        tokens = tokenizer.encode(user_input, prepend="<|bos|>")
        tokens = torch.tensor(tokens, dtype=torch.long, device=device).unsqueeze(0)
        
        print0("Assistant: ", end="", flush=True)
        
        with torch.inference_mode():
            # Create fresh KV cache for this turn
            kv_cache = KVCache(
                batch_size=1,
                seq_len=m.sequence_len,
                device=device,
                dtype=dtype,
                **kv_model_kwargs,
            )
            
            # --- PREFILL ---
            # Process the whole prompt to populate KV cache and memory state buffer
            logits, state = model(tokens, state=state, kv_cache=kv_cache, return_state=True)
            next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            
            decoded = tokenizer.decode(next_token[0].tolist())
            print(decoded, end="", flush=True)
            
            # --- DECODE ---
            # Autoregressive generation using only 1 token per pass 
            # Fixes the O(T^2) duplicate token ingestion bug
            curr_token = next_token
            for _ in range(args.max_new_tokens - 1):
                logits, state = model(curr_token, state=state, kv_cache=kv_cache, return_state=True)
                next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
                
                decoded = tokenizer.decode(next_token[0].tolist())
                print(decoded, end="", flush=True)
                
                curr_token = next_token
                
                if next_token.item() == tokenizer.eot_token_id:
                    break
                    
        print("\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True, help="Path to pretrained model checkpoint")
    parser.add_argument("--state_path", type=str, required=True, help="Path to memory state file")
    parser.add_argument("--max_new_tokens", type=int, default=50, help="Max tokens to generate")
    args = parser.parse_args()
    
    query(args)
