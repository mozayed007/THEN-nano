"""
Generic Ingestion Script using the Portable Memory Architecture.

Usage: 
python -m portable_memory.scripts.ingest \
    --model_name "Qwen/Qwen2.5-0.5B" \
    --data_path data/episodes.txt
"""

import os
import torch
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer
from portable_memory.memory_manager import DiskTieredMemory
from portable_memory.attention_hooks import install_memory_hooks

def ingest(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Ingesting on {device}...")

    # 1. Load Model (Frozen) using HuggingFace
    print(f"Loading HuggingFace model {args.model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name, 
        device_map=device, 
        torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32
    )
    model.eval()
    
    # 2. Prepare Data
    print(f"Reading data from {args.data_path}...")
    with open(args.data_path, 'r', encoding='utf-8') as f:
        text = f.read()
    
    # Tokenize
    tokens = tokenizer(text, return_tensors="pt")["input_ids"].to(device)
    
    # 3. Inject Portable Hooks
    # Determine the target layers based on typical HuggingFace structure
    # For instance, models like Llama, Qwen, Mistral usually expose `model.model.layers`
    if hasattr(model, 'model') and hasattr(model.model, 'layers'):
        decoder_layers = model.model.layers
    elif hasattr(model, 'transformer') and hasattr(model.transformer, 'h'):
        # GPT-2 / GPT-Neo like
        decoder_layers = model.transformer.h
    else:
        raise ValueError("Could not automatically identify Decoder layers in model structure.")
        
    output_path = args.output_path or "portable_memory_state.dat"
    memory_state = {
        'memory_manager': DiskTieredMemory(filepath=output_path, d_model=model.config.hidden_size, device=device),
        'buffer': torch.empty(1, 0, model.config.hidden_size, device=device, dtype=model.dtype)
    }
    
    print(f"Injecting LiveMem hooks into {len(decoder_layers)} layers...")
    handles = install_memory_hooks(
        model=model, 
        memory_state=memory_state, 
        d_model=model.config.hidden_size, 
        n_heads=model.config.num_attention_heads, 
        hook_target_modules=decoder_layers
    )

    # 4. Ingest Loop
    chunk_size = args.chunk_size
    seq_len = tokens.size(1)
    
    print(f"Ingesting {seq_len} tokens in chunks of {chunk_size}...")
    
    with torch.inference_mode():
        for i in range(0, seq_len, chunk_size):
            chunk = tokens[:, i:i+chunk_size]
            _ = model(chunk)
            print(f"Processed chunk {i}-{min(i+chunk_size, seq_len)} | Stored traces: {memory_state['memory_manager'].head}")

    # 5. Save State
    memory_state['memory_manager'].save()
    
    # We strip the memory manager out of the state dict and just save the leftover buffer
    buffer_state = {'buffer': memory_state.get('buffer', None)}
    buffer_path = output_path.replace(".dat", "_buffer.pt")
    torch.save(buffer_state, buffer_path)
    
    print(f"Disk memory stream saved to {output_path}")
    print(f"Leftover buffer state saved to {buffer_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True, help="HuggingFace Model Hub ID")
    parser.add_argument("--data_path", type=str, required=True, help="Path to text file to ingest")
    parser.add_argument("--output_path", type=str, default=None, help="Where to save the memory state (.dat)")
    parser.add_argument("--chunk_size", type=int, default=512, help="Context window size for ingestion chunks")
    args = parser.parse_args()
    
    ingest(args)
