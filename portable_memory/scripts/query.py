"""
Generic Query Script using the Portable Memory Architecture.

Usage: 
python -m portable_memory.scripts.query \
    --model_name "Qwen/Qwen2.5-0.5B" \
    --state_path portable_memory_state.dat
"""

import os
import torch
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer
from portable_memory.memory_manager import DiskTieredMemory, adapter_sidecar_path, buffer_sidecar_path, require_dat_path
from portable_memory.model_wrapper import InferenceEngine

def _legacy_buffer_path(state_path):
    return state_path.replace(".dat", "_buffer.pt")

def _first_existing_buffer_path(state_path):
    candidates = [buffer_sidecar_path(state_path), _legacy_buffer_path(state_path)]
    return next((path for path in candidates if os.path.exists(path)), candidates[0])

def get_target_layers(model):
    """Helper to detect the target layers across standard HF models."""
    if hasattr(model, 'model') and hasattr(model.model, 'layers'):
        return model.model.layers
    elif hasattr(model, 'transformer') and hasattr(model.transformer, 'h'):
        return model.transformer.h
    raise ValueError("Could not automatically identify Decoder layers in model structure.")

def query(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 1. Load Model via HuggingFace
    print(f"Loading HuggingFace model {args.model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name, 
        device_map=device, 
        torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32
    )
    
    # 2. Load Memory State
    print(f"Loading memory stream from {args.state_path}...")
    require_dat_path(args.state_path)
    memory_manager = DiskTieredMemory(filepath=args.state_path, max_traces=100_000, d_model=model.config.hidden_size, device=device)
    print(f"Loaded disk state with {memory_manager.head} traces.")

    try:
        # 3. Create InferenceEngine Wrapper
        # Instead of manual PREFILL vs DECODE logic in the script, the wrapper handles it.
        engine = InferenceEngine(model, tokenizer, memory_manager, get_target_layers)
        adapter_path = adapter_sidecar_path(args.state_path)
        if not os.path.exists(adapter_path):
            raise FileNotFoundError(
                f"Missing portable memory adapter sidecar: {adapter_path}. "
                "Re-run portable_memory.scripts.ingest so query uses the same KDA/DSA projections as ingest."
            )
        adapter_state = torch.load(adapter_path, map_location=device)
        model.then_modules.load_state_dict(adapter_state)
        buffer_path = _first_existing_buffer_path(args.state_path)
        if os.path.exists(buffer_path):
            buffer_state = torch.load(buffer_path, map_location=device)
            buffer = buffer_state.get('buffer', None)
            if buffer is not None:
                engine.memory_state['buffer'] = buffer.to(device=device, dtype=model.dtype)
        print(f"Loaded portable memory adapter from {adapter_path}")

        # 4. Interactive Loop
        print("\nReady for queries! (Type 'exit' to quit)")
        print("-" * 50)

        while True:
            user_input = input("User: ")
            if user_input.lower() in ["exit", "quit"]:
                break

            print("Assistant: ", end="", flush=True)

            # Stream the generation
            for token_text in engine.generate(user_input, max_new_tokens=args.max_new_tokens):
                print(token_text, end="", flush=True)

            print("\n")
    finally:
        memory_manager.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True, help="HuggingFace Model Hub ID")
    parser.add_argument("--state_path", type=str, required=True, help="Path to memory state file (.dat)")
    parser.add_argument("--max_new_tokens", type=int, default=50, help="Max tokens to generate")
    args = parser.parse_args()
    
    query(args)
