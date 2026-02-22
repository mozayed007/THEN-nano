import torch
import torch.nn as nn
import torch.nn.functional as F

class HybridTHENAttention(nn.Module):
    """
    Standalone Hybrid THEN Attention module.
    Implements Knowledge Distillation Attention (KDA) for compression 
    and Dense Sparse Attention (DSA) for retrieval.
    
    Can be injected into any Transformer layer via PyTorch forward hooks.
    """
    def __init__(self, d_model, n_heads, config=None, ratio=3, chunk_size=16):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.ratio = ratio
        self.chunk_size = chunk_size
        
        # In a real deployed system, these get loaded from a custom checkpoint 
        # or adapter weights. Initialize randomly for demonstration.
        self.kda = nn.Linear(d_model, d_model, bias=False)  # Sim KDA compression
        self.dsa = nn.Linear(d_model, d_model, bias=False)  # Sim DSA retrieval

    def forward(self, x, memory_state, layer_idx=0):
        """
        Applies KDA or DSA logic to the hidden states `x` (B, T, D)
        using the shared `memory_state` dictionary.
        """
        # Ensure buffer exists
        if 'buffer' not in memory_state:
            memory_state['buffer'] = torch.empty(x.size(0), 0, x.size(2), device=x.device, dtype=x.dtype)
            
        if layer_idx % (self.ratio + 1) < self.ratio:
            # --- KDA: Compress/encode trace ---
            compressed = self.kda(x)
            
            # Append current compressed tokens to buffer
            memory_state['buffer'] = torch.cat([memory_state['buffer'], compressed], dim=1)
            
            # Flush buffer to traces when a chunk is full
            while memory_state['buffer'].size(1) >= self.chunk_size:
                chunk = memory_state['buffer'][:, :self.chunk_size, :]
                trace = torch.mean(chunk, dim=1)  # (B, D)
                
                # Interface with DiskTieredMemory or normal list
                if 'memory_manager' in memory_state:
                    memory_state['memory_manager'].append(trace)
                elif 'traces' in memory_state:
                    memory_state['traces'].append(trace)
                    
                # Slide buffer
                memory_state['buffer'] = memory_state['buffer'][:, self.chunk_size:, :]
                
            return compressed
            
        else:
            # --- DSA: Retrieve/abstract semantics ---
            if 'memory_manager' in memory_state and memory_state['memory_manager'].head > 0:
                q_mem = self.dsa(x)
                attn_out = memory_state['memory_manager'].retrieve(q_mem)
                fused = x + attn_out
                return fused
                
            elif 'traces' in memory_state and len(memory_state['traces']) > 0:
                memory = torch.stack(memory_state['traces'], dim=1)
                q_mem = self.dsa(x)
                attn_out = F.scaled_dot_product_attention(q_mem, memory, memory)
                fused = x + attn_out
                return fused
                
            else:
                return x


def install_memory_hooks(model, memory_state, d_model, n_heads, hook_target_modules, ratio=3, chunk_size=16):
    """
    Injects THEN attention logic into standard transformer layers.
    
    Args:
        model: HuggingFace AutoModelForCausalLM (or similar)
        memory_state (dict): Shared dict containing 'memory_manager' and 'buffer'.
        d_model (int): Hidden dimension size.
        n_heads (int): Number of attention heads.
        hook_target_modules (list of nn.Module): The layers to attach hooks to 
            (e.g., `model.model.layers`).
    """
    # Create the shared attention module (in reality, each layer might have its own KDA/DSA,
    # or they are tied. Here we instantiate one tied module for simplicity or you can 
    # adjust the logic to have an nn.ModuleList).
    # For now, we create a ModuleList so each layer has its own projection weights.
    num_layers = len(hook_target_modules)
    then_modules = nn.ModuleList([
        HybridTHENAttention(d_model, n_heads, ratio=ratio, chunk_size=chunk_size).to(
            next(hook_target_modules[0].parameters()).device, 
            dtype=next(hook_target_modules[0].parameters()).dtype
        )
        for _ in range(num_layers)
    ])
    
    # Store them on the model so they get saved/loaded alongside it
    model.then_modules = then_modules

    hook_handles = []

    for i, target_layer in enumerate(hook_target_modules):
        # We define a closure that binds `i` to the current layer index
        def make_hook(layer_idx):
            def forward_hook(module, args, kwargs, output):
                # The output from a standard HF Decoder layer is a tuple.
                # output[0] is the hidden_states: (B, T, D)
                hidden_states = output[0] if isinstance(output, tuple) else output
                
                # Apply our memory logic
                new_hidden = model.then_modules[layer_idx](hidden_states, memory_state, layer_idx=layer_idx)
                
                # Re-pack the tuple if necessary
                if isinstance(output, tuple):
                    return (new_hidden,) + output[1:]
                return new_hidden
                
            return forward_hook
            
        handle = target_layer.register_forward_hook(make_hook(i), with_kwargs=True)
        hook_handles.append(handle)
        
    return hook_handles
