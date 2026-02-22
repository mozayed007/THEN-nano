import torch
from .attention_hooks import install_memory_hooks

class InferenceEngine:
    """
    Generic wrapper for HuggingFace models (or custom models) to handle 
    LiveMem's DiskTieredMemory and the Prefill vs Decode split state machine.
    """
    def __init__(self, model, tokenizer, memory_manager, get_layers_fn):
        """
        Args:
            model: HuggingFace model (e.g. from AutoModelForCausalLM). 
            tokenizer: HuggingFace tokenizer.
            memory_manager: An instance of DiskTieredMemory.
            get_layers_fn (callable): A function that takes the `model` and returns 
                an iterable of layer modules to hook into. 
                e.g., `lambda m: m.model.layers` for Llama.
        """
        self.device = next(model.parameters()).device
        self.model = model
        self.tokenizer = tokenizer
        self.memory_state = {
            'memory_manager': memory_manager,
            'buffer': torch.empty(1, 0, model.config.hidden_size, device=self.device, dtype=model.dtype)
        }
        
        # Install the PyTorch hooks into the residual stream
        layers = get_layers_fn(model)
        d_model = model.config.hidden_size
        n_heads = model.config.num_attention_heads
        
        print("Injecting THEN generic memory hooks into model layers...")
        self.handles = install_memory_hooks(
            model=model, 
            memory_state=self.memory_state, 
            d_model=d_model, 
            n_heads=n_heads, 
            hook_target_modules=layers
        )

    def generate(self, prompt: str, max_new_tokens: int = 50, temperature: float = 1.0, top_k: int = 50):
        """
        Autoregressive generation resolving the $O(T^2)$ duplicate ingestion bug.
        By cleanly splitting into PREFILL (full context with past_key_values lookup) 
        and DECODE (step-by-step token generation using KV cache).
        """
        self.model.eval()
        
        # 1. Tokenize prompt
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        input_ids = inputs["input_ids"]
        
        # 2. --- PREFILL PHASE ---
        # Run prompt through the model to extract the initial kv_cache.
        # This inherently populates the DiskTieredMemory buffer through the forward_hooks.
        with torch.inference_mode():
            outputs = self.model(input_ids, use_cache=True)
            past_key_values = outputs.past_key_values
            next_token_logits = outputs.logits[:, -1, :]
            
            # Sample next token
            next_token = self._sample(next_token_logits, temperature, top_k)
            yield self.tokenizer.decode(next_token.tolist())
            
            curr_token = next_token.unsqueeze(-1)
            
            # 3. --- DECODE PHASE ---
            for _ in range(max_new_tokens - 1):
                # Pass only the newly generated token + past KV cache
                outputs = self.model(curr_token, past_key_values=past_key_values, use_cache=True)
                past_key_values = outputs.past_key_values
                next_token_logits = outputs.logits[:, -1, :]
                
                next_token = self._sample(next_token_logits, temperature, top_k)
                yield self.tokenizer.decode(next_token.tolist())
                
                curr_token = next_token.unsqueeze(-1)
                if next_token.item() == self.tokenizer.eos_token_id:
                    break
                    
    def _sample(self, logits, temperature, top_k):
        """Helper to sample token from logits."""
        if top_k is not None and top_k > 0:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = -float('Inf')
            
        if temperature > 0:
            logits = logits / temperature
            probs = torch.nn.functional.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1).squeeze(-1)
        else:
            next_token = torch.argmax(logits, dim=-1)
        return next_token
