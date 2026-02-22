import os
import torch
import torch.nn.functional as F
import numpy as np

class DiskTieredMemory:
    """
    Hardware-Native Memory Manager for Live Memory (THEN) - Portable Version.
    
    Instead of holding $O(N)$ traces in VRAM as python lists, this manager streams
    traces to an unbuffered memory-mapped file `state.dat` on NVMe/SSD.
    
    This allows Ingestion to build massive context, and Query loops to perform
    batch-streamed attention over the full history without OOMs.
    """
    def __init__(self, filepath: str, max_traces: int = 100_000, d_model: int = 768, dtype=np.float32, device: str = "cuda"):
        self.filepath = filepath
        self.max_traces = max_traces
        self.d_model = d_model
        self.dtype = dtype
        self.device = device
        
        # Determine internal tracking
        self.head = 0  # Number of traces written so far
        self.shape = (max_traces, d_model)
        
        # Create or attach to memmap
        if not os.path.exists(filepath):
            # Explicitly create an empty file of exact size before memmapping
            with open(filepath, "wb") as f:
                f.seek((max_traces * d_model * np.dtype(dtype).itemsize) - 1)
                f.write(b'\0')
        
        # Load memmap
        self.memmap = np.memmap(filepath, dtype=dtype, mode='r+', shape=self.shape)
        
    def append(self, trace: torch.Tensor):
        """
        Append a single trace or batch of traces (B, T, D) safely to disk.
        """
        assert trace.size(-1) == self.d_model, f"Expected D={self.d_model}, got {trace.size(-1)}"
        
        # Flatten time and batch if necessary. Assuming (B, D) or (B, T, D)
        if trace.dim() == 3:
            trace = trace.reshape(-1, self.d_model)
        elif trace.dim() == 2:
            pass
        elif trace.dim() == 1:
            trace = trace.unsqueeze(0)
            
        num_new = trace.size(0)
        if self.head + num_new > self.max_traces:
            print(f"Warning: Memory full (head={self.head}, max={self.max_traces}). Dropping traces.")
            num_new = self.max_traces - self.head
            if num_new <= 0: return
            trace = trace[:num_new]
            
        # Move back to CPU numpy and write straight into memmap
        trace_np = trace.detach().cpu().to(torch.float32).numpy()
        self.memmap[self.head : self.head + num_new] = trace_np
        self.head += num_new

    def retrieve(self, query: torch.Tensor, top_k: int = 64, chunk_size: int = 4096) -> torch.Tensor:
        """
        Retrieve context vector by streaming attention across disk.
        Query shape: (B, T, D) -> returns Context vector (B, T, D).
        """
        if self.head == 0:
            return torch.zeros_like(query)
            
        B, T, D = query.shape
        # To avoid enormous internal memory, we iterate through disk chunks
        # keeping a running top-k score.
        
        # We need to maintain the highest top_k (score, index) pairs for each token.
        # Shape: (B, T, top_k)
        global_top_scores = torch.full((B, T, top_k), float('-inf'), device=self.device)
        global_top_indices = torch.zeros((B, T, top_k), dtype=torch.long, device=self.device)
        
        # Stream from disk
        for start_idx in range(0, self.head, chunk_size):
            end_idx = min(start_idx + chunk_size, self.head)
            
            # Load chunk into VRAM (zero-copy overhead from OS page cache)
            chunk_np = self.memmap[start_idx:end_idx] # (chunk, D)
            chunk_pt = torch.from_numpy(chunk_np).to(self.device).to(query.dtype)
            
            # Score: Query (B, T, D) @ Chunk.T (D, chunk) -> (B, T, chunk)
            # Dividing by sqrt(D) for scaled dot product 
            scores = torch.matmul(query, chunk_pt.transpose(0, 1)) / (self.d_model ** 0.5)
            
            # Get local top-k from this chunk
            k_local = min(top_k, scores.size(-1))
            local_top_scores, local_top_indices = torch.topk(scores, k_local, dim=-1)
            
            # Map local indices back to absolute disk offsets
            local_top_indices_abs = local_top_indices + start_idx
            
            # Merge with global running top-k
            # Concat global (B, T, top_k) and local (B, T, chunk_top_k) -> (B, T, top_k + chunk_top_k)
            merged_scores = torch.cat([global_top_scores, local_top_scores], dim=-1)
            merged_indices = torch.cat([global_top_indices, local_top_indices_abs], dim=-1)
            
            # Take top-k of merged
            global_top_scores, sort_idx = torch.topk(merged_scores, top_k, dim=-1)
            # Gather corresponding absolute indices
            global_top_indices = merged_indices.gather(-1, sort_idx)

        # We now have the absolute highest Top-K indices across the entire disk length.
        # We need to load *only* those specific value vectors to finalize the attention sum.
        
        # Flatten indices to unique set to minimizing redundant disk reads
        flat_indices = global_top_indices.reshape(-1).unique().cpu().numpy()
        
        # Load only exact required lines from memmap 
        required_values_np = self.memmap[flat_indices] 
        required_values_pt = torch.from_numpy(required_values_np).to(device=self.device, dtype=query.dtype) # (U, D)
        
        # Create a mapping from absolute index -> local (U, D) tensor index
        abs_to_local_map = torch.zeros(self.head, dtype=torch.long, device=self.device)
        abs_to_local_map[torch.from_numpy(flat_indices).to(self.device)] = torch.arange(len(flat_indices), device=self.device)
        
        # Remap our global_top_indices to index into the small required_values_pt matrix
        mapped_val_indices = abs_to_local_map[global_top_indices] # (B, T, top_k)
        
        # Gather final values: shape (B, T, top_k, D)
        final_values = required_values_pt[mapped_val_indices] 
        
        # Apply Softmax to scores
        attn_weights = F.softmax(global_top_scores, dim=-1) # (B, T, top_k)
        
        # Weighted sum: (B, T, 1, top_k) @ (B, T, top_k, D) -> (B, T, 1, D) -> (B, T, D)
        context = torch.matmul(attn_weights.unsqueeze(-2), final_values).squeeze(-2)
        
        return context

    def save(self):
        """Memmap flushes automatically, but explicitly calling ensures os sync."""
        self.memmap.flush()

    def reload(self):
        """Updates internal views if file changed externally."""
        self.memmap = np.memmap(self.filepath, dtype=self.dtype, mode='r+', shape=self.shape)
