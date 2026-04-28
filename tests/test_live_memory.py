import os
import tempfile
import torch
import unittest
from nanochat.gpt import GPTConfig, THENGPT
from nanochat.memory_manager import DiskTieredMemory
from nanochat.common import COMPUTE_DTYPE

class TestLiveMemoryFlow(unittest.TestCase):
    def setUp(self):
        self.device = "cpu"
        self.config = GPTConfig(
            n_layer=4,
            n_head=4,
            n_kv_head=4,
            n_embd=64,
            vocab_size=100, # Small vocab
            sequence_len=32
        )
        self.model = THENGPT(self.config)
        self.model.to(self.device)
        self.model.cos = self.model.cos.to(COMPUTE_DTYPE)
        self.model.sin = self.model.sin.to(COMPUTE_DTYPE)
        self.model.eval()

    def test_ingest_state_growth(self):
        """Test that ingesting data increases the memory state traces."""
        # 1. Create dummy input
        input_ids = torch.randint(0, 100, (1, 16)).to(self.device)
        
        # 2. Ingest
        state = None
        with torch.inference_mode():
            _, state = self.model(input_ids, state=state, return_state=True)
            
        # 3. Check state
        self.assertIsNotNone(state)
        self.assertIn('traces', state)
        # traces are appended per layer based on ratio. 
        # In THENGPT default, ratio=3. n_layer=4.
        # layer 0: 0 % 4 = 0 < 3 -> KDA (Compress) -> Appends trace
        # layer 1: 1 % 4 = 1 < 3 -> KDA (Compress) -> Appends trace
        # layer 2: 2 % 4 = 2 < 3 -> KDA (Compress) -> Appends trace
        # layer 3: 3 % 4 = 3 < 3 is false -> retrieval branch executes
        # So we expect traces and at least one retrieval-capable layer in the stack.
        self.assertTrue(len(state['traces']) > 0)
        print(f"Traces captured: {len(state['traces'])}")

    def test_query_uses_state(self):
        """Test that query accepts state and doesn't crash."""
        # 1. Create dummy state
        state = {'traces': [torch.randn(1, 64, device=self.device, dtype=COMPUTE_DTYPE) for _ in range(5)]}
        
        # 2. Query
        input_ids = torch.randint(0, 100, (1, 5)).to(self.device)
        with torch.inference_mode():
            logits, new_state = self.model(input_ids, state=state, return_state=True)
            
        # 3. Check output
        self.assertEqual(logits.shape, (1, 5, 100))
        self.assertGreaterEqual(len(new_state['traces']), 5)
        self.assertIn('buffer', new_state)
        self.assertGreater(new_state['buffer'].shape[1], 0)

    def test_disk_tiered_memory_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "state.dat")
            memory = DiskTieredMemory(filepath=path, max_traces=8, d_model=64, device=self.device)
            trace = torch.zeros(1, 64, dtype=torch.float32)
            trace[0, 7] = 1.0
            memory.append(trace)
            memory.save()

            reloaded = DiskTieredMemory(filepath=path, max_traces=8, d_model=64, device=self.device)
            self.assertEqual(reloaded.head, 1)

            query = trace.unsqueeze(1)
            context = reloaded.retrieve(query, top_k=1, chunk_size=1)

            self.assertEqual(context.shape, query.shape)
            self.assertTrue(torch.allclose(context.squeeze(1), trace, atol=1e-5, rtol=1e-4))

    def test_save_load_consistency(self):
        """Test that state can be saved and loaded identically."""
        # 1. Generate state
        state = {'traces': [torch.randn(1, 64) for _ in range(3)]}
        
        # 2. Save
        path = "temp_test_state.pt"
        torch.save(state, path)
        
        # 3. Load
        loaded_state = torch.load(path)
        
        # 4. Compare
        self.assertEqual(len(state['traces']), len(loaded_state['traces']))
        self.assertTrue(torch.allclose(state['traces'][0], loaded_state['traces'][0]))
        
        # Cleanup
        if os.path.exists(path):
            os.remove(path)

if __name__ == "__main__":
    unittest.main()
