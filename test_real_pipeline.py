import numpy as np
import torch
import unittest
import sys
import os

# Ensure we can import from local
sys.path.append(os.getcwd())

from datasets.buffer_dataset import BufferDataset

class TestBufferDataset(unittest.TestCase):
    def setUp(self):
        # Create Dummy Data
        self.B = 2
        self.T = 100
        self.D_base = 7 # Pos(3) + Quat(4)
        self.D_joint = 7
        self.D_obj = 7 
        
        # Create valid quaternions
        quats = np.zeros((self.B, self.T, 4))
        quats[:, :, 3] = 1.0 # w=1
        
        # Batched Dictionary
        # Note: FlexibleDataset usually expects (T, 3) for body_pos in raw, or (T, 1, 3)
        # BufferDataset handles (T, 1, 3) -> (T, 3) standard.
        self.dict_buffer = {
            'body_pos_w': np.random.randn(self.B, self.T, 1, 3).astype(np.float32), 
            'body_quat_w': np.random.randn(self.B, self.T, 1, 4).astype(np.float32), # w last
            'joint_pos': np.random.randn(self.B, self.T, self.D_joint).astype(np.float32),
            'object_pos_w': np.random.randn(self.B, self.T, 3).astype(np.float32),
            'object_quat_w': quats.astype(np.float32),
            'task_params': np.random.randn(self.B, 5).astype(np.float32) # (B, D)
        }
        
        # Ensure Quats are normalized roughly (not strictly needed for just running code)
        
        # List of Dictionaries
        self.list_buffer = []
        for i in range(self.B):
            d = {
                'body_pos_w': self.dict_buffer['body_pos_w'][i],
                'body_quat_w': self.dict_buffer['body_quat_w'][i],
                'joint_pos': self.dict_buffer['joint_pos'][i],
                'object_pos_w': self.dict_buffer['object_pos_w'][i],
                'object_quat_w': self.dict_buffer['object_quat_w'][i],
                'task_params': self.dict_buffer['task_params'][i]
            }
            self.list_buffer.append(d)
        
        # Minimal configuration matching default feature_order
        self.config = {
            "num_observations": 10, # Doesn't matter much for functionality check
            "num_features": 48, # Needs to match what features produce? 
                                # FlexibleDataset calculates total dim based on features. 
                                # This config value is used for splitting cond/target.
            "state_history": 2,
            "num_timesteps": 10,
            "downsample": 1,
            "stride": 1,
            "start_timestep": 0
            # "feature_order" uses default if not specified
        }

    def test_list_input(self):
        print("\n=== Testing List Input ===")
        print("Initializing BufferDataset with LIST of dicts...")
        ds = BufferDataset(self.list_buffer, self.config, calculate_stats=True)
        print(f"Dataset length: {len(ds)}")
        self.assertTrue(len(ds) > 0, "Dataset should not be empty")
        
        # Verify internal conversion
        self.assertIsInstance(ds.data_buffer, dict, "Internal buffer should be converted to dict")
        self.assertEqual(len(ds.data_buffer['body_pos_w']), self.B, "Should have B entries in converted list")
        
        # Access item
        print("Accessing Item 0...")
        item = ds[0]
        future, current, task, anchor = item
        print(f"Task Shape: {task.shape}")
        
        self.assertIsInstance(future, torch.Tensor)
        self.assertIsInstance(task, torch.Tensor)
        self.assertEqual(task.shape[0], 5, "Task params dimension mismatch")

    def test_dict_input(self):
        print("\n=== Testing Batched Dict Input ===")
        print("Initializing BufferDataset with BATCHED dict...")
        ds = BufferDataset(self.dict_buffer, self.config, calculate_stats=True)
        self.assertTrue(len(ds) > 0)
        
        # Access item to ensure indexing works
        item = ds[0]
        self.assertIsNotNone(item)
        print("Batch Dict access successful.")

    def test_external_task_params(self):
         print("\n=== Testing External Task Params ===")
         # Disable stats to check raw values
         ds = BufferDataset(self.dict_buffer, self.config, calculate_stats=False)
         
         # Get item 0, which corresponds to batch 0
         # Check indices to be sure
         idx_info = ds.indices[0] # (file_idx, batch_idx, t)
         batch_idx = idx_info[1]
         
         _, _, task_t, _ = ds[0]
         
         expected = self.dict_buffer['task_params'][batch_idx]
         
         # Note: BufferDataset._normalize is called even if calculate_stats=False, 
         # but if stats are empty, _normalize returns value as is.
         
         print(f"Expected: {expected}")
         print(f"Got: {task_t.numpy()}")
         
         self.assertTrue(np.allclose(task_t.numpy(), expected, atol=1e-5), "Task params values do not match external input")
         print("Task params match confirmed.")

    def test_speed_initialization(self):
        print("\n=== Testing Initialization Speed ===")
        import time
        start = time.time()
        ds = BufferDataset(self.dict_buffer, self.config, calculate_stats=True)
        end = time.time()
        print(f"Init time (B={self.B}, T={self.T}): {end-start:.4f}s")

if __name__ == '__main__':
    unittest.main()
