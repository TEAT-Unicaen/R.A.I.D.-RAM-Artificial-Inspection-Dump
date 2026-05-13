import json
import torch
from torch.utils.data import Dataset
import mmap
import numpy as np

import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import config as cfg

class RamDumpDataset(Dataset):
    def __init__(
        self,
        bin_path,
        meta_path,
        chunk_size=cfg.DEFAULT_CHUNK_SIZE,
        offset=cfg.DEFAULT_DATASET_OFFSET,
    ):
        self.bin_path = bin_path
        self.chunk_size = chunk_size
        self.offset = min(offset, chunk_size)
        self.samples = []
        self.ram_data = None
        self.f = None  # Keep file handle alive to prevent mmap invalidation
        
        with open(meta_path, 'r') as f:
            self.metadata = json.load(f)
            
        self.label_map = {
            "ENCRYPTED": 1,
            "COMPRESSED": 0,
            "BINARY_TEXT": 0,
            "BINARY_IMAGE": 0,
            "BINARY_OTHER": 0,
            "BINARY_PDF": 0,
            "BASE64": 0,
            "DECODED": 0,
            "SYSTEM": 0,
            "NOISE": 0
        }
        self._prepare_samples()
        self._build_full_label_mask()

    def _prepare_samples(self):
        """Generate sample start positions (O(n) once at init)."""
        bin_size = self.metadata[-1]['de']
        current_pos = 0

        while current_pos + self.chunk_size <= bin_size:
            self.samples.append(current_pos)
            current_pos += self.offset

    def _build_full_label_mask(self):
        """Pre-compute label mask for entire file (O(1) lookup in __getitem__)."""
        bin_size = self.metadata[-1]['de']
        print(f"Building full label mask ({bin_size / 1e6:.1f}MB)...", flush=True)
        
        # Initialize with -1 (padding/unlabeled)
        self.full_label_mask = np.full(bin_size, -1, dtype=np.float32)
        
        # Fill in labels from metadata in single pass
        for entry in self.metadata:
            label_val = self.label_map.get(entry['t'], 0)
            start, end = entry['ds'], entry['de']
            self.full_label_mask[start:end] = label_val
        
        print(f"Label mask built successfully.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        if self.ram_data is None:
            self.f = open(self.bin_path, 'rb')
            self.ram_data = mmap.mmap(self.f.fileno(), 0, access=mmap.ACCESS_READ)
        
        data_start = self.samples[idx]  # O(1) direct access
        chunk = self.ram_data[data_start : data_start + self.chunk_size]
        
        x = torch.from_numpy(np.frombuffer(chunk, dtype=np.uint8).astype(np.int64))
        # O(1) slice from pre-computed label mask
        y = torch.from_numpy(self.full_label_mask[data_start : data_start + self.chunk_size].copy()).float()
        
        return x, y, data_start

    def __del__(self):
        """Clean up mmap and file resources on deletion."""
        if self.ram_data is not None:
            try:
                self.ram_data.close()
            except Exception:
                pass
        if self.f is not None:
            try:
                self.f.close()
            except Exception:
                pass