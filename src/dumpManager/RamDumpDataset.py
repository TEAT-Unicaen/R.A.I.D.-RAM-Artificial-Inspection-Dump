import json
import torch
from torch.utils.data import Dataset
import mmap

class RamDumpDataset(Dataset):
    def __init__(self, bin_path, meta_path, chunk_size=512):
        self.bin_path = bin_path
        self.chunk_size = chunk_size
        self.samples = []
        self.ram_data = None
        
        with open(meta_path, 'r') as f:
            self.metadata = json.load(f)
            
        self.label_map = {
            "ENCRYPTED": 1, "COMPRESSED": 1, "BINARY_TEXT": 0,
            "BINARY_IMAGE": 1, "BINARY_OTHER": 0, "BINARY_PDF": 1,
            "BASE64": 0, "DECODED": 0, "SYSTEM": 0, "NOISE": 0
        }
        self._prepare_samples()

    def _prepare_samples(self):
        for entry in self.metadata:
            label_str = entry['type']
            if label_str not in self.label_map: continue
            label = self.label_map[label_str]
            start, end = entry['data_start'], entry['data_end']
            current_pos = start
            while current_pos + self.chunk_size <= end:
                self.samples.append((current_pos, label))
                current_pos += self.chunk_size

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        if self.ram_data is None:
            f = open(self.bin_path, 'rb')
            self.ram_data = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
        
        offset, label = self.samples[idx]
        chunk = self.ram_data[offset : offset + self.chunk_size]
        
        x = torch.tensor(list(chunk), dtype=torch.long)
        y = torch.tensor(label, dtype=torch.float)
        
        return x, y