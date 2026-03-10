import json
import torch
from torch.utils.data import Dataset
import mmap

class RamDumpDataset(Dataset):
    def __init__(self, bin_path, meta_path, chunk_size=512, offset=512):
        self.bin_path = bin_path
        self.chunk_size = chunk_size
        self.offset = min(offset, chunk_size)
        self.samples = []
        self.ram_data = None
        
        with open(meta_path, 'r') as f:
            self.metadata = json.load(f)
            
        self.label_map = {
            "ENCRYPTED": 1,
            "COMPRESSED": 1,
            "BINARY_TEXT": 0,
            "BINARY_IMAGE": 1,
            "BINARY_OTHER": 0,
            "BINARY_PDF": 1,
            "BASE64": 0,
            "DECODED": 0,
            "SYSTEM": 0,
            "NOISE": 0
        }
        self._prepare_samples()

    def _prepare_samples(self):
        bin_size = self.metadata[-1]['de']
        current_pos = 0
        meta_idx = 0
        num_meta = len(self.metadata)

        while current_pos + self.chunk_size <= bin_size:
            while meta_idx < num_meta and self.metadata[meta_idx]['de'] <= current_pos:
                meta_idx += 1

            self.samples.append((current_pos, meta_idx))
            current_pos += self.offset

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        if self.ram_data is None:
            f = open(self.bin_path, 'rb')
            self.ram_data = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
        
        data_start, meta_idx = self.samples[idx]
        segment_end = data_start + self.chunk_size
        chunk = self.ram_data[data_start : data_start + self.chunk_size]
        
        x = torch.tensor(list(chunk), dtype=torch.long)
        y = torch.zeros(self.chunk_size, dtype=torch.float)

        temp_idx = meta_idx
        while temp_idx < len(self.metadata):
            entry = self.metadata[temp_idx]
            if entry['ds'] >= segment_end:
                break
            if entry['de'] <= data_start:
                temp_idx += 1
                continue
            
            overlap_start = max(data_start, entry['ds']) - data_start
            overlap_end   = min(segment_end, entry['de']) - data_start
            
            if overlap_start < overlap_end:
                label_val = self.label_map.get(entry['t'], 0)
                y[overlap_start:overlap_end] = label_val
            
            temp_idx += 1
        
        
        return x, y