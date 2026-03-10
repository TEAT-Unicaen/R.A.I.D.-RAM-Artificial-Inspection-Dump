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
            
        # 0: clear | 1: encrypted-like
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
            segment_start = current_pos
            segment_end = current_pos + self.chunk_size
            
            type_counts = {}
            
            temp_idx = meta_idx
            while temp_idx < num_meta:
                entry = self.metadata[temp_idx]
                
                if entry['ds'] >= segment_end:
                    break
                    
                #intersection
                overlap_start = max(segment_start, entry['ds'])
                overlap_end = min(segment_end, entry['de'])
                
                if overlap_start < overlap_end:
                    overlap_len = overlap_end - overlap_start
                    label_val = self.label_map.get(entry['t'], 0)
                    type_counts[label_val] = type_counts.get(label_val, 0) + overlap_len
                    
                    if entry['de'] <= segment_start:
                        meta_idx = temp_idx
                
                temp_idx += 1
 
            if type_counts:
                major_type = max(type_counts, key=type_counts.get)
            else:
                major_type = 0
                
            self.samples.append((current_pos, major_type))
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