"""
RAM Dump Dataset Preprocessor module.
This module handles loading and preprocessing of generated RAM dumps for training.
It reads binary dump files and their associated metadata to create training datasets.
"""

import os
import json
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np


# Define the label mapping for RAM dump segment types
LABEL_MAPPING = {
    "BINARY_TEXT": 0,
    "BINARY_IMAGE": 1,
    "BINARY_OTHER": 2,
    "ENCRYPTED": 3,
    "DECODED": 4,
    "BASE64": 5,
    "COMPRESSED": 6,
    "SYSTEM": 7,
    "NOISE": 8
}

NUM_CLASSES = len(LABEL_MAPPING)


class RAMDumpDataset(Dataset):
    """
    PyTorch Dataset for RAM dump segments.
    
    Loads binary RAM dump data and extracts segments based on metadata.
    Each segment is padded/truncated to a fixed sequence length.
    """
    
    def __init__(self, dump_path: str, metadata_path: str, sequence_length: int = 8192, 
                 exclude_noise: bool = True, normalize: bool = True):
        """
        Initialize the RAM dump dataset.
        
        Args:
            dump_path: Path to the binary RAM dump file (.bin)
            metadata_path: Path to the metadata JSON file
            sequence_length: Fixed length for each segment (will pad/truncate)
            exclude_noise: Whether to exclude NOISE segments from training
            normalize: Whether to normalize byte values to [0, 1]
        """
        self.sequence_length = sequence_length
        self.normalize = normalize
        
        # Load the binary dump
        with open(dump_path, 'rb') as f:
            self.dump_data = f.read()
        
        # Load metadata
        with open(metadata_path, 'r', encoding='utf-8') as f:
            all_metadata = json.load(f)
        
        # Filter out noise if requested
        if exclude_noise:
            self.metadata = [m for m in all_metadata if m['type'] != 'NOISE']
        else:
            self.metadata = all_metadata
        
        # Validate labels
        self.samples = []
        for entry in self.metadata:
            label_str = entry['type']
            if label_str in LABEL_MAPPING:
                self.samples.append({
                    'data_start': entry['data_start'],
                    'data_end': entry['data_end'],
                    'label': LABEL_MAPPING[label_str],
                    'type': label_str
                })
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Extract the segment from the dump
        start = sample['data_start']
        end = sample['data_end']
        segment = self.dump_data[start:end]
        
        # Convert to numpy array of bytes
        data = np.frombuffer(segment, dtype=np.uint8).astype(np.float32)
        
        # Normalize to [0, 1] range
        if self.normalize:
            data = data / 255.0
        
        # Pad or truncate to fixed sequence length
        if len(data) < self.sequence_length:
            # Pad with zeros
            padded = np.zeros(self.sequence_length, dtype=np.float32)
            padded[:len(data)] = data
            data = padded
        elif len(data) > self.sequence_length:
            # Truncate (take first sequence_length bytes)
            data = data[:self.sequence_length]
        
        # Convert to tensor
        tensor = torch.from_numpy(data)
        label = torch.tensor(sample['label'], dtype=torch.long)
        
        return tensor, label


class MultiDumpDataset(Dataset):
    """
    Dataset that combines multiple RAM dump files.
    Useful when you have multiple generated dumps for training.
    """
    
    def __init__(self, dump_dir: str, sequence_length: int = 8192, 
                 exclude_noise: bool = True, normalize: bool = True):
        """
        Initialize dataset from a directory containing multiple dumps.
        
        Args:
            dump_dir: Directory containing .bin files and their .json metadata
            sequence_length: Fixed length for each segment
            exclude_noise: Whether to exclude NOISE segments
            normalize: Whether to normalize byte values
        """
        self.samples = []
        self.dumps = {}  # Store loaded dump data
        self.sequence_length = sequence_length
        self.normalize = normalize
        
        # Find all bin files and their metadata
        for filename in os.listdir(dump_dir):
            if filename.endswith('.bin'):
                bin_path = os.path.join(dump_dir, filename)
                json_path = os.path.join(dump_dir, filename.replace('.bin', '.json'))
                
                # Try to find metadata file
                if not os.path.exists(json_path):
                    # Check for metadata.json if the bin file is ram_dump.bin
                    json_path = os.path.join(dump_dir, 'metadata.json')
                
                if os.path.exists(json_path):
                    self._load_dump(bin_path, json_path, exclude_noise)
    
    def _load_dump(self, bin_path: str, json_path: str, exclude_noise: bool):
        """Load a single dump file and its metadata."""
        dump_id = bin_path
        
        # Load binary data
        with open(bin_path, 'rb') as f:
            self.dumps[dump_id] = f.read()
        
        # Load metadata
        with open(json_path, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        
        # Process entries
        for entry in metadata:
            label_str = entry['type']
            if exclude_noise and label_str == 'NOISE':
                continue
            if label_str in LABEL_MAPPING:
                self.samples.append({
                    'dump_id': dump_id,
                    'data_start': entry['data_start'],
                    'data_end': entry['data_end'],
                    'label': LABEL_MAPPING[label_str],
                    'type': label_str
                })
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        dump_data = self.dumps[sample['dump_id']]
        
        # Extract segment
        start = sample['data_start']
        end = sample['data_end']
        segment = dump_data[start:end]
        
        # Convert to numpy
        data = np.frombuffer(segment, dtype=np.uint8).astype(np.float32)
        
        if self.normalize:
            data = data / 255.0
        
        # Pad or truncate
        if len(data) < self.sequence_length:
            padded = np.zeros(self.sequence_length, dtype=np.float32)
            padded[:len(data)] = data
            data = padded
        elif len(data) > self.sequence_length:
            data = data[:self.sequence_length]
        
        tensor = torch.from_numpy(data)
        label = torch.tensor(sample['label'], dtype=torch.long)
        
        return tensor, label


def preprocessDumpDataset(dump_dir: str, sequence_length: int = 8192, 
                          exclude_noise: bool = True) -> Dataset:
    """
    Create a dataset from RAM dump files in a directory.
    
    This is the main entry point for loading dump data for training.
    
    Args:
        dump_dir: Path to directory containing dump files
                  Should contain .bin files and corresponding .json metadata
        sequence_length: Fixed sequence length for each sample
        exclude_noise: Whether to exclude NOISE segments
    
    Returns:
        PyTorch Dataset ready for training
    """
    # Check if it's a single dump or multiple
    bin_files = [f for f in os.listdir(dump_dir) if f.endswith('.bin')]
    
    if len(bin_files) == 1:
        # Single dump file
        bin_path = os.path.join(dump_dir, bin_files[0])
        json_path = os.path.join(dump_dir, 'metadata.json')
        
        if not os.path.exists(json_path):
            json_path = os.path.join(dump_dir, bin_files[0].replace('.bin', '.json'))
        
        return RAMDumpDataset(bin_path, json_path, sequence_length, exclude_noise)
    else:
        # Multiple dump files
        return MultiDumpDataset(dump_dir, sequence_length, exclude_noise)


def getDataLoader(dump_dir: str, batch_size: int = 32, sequence_length: int = 8192,
                  shuffle: bool = True, num_workers: int = 0, 
                  exclude_noise: bool = True) -> DataLoader:
    """
    Create a DataLoader for RAM dump training.
    
    Args:
        dump_dir: Path to directory containing dump files
        batch_size: Batch size for training
        sequence_length: Fixed sequence length for each sample
        shuffle: Whether to shuffle the data
        num_workers: Number of worker processes for data loading
        exclude_noise: Whether to exclude NOISE segments
    
    Returns:
        PyTorch DataLoader ready for training
    """
    dataset = preprocessDumpDataset(dump_dir, sequence_length, exclude_noise)
    
    return DataLoader(
        dataset, 
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True
    )


def getClassWeights(dump_dir: str, exclude_noise: bool = True) -> torch.Tensor:
    """
    Calculate class weights for handling imbalanced datasets.
    
    Args:
        dump_dir: Path to directory containing dump files
        exclude_noise: Whether to exclude NOISE segments
    
    Returns:
        Tensor of class weights for use with CrossEntropyLoss
    """
    dataset = preprocessDumpDataset(dump_dir, exclude_noise=exclude_noise)
    
    # Count samples per class
    class_counts = [0] * NUM_CLASSES
    for i in range(len(dataset)):
        _, label = dataset[i]
        class_counts[label.item()] += 1
    
    # Calculate weights (inverse frequency)
    total = sum(class_counts)
    weights = []
    for count in class_counts:
        if count > 0:
            weights.append(total / (NUM_CLASSES * count))
        else:
            weights.append(0.0)
    
    return torch.tensor(weights, dtype=torch.float32)
