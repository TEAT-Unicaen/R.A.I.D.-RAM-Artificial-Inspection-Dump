"""
Configuration class for training parameters.
Centralizes all hyperparameters and training settings.
"""

import configparser
import os

class TrainingConfig:
    """Training configuration class"""
    
    def __init__(self, config_path='config.cfg'):
        """
        Initialize configuration from file or use defaults.
        
        Args:
            config_path (str): Path to the configuration file
        """
        
        config = configparser.ConfigParser()
        config.read(config_path)
        
        self.debug = config['general'].getboolean('debug', fallback=False)
        self.use_cpu = config['general'].getboolean('use_cpu', fallback=False)
        
        self.seed = config['training'].getint('seed', fallback=42)
        self.batch_size = config['training'].getint('batch_size', fallback=128)
        self.lr = config['training'].getfloat('lr', fallback=3e-3)
        self.epochs = config['training'].getint('epochs', fallback=200)
        
        self.embed_dim = config['model'].getint('embed_dim', fallback=768)
        self.depth = config['model'].getint('depth', fallback=6)
        self.heads = config['model'].getint('heads', fallback=8)
        self.dropout = config['model'].getfloat('dropout', fallback=0.1)

    def __repr__(self):
        """String representation of the configuration"""
        return (f"TrainingConfig(seed={self.seed}, batch_size={self.batch_size}, "
                f"embed_dim={self.embed_dim}, depth={self.depth}, heads={self.heads}, "
                f"dropout={self.dropout}, lr={self.lr}, epochs={self.epochs})")
