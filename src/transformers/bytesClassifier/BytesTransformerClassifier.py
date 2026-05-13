import torch.nn as nn
import math

from .PositionalEncoding import PositionalEncoding

class BytesTransformerClassifier(nn.Module):
    def __init__(
        self,
        dim_model: int = 128,
        num_heads: int = 4,
        num_layers: int = 4,
        dim_ff: int = 512,
        dropout: float = 0.1,
        max_len: int = 5000,
        local_conv_kernel_size: int = 3,
        padding_idx: int = 256,
        vocab_size: int = 257,
        classifier_hidden_dim: int | None = None,
    ):
        """
        Args:
            padding_idx: Embedding padding index (default: 256, representing out-of-range byte value).
            vocab_size:  Size of byte vocabulary (default: 257, for 256 byte values + 1 padding).
            dim_model:  Embedding/hidden dimension throughout the model.
            num_heads:  Number of self-attention heads.
            num_layers: Number of TransformerEncoder layers.
            dim_ff:     Inner dimension of the feed-forward sublayer.
            dropout:    Dropout probability applied throughout.
            max_len:    Maximum sequence length handled by the positional encoding.
        """
        super().__init__()
        self.padding_idx = padding_idx
        self.vocab_size = vocab_size
        if local_conv_kernel_size < 1 or local_conv_kernel_size % 2 == 0:
            raise ValueError("local_conv_kernel_size must be a positive odd integer")
        
        self.embedding = nn.Embedding(self.vocab_size, dim_model, padding_idx=self.padding_idx)
        self.pos_encoder = PositionalEncoding(dim_model, max_len=max_len, dropout=dropout)

        #Convolution related
        self.local_conv = nn.Conv1d(
            in_channels=dim_model,
            out_channels=dim_model,
            kernel_size=local_conv_kernel_size,
            padding=local_conv_kernel_size // 2,
        )
        nn.init.zeros_(self.local_conv.weight)
        if self.local_conv.bias is not None:
            nn.init.zeros_(self.local_conv.bias)
        self.local_conv_activation = nn.SiLU()
        self.local_norm = nn.LayerNorm(dim_model)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim_model,
            nhead=num_heads,
            dim_feedforward=dim_ff,
            dropout=dropout,
            batch_first=True
        )
        
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        classifier_hidden_dim = classifier_hidden_dim or dim_model * 2
        # MLP plus expressif pour mieux séparer les patterns locaux et globaux.
        self.classifier = nn.Sequential(
            nn.Linear(dim_model, classifier_hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(classifier_hidden_dim, dim_model),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_model, 1)          # Sortie unique (Logit)
        )
        
        self.dim_model = dim_model

    def forward(self, x):
        """
        Retourne les logits pour chaque byte de la séquence.
        """

        # --- Étape 1 : Gestion du Masque de Padding ---
        src_key_padding_mask = (x == self.padding_idx)

        # --- Étape 2 : Embedding + Encodage Positionnel ---
        x = self.embedding(x) * math.sqrt(self.dim_model)

        # --- Étape 2b : Motifs locaux ---
        local_features = self.local_conv(x.transpose(1, 2)).transpose(1, 2)
        x = x + self.local_conv_activation(local_features)
        x = self.local_norm(x)

        # --- Étape 2c : Positional Encoding ---
        x = self.pos_encoder(x)
        
        # --- Étape 3 : Passage dans le Transformer ---
        x = self.transformer_encoder(x, src_key_padding_mask=src_key_padding_mask)
        
        # --- Étape 4 : Classification pour chaque byte ---
        # Sortie : (Batch_Size, Seq_Len, 1)
        logits = self.classifier(x)
        
        return logits.squeeze(-1)  # (Batch_Size, Seq_Len)
