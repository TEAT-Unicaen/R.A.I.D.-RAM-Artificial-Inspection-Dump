import torch.nn as nn
import math

from .PositionalEncoding import PositionalEncoding

class BytesTransformerClassifier(nn.Module):
    def __init__(self, dim_model: int=128, num_heads: int=4, num_layers: int=4, dim_ff: int=512, dropout: float=0.1):
        """
        Args:
            dim_model:  Embedding/hidden dimension throughout the model.
            num_heads:  Number of self-attention heads.
            num_layers: Number of TransformerEncoder layers.
            dim_ff:     Inner dimension of the feed-forward sublayer.
            dropout:    Dropout probability applied throughout.
        """
        super().__init__()
        self.padding_idx = 256 # Valeur pour le padding (octet hors plage)
        self.vocab_size = 257 # 256 valeurs d'octets + 1 pour le padding
        
        self.embedding = nn.Embedding(self.vocab_size, dim_model, padding_idx=self.padding_idx)
        self.pos_encoder = PositionalEncoding(dim_model, dropout=dropout)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim_model,
            nhead=num_heads,
            dim_feedforward=dim_ff,
            dropout=dropout,
            batch_first=True
        )
        
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # MLP
        self.classifier = nn.Sequential(
            nn.Linear(dim_model, dim_model), # Projection intermédiaire
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
        x = self.pos_encoder(x)
        
        # --- Étape 3 : Passage dans le Transformer ---
        x = self.transformer_encoder(x, src_key_padding_mask=src_key_padding_mask)
        
        # --- Étape 4 : Classification pour chaque byte ---
        # Sortie : (Batch_Size, Seq_Len, 1)
        logits = self.classifier(x)
        
        return logits.squeeze(-1)  # (Batch_Size, Seq_Len)
