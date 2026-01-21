import torch.nn as nn
import math

from .PositionalEncoding import PositionalEncoding

class BytesTransformerClassifier(nn.Module):
    def __init__(self, dim_model: int=128, num_heads: int=4, num_layers: int=4, dim_ff: int=512, dropout: float=0.1):
        """
        Args:
            Dim_model: Dimension of the model
            Num_heads: Number of attention heads
            Num_layers: Number of transformer layers
            Dim_ff: 
            Dropout: Dropout rate
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
        En gros c'est relou à lire mais pas compliqué : t'as un vecteur par byte, des fois incomplet donc t'as du bruit (padding).
        Sauf que le classifier il faut qu'un vecteur donc on sépare le vrai du faux avec un masque
        Et après on fait une moyenne des vrais vecteurs uniquement (Global Average Pooling masqué)
        """

        # --- Étape 1 : Gestion du Masque de Padding ---
        # Crée un masque Booléen : True là où x est du padding (valeur 256)
        # Forme : (Batch_Size, Seq_Len)
        src_key_padding_mask = (x == self.padding_idx)

        # --- Étape 2 : Embedding + Encodage Positionnel ---
        # Scaling par sqrt(d_model) est standard dans "Attention Is All You Need"
        x = self.embedding(x) * math.sqrt(self.dim_model)
        x = self.pos_encoder(x)
        
        # --- Étape 3 : Passage dans le Transformer ---
        # Le masque empêche l'attention de regarder les tokens de padding
        # Sortie : (Batch_Size, Seq_Len, dim_model)
        x = self.transformer_encoder(x, src_key_padding_mask=src_key_padding_mask)
        
        # --- Étape 4 : Global Average Pooling (Moyennage intelligent) ---
        # On ne veut pas faire la moyenne des vecteurs de padding (qui sont des bruits).
        # On crée un masque multiplicatif (0 pour le padding, 1 pour les données réelles)
        mask_expanded = (~src_key_padding_mask).unsqueeze(-1).float() # (Batch, Seq, 1)  --> la vague ça inverse le masque booléen
        
        # On met à zéro les vecteurs de padding
        x_masked = x * mask_expanded
        
        # Somme sur l'axe de la séquence
        sum_embeddings = x_masked.sum(dim=1) # (Batch, dim_model)
        
        # On compte le nombre de vrais tokens par séquence pour diviser correctement
        # clamp(min=1) évite la division par zéro si une séquence est vide (cas limite)
        valid_token_counts = mask_expanded.sum(dim=1).clamp(min=1) 
        
        pooled_output = sum_embeddings / valid_token_counts
        
        # --- Étape 5 : Classification ---
        logits = self.classifier(pooled_output)
        
        return logits