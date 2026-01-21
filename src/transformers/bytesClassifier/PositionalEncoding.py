import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    """
    sans cette classe, un modèle Transformer voit les mots comme un "sac de mots" sans ordre, car il traite tout en parallèle
    """
    def __init__(self, d_model: int, max_len:int =5000, dropout: float=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)

        # Créer une échelle de fréquence afin d'obtenir les relations de voisinage
        # donc des mots proches en terme de distance et pas de sens, auront des petites valeurs de distance et inversement
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # Ajouter une dimension batch (afin de rendre compatible avec l'entrée du modèle)
        pe = pe.unsqueeze(0)

        # Ajout au fichier .pth
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)