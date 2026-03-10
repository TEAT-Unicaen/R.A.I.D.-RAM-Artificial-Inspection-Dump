import math
import torch
import torch.nn as nn

from .PositionalEncoding import PositionalEncoding


class BytesTransformerClassifier(nn.Module):
    def __init__(
        self,
        dim_model: int = 128,
        num_heads: int = 4,
        num_layers: int = 4,
        dim_ff: int = 512,
        dropout: float = 0.1,
    ):
        """
        Transformer-based binary classifier operating on raw byte sequences.

        Args:
            dim_model:  Embedding/hidden dimension throughout the model.
            num_heads:  Number of self-attention heads.
            num_layers: Number of TransformerEncoder layers.
            dim_ff:     Inner dimension of the feed-forward sublayer.
            dropout:    Dropout probability applied throughout.
        """
        super().__init__()

        self.padding_idx = 256   # Reserved token — outside the 0-255 byte range
        self.vocab_size  = 257   # 256 byte values + 1 padding token
        self.dim_model   = dim_model

        # ── Embedding ────────────────────────────────────────────────────────
        self.embedding   = nn.Embedding(self.vocab_size, dim_model, padding_idx=self.padding_idx)
        self.pos_encoder = PositionalEncoding(dim_model, dropout=dropout)

        # ── [CLS] token (learnable) ──────────────────────────────────────────
        # Prepended to every sequence; its final hidden state is used for
        # classification instead of average-pooling over all positions.
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim_model))

        # ── Transformer Encoder ──────────────────────────────────────────────
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim_model,
            nhead=num_heads,
            dim_feedforward=dim_ff,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # ── Post-encoder normalisation ───────────────────────────────────────
        self.norm = nn.LayerNorm(dim_model)

        # ── Classifier MLP ───────────────────────────────────────────────────
        self.classifier = nn.Sequential(
            nn.Linear(dim_model, dim_model * 2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_model * 2, dim_model),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_model, 1),   # Raw logit — apply sigmoid externally
        )

        # ── Weight initialisation ────────────────────────────────────────────
        self.apply(self._init_weights)
        # [CLS] token starts near zero; small normal noise breaks symmetry
        nn.init.normal_(self.cls_token, mean=0.0, std=0.02)

    # ─────────────────────────────────────────────────────────────────────────
    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            # Restore zero vector for the padding index after re-init
            if module.padding_idx is not None:
                with torch.no_grad():
                    module.weight[module.padding_idx].zero_()

    # ─────────────────────────────────────────────────────────────────────────
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: LongTensor of shape (batch, seq_len) containing byte values in
               [0, 255]; positions filled with 256 are treated as padding.

        Returns:
            logits: FloatTensor of shape (batch, 1).
        """
        batch_size = x.size(0)

        # ── Step 1 · Padding mask ────────────────────────────────────────────
        # True where x is the padding token (ignored by attention).
        # Shape: (batch, seq_len)
        byte_padding_mask = (x == self.padding_idx)

        # ── Step 2 · Embedding + positional encoding ─────────────────────────
        # Scale as in "Attention Is All You Need".
        x = self.embedding(x) * math.sqrt(self.dim_model)  # (batch, seq, d)
        x = self.pos_encoder(x)

        # ── Step 3 · Prepend [CLS] token ────────────────────────────────────
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)  # (batch, 1, d)
        x = torch.cat([cls_tokens, x], dim=1)                   # (batch, 1+seq, d)

        # Extend the padding mask: [CLS] is never masked (False = keep).
        cls_mask = torch.zeros(batch_size, 1, dtype=torch.bool, device=x.device)
        padding_mask = torch.cat([cls_mask, byte_padding_mask], dim=1)  # (batch, 1+seq)

        # ── Step 4 · Transformer encoder ────────────────────────────────────
        x = self.transformer_encoder(x, src_key_padding_mask=padding_mask)
        # Shape: (batch, 1+seq, d)

        # ── Step 5 · Extract [CLS] representation ───────────────────────────
        cls_output = x[:, 0, :]          # (batch, d)
        cls_output = self.norm(cls_output)

        # ── Step 6 · Classify ────────────────────────────────────────────────
        logits = self.classifier(cls_output)  # (batch, 1)

        return logits