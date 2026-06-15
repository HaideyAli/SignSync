import torch
import torch.nn as nn
import math

NUM_CLASSES = 100
INPUT_DIM   = 258   # landmark values per frame
SEQ_LEN     = 30    # frames per sequence


class LSTMClassifier(nn.Module):
    # Reads the 30-frame sequence forwards and backwards, then predicts the sign class
    def __init__(self, input_dim=INPUT_DIM, hidden_dim=256,
                 num_layers=2, num_classes=NUM_CLASSES, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim, hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, 128),  # *2 for bidirectional
            nn.ReLU(),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        # x: (batch, 30, 258)
        _, (hidden, _) = self.lstm(x)
        # hidden: (num_layers*2, batch, hidden_dim) — take last layer, both directions
        fwd = hidden[-2]   # last forward layer
        bwd = hidden[-1]   # last backward layer
        out = torch.cat([fwd, bwd], dim=1)   # (batch, hidden_dim*2)
        return self.classifier(out)


class PositionalEncoding(nn.Module):
    # Injects position information into the sequence so the Transformer knows frame order
    def __init__(self, d_model: int, max_len: int = SEQ_LEN, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x):
        return self.dropout(x + self.pe[:, : x.size(1)])


class TransformerClassifier(nn.Module):
    # Projects landmarks into embedding space, adds position info, applies attention, classifies
    def __init__(self, input_dim=INPUT_DIM, d_model=128, nhead=4,
                 num_layers=2, num_classes=NUM_CLASSES, dropout=0.3):
        super().__init__()
        self.input_proj   = nn.Linear(input_dim, d_model)
        self.pos_encoding = PositionalEncoding(d_model, dropout=dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=256, dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(d_model, num_classes),
        )

    def forward(self, x):
        # x: (batch, 30, 258)
        x = self.pos_encoding(self.input_proj(x))   # (batch, 30, d_model)
        x = self.transformer(x)                      # (batch, 30, d_model)
        x = x.mean(dim=1)                            # mean pool over time → (batch, d_model)
        return self.classifier(x)


# Returns the right model given an arch string — used by train.py and evaluate.py
def build_model(arch: str) -> nn.Module:
    if arch == "lstm":
        return LSTMClassifier()
    if arch == "transformer":
        return TransformerClassifier()
    raise ValueError(f"Unknown arch '{arch}'. Choose 'lstm' or 'transformer'.")
