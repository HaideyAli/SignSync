import torch
import torch.nn as nn
import math

NUM_CLASSES = 100
INPUT_DIM   = 516   # 258 landmark positions + 258 frame-to-frame velocity deltas
SEQ_LEN     = 30


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
            nn.Linear(hidden_dim * 2, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        _, (hidden, _) = self.lstm(x)
        fwd = hidden[-2]
        bwd = hidden[-1]
        return self.classifier(torch.cat([fwd, bwd], dim=1))


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
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return self.dropout(x + self.pe[:, : x.size(1)])


class TransformerClassifier(nn.Module):
    # Projects landmarks into embedding space, adds position info, applies attention, classifies
    def __init__(self, input_dim=INPUT_DIM, d_model=256, nhead=8,
                 num_layers=2, num_classes=NUM_CLASSES, dropout=0.3):
        super().__init__()
        self.input_proj   = nn.Linear(input_dim, d_model)
        self.pos_encoding = PositionalEncoding(d_model, dropout=dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=512, dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(d_model, num_classes),
        )

    def forward(self, x):
        x = self.pos_encoding(self.input_proj(x))
        x = self.transformer(x)
        x = x.mean(dim=1)
        return self.classifier(x)


def build_model(arch: str, num_classes: int = NUM_CLASSES) -> nn.Module:
    if arch == "lstm":
        return LSTMClassifier(num_classes=num_classes)
    if arch == "transformer":
        return TransformerClassifier(num_classes=num_classes)
    raise ValueError(f"Unknown arch '{arch}'. Choose 'lstm' or 'transformer'.")
