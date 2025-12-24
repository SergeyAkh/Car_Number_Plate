import torch.nn as nn

class TinyOCR(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d((2,2)),   # H=32 -> 16
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d((2,2)),   # H=16 -> 8
        )
        self.rnn = nn.LSTM(
            input_size=64 * 8,
            hidden_size=128,
            num_layers=2,
            dropout=0.3,
            batch_first=False,
            bidirectional=True
        )
        self.fc = nn.Linear(256, num_classes)

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.conv(x)                    # B × 64 × 8 × W/4
        x = x.permute(3, 0, 1, 2).contiguous()
        Tt, B, C2, H2 = x.shape
        x = x.view(Tt, B, C2 * H2)          # T × B × (64*8)
        x, _ = self.rnn(x)                  # T × B × 256
        x = self.fc(x)                      # T × B × num_classes
        return x