import torch
import torch.nn as nn
from torchvision import models
from config import CHARSET


# class CRNN(nn.Module):
#     def __init__(self, num_classes=len(CHARSET) + 1):  # +1 for CTC blank
#         super().__init__()
#
#         # ---------- CNN BACKBONE (ResNet18) ----------
#         backbone = models.resnet18(weights=None)
#
#         # modify first conv to accept 1 channel instead of 3
#         backbone.conv1 = nn.Conv2d(
#             in_channels=1,  # grayscale
#             out_channels=64,
#             kernel_size=7,
#             stride=1,
#             padding=3,
#             bias=False
#         )
#
#         # remove maxpool (keeps spatial resolution)
#         backbone.maxpool = nn.Identity()
#
#         # Remove downsampling in Layer2
#         backbone.layer2[0].conv1.stride = (1, 1)
#         backbone.layer2[0].downsample[0].stride = (1, 1)
#
#         # Remove downsampling in Layer3
#         backbone.layer3[0].conv1.stride = (1, 1)
#         backbone.layer3[0].downsample[0].stride = (1, 1)
#
#         # Remove downsampling in Layer4
#         backbone.layer4[0].conv1.stride = (1, 1)
#         backbone.layer4[0].downsample[0].stride = (1, 1)
#         # take only conv layers (exclude avgpool and fc)
#         self.cnn = nn.Sequential(
#             backbone.conv1,
#             backbone.bn1,
#             backbone.relu,
#             # backbone.maxpool,  <-- removed
#             backbone.layer1,
#             backbone.layer2,
#             backbone.layer3,
#             backbone.layer4,
#         )
#
#         # CNN output: (B, 512, H/4, W/4)
#         # Reduce channels 512 → 256 for LSTM
#         self.map_to_seq = nn.Conv2d(512, 256, kernel_size=1)
#         self.pool_h = nn.AdaptiveAvgPool2d((1, None))
#         # ---------- RNN (BiLSTM) ----------
#         self.lstm = nn.LSTM(
#             input_size=256,
#             hidden_size=256,
#             num_layers=2,
#             bidirectional=True,
#             batch_first=True
#         )
#         # output features: 512 per timestep (256 × 2 directions)
#
#         # ---------- CLASSIFIER ----------
#         self.fc = nn.Linear(512, num_classes)  # CTC classes
#
#     def forward(self, x):
#         conv = self.cnn(x)
#         conv = self.map_to_seq(conv)  # B, 256, H/4, W/4
#         conv = self.pool_h(conv)
#         # collapse H dimension (should be 1 after resizing)
#         seq = conv.squeeze(2)         # B, 256, W/4
#         seq = seq.permute(0, 2, 1)    # B, T, feat  — correct for batch_first=True LSTM
#
#         lstm_out, _ = self.lstm(seq)  # B, T, 512
#
#         out = self.fc(lstm_out)       # B, T, classes
#         return out

class CRNN(nn.Module):
    def __init__(self, num_classes=len(CHARSET)+1):
        super().__init__()

        # ---------- CNN BACKBONE (pretrained ResNet18) ----------
        backbone = models.resnet18(weights='IMAGENET1K_V1')

        # модифицируем первый слой под 1 канал
        backbone.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=64,
            kernel_size=7,
            stride=1,
            padding=3,
            bias=False
        )

        # убираем maxpool
        backbone.maxpool = nn.Identity()

        # отключаем downsampling в слоях 2-4
        for layer in [backbone.layer2, backbone.layer3, backbone.layer4]:
            layer[0].conv1.stride = (1, 1)
            if layer[0].downsample is not None:
                layer[0].downsample[0].stride = (1, 1)

        # оставляем только conv слои
        self.cnn = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.layer1,
            backbone.layer2,
            backbone.layer3,
            backbone.layer4
        )

        # reduce channels 512 -> 256 для LSTM
        self.map_to_seq = nn.Conv2d(512, 256, kernel_size=1)
        self.pool_h = nn.AdaptiveAvgPool2d((1, None))

        # ---------- RNN (BiLSTM) ----------
        self.lstm = nn.LSTM(
            input_size=256,
            hidden_size=256,
            num_layers=2,
            bidirectional=True,
            batch_first=True
        )

        # ---------- CLASSIFIER ----------
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        conv = self.cnn(x)
        conv = self.map_to_seq(conv)
        conv = self.pool_h(conv)          # (B, 256, 1, W)
        seq = conv.squeeze(2).permute(0, 2, 1)  # (B, T, 256)
        lstm_out, _ = self.lstm(seq)
        out = self.fc(lstm_out)           # (B, T, num_classes)
        return out