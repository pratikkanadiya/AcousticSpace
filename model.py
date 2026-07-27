import torch
import torch.nn as nn


class SpecBranch(nn.Module):

    def __init__(self, in_channels: int = 1, out_features: int = 512):
        super().__init__()

        channels = [in_channels, 32, 64, 128, 256, 512]

        blocks = []
        for i in range(5):
            blocks.append(
                nn.Sequential(
                    nn.Conv2d(
                        channels[i],
                        channels[i + 1],
                        kernel_size=3,
                        padding=1,
                        bias=False,
                    ),
                    nn.BatchNorm2d(channels[i + 1]),
                    nn.ReLU(inplace=True),

                    nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True),
                )
            )
        self.conv_blocks = nn.Sequential(*blocks)

        self.global_pool = nn.AdaptiveAvgPool2d(output_size=1)

        assert channels[-1] == out_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_blocks(x)         
        x = self.global_pool(x)       
        x = torch.flatten(x, 1)        
        return x


class RoomBranch(nn.Module):

    def __init__(self, in_channels: int = 1, out_features: int = 128):
        super().__init__()

        channels = [in_channels, 32, 64, 128]

        blocks = []
        for i in range(len(channels) - 1):
            blocks.append(
                nn.Sequential(
                    nn.Conv1d(
                        channels[i],
                        channels[i + 1],
                        kernel_size=7,
                        padding=3,
                        bias=False,
                    ),
                    nn.BatchNorm1d(channels[i + 1]),
                    nn.ReLU(inplace=True),
                    nn.MaxPool1d(kernel_size=4, stride=4, ceil_mode=True),
                )
            )
        self.conv_blocks = nn.Sequential(*blocks)

        self.global_pool = nn.AdaptiveAvgPool1d(output_size=1)

        assert channels[-1] == out_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            x = x.unsqueeze(1)
        x = self.conv_blocks(x)          # (B, 128, L')
        x = self.global_pool(x)          # (B, 128, 1)
        x = torch.flatten(x, 1)          # (B, 128)
        return x


class AcousticBranch(nn.Module):

    def __init__(self, in_features: int = 9, out_features: int = 32):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_features, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Linear(64, out_features),
            nn.BatchNorm1d(out_features),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)


class ClassifierHead(nn.Module):

    def __init__(self, in_features: int = 672, num_classes: int = 2, dropout: float = 0.3):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),

            nn.Linear(256, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),

            nn.Linear(64, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(x)


class MultiModalAcousticNet(nn.Module):

    def __init__(self, spec_in_channels: int = 1, room_in_channels: int = 1, acoustic_in_features: int = 9, num_classes: int = 2, dropout: float = 0.3,):
        super().__init__()

        self.spec_branch = SpecBranch(in_channels=spec_in_channels, out_features=512)
        self.room_branch = RoomBranch(in_channels=room_in_channels, out_features=128)
        self.acoustic_branch = AcousticBranch(in_features=acoustic_in_features, out_features=32)

        fused_dim = 512 + 128 + 32  # 672
        self.classifier = ClassifierHead(
            in_features=fused_dim, num_classes=num_classes, dropout=dropout
        )

        self._init_weights()

    def forward(self, spec: torch.Tensor, room: torch.Tensor, acoustic: torch.Tensor) -> torch.Tensor:
        spec_feat = self.spec_branch(spec)      
        room_feat = self.room_branch(room)           
        acoustic_feat = self.acoustic_branch(acoustic)  

        fused = torch.cat([spec_feat, room_feat, acoustic_feat], dim=1) 
        logits = self.classifier(fused)              
        return logits

    def _init_weights(self):
        """Kaiming init for Conv/Linear (ReLU-friendly), standard init for BatchNorm."""
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Conv2d)):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="relu")
                nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)