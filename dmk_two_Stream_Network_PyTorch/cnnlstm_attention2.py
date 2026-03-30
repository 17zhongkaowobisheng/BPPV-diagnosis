import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import ResNet101_Weights
import torch.nn.functional as F

class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction_ratio),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction_ratio, in_channels)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x).squeeze())
        max_out = self.fc(self.max_pool(x).squeeze())
        channel_attn = self.sigmoid(avg_out + max_out).unsqueeze(-1).unsqueeze(-1)
        return x * channel_attn


class SpatialAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=7, padding=3)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        spatial_attn = self.sigmoid(self.conv(torch.cat([avg_out, max_out], dim=1)))
        return x * spatial_attn


class CBAM(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.channel_att = ChannelAttention(in_channels)
        self.spatial_att = SpatialAttention()

    def forward(self, x):
        x = self.channel_att(x)
        x = self.spatial_att(x)
        return x


class TemporalAttention(nn.Module):
    """新增时间注意力模块"""

    def __init__(self, hidden_size):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.Tanh(),
            nn.Linear(hidden_size // 2, 1)
        )
        self.softmax = nn.Softmax(dim=1)

    def forward(self, lstm_out):
        # lstm_out shape: (seq_len, batch, hidden_size)
        attn_weights = self.softmax(self.attn(lstm_out).squeeze(-1))  # (seq_len, batch)
        return torch.sum(lstm_out * attn_weights.unsqueeze(-1), dim=0)


class CBAM_LSTM(nn.Module):
    def __init__(self, num_classes=5):
        super().__init__()

        # 改进的ResNet骨干网络
        base_model = models.resnet101(weights=ResNet101_Weights.DEFAULT)
        self.resnet_conv = nn.Sequential(*list(base_model.children())[:-2])

        # 多尺度CBAM注意力
        self.cbam1 = CBAM(256)  # layer1输出通道数
        self.cbam2 = CBAM(512)  # layer2输出通道数
        self.cbam3 = CBAM(2048)  # layer4输出通道数

        # 特征融合
        self.feature_fusion = nn.Sequential(
            nn.Conv2d(2048 + 512 + 256, 512, kernel_size=1),
            nn.BatchNorm2d(512),
            nn.ReLU()
        )

        # 时空特征处理
        self.lstm = nn.LSTM(
            input_size=512,
            hidden_size=256,
            num_layers=2,
            bidirectional=True,
            dropout=0.3
        )
        self.temporal_attn = TemporalAttention(hidden_size=512)

        # 分类头
        self.classifier = nn.Sequential(
            nn.Linear(512, 128),
            nn.GELU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )

    def forward(self, x_3d):
        batch_size, timesteps, C, H, W = x_3d.size()
        lstm_input = []

        # 空间特征提取
        for t in range(timesteps):
            # 基础特征提取
            x = self.resnet_conv(x_3d[:, t])

            # # 多尺度注意力
            # with torch.no_grad():  # 冻结底层参数
            layer1_out = self.cbam1(self.resnet_conv[:5](x_3d[:, t]))
            layer2_out = self.cbam2(self.resnet_conv[5](layer1_out))
            layer3_out = self.cbam3(x)

            # 特征融合
            h, w = layer3_out.size(2), layer3_out.size(3)
            fused = torch.cat([
                F.adaptive_avg_pool2d(layer1_out, (h, w)),
                F.adaptive_avg_pool2d(layer2_out, (h, w)),
                layer3_out
            ], dim=1)
            fused = self.feature_fusion(fused)

            # 全局特征
            pooled = F.adaptive_avg_pool2d(fused, (1, 1)).squeeze()
            lstm_input.append(pooled)

        # 时域建模
        lstm_input = torch.stack(lstm_input)  # [timesteps, batch, features]
        lstm_out, _ = self.lstm(lstm_input)  # [timesteps, batch, hidden*2]

        # 时间注意力
        temporal_feat = self.temporal_attn(lstm_out)  # [batch, hidden*2]

        # 分类
        return self.classifier(temporal_feat)


# 示例用法
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CBAM_LSTM(num_classes=5).to(device)

    # 模拟输入: (batch, timesteps, channels, height, width)
    dummy_input = torch.randn(2, 16, 3, 224, 224).to(device)

    output = model(dummy_input)
    print(f"Output shape: {output.shape}")  # 应为torch.Size([2, 5])