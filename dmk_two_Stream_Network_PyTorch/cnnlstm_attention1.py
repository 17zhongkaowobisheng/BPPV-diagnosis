# import torch
# import torch.nn as nn
# import torchvision.models as models
# from torchvision.models import ResNet34_Weights
# import torch.nn.functional as F
#
# # ================== 注意力模块定义 ==================
# class ChannelAttention(nn.Module):
#     def __init__(self, in_channels, reduction_ratio=16):
#         super().__init__()
#         self.avg_pool = nn.AdaptiveAvgPool2d(1)
#         self.max_pool = nn.AdaptiveMaxPool2d(1)
#         self.fc = nn.Sequential(
#             nn.Linear(in_channels, in_channels // reduction_ratio),
#             nn.ReLU(inplace=True),
#             nn.Linear(in_channels // reduction_ratio, in_channels)
#         )
#         self.sigmoid = nn.Sigmoid()
#
#     def forward(self, x):
#         avg_out = self.fc(self.avg_pool(x).squeeze())
#         max_out = self.fc(self.max_pool(x).squeeze())
#         channel_attn = self.sigmoid(avg_out + max_out).unsqueeze(-1).unsqueeze(-1)
#         return x * channel_attn
#
#
# class SpatialAttention(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.conv = nn.Conv2d(2, 1, kernel_size=7, padding=3)
#         self.sigmoid = nn.Sigmoid()
#
#     def forward(self, x):
#         avg_out = torch.mean(x, dim=1, keepdim=True)
#         max_out, _ = torch.max(x, dim=1, keepdim=True)
#         spatial_attn = self.sigmoid(self.conv(torch.cat([avg_out, max_out], dim=1)))
#         return x * spatial_attn
#
#
# class CBAM(nn.Module):
#     def __init__(self, in_channels):
#         super().__init__()
#         self.channel_att = ChannelAttention(in_channels)
#         self.spatial_att = SpatialAttention()
#
#     def forward(self, x):
#         x = self.channel_att(x)
#         x = self.spatial_att(x)
#         return x

#
# class TemporalAttention(nn.Module):
#     """时间注意力模块"""
#
#     def __init__(self, hidden_size):
#         super().__init__()
#         self.attn = nn.Sequential(
#             nn.Linear(hidden_size, hidden_size // 2),
#             nn.Tanh(),
#             nn.Linear(hidden_size // 2, 1)
#         )
#         self.softmax = nn.Softmax(dim=1)
#
#     def forward(self, lstm_out):
#         attn_weights = self.softmax(self.attn(lstm_out).squeeze(-1))
#         return torch.sum(lstm_out * attn_weights.unsqueeze(-1), dim=0)

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from load_data import RGB_FRAME_NUM

# class CNNLSTM(nn.Module):
#     def __init__(self, num_classes=5, n_segment=RGB_FRAME_NUM):
#         super().__init__()
#
#         # 加载ResNet34预训练模型
#         base_model = models.resnet34(weights=ResNet34_Weights.DEFAULT)
#
#         # 修改ResNet为Temporal ResNet
#         self.resnet = make_temporal_modeling(base_model, n_segment=n_segment)
#
#         # 多尺度注意力模块
#         self.cbam1 = CBAM(64)  # layer1输出
#         self.cbam2 = CBAM(128)  # layer2输出
#         self.cbam3 = CBAM(256)  # layer3输出
#
#         # 特征融合层
#         self.feature_fusion = nn.Sequential(
#             nn.Conv2d(64 + 128 + 256, 256, kernel_size=1),
#             nn.BatchNorm2d(256),
#             nn.ReLU()
#         )
#
#         # 时空特征处理
#         self.lstm = nn.LSTM(
#             input_size=256,
#             hidden_size=128,
#             num_layers=3,
#             bidirectional=True,
#             dropout=0.3
#         )
#
#         # 分类头
#         self.classifier = nn.Sequential(
#             nn.Linear(256, 128),
#             nn.Linear(128, 64),
#             nn.GELU(),
#             nn.Dropout(0.5),
#             nn.Linear(64, 32),
#             nn.Linear(32, num_classes)
#         )
#
#     def forward(self, x_3d):
#         batch_size, timesteps, C, H, W = x_3d.size()
#
#         # 空间特征提取
#         lstm_input = []
#         for t in range(timesteps):
#             x = x_3d[:, t]  # [batch_size, C, H, W]
#             x = self.resnet.conv1(x)
#             x = self.resnet.bn1(x)
#             x = self.resnet.relu(x)
#             x = self.resnet.maxpool(x)
#
#             # 多尺度特征提取
#             layer1_out = self.resnet.layer1(x)
#             layer2_out = self.resnet.layer2(layer1_out)
#             layer3_out = self.resnet.layer3(layer2_out)
#
#             # 特征融合
#             h, w = layer3_out.size(2), layer3_out.size(3)
#             fused = torch.cat([
#                 F.adaptive_avg_pool2d(layer1_out, (h, w)),
#                 F.adaptive_avg_pool2d(layer2_out, (h, w)),
#                 layer3_out
#             ], dim=1)
#             fused = self.feature_fusion(fused)
#
#             # 全局特征
#             pooled = F.adaptive_avg_pool2d(fused, (1, 1)).squeeze()
#             lstm_input.append(pooled)
#
#         # 时域建模
#         lstm_input = torch.stack(lstm_input, dim=0)  # [timesteps, batch_size, features]
#         lstm_out, _ = self.lstm(lstm_input)  # [timesteps, batch_size, hidden*2]
#         lstm_out = lstm_out[-1, :, :]  # 取最后一个时间步的输出
#
#         # 分类
#         return self.classifier(lstm_out)


#
#
# class CNNLSTM(nn.Module):
#     def __init__(self, num_classes=5):
#         super().__init__()
#
#         # 加载ResNet34预训练模型
#         base_model = models.resnet34(weights=ResNet34_Weights.DEFAULT)
#
#         # 划分不同卷积阶段
#         self.conv1 = nn.Sequential(base_model.conv1, base_model.bn1, base_model.relu)
#         self.layer1 = nn.Sequential(base_model.maxpool, base_model.layer1)  # 输出64通道
#         self.layer2 = base_model.layer2  # 输出128通道
#         self.layer3 = base_model.layer3  # 输出256通道
#
#         # 多尺度注意力模块
#         self.cbam1 = CBAM(64)  # layer1输出
#         self.cbam2 = CBAM(128)  # layer2输出
#         self.cbam3 = CBAM(256)  # layer3输出
#
#         # 特征融合层
#         self.feature_fusion = nn.Sequential(
#             nn.Conv2d(64 + 128 + 256, 256, kernel_size=1),
#             nn.BatchNorm2d(256),
#             nn.ReLU()
#         )
#
#         # 时空特征处理
#         self.lstm = nn.LSTM(
#             input_size=256,
#             hidden_size=128,
#             num_layers=3,
#             bidirectional=True,
#             dropout=0.3
#
#         )
#         # self.temporal_attn = TemporalAttention(hidden_size=256)
#
#         # 分类头
#         self.classifier = nn.Sequential(
#             nn.Linear(256, 128),
#             nn.Linear(128, 64),
#             nn.GELU(),
#             nn.Dropout(0.5),
#             nn.Linear(64, 32),
#             nn.Linear(32, num_classes)
#         )
#
#     def forward(self, x_3d):
#         batch_size, timesteps, C, H, W = x_3d.size()
#         lstm_input = []
#
#         # 空间特征提取
#         for t in range(timesteps):
#             # 多尺度特征提取
#             # with torch.no_grad():  # 冻结底层参数
#             x = self.conv1(x_3d[:, t])
#             x = self.layer1(x)
#             layer1_out = self.cbam1(x)
#
#             x = self.layer2(layer1_out)
#             layer2_out = self.cbam2(x)
#
#             x = self.layer3(layer2_out)
#             layer3_out = self.cbam3(x)
#
#             # 特征融合3
#             h, w = layer3_out.size(2), layer3_out.size(3)
#             fused = torch.cat([
#                 F.adaptive_avg_pool2d(layer1_out, (h, w)),
#                 F.adaptive_avg_pool2d(layer2_out, (h, w)),
#                 layer3_out
#             ], dim=1)
#             fused = self.feature_fusion(fused)
#
#             # 全局特征
#             pooled = F.adaptive_avg_pool2d(fused, (1, 1)).squeeze()#将F.adaptive_avg_pool2d(fused, (1, 1))中所有为1的维度删掉
#             lstm_input.append(pooled)
#
#         # 时域建模：将特征图展平为一维向量，然后输入到LSTM（RNN或其变体）中。
#         lstm_input = torch.stack(lstm_input)  # [timesteps, batch, features]
#         lstm_out, _ = self.lstm(lstm_input)  # [timesteps, batch, hidden*2]
#         lstm_out=lstm_out[-1,:,:]
#
#         # 分类
#         return self.classifier(lstm_out)
        #
        # # 时间注意力
        # temporal_feat = self.temporal_attn(lstm_out)  # [batch, hidden*2]

        # # 分类
        # return self.classifier(temporal_feat)


import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from load_data import RGB_FRAME_NUM

class TAM(nn.Module):
    def __init__(self, in_channels, n_segment, kernel_size=3, stride=1, padding=1):
        super(TAM, self).__init__()
        self.in_channels = in_channels
        self.n_segment = n_segment
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        self.G = nn.Sequential(
            nn.Linear(n_segment, n_segment * 2, bias=False),
            nn.BatchNorm1d(n_segment * 2),
            nn.ReLU(inplace=True),
            nn.Linear(n_segment * 2, kernel_size, bias=False),
            nn.Softmax(-1)
        )

        self.L = nn.Sequential(
            nn.Conv1d(in_channels, in_channels // 4, kernel_size,
                      stride=1, padding=kernel_size // 2, bias=False),
            nn.BatchNorm1d(in_channels // 4),
            nn.ReLU(inplace=True),
            nn.Conv1d(in_channels // 4, in_channels, 1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        nt, c, h, w = x.size()
        t = self.n_segment
        n_batch = nt // t

        new_x = x.view(n_batch, t, c, h, w).permute(0, 2, 1, 3, 4).contiguous()
        out = F.adaptive_avg_pool2d(new_x.view(n_batch * c, t, h, w), (1, 1))#使用全局空间平均池化对特征图进行如下压缩
        out = out.view(-1, t)

        conv_kernel = self.G(out.view(-1, t)).view(n_batch * c, 1, -1, 1)
        local_activation = self.L(out.view(n_batch, c, t)).view(n_batch, c, t, 1, 1)
        new_x = new_x * local_activation

        out = F.conv2d(new_x.view(1, n_batch * c, t, h * w), conv_kernel,
                       stride=(self.stride, 1), padding=(self.padding, 0),
                       groups=n_batch * c)

        return out.view(n_batch, c, t, h, w).permute(0, 2, 1, 3, 4).contiguous().view(nt, c, h, w)


class TemporalBottleneck(nn.Module):
    def __init__(self, net, n_segment=8, t_kernel_size=3, t_stride=1, t_padding=1):
        super(TemporalBottleneck, self).__init__()
        self.net = net
        self.n_segment = n_segment
        self.tam = TAM(net.conv1.out_channels, n_segment, t_kernel_size, t_stride, t_padding)

    def forward(self, x):
        identity = x
        out = self.net.conv1(x)
        out = self.net.bn1(out)
        out = self.net.relu(out)
        out = self.tam(out)  # 添加时间注意力

        out = self.net.conv2(out)
        out = self.net.bn2(out)
        out = self.net.relu(out)

        out = self.net.conv3(out)
        out = self.net.bn3(out)

        if self.net.downsample is not None:
            identity = self.net.downsample(x)

        out += identity
        return self.net.relu(out)


def make_temporal_modeling(net, n_segment=8, t_kernel_size=3, t_stride=1, t_padding=1):
    def make_block_temporal(stage, n_segment):
        blocks = list(stage.children())
        for i, b in enumerate(blocks):
            if isinstance(b, torchvision.models.resnet.Bottleneck):
                blocks[i] = TemporalBottleneck(b, n_segment, t_kernel_size, t_stride, t_padding)
        return nn.Sequential(*blocks)

    net.layer1 = make_block_temporal(net.layer1, n_segment)
    net.layer2 = make_block_temporal(net.layer2, n_segment)
    net.layer3 = make_block_temporal(net.layer3, n_segment)
    net.layer4 = make_block_temporal(net.layer4, n_segment)
    return net


class CNNLSTM(nn.Module):
    def __init__(self, n_segment=40, num_classes=5):
        super(CNNLSTM, self).__init__()
        self.n_segment = n_segment

        # 使用预训练ResNet-34作为骨干网络
        # backbone = torchvision.models.resnet34(pretrained=True)
        backbone = torchvision.models.resnet18()
        # 修改第一层卷积的输入通道数为1
        backbone.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        backbone.bn1 = nn.BatchNorm2d(64)
        backbone.relu = nn.ReLU(inplace=True)
        backbone.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.backbone = make_temporal_modeling(backbone, n_segment)

        # 替换最后的全连接层
        in_features = backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(0.5),  # 增加Dropout
            nn.Linear(in_features, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

        # # 空间注意力参数
        # self.phi = nn.Conv2d(1, 1, kernel_size=1)

    def forward(self, x):
        # 输入形状: [B, T, C, H, W] -> [B*T, C, H, W]
        B, T, C, H, W = x.size()
        x = x.view(B * T, C, H, W)

        # 通过骨干网络
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)

        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)

        # 时空特征聚合
        x = self.backbone.avgpool(x)
        x = torch.flatten(x, 1)
        x = x.view(B, T, -1)  # [B, T, D]

        # 时间维度平均池化
        x = x.mean(dim=1)  # [B, D]

        # 分类头
        return self.backbone.fc(x)


# 使用示例
if __name__ == "__main__":
    model = CNNLSTM(n_segment=RGB_FRAME_NUM, num_classes=5)
    input_tensor = torch.randn(10, 100, 1, 112, 112)  # [B, T, C, H, W]
    output = model(input_tensor)
    print(f"Output shape: {output.shape}")  # 应为[10, 5]










