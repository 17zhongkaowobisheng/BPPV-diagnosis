import torchvision
import torch.nn as nn
import torchvision.models as models
import torch
from torchvision.models import ResNet101_Weights
import torch.nn.functional as F
import load_data
# from LoadUCF101Data import UCF101Data, SAMPLE_FRAME_NUM  # 假设这些模块存在
from torchvision.models.resnet import ResNet



###方法一
# # 定义光流网络类
# class OpticalFlowStreamNet(nn.Module):
#     def __init__(self):
#         super(OpticalFlowStreamNet, self).__init__()
#
#
#         self.OpticalFlow_stream = models.resnet101()  # 模型选用resnet101
#         # 改变resnet101的第一层和最后一层
#         self.OpticalFlow_stream.conv1 = nn.Conv2d(LoadUCF101Data.SAMPLE_FRAME_NUM * 2, 64, kernel_size=7, stride=2, padding=3,bias=False)
#         # self.OpticalFlow_stream.conv5 = nn.Conv2d(LoadUCF101Data.SAMPLE_FRAME_NUM * 2, 64, kernel_size=7, stride=2, padding=3, bias=False)
#         self.OpticalFlow_stream.fc1 = nn.Linear(in_features=2048, out_features=256)
#         self.OpticalFlow_stream.fc2 = nn.Linear(in_features=256, out_features=64)
#         self.OpticalFlow_stream.fc3 = nn.Linear(in_features=64, out_features=5)
#         self.bn1 = nn.BatchNorm2d(64)
#         self.bn2 = nn.BatchNorm2d(64)
#         self.bn3 = nn.BatchNorm2d(128)
#         self.bn4 = nn.BatchNorm2d(256)
#         self.bn5 = nn.BatchNorm2d(512)
#         self.dropout = nn.Dropout(p=0.5)
#
#     def forward(self, x):
#         # streamOpticalFlow_out = self.OpticalFlow_stream(x)
#         # return streamOpticalFlow_out
#         x = self.OpticalFlow_stream.conv1(x)
#         x = self.bn1(x)
#         x = self.OpticalFlow_stream.bn1(x)
#         x = self.OpticalFlow_stream.relu(x)
#         x = self.OpticalFlow_stream.maxpool(x)
#         x = self.OpticalFlow_stream.layer1(x)
#         x = self.bn2(x)
#         x = self.OpticalFlow_stream.layer2(x)
#         x = self.bn3(x)
#         x = self.OpticalFlow_stream.layer3(x)
#         x = self.bn4(x)
#         x = self.OpticalFlow_stream.layer4(x)
#         x = self.bn5(x)
#         x = self.OpticalFlow_stream.avgpool(x)
#         x = torch.flatten(x, 1)
#         x = self.OpticalFlow_stream.fc1(x)
#         x = self.dropout(x)  # 在 fc1 后添加 dropout
#         x = self.OpticalFlow_stream.fc2(x)
#         x = self.dropout(x)  # 在 fc2 后添加 dropout
#         x = self.OpticalFlow_stream.fc3(x)
#         return x
# # # 定RGB网络类
# class RGBStreamNet(nn.Module):
#     def __init__(self):
#         super(RGBStreamNet, self).__init__()
#         # 模型选用resnet101,并使用预训练模型
#         self.RGB_stream = models.resnet101(pretrained=True)
#         self.RGB_stream.fc1 = nn.Linear(in_features=2048, out_features=256)
#         self.RGB_stream.fc2 = nn.Linear(in_features=256, out_features=64)
#         self.RGB_stream.fc3 = nn.Linear(in_features=64, out_features=5)
#         self.dropout = nn.Dropout(p=0.5)
#         self.bn1 = nn.BatchNorm2d(64)
#         self.bn2 = nn.BatchNorm2d(64)
#         self.bn3 = nn.BatchNorm2d(128)
#         self.bn4 = nn.BatchNorm2d(256)
#         self.bn5 = nn.BatchNorm2d(512)
#
#
#     def forward(self, x):
#         # streamRGB_out = self.RGB_stream(x)
#         # return streamRGB_out
#         x = self.RGB_stream.conv1(x)
#         x = self.bn1(x)
#         x = self.RGB_stream.bn1(x)
#         x = self.RGB_stream.relu(x)
#         x = self.RGB_stream.maxpool(x)
#         x = self.RGB_stream.layer1(x)
#         x = self.bn2(x)
#         x = self.RGB_stream.layer2(x)
#         x = self.bn3(x)
#         x = self.RGB_stream.layer3(x)
#         x = self.bn4(x)
#         x = self.RGB_stream.layer4(x)
#         x = self.bn5(x)
#         x = self.RGB_stream.avgpool(x)
#         x = torch.flatten(x, 1)
#         x = self.RGB_stream.fc1(x)
#         x = self.dropout(x)  # 在 fc1 后添加 dropout
#         x = self.RGB_stream.fc2(x)
#         x = self.dropout(x)  # 在 fc2 后添加 dropout
#         x = self.RGB_stream.fc3(x)
#         return x
# # 定义双流网络类
# class TwoStreamNet(nn.Module):
#     def __init__(self, alpha=0.2):
#         super(TwoStreamNet, self).__init__()
#         self.alpha = alpha
#         self.rgb_branch = RGBStreamNet()
#         self.opticalFlow_branch = OpticalFlowStreamNet()
#
#     def forward(self, x_rgb, x_opticalFlow):
#         rgb_out = self.rgb_branch(x_rgb)
#         opticalFlow_out = self.opticalFlow_branch(x_opticalFlow)
#         # 相加融合，并采用softmax函数
#         final_out = nn.Softmax(dim=1)(rgb_out + opticalFlow_out)
#         # final_out = nn.Softmax(dim=1)(self.alpha*rgb_out + (1 - self.alpha)*opticalFlow_out)
#         return final_out






###方法二
# class OpticalFlowStreamNet(nn.Module):
#     def __init__(self):
#         super(OpticalFlowStreamNet, self).__init__()
#         # 使用 resnet101
#         self.OpticalFlow_stream = models.resnet101()
#         # 调整输入通道数以适应光流数据
#         self.OpticalFlow_stream.conv1 = nn.Conv2d(LoadUCF101Data.SAMPLE_FRAME_NUM * 2, 64, kernel_size=7, stride=2, padding=3, bias=False)
#         # 构建更复杂的全连接层，增强网络表达能力
#         self.fc = nn.Sequential(
#             nn.Linear(2048, 1024),
#             nn.BatchNorm1d(1024),
#             nn.ReLU(),
#             nn.Dropout(p=0.3),
#             nn.Linear(1024, 512),
#             nn.BatchNorm1d(512),
#             nn.ReLU(),
#             nn.Dropout(p=0.3),
#             nn.Linear(512, 5)
#         )
#
#     def forward(self, x):
#         x = self.OpticalFlow_stream.conv1(x)
#         x = self.OpticalFlow_stream.bn1(x)
#         x = self.OpticalFlow_stream.relu(x)
#         x = self.OpticalFlow_stream.maxpool(x)
#         x = self.OpticalFlow_stream.layer1(x)
#         x = self.OpticalFlow_stream.layer2(x)
#         x = self.OpticalFlow_stream.layer3(x)
#         x = self.OpticalFlow_stream.layer4(x)
#         x = self.OpticalFlow_stream.avgpool(x)
#         x = torch.flatten(x, 1)
#         x = self.fc(x)
#         return x
# class RGBStreamNet(nn.Module):
#     def __init__(self):
#         super(RGBStreamNet, self).__init__()
#         # 使用预训练的 resnet101
#         self.RGB_stream = models.resnet101(pretrained=True)
#         # 构建更复杂的全连接层
#         self.fc = nn.Sequential(
#             nn.Linear(2048, 1024),
#             nn.BatchNorm1d(1024),
#             nn.ReLU(),
#             nn.Dropout(p=0.3),
#             nn.Linear(1024, 512),
#             nn.BatchNorm1d(512),
#             nn.ReLU(),
#             nn.Dropout(p=0.3),
#             nn.Linear(512, 5)
#         )
#
#     def forward(self, x):
#         x = self.RGB_stream.conv1(x)
#         x = self.RGB_stream.bn1(x)
#         x = self.RGB_stream.relu(x)
#         x = self.RGB_stream.maxpool(x)
#         x = self.RGB_stream.layer1(x)
#         x = self.RGB_stream.layer2(x)
#         x = self.RGB_stream.layer3(x)
#         x = self.RGB_stream.layer4(x)
#         x = self.RGB_stream.avgpool(x)
#         x = torch.flatten(x, 1)
#         x = self.fc(x)
#         return x
# class TwoStreamNet(nn.Module):
#     def __init__(self):
#         super(TwoStreamNet, self).__init__()
#
#         self.rgb_branch = RGBStreamNet()
#         self.opticalFlow_branch = OpticalFlowStreamNet()
#
#     def forward(self, x_rgb, x_opticalFlow):
#         rgb_out = self.rgb_branch(x_rgb)
#         opticalFlow_out = self.opticalFlow_branch(x_opticalFlow)
#         return rgb_out + opticalFlow_out



###方法三
#加入注意力机制
# class TemporalAttention(nn.Module):
#     """时间注意力机制模块"""
#     def __init__(self, hidden_size):
#         super(TemporalAttention, self).__init__()
#         self.attn = nn.Sequential(
#             nn.Linear(hidden_size, hidden_size // 2),  # 压缩维度减少计算量
#             nn.Tanh(),
#             nn.Linear(hidden_size // 2, 1)             # 每个时间步输出一个注意力分数
#         )
#         self.softmax = nn.Softmax(dim=1)  # 沿时间步维度做归一化
#
#     def forward(self, lstm_out):
#         # lstm_out形状: (batch_size, time_steps, hidden_size)
#         attn_scores = self.attn(lstm_out)          # (batch_size, time_steps, 1)
#         attn_scores = attn_scores.squeeze(2)       # (batch_size, time_steps)
#         attn_weights = self.softmax(attn_scores)   # 归一化权重
#         context = torch.bmm(attn_weights.unsqueeze(1), lstm_out)  # (batch_size, 1, hidden_size)
#         return context.squeeze(1), attn_weights    # 返回上下文向量和权重（可选）
#
#
#
# class OpticalFlowStreamNet(nn.Module):
#     def __init__(self, hidden_size=512, num_layers=2, bidirectional=True):
#         super(OpticalFlowStreamNet, self).__init__()
#         # 使用 resnet101
#         self.OpticalFlow_stream = models.resnet101()
#         # # 调整输入通道数以适应光流数据
#         # self.OpticalFlow_stream.conv1 = nn.Conv2d(LoadUCF101Data.SAMPLE_FRAME_NUM * 2, 64, kernel_size=7, stride=2, padding=3, bias=False)
#         self.OpticalFlow_stream.conv1 = nn.Conv2d(2, 64, kernel_size=7, stride=2, padding=3, bias=False)  # 输入光流u/v两通道
#
#         # 改进点2：增强时序建模能力
#         self.lstm = nn.LSTM(
#             input_size=2048,  # 修正输入维度匹配特征通道
#             hidden_size=hidden_size,
#             num_layers=num_layers,
#             bidirectional=bidirectional,  # 双向LSTM捕捉前后时序
#             dropout=0.3 if num_layers > 1 else 0,
#             batch_first=True
#         )
#         # 注意力机制（考虑双向输出的维度）
#         self.attention = TemporalAttention(hidden_size * 2 if bidirectional else hidden_size)
#         # 分类层
#         lstm_output_size = hidden_size * 2 if bidirectional else hidden_size
#         self.fc = nn.Linear(lstm_output_size, 5)
#         # # 改进点3：增强特征融合
#         # self.attention = nn.Sequential(
#         #     nn.Linear(hidden_size * 2, 128),
#         #     nn.Tanh(),
#         #     nn.Linear(128, 1, bias=False)
#         # )
#         # self.fc = nn.Sequential(
#         #     nn.LayerNorm(hidden_size * 2),
#         #     nn.Linear(hidden_size * 2, 5)
#         # )
#         # self.fc = nn.Linear(hidden_size, 5)
#
#     def forward(self, x):
#         # 输入x形状: (batch_size, 2*T, H, W) → 拆分为T个时间步
#         batch_size, C, H, W = x.size()
#         T = LoadUCF101Data.SAMPLE_FRAME_NUM  # 时间步数（假设输入通道为2*T）
#         # 调整形状为 (batch*T, 2, H, W)
#         x = x.view(batch_size * T, 2, H, W)
#         x = self.OpticalFlow_stream.conv1(x)
#         x = self.OpticalFlow_stream.bn1(x)
#         x = self.OpticalFlow_stream.relu(x)
#         x = self.OpticalFlow_stream.maxpool(x)
#         x = self.OpticalFlow_stream.layer1(x)
#         x = self.OpticalFlow_stream.layer2(x)
#         x = self.OpticalFlow_stream.layer3(x)
#         x = self.OpticalFlow_stream.layer4(x)
#         x = self.OpticalFlow_stream.avgpool(x)
#         x = torch.flatten(x, 1)  # (batch*T, 2048)
#         # 恢复为时间序列 (batch, T, 2048)
#         x = x.view(batch_size, T, -1)
#         # 双向LSTM处理
#         lstm_out, _ = self.lstm(x)  # (batch, T, hidden_size*2)
#         out= self.fc(lstm_out[:,-1,:])
#
#         # # 时间注意力加权
#         # context, _ = self.attention(lstm_out)  # (batch, hidden_size*2)
#
#         # # 分类
#         # out = self.fc(context)
#         return out

        # # 修改后的维度变换
        # x = x.flatten(2)  # (B,2048,7,7) → (B,2048,7*7)
        # x = x.permute(0, 2, 1)  # (B,7*7,2048)
        # x = x.permute(0, 2, 1)  # (B,2048,7*7) → 视为 (B, seq_len=2048, input_size=49)

        # x = self.OpticalFlow_stream.avgpool(x)
        #从第 1 维度（即 C 维度）开始展平，将 C、H 和 W 维度合并为一个维度，输出张量的形状为 (batch_size, C * H * W)。
        # x = torch.flatten(x, 1)

        # # 需要将输出扩展维度以适应LSTM输入
        # x = x.unsqueeze(1)  # 添加一个时间维度，
        # lstm_out, _ = self.lstm(x)  # LSTM处理
        # lstm_out = lstm_out[:, -1, :]  # 只取最后一个时刻的输出
        # out = self.fc(lstm_out)

        # # 改进时序建模
        # lstm_out, _ = self.lstm(x)  # [B,49,hidden_size*2]
        #
        # # 注意力机制（增强关键帧识别）
        # attn_weights = torch.softmax(self.attention(lstm_out), dim=1)
        # context_vector = torch.sum(attn_weights * lstm_out, dim=1)
        #
        # return self.fc(context_vector)
        # lstm_out, _ = self.lstm(x)
        # lstm_out = lstm_out[:, -1, :]  # 取最后一个时间步的输出
        # out = self.fc(lstm_out)
        #
        # return out
# class RGBStreamNet(nn.Module):
#     def __init__(self, hidden_size=512, num_layers=2,bidirectional=True):
#         super(RGBStreamNet, self).__init__()
#         # 使用预训练的 resnet101
#         self.RGB_stream = models.resnet101(weights=ResNet101_Weights.DEFAULT)
#         self.RGB_stream.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)  # 输入rgb3通道
#         # 改进点2：增强时序建模能力
#         self.lstm = nn.LSTM(
#             input_size=2048,  # 修正输入维度匹配特征通道
#             hidden_size=hidden_size,
#             num_layers=num_layers,
#             bidirectional=bidirectional,  # 双向LSTM捕捉前后时序
#             dropout=0.3 if num_layers > 1 else 0,
#             batch_first=True
#         )
#         # 注意力机制（考虑双向输出的维度）
#         self.attention = TemporalAttention(hidden_size * 2 if bidirectional else hidden_size)
#         # 分类层
#         lstm_output_size = hidden_size * 2 if bidirectional else hidden_size
#         self.fc = nn.Linear(lstm_output_size, 5)
#
#         # # 构建更复杂的全连接层
#         # self.fc = nn.Sequential(
#         #     nn.Linear(2048, 1024),
#         #     nn.BatchNorm1d(1024),
#         #     nn.ReLU(),
#         #     nn.Dropout(p=0.3),
#         #     nn.Linear(1024, 512),
#         #     nn.BatchNorm1d(512),
#         #     nn.ReLU(),
#         #     nn.Dropout(p=0.3),
#         #     nn.Linear(512, 5)
#         #         )
#
#
#
#         # 添加 LSTM 层
#         # self.lstm = nn.LSTM(input_size=2048, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
#         # self.fc = nn.Linear(hidden_size, 5)
#
#     def forward(self, x):
#         batch_size, C, H, W = x.size()
#         T = LoadUCF101Data.SAMPLE_FRAME_NUM  # 时间步数（假设输入通道为3*T）
#         # 调整形状为 (batch*T, 2, H, W)
#         x = x.view(batch_size * T, 1, H, W)
#
#
#         x = self.RGB_stream.conv1(x)
#         x = self.RGB_stream.bn1(x)
#         x = self.RGB_stream.relu(x)
#         x = self.RGB_stream.maxpool(x)
#         x = self.RGB_stream.layer1(x)
#         x = self.RGB_stream.layer2(x)
#         x = self.RGB_stream.layer3(x)
#         x = self.RGB_stream.layer4(x)
#         x = self.RGB_stream.avgpool(x)
#         x = torch.flatten(x, 1)
#
#         # 恢复为时间序列 (batch, T, 2048)
#         x = x.view(batch_size, T, -1)
#         # 双向LSTM处理
#         lstm_out, _ = self.lstm(x)
#         # # 时间注意力加权
#         # context, _ = self.attention(lstm_out)
#         # # 分类
#         # out = self.fc(context)
#         out = self.fc(lstm_out[:,-1,:])
#
#         return out
# class TwoStreamNet(nn.Module):
#     def __init__(self, hidden_size=512, num_layers=2,bidirectional=True):
#         super(TwoStreamNet, self).__init__()
#
#         self.rgb_branch = RGBStreamNet(hidden_size, num_layers)
#         self.opticalFlow_branch = OpticalFlowStreamNet(hidden_size, num_layers,bidirectional)
#
#     def forward(self, x_rgb, x_opticalFlow):
#         rgb_out = self.rgb_branch(x_rgb)
#         opticalFlow_out = self.opticalFlow_branch(x_opticalFlow)
#         return rgb_out + opticalFlow_out

#方法四
#时间注意力
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

#空间注意力cbam
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
# 自定义Bottleneck，加入CBAM
class BottleneckWithCBAM(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BottleneckWithCBAM, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        self.conv1 = nn.Conv2d(inplanes, width, 1, bias=False)
        self.bn1 = norm_layer(width)
        self.conv2 = nn.Conv2d(width, width, 3, stride, dilation, dilation, groups, bias=False)
        self.bn2 = norm_layer(width)
        self.conv3 = nn.Conv2d(width, planes * self.expansion, 1, bias=False)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.cbam = CBAM(planes * self.expansion)  # 添加CBAM

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        out = self.cbam(out)  # 应用CBAM

        return out

class RGBStreamNet(nn.Module):
    def __init__(self, n_segment=40):
        super(RGBStreamNet, self).__init__()
        backbone = torchvision.models.resnet18()
        backbone.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.backbone = make_temporal_modeling(backbone, n_segment)

    def forward(self, x):
        B, T, C, H, W = x.size()
        x = x.view(B * T, C, H, W)
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)

        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)
        x = self.backbone.avgpool(x)   ###（2000，512，1，1）
        x = torch.flatten(x, 1)   ###（2000，512）   x 的维度变为 [B*T, C*H*W] 。即从第 1 维（索引从 0 开始）开始，将后面的维度全部展平成一个一维向量。
        x = x.view(B, T, -1).mean(dim=1)  # （10，512）  视图变换
        return x

class OpticalFlowStreamNet(nn.Module):
    def __init__(self):
        super(OpticalFlowStreamNet, self).__init__()
        self.backbone = ResNet(BottleneckWithCBAM, [3, 4, 23, 3])
        self.backbone.conv1 = nn.Conv2d(load_data.SAMPLE_FRAME_NUM * 2, 64, kernel_size=7, stride=2, padding=3, bias=False)

    def forward(self, x):
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)

        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)  #（10，2048，4，4）
        x = self.backbone.avgpool(x)   #（10，2048，1，1）
        x = torch.flatten(x, 1)  #（10，2048）
        return x

class TwoStreamNet(nn.Module):
    def __init__(self, n_segment=40, num_classes=5):
        super(TwoStreamNet, self).__init__()
        self.rgb_branch = RGBStreamNet(n_segment)
        self.optical_branch = OpticalFlowStreamNet()
        self.fusion_fc = nn.Sequential(
            nn.Linear(512 + 2048, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes),
            nn.Softmax(dim=1)  # 添加 softmax 层
        )
        # ⭐ 新增：特征映射层
        self.feature_proj = nn.Linear(2560, 128)

    def get_fusion_feature(self, x_rgb, x_flow):
        rgb_feat = self.rgb_branch(x_rgb)  # (B, 512)
        flow_feat = self.optical_branch(x_flow)  # (B, 2048)
        fusion_feat = torch.cat([rgb_feat, flow_feat], dim=1)  # (B, 2560)
        return fusion_feat
        # # ⭐ 映射到128维
        # feature_128 = self.feature_proj(fusion_feat)
        # return feature_128  # 不送入 fusion_fc，返回中间层特征

    def forward(self, x_rgb, x_flow):
        rgb_feat = self.rgb_branch(x_rgb)     # （10, 512）
        flow_feat = self.optical_branch(x_flow)  # （10, 2048）
        fusion_feat = torch.cat([rgb_feat, flow_feat], dim=1)  ###（10，2560）
        out = self.fusion_fc(fusion_feat)  # （10, 5）
        return out

if __name__=="__main__":
    # 创建一个维度为 torch.Size([10, 200, 1, 112, 112]) 的张量
    RGB_images = torch.randn(10, 200, 1, 112, 112)

    # 创建一个维度为 torch.Size([10, 100, 112, 112]) 的张量
    OpticalFlow_images = torch.randn(10, 100, 112, 112)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    twoStreamNet = TwoStreamNet().to(device)
    RGB_images = RGB_images.to(device)
    OpticalFlow_images = OpticalFlow_images.to(device)
    output = twoStreamNet(RGB_images, OpticalFlow_images)






