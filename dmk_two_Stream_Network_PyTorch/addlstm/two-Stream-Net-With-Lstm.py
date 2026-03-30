import torch
import torch.nn as nn
import torchvision.models as models


# Optical Flow Stream with LSTM
class OpticalFlowStreamNetWithLSTM(nn.Module):
    def __init__(self, lstm_input_size=2048, hidden_size=512, num_layers=1, num_classes=5):
        super(OpticalFlowStreamNetWithLSTM, self).__init__()
        # ResNet101 Backbone for Optical Flow stream
        self.OpticalFlow_stream = models.resnet101()
        self.OpticalFlow_stream.conv1 = nn.Conv2d(LoadUCF101Data.SAMPLE_FRAME_NUM * 2, 64, kernel_size=7, stride=2,
                                                  padding=3, bias=False)

        # LSTM Layer
        self.lstm = nn.LSTM(lstm_input_size, hidden_size, num_layers, batch_first=True)

        # Fully connected layer after LSTM
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        # x shape: (batch_size, seq_len, channels, height, width)
        batch_size, seq_len, c, h, w = x.size()

        # 通过ResNet101提取特征
        x = self.OpticalFlow_stream.conv1(x.view(batch_size * seq_len, c, h, w))  # 把序列维度合并到batch中
        x = self.OpticalFlow_stream.bn1(x)
        x = self.OpticalFlow_stream.relu(x)
        x = self.OpticalFlow_stream.maxpool(x)
        x = self.OpticalFlow_stream.layer1(x)
        x = self.OpticalFlow_stream.layer2(x)
        x = self.OpticalFlow_stream.layer3(x)
        x = self.OpticalFlow_stream.layer4(x)
        x = self.OpticalFlow_stream.avgpool(x)
        x = torch.flatten(x, 1)

        # 重塑为LSTM输入格式 (batch_size, seq_len, feature_dim)
        x = x.view(batch_size, seq_len, -1)

        # LSTM层
        lstm_out, _ = self.lstm(x)  # LSTM的输出，忽略LSTM的隐藏状态

        # 选择最后一个时刻的输出作为分类输入
        lstm_out = lstm_out[:, -1, :]  # 取序列的最后一帧输出

        # 通过全连接层进行分类
        x = self.fc(lstm_out)
        return x


# RGB Stream with LSTM
class RGBStreamNetWithLSTM(nn.Module):
    def __init__(self, lstm_input_size=2048, hidden_size=512, num_layers=1, num_classes=5):
        super(RGBStreamNetWithLSTM, self).__init__()
        # ResNet101 Backbone for RGB stream
        self.RGB_stream = models.resnet101(pretrained=True)

        # LSTM Layer
        self.lstm = nn.LSTM(lstm_input_size, hidden_size, num_layers, batch_first=True)

        # Fully connected layer after LSTM
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        # x shape: (batch_size, seq_len, channels, height, width)
        batch_size, seq_len, c, h, w = x.size()

        # 通过ResNet101提取特征
        x = self.RGB_stream.conv1(x.view(batch_size * seq_len, c, h, w))  # 把序列维度合并到batch中
        x = self.RGB_stream.bn1(x)
        x = self.RGB_stream.relu(x)
        x = self.RGB_stream.maxpool(x)
        x = self.RGB_stream.layer1(x)
        x = self.RGB_stream.layer2(x)
        x = self.RGB_stream.layer3(x)
        x = self.RGB_stream.layer4(x)
        x = self.RGB_stream.avgpool(x)
        x = torch.flatten(x, 1)

        # 重塑为LSTM输入格式 (batch_size, seq_len, feature_dim)
        x = x.view(batch_size, seq_len, -1)

        # LSTM层
        lstm_out, _ = self.lstm(x)  # LSTM的输出，忽略LSTM的隐藏状态

        # 选择最后一个时刻的输出作为分类输入
        lstm_out = lstm_out[:, -1, :]  # 取序列的最后一帧输出

        # 通过全连接层进行分类
        x = self.fc(lstm_out)
        return x


# 双流网络 (Two-Stream Network with LSTM)
class TwoStreamNetWithLSTM(nn.Module):
    def __init__(self):
        super(TwoStreamNetWithLSTM, self).__init__()

        self.rgb_branch = RGBStreamNetWithLSTM()
        self.opticalFlow_branch = OpticalFlowStreamNetWithLSTM()

    def forward(self, x_rgb, x_opticalFlow):
        rgb_out = self.rgb_branch(x_rgb)
        opticalFlow_out = self.opticalFlow_branch(x_opticalFlow)

        # 融合RGB流和光流流的输出
        return rgb_out + opticalFlow_out

