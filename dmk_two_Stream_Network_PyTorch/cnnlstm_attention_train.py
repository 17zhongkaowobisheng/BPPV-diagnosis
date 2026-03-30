"""
训练文件
"""
from sklearn.model_selection import KFold
from sympy.stats import moment
from torch.utils.data import DataLoader

from load_data import trainset_loader, testset_loader, combined_loader,TRAIN_BATCH_SIZE,TEST_BATCH_SIZE

from cnnlstm_attention1 import CNNLSTM
import torch
import torch.optim as optim
import os
import socket
import torch.nn.functional as F
from tqdm import tqdm
from datetime import datetime
from tensorboardX import SummaryWriter
from torchvision import transforms

#
# 超参数
EPOCH = 50  # 总训练轮数
LEARNING_RATE = 1e-2 # 初始学习率
MOMENTUM = 0.9  # 动量因子
N_TEST = 1  # 每N_TEST轮测试一次
N_SAVE = 1  # 每N_SAVE轮保存一次模型
K_FOLD = 5  # 交叉验证的折数
SAVE_DIR = "./model/TSM_segment40"


# 选择设备
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = torch.device( "cpu")

# # 双流网络模型
# twoStreamNet = TwoStreamNet().to(device)

## cnnLstm模型
cnnLstm = CNNLSTM().to(device)

#加载最优模型
# state = torch.load(SAVE_DIR + "/30/connect1.pth", weights_only=True)
# twoStreamNet.load_state_dict(state['model'])
#
#加载上次训练的模型
j=50
state = torch.load(SAVE_DIR + "/checkpoint-{}.pth".format(j), weights_only=True)
cnnLstm.load_state_dict(state['model'])




#ADM优化器
optimizer=optim.Adam(params=cnnLstm.parameters(), lr=LEARNING_RATE)



scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.9)
optimizer.load_state_dict(state['optimizer'])
# 保存模型
def save_checkpoint(path, model, optimizer):
    state = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict()
    }
    torch.save(state, path)


# 测试函数
def test(i_epoch, epoch, writer):
    cnnLstm.eval()

    runing_correct = 0
    runing_loss = 0
    #上下文管理器来禁用梯度计算，减少内存消耗并提高计算效率。
    with torch.no_grad():
        for data in tqdm(testset_loader):
        # for data in testset_loader:
            # 加载batch_size个数据
            RGB_images, OpticalFlow_images, label = data
            RGB_images = RGB_images.to(device)

            label = label.to(device)
            # 计算batch_size损失
            output = cnnLstm(RGB_images)
            loss = F.cross_entropy(output, label)
            runing_loss += loss.item() * len(label)
            # 计算测试集batch_size中正确的个数
            max_value, max_index = output.max(1, keepdim=True)
            runing_correct += max_index.eq(label.view_as(max_index)).sum().item()



        # 计算每个epoch上的平均损失
        epoch_loss = runing_loss / len(testset_loader.dataset)
        epoch_acc = (runing_correct * 1.0 * 100 / len(testset_loader.dataset))  # 计算准确度



        writer.add_scalar('data/test_loss_epoch', epoch_loss, i_epoch-1)
        writer.add_scalar('data/test_acc_epoch', epoch_acc, i_epoch-1)

        print("[{}] Epoch: {}/{} Loss: {} Acc: {}".format('test', i_epoch, epoch, epoch_loss, epoch_acc))
        # scheduler.step(epoch_loss)  # 学习率衰减策略，传入平均损失



# 训练函数
def train(epoch, N_TEST, N_SAVE, save_dir):
    # 存入日志文件，方便使用tensorboard工具观看训练变化
    log_dir = os.path.join(save_dir, 'models', datetime.now().strftime('%b%d_%H-%M-%S') + '_' + socket.gethostname())
    writer = SummaryWriter(log_dir=log_dir)
    for i in range(epoch):
        cnnLstm.train()

        runing_correct = 0
        runing_loss = 0
        for data in tqdm(trainset_loader):
        # for data in trainset_loader:
            # 加载bach_size个数据
            RGB_images, OpticalFlow_images, label = data
            RGB_images = RGB_images.to(device)
            # print(RGB_images.size())
            OpticalFlow_images = OpticalFlow_images.to(device)
            label = label.to(device)
            # 更新一次模型参数
            optimizer.zero_grad()
            output = cnnLstm(RGB_images)
            loss = F.cross_entropy(output, label)
            loss.backward()
            optimizer.step()
            # 计算当前训练损失
            runing_loss += loss.item() * len(label)
            # 计算当前正确个数
            max_value, max_index = output.max(1, keepdim=True)
            runing_correct += max_index.eq(label.view_as(max_index)).sum().item()


        scheduler.step()
        print("第%d个epoch的学习率：%f" % (i + 1, optimizer.param_groups[0]['lr']))

        # 计算每个epoch上的平均损失
        epoch_loss = runing_loss / len(trainset_loader.dataset)
        epoch_acc = (runing_correct * 1.0 * 100 / len(trainset_loader.dataset))  # 计算准确度
        # scheduler.step(epoch_loss)  # 学习率衰减策略，传入平均损失
        # 写入日志文件中
        writer.add_scalar('data/train_loss_epoch', epoch_loss, i)
        writer.add_scalar('data/train_acc_epoch', epoch_acc, i)
        # 打印当前轮结果
        print("[{}] Epoch: {}/{} Loss: {} Acc: {}".format('train', i + 1, epoch, epoch_loss, epoch_acc))
        # 训练N_TEST轮测试一次
        if (i+1) % N_TEST == 0:
            test(i + 1, epoch, writer)
        # 训练N_SAVE轮测试一次，保存一次模型
        if (i+1) % N_SAVE == 0:
            save = save_dir + "/checkpoint-%i.pth"
            save_checkpoint(save % (i+j+1), cnnLstm, optimizer)
            print("Save checkpoint-{}.pth at {}\n".format(i+j+1, "./model"))



if __name__ == '__main__':
    # 训练和验证
    train(EPOCH, N_TEST, N_SAVE, SAVE_DIR)