"""
训练文件
"""
###
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_fscore_support  # 新增导入
from load_data import trainset_loader, testset_loader
from MSTA_DSN import TwoStreamNet
import torch
import torch.optim as optim
import os
import socket
from tqdm import tqdm
from datetime import datetime
from tensorboardX import SummaryWriter
import torch.nn.functional as F


# 超参数
EPOCH = 100
LEARNING_RATE = 1e-3
MOMENTUM = 0.9
N_TEST = 1
N_SAVE = 1
SAVE_DIR = "./model/IEEE"

# 初始化绘图参数
plt.style.use('ggplot')
plt.rcParams.update({
    'font.size': 14,
    'font.family': 'serif',
    'lines.linewidth': 2,
    'axes.linewidth': 1.5,
    'xtick.major.width': 1.5,
    'ytick.major.width': 1.5,
    'figure.dpi': 300
})

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
twoStreamNet = TwoStreamNet().to(device)

# 加载上次训练的模型
start_epoch = 1  # 从第start_epoch轮开始
checkpoint_path = os.path.join("./model/", f"checkpoint-120.pth")

rgb_params = list(twoStreamNet.rgb_branch.parameters())
flow_params = list(twoStreamNet.optical_branch.parameters())

# 需要同时更新两个流的参数
optimizer = optim.Adam([
    {'params': rgb_params, 'lr': LEARNING_RATE},              # RGB分支
    {'params': flow_params, 'lr': LEARNING_RATE * 0.5}        # 光流分支
])
if os.path.exists(checkpoint_path):
    state = torch.load(checkpoint_path)
    twoStreamNet.load_state_dict(state['model'])
    print("加载模型成功")

scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.9)

def save_checkpoint(path, model, optimizer):
    state = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict()
    }
    torch.save(state, path)


def test(i_epoch, epoch, writer):
    twoStreamNet.eval()
    runing_correct = 0
    runing_loss = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for data in tqdm(testset_loader):
            RGB_images, OpticalFlow_images, label,_ = data
            RGB_images = RGB_images.to(device)
            OpticalFlow_images = OpticalFlow_images.to(device)
            label = label.to(device)

            output = twoStreamNet(RGB_images, OpticalFlow_images)
            loss = F.cross_entropy(output, label)

            runing_loss += loss.item() * len(label)
            preds = output.argmax(dim=1)
            runing_correct += preds.eq(label).sum().item()

            # 收集预测结果和标签
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(label.cpu().numpy())

    # 计算各项指标
    epoch_loss = runing_loss / len(testset_loader.dataset)
    epoch_acc = (runing_correct * 100.0) / len(testset_loader.dataset)

    # 计算Recall和F1 Score
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='macro', zero_division=0)

    # 记录到TensorBoard
    writer.add_scalar('data/test_loss_epoch', epoch_loss, i_epoch - 1)
    writer.add_scalar('data/test_acc_epoch', epoch_acc, i_epoch - 1)
    writer.add_scalar('data/test_recall_epoch', recall, i_epoch - 1)
    writer.add_scalar('data/test_f1_epoch', f1, i_epoch - 1)

    print(f"[test] Epoch: {i_epoch}/{epoch} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.2f}% "
          f"Recall: {recall:.4f} F1: {f1:.4f}")

    return epoch_loss, epoch_acc, recall, f1


def train(epoch, N_TEST, N_SAVE, save_dir):
    log_dir = os.path.join(save_dir, 'models', datetime.now().strftime('%b%d_%H-%M-%S') + '_' + socket.gethostname())
    writer = SummaryWriter(log_dir=log_dir)

    # 初始化记录列表
    train_losses, train_accs = [], []
    test_losses, test_accs, test_recalls, test_f1s = [], [], [], []

    for i in range(start_epoch - 1, epoch):
        twoStreamNet.train()
        runing_correct = 0
        runing_loss = 0

        for data in tqdm(trainset_loader):
            RGB_images, OpticalFlow_images, label,_= data
            RGB_images = RGB_images.to(device)
            OpticalFlow_images = OpticalFlow_images.to(device)
            label = label.to(device)

            optimizer.zero_grad()
            output = twoStreamNet(RGB_images, OpticalFlow_images)
            loss = F.cross_entropy(output, label)
            loss.backward()
            optimizer.step()

            runing_loss += loss.item() * len(label)
            preds = output.argmax(dim=1)
            runing_correct += preds.eq(label).sum().item()

        # 计算训练指标
        epoch_loss = runing_loss / len(trainset_loader.dataset)
        epoch_acc = (runing_correct * 100.0) / len(trainset_loader.dataset)
        train_losses.append(epoch_loss)
        train_accs.append(epoch_acc)

        # 记录训练指标
        writer.add_scalar('data/train_loss_epoch', epoch_loss, i)
        writer.add_scalar('data/train_acc_epoch', epoch_acc, i)
        print(f"[train] Epoch: {i + 1}/{epoch} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.2f}%")

        # 测试阶段
        if (i + 1) % N_TEST == 0:
            test_loss, test_acc, test_recall, test_f1 = test(i + 1, epoch, writer)
            test_losses.append(test_loss)
            test_accs.append(test_acc)
            test_recalls.append(test_recall)
            test_f1s.append(test_f1)

        #保存模型
        if (i + 1) % N_SAVE == 0:
            save_path = os.path.join(save_dir, f"checkpoint-{i +1}.pth")
            save_checkpoint(save_path, twoStreamNet, optimizer)

        scheduler.step()
        print(f"当前学习率：{optimizer.param_groups[0]['lr']:.6f}")

    # 绘制并保存曲线
    plt.figure(figsize=(12, 6))

    # 损失曲线
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, 'b-', label='Train Loss')
    plt.plot(test_losses, 'b--', label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.title('Loss Curve')

    # 准确率曲线
    plt.subplot(1, 2, 2)
    plt.plot(train_accs, 'r-', label='Train Accuracy')
    plt.plot(test_accs, 'r--', label='Test Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.title('Accuracy Curve')

    plt.tight_layout()

    # 保存图片
    save_path = r"训练过程.png"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()

    print(f"结果已保存至：{save_path}")


if __name__ == '__main__':
    train(EPOCH, N_TEST, N_SAVE, SAVE_DIR)