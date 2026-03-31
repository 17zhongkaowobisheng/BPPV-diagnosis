"""
测试
"""
from MSTA_DSN import TwoStreamNet
from load_data import testset
import load_data
from load_data import  testset_loader
from torch.utils.data import DataLoader
import torch
import time

# 测试集
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# 网络模型
twoStreamNet = TwoStreamNet().to(device)

if __name__ == '__main__':

    # 加载模型文件
    state = torch.load('model/self_created/best.pth')
    # state = torch.load('model/public/best.pth')
    twoStreamNet.load_state_dict(state['model'], strict=False)

    # 将选好模型的双流网络选择设备去运行，并评估
    twoStreamNet = twoStreamNet.to(device)
    twoStreamNet.eval()
    total_samples = 0
    # ===== 开始计时 =====
    start_time = time.time()

    with torch.no_grad():
        t = 0
        for RGB_img, opticalFlowStackedImg, actual_label, video_name in testset_loader:

            RGB_img = RGB_img.to(device)
            opticalFlowStackedImg = opticalFlowStackedImg.to(device)

            # GPU同步（保证时间准确）
            if device.type == 'cuda':
                torch.cuda.synchronize()

            output = twoStreamNet(RGB_img, opticalFlowStackedImg)

            if device.type == 'cuda':
                torch.cuda.synchronize()

            prob = output
            max_value, max_index = torch.max(prob, 1)
            pred_class = max_index

            total_samples += RGB_img.size(0)

            for i, (act, pred) in enumerate(zip(actual_label, pred_class)):
                # '视频：', video_name[i],
                print('视频：', video_name[i],
                      '真实类别：', load_data.classInd[act],
                      '预测类别：', load_data.classInd[pred],
                      '概率：', max_value[i].item())
                if load_data.classInd[act] != load_data.classInd[pred]:
                    t += 1
        x = (115-t)/115
        print("预测概率值",x)

    # ===== 结束计时 =====
    end_time = time.time()

    total_time = end_time - start_time
    avg_time = total_time / total_samples
    fps = total_samples / total_time

    print("\n==============================")
    print("测试完成")
    print("测试样本数量:", total_samples)
    print("总测试时间: %.4f 秒" % total_time)
    print("平均每个样本耗时: %.6f 秒" % avg_time)
    print("推理速度 (FPS): %.2f" % fps)
    print("==============================")











"""
对整个测试集进行推理，并生成 t-SNE 散点图可视化
"""
import pandas as pd

# import pandas as pd
#
# from two_Stream_Net import TwoStreamNet
# from lstm_Load import trainset_loader, classInd
# from sklearn.manifold import TSNE
# from torch.utils.data import DataLoader
# import matplotlib.pyplot as plt
# import numpy as np
# import torch

# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# twoStreamNet = TwoStreamNet().to(device)
#
# if __name__ == '__main__':
#     # 加载模型
#     state = torch.load('model/final1/checkpoint-100.pth', map_location=device)
#     twoStreamNet.load_state_dict(state['model'])
#     twoStreamNet.eval()
#
#     # 提取特征和标签
#     all_features = []
#     all_labels = []
#
#     with torch.no_grad():
#         for RGB_img, opticalFlowStackedImg, actual_label in trainset_loader:
#             RGB_img = RGB_img.to(device)
#             opticalFlowStackedImg = opticalFlowStackedImg.to(device)
#
#             # 提取 batch 中所有样本的融合特征
#             fusion_feat = twoStreamNet.get_fusion_feature(RGB_img, opticalFlowStackedImg)  # shape: [B, 2560]
#
#             all_features.append(fusion_feat.cpu().numpy())  # shape: [B, 2560]
#             all_labels.append(actual_label.cpu().numpy())  # shape: [B]
#
#     # 拼接为整体数组
#     all_features = np.concatenate(all_features, axis=0)
#     all_labels = np.concatenate(all_labels, axis=0)
#     print("总样本数量：", all_features.shape[0])  # ✅ 应该等于 testset 的长度
#
#     # t-SNE 降维到2维
#     tsne = TSNE(n_components=2, random_state=42, perplexity=30)
#     tsne_results = tsne.fit_transform(all_features)
#
#     # 可视化散点图
#     plt.figure(figsize=(8, 6))
#     label_names = ['left', 'leftRear', 'right', 'rightRear', 'normal']
#     colors = ['red', 'orange', 'blue', 'green', 'purple']
#
#     for i in range(5):
#         idx = all_labels == i
#         plt.scatter(tsne_results[idx, 0], tsne_results[idx, 1], c=colors[i], label=label_names[i], alpha=0.7)
#
#     plt.title("t-SNE Visualization of BPPV Fusion Features")
#     plt.xlabel("t-SNE Dimension 1")
#     plt.ylabel("t-SNE Dimension 2")
#     plt.legend()
#     plt.grid(True)
#     plt.tight_layout()
#     plt.savefig("tsne_bppv_result.png")
#     plt.show()












# ##绘制结果可视化散点图
# from two_Stream_Net import TwoStreamNet
# from lstm_Load import trainset_loader, classInd
# from sklearn.manifold import TSNE
# from torch.utils.data import DataLoader
# import matplotlib.pyplot as plt
# import numpy as np
# import torch
# import seaborn as sns
# import matplotlib as mpl
# # 设置图形样式和字体
# plt.style.use('seaborn-v0_8-whitegrid')
# mpl.rcParams.update({
#     'font.family': 'serif',
#     'font.size': 10,
#     'axes.titlesize': 14,
#     'axes.labelsize': 12,
#     'grid.color': 'lightgrey',
#     'grid.linestyle': '--',
#     'grid.linewidth': 0.5
# })
#
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# twoStreamNet = TwoStreamNet().to(device)
#
# if __name__ == '__main__':
#     # 加载模型
#     state = torch.load('model/final1/checkpoint-100.pth', map_location=device)
#     twoStreamNet.load_state_dict(state['model'],strict=False)
#     twoStreamNet.eval()
#
#     # 提取特征和标签
#     all_features = []
#     all_labels = []
#
#     with torch.no_grad():
#         for RGB_img, opticalFlowStackedImg, actual_label,_ in trainset_loader:
#             RGB_img = RGB_img.to(device)
#             opticalFlowStackedImg = opticalFlowStackedImg.to(device)
#
#             # 提取 batch 中所有样本的融合特征
#             fusion_feat = twoStreamNet.get_fusion_feature(RGB_img, opticalFlowStackedImg)  # shape: [B, 2560]
#
#             all_features.append(fusion_feat.cpu().numpy())  # shape: [B, 2560]
#             all_labels.append(actual_label.cpu().numpy())  # shape: [B]
#
#     # 拼接为整体数组
#     all_features = np.concatenate(all_features, axis=0)
#     all_labels = np.concatenate(all_labels, axis=0)
#     print("总样本数量：", all_features.shape[0])  #  应该等于 testset 的长度
#
#     # t-SNE 降维到2维
#     tsne = TSNE(n_components=2, random_state=42, perplexity=5, learning_rate='auto', init='random', n_iter=2000)
#     tsne_results = tsne.fit_transform(all_features)
#
#     # 创建数据框
#     tsne_df = pd.DataFrame({
#         't-SNE Dimension 1': tsne_results[:, 0],
#         't-SNE Dimension 2': tsne_results[:, 1],
#         'Class': all_labels
#     })
#
#     # 绘制散点图
#     plt.figure(figsize=(10, 8))
#     label_names = ['left', 'leftRear', 'right', 'rightRear', 'normal']
#     colors = ['#FF6B6B', '#F9D423', '#4DABF7', '#30638E', '#8B008B']  # 更柔和的配色方案
#     # label_names = ['left', 'leftRear', 'right', 'rightRear']
#     # colors = ['#FF6B6B', '#F9D423', '#4DABF7', '#30638E']  # 更柔和的配色方案
#
#
#     for i in range(5):
#         subset = tsne_df[tsne_df['Class'] == i]
#         plt.scatter(subset['t-SNE Dimension 1'], subset['t-SNE Dimension 2'],
#                     c=colors[i], label=label_names[i], alpha=0.7, edgecolors='black', linewidth=0.5)
#
#     plt.title("t-SNE Visualization of Self-created Dataset Fusion Features", fontsize=14, fontweight='bold')
#     plt.xlabel("t-SNE Dimension 1", fontsize=12)
#     plt.ylabel("t-SNE Dimension 2", fontsize=12)
#     plt.legend(title='Class', loc='upper right')
#     plt.grid(True)
#     plt.tight_layout()
#
#     # 保存和显示图像
#     plt.savefig("Figure11_1.png", dpi=300, bbox_inches='tight')
#     plt.show()

#
# ### 特征图绘制
#
# import os
# os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
#
# import torch
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
#
# from two_Stream_Net import TwoStreamNet
# from lstm_Load import testset_loader
# import lstm_Load
#
# def main():
#
#     # =========================
#     # 设备
#     # =========================
#     device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#
#     # =========================
#     # 加载模型
#     # =========================
#     model = TwoStreamNet().to(device)
#
#     state = torch.load('model/IEEE/checkpoint-20.pth', map_location=device)
#     model.load_state_dict(state['model'], strict=False)
#
#     model.eval()
#
#     # =========================
#     # 提取特征
#     # =========================
#
#     all_features = []
#     all_labels = []
#
#     with torch.no_grad():
#
#         for RGB_img, opticalFlowStackedImg, label in testset_loader:
#
#             RGB_img = RGB_img.to(device)
#             opticalFlowStackedImg = opticalFlowStackedImg.to(device)
#
#             # 提取融合特征 (2560维)
#             feature = model.get_fusion_feature(RGB_img, opticalFlowStackedImg)
#
#             feature = feature.cpu().numpy()
#
#             all_features.append(feature)
#             all_labels.append(label.numpy())
#
#     all_features = np.concatenate(all_features, axis=0)
#     all_labels = np.concatenate(all_labels, axis=0)
#
#     print("特征提取完成")
#     print("Feature shape:", all_features.shape)
#
#     # 保存特征
#     np.save("features.npy", all_features)
#     np.save("labels.npy", all_labels)
#
#     # =========================
#     # 画 Feature Heatmap
#     # =========================
#
#     # 选择一个类别（例如0）的前十个样本
#     class_id = 4
#     idx = np.where(all_labels == class_id)[0]
#     idx = np.random.choice(idx, 10, replace=False)
#
#     selected_features = all_features[idx]  # (10,128)
#
#     num_samples = selected_features.shape[0]
#
#     fig, axes = plt.subplots(
#         1,
#         num_samples,
#         figsize=(num_samples * 1.2, 6),
#         constrained_layout=True
#     )
#
#     for i in range(num_samples):
#
#         feature = selected_features[i].reshape(128, 1)
#
#         im = axes[i].imshow(
#             feature,
#             cmap="jet",
#             aspect="auto",
#             vmin=-np.max(np.abs(feature)),
#             vmax=np.max(np.abs(feature))
#         )
#
#         axes[i].set_title(f"X{i + 1}", fontsize=10)
#
#         axes[i].set_xticks([])
#
#         if i == 0:
#             axes[i].set_ylabel("Feature Dimension")
#
#         axes[i].set_yticks(np.linspace(0, 127, 7).astype(int))
#
#     # colorbar
#     cbar = fig.colorbar(im, ax=axes.ravel().tolist())
#     cbar.ax.set_ylabel("Activation", rotation=270, labelpad=15)
#
#     plt.suptitle(
#         "Learned Feature Visualization",
#         fontsize=16
#     )
#
#     save_path = "rightRear_feature.png"
#
#     plt.savefig(
#         save_path,
#         dpi=600,
#         bbox_inches='tight'
#     )
#
#     plt.close()
#
#     print(f"Heatmap 已保存至: {save_path}")
#
#
# if __name__ == '__main__':
#
#     # Windows 多进程兼容
#     import torch.multiprocessing
#     torch.multiprocessing.set_start_method('spawn', force=True)
#
#     main()