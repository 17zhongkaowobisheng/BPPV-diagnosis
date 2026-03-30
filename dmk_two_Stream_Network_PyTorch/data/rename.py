# import os
#
# # 定义目录路径
# directory = r"D:\PyCharm Community Edition 2024.3.1.1\project\dmk_two_Stream_Network_PyTorch\data\OpticalFlow\right"
#
# # 遍历目录中的所有文件
# for filename in os.listdir(directory):
#     # if os.path.isfile(os.path.join(directory, filename)):
#         # 检查文件名是否以_result结尾
#     if filename.endswith('_result'):
#         # 构建新的文件名
#         new_filename = filename[:-7]  # 去掉最后的_result
#         # 重命名文件
#         os.rename(os.path.join(directory, filename), os.path.join(directory, new_filename))
# import os
#
# # 定义视频所在的目录
# video_dir = r"D:\PyCharm Community Edition 2024.3.1.1\project\dmk_two_Stream_Network_PyTorch\data\OpticalFlow\leftRear"
#
# # 获取目录下所有的.mp4文件
# video_files = [f for f in os.listdir(video_dir)]
#
# # 按读取顺序对视频文件进行排序
# video_files.sort()
#
# # 遍历视频文件并重新命名
# for i, old_name in enumerate(video_files, start=1):
#     # 构建新的文件名
#     new_name = f"{i}"
#     # 构建旧文件的完整路径
#     old_path = os.path.join(video_dir, old_name)
#     # 构建新文件的完整路径
#     new_path = os.path.join(video_dir, new_name)
#     # 重命名文件
#     os.rename(old_path, new_path)
#     print(f"已将 {old_name} 重命名为 {new_name}")



###名字更换为1-100
import os

# # ===== 直接在这里修改路径 =====
# parent_dir = r"D:\PyCharm Community Edition 2024.3.1.1\project\dmk_two_Stream_Network_PyTorch\data\RGB\rightRear"  # 你的目标文件夹
# max_num = 100  # 最多重命名多少个
#
#
# def safe_rename_folders(parent_dir, max_num=100):
#     # 获取所有子文件夹
#     folders = [f for f in os.listdir(parent_dir) if os.path.isdir(os.path.join(parent_dir, f))]
#
#     # 排序，保证顺序稳定
#     folders.sort()
#
#     # 限制最大数量
#     folders = folders[:max_num]
#
#     temp_names = []
#
#     # 第一步：改为临时名字，避免冲突
#     for i, folder in enumerate(folders):
#         old_path = os.path.join(parent_dir, folder)
#         temp_name = f"temp_rename_{i}"
#         temp_path = os.path.join(parent_dir, temp_name)
#
#         os.rename(old_path, temp_path)
#         temp_names.append(temp_name)
#
#     # 第二步：改为 1,2,3...
#     for i, temp_name in enumerate(temp_names, start=1):
#         temp_path = os.path.join(parent_dir, temp_name)
#         new_path = os.path.join(parent_dir, str(i))
#
#         os.rename(temp_path, new_path)
#         print(f"{temp_name} -> {i}")
#
#     print("重命名完成！")
#
#
# safe_rename_folders(parent_dir, max_num)
#

###名字从left到right
import os

# ===============================
# 修改这里：视频文件所在文件夹
# ===============================
folder_path = r"D:\PyCharm Community Edition 2024.3.1.1\project\dmk_two_Stream_Network_PyTorch\data\IEEE\不许动\right1_9s"   # 改成你的文件夹路径


for filename in os.listdir(folder_path):

    # 只处理视频文件
    if filename.endswith(".mp4"):

        # 判断是否包含 right
        if "right" in filename:

            old_path = os.path.join(folder_path, filename)

            # 替换 right 为 left
            new_filename = filename.replace("right", "left")
            new_path = os.path.join(folder_path, new_filename)

            os.rename(old_path, new_path)

            print(f"{filename}  ->  {new_filename}")

print("重命名完成！")