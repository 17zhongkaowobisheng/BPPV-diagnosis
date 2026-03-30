import os

# 指定目录路径
directory_path = r'D:\PyCharm Community Edition 2024.3.1.1\project\dmk_two_Stream_Network_PyTorch\data\RGB\rightRear'

# 获取顶级子文件夹列表
top_level_dirs = os.listdir(directory_path)

for dir in top_level_dirs:
    dir_path = os.path.join(directory_path, dir)
    # 确保是文件夹
    if os.path.isdir(dir_path):
        # 计算每个文件夹中的PNG图片数量
        png_count = len([f for f in os.listdir(dir_path) if f.endswith('.png')])
        print(f'文件夹 {dir_path} 中的PNG图片数量为: {png_count}')