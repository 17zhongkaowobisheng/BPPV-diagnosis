# import os
#
# base_path = 'UCF101'
# folder_num_map = {}
# folders = os.listdir(base_path)
# for index, folder in enumerate(folders):
#     folder_num_map[folder] = index + 1
#
# with open('output.txt', 'w') as f:
#     for folder in folders:
#         folder_path = os.path.join(base_path, folder)
#         if os.path.isdir(folder_path):
#             videos = os.listdir(folder_path)
#             for video in videos:
#                 full_path = os.path.join(folder_path, video)
#
#                 line = f'{full_path} {folder_num_map[folder]}\n'
#                 f.write(line)
import os

base_path = 'IEEE/BPPV'
folder_num_map = {}
folders = os.listdir(base_path)
for index, folder in enumerate(folders):
    folder_num_map[folder] = index + 1

with open('output.txt', 'w') as f:
    for folder in folders:
        folder_path = os.path.join(base_path, folder)
        if os.path.isdir(folder_path):
            videos = os.listdir(folder_path)
            for video in videos:
                full_path = os.path.join(folder_path, video)
                relative_path = os.path.relpath(full_path, base_path)
                line = f'{relative_path} {folder_num_map[folder]}\n'
                f.write(line)