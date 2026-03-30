# import cv2
# import os
#
# # 输入视频路径
# input_folder = r"D:\硕士\眼震\结果\瞳孔跟踪"
# # 输出路径
# output_folder = r"D:\硕士\眼震\结果\瞳孔跟踪"
#
# # 采样间隔（跳过 2 帧取 1 帧）
# sampling_interval = 0
#
# # 遍历输入文件夹中的所有视频文件
# for video_filename in os.listdir(input_folder):
#     video_path = os.path.join(input_folder, video_filename)
#
#     # 创建输出子文件夹
#     output_subfolder = os.path.join(output_folder, os.path.splitext(video_filename)[0])
#     os.makedirs(output_subfolder, exist_ok=True)
#
#     # 打开视频文件
#     cap = cv2.VideoCapture(video_path)
#
#     if not cap.isOpened():
#         print(f"Error: Could not open video {video_path}")
#         continue
#
#     # 获取视频帧率和总帧数
#     fps = cap.get(cv2.CAP_PROP_FPS)
#     total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
#
#     frame_count = 0
#     saved_frame_count = 0
#
#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             break
#
#         # 根据采样间隔保存帧
#         if frame_count % (sampling_interval + 1) == 0:
#             output_frame_path = os.path.join(output_subfolder, f"frame_{saved_frame_count:04d}.png")
#             cv2.imwrite(output_frame_path, frame)
#             saved_frame_count += 1
#
#         frame_count += 1
#
#     cap.release()
#     print(f"Processed {video_filename}: {saved_frame_count} frames saved")

import cv2
import os
def save_all_frames(video_path, out_dir):
    os.makedirs(out_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        cv2.imwrite(os.path.join(out_dir, f"{idx:05d}.jpg"), frame)
        idx += 1

    cap.release()
    print(f"Saved {idx} frames to {out_dir}")

# def extract_frames(video_path, output_folder, frame_interval=1):
#     """
#     从视频文件中抽取所有帧并保存为图片。
#
#     参数:
#         video_path: 视频文件路径
#         output_folder: 保存帧图片的文件夹
#         frame_interval: 每隔多少帧保存一次（默认为1，即保存所有帧）
#     """
#     # 创建输出文件夹（如果不存在）
#     if not os.path.exists(output_folder):
#         os.makedirs(output_folder)
#
#     # 打开视频文件
#     cap = cv2.VideoCapture(video_path)
#
#     if not cap.isOpened():
#         print(f"无法打开视频文件: {video_path}")
#         print("可能的编码问题，请尝试安装必要的编解码器或转换视频格式。")
#         return
#
#     # 获取视频的帧率和总帧数
#     fps = cap.get(cv2.CAP_PROP_FPS)
#     total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
#     print(f"视频帧率: {fps} FPS")
#     print(f"总帧数: {total_frames}")
#
#     # 初始化帧计数器
#     frame_count = 0
#     saved_count = 0
#
#     while True:
#         ret, frame = cap.read()
#
#         if not ret:
#             break  # 视频结束
#
#         # 每隔frame_interval帧保存一次
#         if frame_count % frame_interval == 0:
#             # 构造输出文件名
#             output_path = os.path.join(output_folder, f"frame_{frame_count:06d}.jpg")
#             cv2.imwrite(output_path, frame, [cv2.IMWRITE_JPEG_QUALITY, 90])  # 保存为JPEG，质量90%
#             saved_count += 1
#             print(f"已保存: {output_path}")
#
#         frame_count += 1
#
#     # 释放资源
#     cap.release()
#     print(f"完成！共保存 {saved_count} 帧到 {output_folder}")


# 示例用法
video_path = "lab/mp4/chc1_nys.mp4"  # 输入视频路径
output_folder = "lab/mp4/chc1_nys"  # 输出帧图片的文件夹
# extract_frames(video_path, output_folder)
save_all_frames(video_path, output_folder)