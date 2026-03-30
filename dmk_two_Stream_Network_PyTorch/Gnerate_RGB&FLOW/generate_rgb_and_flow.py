"""
目的:将原始UCF101视频数据生成RGB+Flow数据
"""
import cv2
import os
import numpy as np


def cal_optical_flow_flat_dir(
    video_dir,
    RGB_Dir_Path,
    OpticalFlow_Dir_Path,
    rgb_save_interval=5,
    flow_save_interval=2,
    resize=(224, 224),
    clip=10.0
):
    os.makedirs(RGB_Dir_Path, exist_ok=True)
    os.makedirs(OpticalFlow_Dir_Path, exist_ok=True)

    for video_name in os.listdir(video_dir):
        if not video_name.lower().endswith(('.mp4', '.avi', '.mov')):
            continue

        video_path = os.path.join(video_dir, video_name)
        video_base = os.path.splitext(video_name)[0]

        rgb_save_dir = os.path.join(RGB_Dir_Path, video_base)
        flow_save_dir = os.path.join(OpticalFlow_Dir_Path, video_base)
        os.makedirs(rgb_save_dir, exist_ok=True)
        os.makedirs(flow_save_dir, exist_ok=True)

        cap = cv2.VideoCapture(video_path)
        prev_gray = None
        frame_idx = 0

        print(f"[INFO] Processing {video_name}")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if resize is not None:
                frame = cv2.resize(frame, resize)

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # ---- Save RGB ----
            if frame_idx % rgb_save_interval == 0:
                cv2.imwrite(
                    os.path.join(rgb_save_dir, f"{video_base}_{frame_idx}.jpg"),
                    frame
                )

            # ---- Save Flow ----
            if prev_gray is not None and frame_idx % flow_save_interval == 0:
                flow = cv2.calcOpticalFlowFarneback(
                    prev_gray, gray,
                    None, 0.5, 3, 15, 3, 5, 1.2, 0
                )

                fx = np.clip(flow[..., 0], -clip, clip)
                fy = np.clip(flow[..., 1], -clip, clip)

                fx_img = ((fx + clip) / (2 * clip) * 255).astype(np.uint8)
                fy_img = ((fy + clip) / (2 * clip) * 255).astype(np.uint8)

                cv2.imwrite(
                    os.path.join(flow_save_dir, f"{video_base}_{frame_idx}_x.jpg"),
                    fx_img
                )
                cv2.imwrite(
                    os.path.join(flow_save_dir, f"{video_base}_{frame_idx}_y.jpg"),
                    fy_img
                )

            prev_gray = gray
            frame_idx += 1

        cap.release()


if __name__ == "__main__":

    # 超参数
    MP4_dir_path = '../data/IEEE/BPPV/Rt_PC_BPPV'  # mp4输入的位置
    RGB_Dir_Path = '../data/IEEE/RGB/Rt_PC_BPPV'  # 生成RGB存放位置
    OpticalFlow_Dir_Path = '../data/IEEE/OpticalFlow/Rt_PC_BPPV'  # 生成flow存放位置

    rgb_save_interval = int(5)  # 生成RGB图像，每隔rgb_save_interval-1帧生成一张
    flow_save_interval = int(2)  # 生成flow图像，每隔flow_save_interval-1帧生成一张,包括x和y方向上的图片

    ###运行
    cal_optical_flow_flat_dir(
        MP4_dir_path,
        RGB_Dir_Path,
        OpticalFlow_Dir_Path,
        rgb_save_interval=1,
        flow_save_interval=1,
        resize=None,
        clip=10.0
    )



