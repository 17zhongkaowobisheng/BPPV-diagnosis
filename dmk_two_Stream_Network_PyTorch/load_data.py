import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import torch
import numpy as np
import os
from PIL import Image
import torchvision.transforms as transforms
from PIL import ImageFilter
import torch
import random

# 超参数
RGB_SAMPLE_RATE = 1  # RGB帧采样率
TRAIN_BATCH_SIZE = 10 #训练batch大小
TEST_BATCH_SIZE = 2  # 测试batch大小
SAMPLE_FRAME_NUM = 50 # 光流对数量(一对包含一个x和一个y方向的光流)
RGB_FRAME_NUM =  100 # RGB帧数量
FLOW_INTERVAL =  1 # 光流生成间隔

# 将 classInd_self.txt 中的动作类别添加到列表中
classInd = []
# with open('TrainTestlist/classInd_public.txt', 'r') as f:
with open('TrainTestlist/classInd_self.txt', 'r') as f:
    all_Class_and_Ind = f.readlines()
    for line in all_Class_and_Ind:
        idx = line[:-1].split(' ')[0]
        className = line[:-1].split(' ')[1]
        classInd.append(className)

# 将 trainlist_IEEE.txt 中划分的训练集动作视频存入 TrainVideoNameList 列表中
TrainVideoNameList = []
# with open('TrainTestlist/trainlist_public.txt', 'r') as f:
with open('TrainTestlist/trainlist_self.txt', 'r') as f:
    all_Class_and_Ind = f.readlines()
    for line in all_Class_and_Ind:
        video_name = line[:-1].split('.')[0]
        video_name = video_name.split('\\')[1]
        TrainVideoNameList.append(video_name)

# 将 testlist_IEEE.txt 中划分的测试集动作视频存入 TestVideoNameList 列表中
TestVideoNameList = []
with open('TrainTestlist/testlist_self.txt', 'r') as f:
# with open('TrainTestlist/testlist_public.txt', 'r') as f:
    all_Class_and_Ind = f.readlines()
    for line in all_Class_and_Ind:
        video_name = line[:-1].split('.')[0]
        video_name = video_name.split('\\')[1]
        TestVideoNameList.append(video_name)


# 定义一个 UCF101Data 数据加载的类
class UCF101Data(Dataset):

    def __init__(self, RBG_root, OpticalFlow_root, data_class="train", transform=None, action_num=5, flow_interval=10):
        """
        :param RBG_root:  RGB 数据集的文件路径
        :param OpticalFlow_root:  OpticalFlow_root 数据集的文件路径
        :param data_class: 需要加载的数据类型
        :param transform: 是否对数据根据 transform 的操作
        :param action_num: 动作的类别个数
        """
        self.filenames = []
        self.transform = transform
        self.num = action_num

        for i in range(0, self.num):    # dmk 修改，本类别数为 5
            # 依次读取 classInd 中的动作名字，并依次生成光流数集中的每类动作的目录
            OpticalFlow_class_path = OpticalFlow_root + '/' + classInd[i]
            # 依次读取 classInd 中的动作名字，并依次生成 RGB 数集中的每类动作的目录
            RGB_class_path = RBG_root + '/' + classInd[i]
            # 将 OpticalFlow_class_path 路径下的文件生成列表，
            # 并与 TrainVideoNameList/TestVideoNameList 列表中内容求交集，即获取第 i 类动作需要训练/测试的全部动作视频
            if data_class == "train":
                TrainOrTest_VideoNameList = list(set(os.listdir(OpticalFlow_class_path)).intersection(set(TrainVideoNameList)))
            if data_class == "test":
                TrainOrTest_VideoNameList = list(set(os.listdir(OpticalFlow_class_path)).intersection(set(TestVideoNameList)))

            # 依次遍历第 i 个 OpticalFlow_class_path 路径下的所有视频文件
            for video_dir in os.listdir(OpticalFlow_class_path):
                # 判断此时路径下的视频文件是否在 TrainOrTest_VideoNameList 列表中
                # 如果在的话，就生成 single_OpticalFlow_video_path 和 signel_RGB_video_path 的路径
                if video_dir in TrainOrTest_VideoNameList:
                    single_OpticalFlow_video_path = OpticalFlow_class_path + '/' + video_dir
                    signel_RGB_video_path = RGB_class_path + '/' + video_dir

                    # 加载 single_OpticalFlow_video_path 中的所有光流文件名字，并放入 fram_list 列表中
                    frame_list = os.listdir(single_OpticalFlow_video_path)
                    # 根据当前视频生成的每个光流图像的索引值来对 frame_list 列表数据进行排序，也即当前视频每帧生成的光流图像顺序
                    # 如 v_ApplyEyeMakeup_g08_c01_2_x.jpg 中的……_2_……的值就是排序位
                    frame_list.sort(key=lambda x: int(x.split("_")[-2]))
                    # 在 fram_list 列表中随机生成一个光流图像的索引值，但此索引值一定是从光流图像的 x 方向光流图像开始的
                    # 首先在【当前视频的光流总个数-SAMPLE_FRAME_NUM * 2 + 1】的范围内随机生成一个索引值，
                    # 这样做的目的是为了防止最后堆叠的光流索引超出当前视频的总光流图像的范围，保证每次不管随机生成几，都能产生 SAMPLE_FRAME_NUM 个堆叠光流
                    # print(single_OpticalFlow_video_path, len(frame_list))
                    ran_frame_idx = np.random.randint(0, len(frame_list) - (SAMPLE_FRAME_NUM * 2) + 1)
                    # 如果不是从 x 值开始的，则继续生成新索引值，直到符合规定为止
                    while ran_frame_idx % flow_interval!= 0:
                        ran_frame_idx = np.random.randint(0, len(frame_list) - SAMPLE_FRAME_NUM * 2 + 1)
                    # 在这里堆叠 SAMPLE_FRAME_NUM 个光流图像的路径在接下来的列表中，包括 x 和 y 方向上的光流，所以共 20 个
                    stacked_OpticalFlow_image_path = []
                    for j in range(ran_frame_idx, ran_frame_idx + SAMPLE_FRAME_NUM * 2):
                        OpticalFlow_image_path = single_OpticalFlow_video_path + '/' + frame_list[j]
                        stacked_OpticalFlow_image_path.append(OpticalFlow_image_path)

                    # # 随机从当前视频中的提取一帧 RGB 图像
                    # rgb_frame_list = os.listdir(signel_RGB_video_path)
                    # rgb_frame_list.sort(key=lambda x: int((x.split("_")[-1]).split(".")[0]))
                    # randm_rgb = np.random.randint(0, len(rgb_frame_list))
                    # RGB_image_path = signel_RGB_video_path + '/' + rgb_frame_list[randm_rgb]



                    # 随机从当前视频中的提取RGB_FRAME_NUM帧 RGB 图像
                    rgb_frame_list = os.listdir(signel_RGB_video_path)
                    rgb_frame_list.sort(key=lambda x: int((x.split("_")[-1]).split(".")[0]))
                    # 计算起始索引
                    rgb_ran_frame_idx = np.random.randint(0, len(rgb_frame_list) - RGB_FRAME_NUM)
                    stacked_RGB_image_path = []
                    for j in range(rgb_ran_frame_idx, rgb_ran_frame_idx + RGB_FRAME_NUM):
                        RGB_image_path = signel_RGB_video_path + '/' + rgb_frame_list[j]
                        stacked_RGB_image_path.append(RGB_image_path)
                    # 将上面生成的数据，弄成这样 (RGB_image_path, stacked_OpticalFlow_image_path, label) 的形式，并添加到 filenames 列表中
                    # 这里即 1 个视频数据的 RGB+Flow+标签生成
                    self.filenames.append((stacked_RGB_image_path, stacked_OpticalFlow_image_path, i))

        self.len = len(self.filenames)  # 得到最后 filenames 列表的长度


    # 重写 Dateset 类中的 __getitem__方法
    # 根据 index 的值，从 filenames 列表中，获取此索引对应的列表内容 (RGB+FLOW+label)
    def __getitem__(self, index):
        stacked_RGB_image_path, stacked_OpticalFlow_image_path, label = self.filenames[index]
        # 创建一个 SAMPLE_FRAME_NUM * 2, 224, 224 的张量
        # stacked_OpticalFlow_image = torch.empty(SAMPLE_FRAME_NUM * 2, 224, 224)
        stacked_OpticalFlow_image = torch.empty(SAMPLE_FRAME_NUM * 2, 112, 112)
        idx = 0
        video_name = os.path.basename(os.path.dirname(stacked_RGB_image_path[0]))

        # 依次处理 stacked_OpticalFlow_image_path 列表中的所有光流图像(20 张)
        for i in stacked_OpticalFlow_image_path:
            # 用 Image.open() 打开第 i 个路径的光流图像
            OpticalFlow_image = Image.open(i)
            # 是否使用 transform 对图像大小和像素值进行修改，判断
            if self.transform is not None:
                OpticalFlow_image = self.transform(OpticalFlow_image)
            stacked_OpticalFlow_image[idx, :, :] = OpticalFlow_image[0, :, :]
            idx += 1

        stacked_RGB_image = torch.empty(RGB_FRAME_NUM, 1, 112, 112)
        idx_rgb = 0
        # 依次处理 stacked_RGB_image_path 列表中的所有 RGB 图像(10 张)
        for i in stacked_RGB_image_path:
            # 用 Image.open() 打开第 i 个路径的 RGB 图像
            RGB_image = Image.open(i)
            # 是否使用 transform 对图像大小和像素值进行修改
            if self.transform is not None:
                RGB_image = self.transform(RGB_image)
            stacked_RGB_image[idx_rgb, :, :, :] = RGB_image  # 假设 RGB 图像为 3 通道
            idx_rgb += 1

        return stacked_RGB_image, stacked_OpticalFlow_image, label,video_name

    # 得到数据集的总长度
    def __len__(self):
        return self.len



# 自定义高斯模糊类
class RandomGaussianBlur:
    def __init__(self, p=0.2):
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            radius = random.uniform(0.1, 2.0)
            return img.filter(ImageFilter.GaussianBlur(radius))
        return img

# 自定义添加高斯噪声类
class AddGaussianNoise:
    def __init__(self, mean=0., std=0.01):
        self.std = std
        self.mean = mean

    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)


# 增加更多的数据增强操作
transform_train = transforms.Compose([
    # transforms.Resize(224),
    # transforms.CenterCrop(224),
    transforms.Resize((112,112)),
    # transforms.RandomHorizontalFlip(),  # 随机水平翻转
    transforms.Grayscale(num_output_channels=1),  # 确保灰度图像只有一个通道
    transforms.RandomRotation(15),  # 随机旋转
    transforms.RandomVerticalFlip(),  # 随机垂直翻转
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  # 颜色抖动
    transforms.ToTensor()
])
# 测试集数据增强
transform_test = transforms.Compose([
    # transforms.Resize(224),
    # transforms.CenterCrop(224),  # 中心裁剪
    transforms.Resize((112, 112)),
    transforms.Grayscale(num_output_channels=1),  # 确保灰度图像只有一个通道
    transforms.ToTensor()
])


# 加载数据集中划分的训练数据集
##自己数据集
# trainset = UCF101Data(RBG_root='data/RGB', OpticalFlow_root='data/OpticalFlow', data_class="train", transform=transform_train, action_num=5)
##公开数据集
# trainset = UCF101Data(RBG_root='data/public/RGB', OpticalFlow_root='data/public/OpticalFlow', data_class="train", transform=transform_train, action_num=4)
# # 加载数据集中划分的测试数据集
# trainset_loader = DataLoader(trainset, batch_size=TRAIN_BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)


# 加载数据集中划分的测试集数据集
###自己数据集
testset = UCF101Data(RBG_root='data/self/RGB', OpticalFlow_root='data/self/OpticalFlow', data_class="test", transform=transform_test,action_num=5)
###公开数据集
# testset = UCF101Data(RBG_root='data/public/RGB', OpticalFlow_root='data/public/OpticalFlow', data_class="test", transform=transform_test,action_num=4)
testset_loader = DataLoader(testset, batch_size=TEST_BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

# # 合并训练和测试数据集
# combined_dataset = torch.utils.data.ConcatDataset([trainset, testset])



if __name__ == '__main__':
    print(len(trainset_loader.dataset))# 打印训练集的长度，即数据集中样本的总数。
    for i, data in enumerate(trainset_loader):# 使用 enumerate 遍历 trainset_loader 迭代器，i 是迭代的索引，data 是每次迭代返回的批次数据。
        # 加载 bach_size 个数据
        RGB_images, OpticalFlow_images, label = data
        if i == 0:
            print(RGB_images.size())
            print(OpticalFlow_images.size())
            print(label.size())
        if i == (len(trainset_loader)-1):
            print(RGB_images.size())
            print(OpticalFlow_images.size())
            print(label.size())