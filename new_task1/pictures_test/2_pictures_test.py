import os
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


class LowLightDataset(Dataset):
    """
    自定义Dataset类，用于加载低光图像及其对应的高质量图像。
    """
    def __init__(self, low_light_dir, high_quality_dir, transform=None):
        """
        初始化函数，加载低光图像和高质量图像路径。

        参数：
        low_light_dir (str): 低光图像所在文件夹路径。
        high_quality_dir (str): 高质量图像所在文件夹路径。
        transform (callable, optional): 对图像进行的转换操作。
        """
        self.low_light_dir = low_light_dir
        self.high_quality_dir = high_quality_dir
        self.transform = transform

        # 获取所有低光图像文件名
        self.low_light_images = os.listdir(low_light_dir)

    def __len__(self):
        """
        返回数据集大小。
        """
        return len(self.low_light_images)

    def __getitem__(self, idx):
        """
        根据索引获取低光图像及其对应的高质量图像。

        参数：
        idx (int): 索引值。

        返回：
        dict: 包含低光图像和高质量图像的字典。
        """
        low_light_image_path = os.path.join(self.low_light_dir, self.low_light_images[idx])
        high_quality_image_path = os.path.join(self.high_quality_dir, self.low_light_images[idx])

        # 打开图像
        low_light_image = Image.open(low_light_image_path).convert('RGB')
        high_quality_image = Image.open(high_quality_image_path).convert('RGB')

        # 应用变换
        if self.transform:
            low_light_image = self.transform(low_light_image)
            high_quality_image = self.transform(high_quality_image)

        return {'low_light': low_light_image, 'high_quality': high_quality_image, 'filename': self.low_light_images[idx]}


# 定义转换操作
transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),  # 随机水平翻转
    transforms.ToTensor(),  # 转换为Tensor
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # 归一化
])

# 低光图像和高质量图像的文件夹路径
low_light_dir = "E:/BaiduNetdiskDownload/LLE_LOL_Part/LowLight"
high_quality_dir = "E:/BaiduNetdiskDownload/LLE_LOL_Part/Taget"

# 输出文件夹路径
output_low_light_dir = "E:/BaiduNetdiskDownload/LLE_LOL_Part/Processed/LowLight"
output_high_quality_dir = "E:/BaiduNetdiskDownload/LLE_LOL_Part/Processed/Taget"

# 创建输出文件夹
os.makedirs(output_low_light_dir, exist_ok=True)
os.makedirs(output_high_quality_dir, exist_ok=True)

# 创建数据集和DataLoader
dataset = LowLightDataset(low_light_dir, high_quality_dir, transform=transform)
batch_size = 16
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)

# 遍历DataLoader并保存处理后的图像
for batch_idx, batch in enumerate(dataloader):
    low_light_images = batch['low_light']
    high_quality_images = batch['high_quality']
    filenames = batch['filename']

    print(f"Processing Batch {batch_idx + 1}/{len(dataloader)}:")
    print(f"  Low Light Images Shape: {low_light_images.shape}")  # [batch_size, 3, H, W]
    print(f"  High Quality Images Shape: {high_quality_images.shape}")  # [batch_size, 3, H, W]

    # 保存每个批次的图像
    for i in range(len(filenames)):
        low_light_image = low_light_images[i]
        high_quality_image = high_quality_images[i]

        # 去归一化处理，将Tensor转换为PIL图像
        inv_normalize = transforms.Normalize(
            mean=[-0.5 / 0.5, -0.5 / 0.5, -0.5 / 0.5],
            std=[1 / 0.5, 1 / 0.5, 1 / 0.5]
        )
        low_light_image = inv_normalize(low_light_image).permute(1, 2, 0).mul(255).byte().numpy()
        high_quality_image = inv_normalize(high_quality_image).permute(1, 2, 0).mul(255).byte().numpy()

        # 保存图像到新文件夹
        low_light_image_pil = Image.fromarray(low_light_image)
        high_quality_image_pil = Image.fromarray(high_quality_image)

        low_light_image_pil.save(os.path.join(output_low_light_dir, filenames[i]))
        high_quality_image_pil.save(os.path.join(output_high_quality_dir, filenames[i]))

print("All images processed and saved successfully.")
