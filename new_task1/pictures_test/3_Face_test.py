import os
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

# 定义路径
FF_PATH = "E:/BaiduNetdiskDownload/FF_test/FF_frames/face_v1"
MANIPULATED = "manipulated_sequences"
ORIGINAL = "original_sequences"

OUTPUT_DIR = "E:/BaiduNetdiskDownload/FF_test/processed_frames"
os.makedirs(OUTPUT_DIR, exist_ok=True)  # 确保输出文件夹存在

# 自定义 Dataset 类
class FaceForensicsDataset(Dataset):
    def __init__(self, ff_path, transform=None):
        """
        初始化数据集
        :param ff_path: 数据集根目录
        :param transform: 数据变换（如水平翻转、归一化等）
        """
        self.data = []
        self.labels = []
        self.transform = transform

        # 遍历伪造图片路径（标签为0）
        manipulated_path = os.path.join(ff_path, MANIPULATED)
        for root, _, files in os.walk(manipulated_path):
            for file in files:
                if file.endswith((".jpg", ".png")):
                    self.data.append(os.path.join(root, file))
                    self.labels.append(0)  # 标签为伪造

        # 遍历真实图片路径（标签为1）
        original_path = os.path.join(ff_path, ORIGINAL)
        for root, _, files in os.walk(original_path):
            for file in files:
                if file.endswith((".jpg", ".png")):
                    self.data.append(os.path.join(root, file))
                    self.labels.append(1)  # 标签为真实

    def __len__(self):
        """返回数据集大小"""
        return len(self.data)

    def __getitem__(self, idx):
        """
        获取指定索引的数据及标签
        :param idx: 索引
        :return: 图像及其标签
        """
        img_path = self.data[idx]
        label = self.labels[idx]

        # 加载图像并进行变换
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        return image, label

# 定义数据变换
transform = transforms.Compose([
    transforms.Resize((128, 128)),  # 调整图像大小
    transforms.RandomHorizontalFlip(p=0.5),  # 随机水平翻转
    transforms.ToTensor(),  # 转换为张量
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # 归一化到 [-1, 1]
])

# 初始化数据集和 DataLoader
dataset = FaceForensicsDataset(FF_PATH, transform=transform)
dataloader = DataLoader(dataset, batch_size=16, shuffle=False, num_workers=0)

# 按批次加载数据并保存预处理后的图片
print("开始处理和保存图片...")
for batch_idx, (images, labels) in enumerate(dataloader):
    # 创建当前批次的保存目录
    batch_dir = os.path.join(OUTPUT_DIR, f"batch_{batch_idx + 1}")
    os.makedirs(batch_dir, exist_ok=True)

    for img_idx, (image, label) in enumerate(zip(images, labels)):
        # 将张量转换为 PIL 图像
        img = transforms.ToPILImage()(image)
        # 保存图像，文件名带上标签，扩展名改为 .png
        img_save_path = os.path.join(batch_dir, f"image_{img_idx}_label_{label.item()}.png")
        img.save(img_save_path)

    print(f"批次 {batch_idx + 1} 已处理完成，保存至 {batch_dir}")

print("所有图片处理完成！")
