import os
import numpy as np
from PIL import Image
import gzip
import torch
from torch.utils.data import Dataset, DataLoader


class MNISTDataset(Dataset):
    """
    自定义 Dataset 类，用于加载本地 MNIST 数据。
    """
    def __init__(self, data_dir, kind='train', transform=None):
        """
        初始化 Dataset，加载 MNIST 数据。

        参数:
            data_dir (str): MNIST 数据目录。
            kind (str): 数据类型，'train' 或 'test'。
            transform (callable, optional): 数据预处理和增强操作。
        """
        if kind == 'train':
            images_file = "train-images-idx3-ubyte.gz"
            labels_file = "train-labels-idx1-ubyte.gz"
        elif kind == 'test':
            images_file = "t10k-images-idx3-ubyte.gz"
            labels_file = "t10k-labels-idx1-ubyte.gz"
        else:
            raise ValueError("参数 kind 只能是 'train' 或 'test'")

        images_path = os.path.join(data_dir, images_file)
        labels_path = os.path.join(data_dir, labels_file)

        # 读取标签
        with gzip.open(labels_path, "rb") as lbpath:
            magic, num = np.frombuffer(lbpath.read(8), dtype=np.uint32).newbyteorder('>')  # 跳过头部
            self.labels = np.frombuffer(lbpath.read(), dtype=np.uint8)

        # 读取图像
        with gzip.open(images_path, "rb") as imgpath:
            magic, num, rows, cols = np.frombuffer(imgpath.read(16), dtype=np.uint32).newbyteorder('>')  # 跳过头部
            self.images = np.frombuffer(imgpath.read(), dtype=np.uint8).reshape(len(self.labels), rows, cols)

        self.transform = transform

    def __len__(self):
        """
        返回数据集大小。
        """
        return len(self.labels)

    def __getitem__(self, idx):
        """
        根据索引返回一对图像和标签。

        参数:
            idx (int): 索引值。

        返回:
            dict: 包含图像和标签的字典。
        """
        image = self.images[idx]
        label = self.labels[idx]

        # 将 NumPy 数组转换为 PIL 图像对象
        image = Image.fromarray(image)

        # 应用转换
        if self.transform:
            image = self.transform(image)

        return {'image': image, 'label': label}


def save_images_as_png(dataloader, output_dir, prefix):
    """
    使用 DataLoader 按批次保存 MNIST 图像为 PNG 格式。

    参数:
        dataloader (DataLoader): 数据加载器。
        output_dir (str): 输出目录。
        prefix (str): 文件名前缀。
    """
    os.makedirs(output_dir, exist_ok=True)

    for batch_idx, batch in enumerate(dataloader):
        images = batch['image'].numpy()  # 转换为 NumPy 格式
        labels = batch['label'].numpy()

        # 创建每个批次的文件夹
        batch_dir = os.path.join(output_dir, f"{prefix}_batch_{batch_idx + 1}")
        os.makedirs(batch_dir, exist_ok=True)

        # 保存当前批次的图像
        for i, (img, label) in enumerate(zip(images, labels)):
            # 转换为 PIL 图像格式
            img = Image.fromarray((img * 255).astype(np.uint8).squeeze(), mode="L")
            img.save(os.path.join(batch_dir, f"img_{i}_label_{label}.png"))

        print(f"已保存 {prefix} 第 {batch_idx + 1} 批次，共保存 {len(images)} 张图像，目录: {batch_dir}")


# 参数配置
data_dir = "E:/mnist_test/pythonProject/data/MNIST/raw/"  # MNIST 本地数据集路径，需包含 .gz 文件
output_train_dir = "./mnist_png_batches/train"  # 输出训练图像存储路径
output_test_dir = "./mnist_png_batches/test"  # 输出测试图像存储路径
batch_size = 100  # 每批次大小

# 创建训练和测试数据集
train_dataset = MNISTDataset(data_dir, kind='train', transform=lambda x: np.array(x) / 255.0)
test_dataset = MNISTDataset(data_dir, kind='test', transform=lambda x: np.array(x) / 255.0)

# 创建 DataLoader
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# 将训练数据按批次保存为 PNG 格式
save_images_as_png(train_loader, output_train_dir, "train")
# 将测试数据按批次保存为 PNG 格式
save_images_as_png(test_loader, output_test_dir, "test")

print("所有图像已保存为 PNG 格式。")
