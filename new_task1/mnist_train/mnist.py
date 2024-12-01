import os
import numpy as np
from tensorflow.keras.datasets import mnist

# 加载MNIST数据集
# mnist.load_data() 返回两个元组：(训练数据, 测试数据)
# 每个元组包含两部分：(图像, 标签)
# 图像大小为28x28灰度图，像素值范围在0到255之间
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# 数据预处理
# 将图像数据归一化到[0, 1]范围，并将其从整数转换为浮点数
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

# 定义保存批次数据的函数
def save_batches(data, labels, batch_size, output_dir, prefix):
    """
    将数据按批次保存到指定目录下

    参数:
        data (np.ndarray): 图像数据，形状为(N, H, W)。
        labels (np.ndarray): 对应的标签，形状为(N,)。
        batch_size (int): 每批的样本数量。
        output_dir (str): 保存批次数据的目录。
        prefix (str): 文件名前缀。
    """
    os.makedirs(output_dir, exist_ok=True)
    num_samples = data.shape[0]
    num_batches = (num_samples + batch_size - 1) // batch_size

    for i in range(num_batches):
        start = i * batch_size
        end = min((i + 1) * batch_size, num_samples)

        # 提取当前批次数据和标签
        batch_data = data[start:end]
        batch_labels = labels[start:end]

        # 保存为.npy文件
        np.save(os.path.join(output_dir, f"{prefix}_data_batch_{i+1}.npy"), batch_data)
        np.save(os.path.join(output_dir, f"{prefix}_labels_batch_{i+1}.npy"), batch_labels)

        print(f"Saved {prefix} batch {i+1}: {end-start} samples to {output_dir}")

# 参数配置
train_output_dir = "mnist_batches/train"
test_output_dir = "mnist_batches/test"
batch_size = 128

# 保存训练和测试批次数据
save_batches(x_train, y_train, batch_size, train_output_dir, "train")
save_batches(x_test, y_test, batch_size, test_output_dir, "test")

print("数据保存完成！")
