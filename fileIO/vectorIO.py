import numpy as np


def load_vec(file_path):
    """
    从TXT文件中读取数据集，并转换为NumPy数组

    参数:
        file_path (str): 数据集文件路径

    返回:
        tuple: (数据(np.ndarray), 维度(int), 数据集大小(int))
    """
    data_list = []

    with open(file_path, 'r') as file:
        # 读取第一行，获取维度和数据集大小
        first_line = file.readline().strip()
        parts = first_line.split()

        if len(parts) >= 2:
            dimension = int(parts[0])
            dataset_size = int(parts[1])
            # 动态调整维数和数据集大小
            if dimension > dataset_size:
                d = dimension
                dimension = dataset_size
                dataset_size = d
            print(f"数据维度: {dimension}, 数据集大小: {dataset_size}")
        else:
            raise ValueError("文件格式错误: 第一行应包含维度和数据集大小")

        # 读取剩余行的数据
        for line in file:
            line = line.strip()
            if line:  # 跳过空行
                # 将每行数据转换为浮点数列表
                values = [float(x) for x in line.split()]

                if len(values) != dimension:
                    print(f"警告: 行 '{line}' 的维度与预期维度 {dimension} 不匹配")

                data_list.append(values)

    # 验证读取的数据集大小是否匹配
    if len(data_list) != dataset_size:
        print(f"警告: 读取到 {len(data_list)} 条数据，预期 {dataset_size} 条")

    # 将列表转换为NumPy数组
    data_array = np.array(data_list, dtype=np.float32)

    return data_array, dimension, dataset_size


# 使用示例
if __name__ == "__main__":
    file_path = "D:\\python\\metricANN\data\\texas\\texas.txt"  # 替换为您的数据集文件路径
    dataset, dim, size = load_vec(file_path)
    print(f"成功读取数据集: {dataset.shape[0]} 条数据, 维度: {dim}")
    print(f"数据类型: {type(dataset)}")
    print(f"数据形状: {dataset.shape}")
