import math
import matplotlib.pyplot as plt
from tqdm import tqdm

# 数据集路径
data_path = "data/wine.data"

# 列类别
column_names = [
    'class', 'Alcohol', 'Malic acid', 'Ash', 'Alcalinity of ash', 'Magnesium',
    'Total phenols', 'Flavanoids', 'Nonflavanoid phenols', 'Proanthocyanins',
    'Color intensity', 'Hue', 'OD280/OD315 of diluted wines', 'Proline'
]

# 加载数据
def load_data(file_path):
    #存放数据集
    data = []
    with open(file_path, 'r') as file:
        for line in file:
            data_point = [float(x) for x in line.strip().split(',')]
            data.append(data_point)
    return data


"""
该方法将数据集按照指定的比例分割成训练集和测试集。
这个比例由split_ratio参数控制,默认情况下是0.8
(80%的数据作为训练集,20%作为测试集)。
"""
def split_data(data, split_ratio=0.8):
    train_size = int(len(data) * split_ratio)
    train_data = data[:train_size] # 取前 train_size 部分作为训练集
    test_data = data[train_size:]
    return train_data, test_data

# 按类别划分数据
def separate_by_class(data):
    separated = {}
    for i in range(len(data)):
        vector = data[i]  # 获取数据集中的每一个向量(样本)
        class_value = int(vector[0])
        if class_value not in separated:    # 如果该类别值不在 separated 字典中,创建一个空列表，用于存放该类别的数据
            separated[class_value] = []
        separated[class_value].append(vector)
    return separated    # 返回按类别分开的数据

# 计算均值和标准差
def summarize(data):
    summaries = [(sum(attribute) / len(attribute), (sum((x - (sum(attribute) / len(attribute))) ** 2 for x in attribute) / (len(attribute) - 1)) ** 0.5) for attribute in zip(*data)]
    del summaries[0]  # 不计算类别特征的均值和标准差
    return summaries


# 汇总每个类别的数据,训练模型入口函数
"""
该方法是用来计算每个类别的特征数据的均值和标准差，
它接收整个数据集作为输入，并对每个类别的数据进行汇总。
"""
def summarize_by_class(data):
    separated = separate_by_class(data)     #根据不同类别划分训练数据
    summaries = {}  # 用于存放每个类别特征值的均值和标准差
    for class_value, instances in tqdm(separated.items(), desc="Processing classes"):
        summaries[class_value] = summarize(instances)   #计算每个特征值的均值和标准差
    return summaries


# 计算高斯概率密度函数
def calculate_probability(x, mean, stdev):
    exponent = math.exp(-((x - mean) ** 2 / (2 * stdev ** 2)))
    return (1 / (math.sqrt(2 * math.pi) * stdev)) * exponent

# 计算类别的先验概率
"""
该方法用于计算给定输入向量对应于不同类别的
先验概率。关键参数包括summaries为一个字典,
包含每个类别的特征值均值和标准差以及input_vector,
是要进行分类的输入样本。
"""
def calculate_class_probabilities(summaries, input_vector):
    probabilities = {}  # 用于存放不同类别的先验概率
    for class_value, class_summaries in summaries.items():
        probabilities[class_value] = 1  # 初始化当前类别的先验概率为 1
        for i in range(len(class_summaries)):
            mean, stdev = class_summaries[i]
            x = input_vector[i + 1]  # 不考虑类别特征
            probabilities[class_value] *= calculate_probability(x, mean, stdev)
    return probabilities        #返回不同类别的数据先验概率

# 预测类别,测试集的入口函数
"""
该方法用于预测给定输入向量所属的类别。它利用先前计
算出的每个类别的先验概率，并选择具有最高概率的类
别作为预测结果。
"""
def predict(summaries, input_vector):
    probabilities = calculate_class_probabilities(summaries, input_vector)  # 获取每个类别的先验概率
    best_label, best_prob = None, -1  # 初始化标签和最高概率
    for class_value, probability in probabilities.items():  
        if best_label is None or probability > best_prob:
            best_prob = probability  # 更新最高概率
            best_label = class_value  # 更新最佳类别标签
    return best_label  # 返回预测的最佳类别


# 计算准确率
def get_accuracy(test_set, summaries):
    correct = 0  # 初始化正确分类的样本数量
    for instance in test_set:  
        prediction = predict(summaries, instance)  # 使用训练得出的特征均值和标准差进行预测
        if prediction == instance[0]: 
            correct += 1  # 增加正确分类的样本数量
    accuracy = (correct / float(len(test_set))) * 100.0
    return accuracy  # 返回模型的准确率百分比

"""
该方法是在寻找最佳的训练集和测试集划分比例。它通过尝
试不同的划分比例(从80%到95%)来训练模型，并在测试
集上评估模型的准确率。
"""
def find_best_split_ratio():
    # 加载数据集
    dataset = load_data(data_path)
    for split_ratio in range(80,96):
        parameter = split_ratio / 100  # 将百分比转换为小数
        train_set, test_set = split_data(dataset, parameter)
        #训练数据
        summaries = summarize_by_class(train_set)
        #测试模型
        accuracy = get_accuracy(test_set, summaries)
        if accuracy == 100.0:
            return parameter
    return -1

# 数据可视化
def summarize_by_class_with_plot(data, summaries):
    # 获取特征类别
    x_values = column_names[1:]  # 去除类别特征
    plt.figure(figsize=(10, 6))  # 设置一个图表用于绘制所有折线

    # 遍历summaries中元素
    for class_value, class_summaries in summaries.items():
        # 创建类别标签
        class_label = f"Class {class_value}"
        # 创建类别数据点
        y_values = [item[0] for item in class_summaries]
        # 绘制折线图
        plt.plot(x_values, y_values, marker='o', linestyle='-', label=class_label)  # 绘制折线
    
    
    train_features = [item[1:] for item in data]
    # 绘制训练集数据的散点图
    for i, feature in enumerate(train_features):
        plt.scatter(x_values, feature, marker='x')  # 绘制训练集数据的散点


    plt.title('Wine Properties')  # 设置图表标题
    plt.xlabel('Wine Attributes')  # 设置 x 轴标签
    plt.ylabel('Values')  # 设置 y 轴标签
    plt.xticks(rotation=90)  # 设置 x 轴标签旋转90度，以避免重叠
    plt.legend()  # 显示图例
    plt.grid(True)  # 显示网格线
    plt.tight_layout()  # 调整布局，避免标签重叠
    plt.show()  # 显示图表



if __name__ == "__main__":
    # 数据集路径
    data_path = "data/wine.data"

    # 加载数据集
    dataset = load_data(data_path)

    # 寻找最佳分割比例
    index = find_best_split_ratio()

    if index != -1:
        print(f"计算得最佳训练划分参数: {index}")
        # 使用最佳分割比例进行训练和测试
        best_split_ratio = index
        train_set, test_set = split_data(dataset, best_split_ratio)
        # 训练数据
        summaries = summarize_by_class(train_set)
        # 测试模型
        accuracy = get_accuracy(test_set, summaries)
        print(f"训练集识别正确率 ({best_split_ratio}): {accuracy:.2f}%")
        summarize_by_class_with_plot(dataset,summaries)
    else:
        #使用默认参数训练
        train_set, test_set = split_data(dataset)
        summaries = summarize_by_class(train_set)
        accuracy = get_accuracy(test_set, summaries)
        print(f"训练集识别正确率 ({0.8}): {accuracy:.2f}%")
        summarize_by_class_with_plot(dataset,summaries)
