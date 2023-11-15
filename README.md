# 简单红酒分类器

这个项目是一个简单的红酒分类器，根据红酒的化学属性将其分类到不同的类别中。

## 数据集

数据集位于 `data/wine.data`，包含以下特征：

- 'class'
- 'Alcohol'
- 'Malic acid'
- 'Ash'
- 'Alcalinity of ash'
- 'Magnesium'
- 'Total phenols'
- 'Flavanoids'
- 'Nonflavanoid phenols'
- 'Proanthocyanins'
- 'Color intensity'
- 'Hue'
- 'OD280/OD315 of diluted wines'
- 'Proline'

## 代码说明

### 数据加载与处理

- `load_data(file_path)`: 加载数据集。
- `split_data(data, split_ratio)`: 将数据集按照指定比例分割成训练集和测试集。
- `separate_by_class(data)`: 按类别划分数据。

### 训练模型

- `summarize(data)`: 计算均值和标准差。
- `summarize_by_class(data)`: 汇总每个类别的数据，计算特征的均值和标准差。

### 模型预测

- `calculate_probability(x, mean, stdev)`: 计算高斯概率密度函数。
- `calculate_class_probabilities(summaries, input_vector)`: 计算给定输入向量对应于不同类别的先验概率。
- `predict(summaries, input_vector)`: 预测输入向量所属的类别。

### 模型评估与可视化

- `get_accuracy(test_set, summaries)`: 计算模型准确率。
- `find_best_split_ratio()`: 寻找最佳的训练集和测试集划分比例。
- `summarize_by_class_with_plot(data, summaries)`: 可视化每个类别的特征数据。

## 使用方法

1. **数据集准备**：确保数据集路径正确，并根据需要进行修改。
2. **模型训练**：运行 `find_best_split_ratio()` 寻找最佳的训练集和测试集划分比例。
3. **模型评估**：使用找到的最佳划分比例运行模型，并查看准确率和特征数据可视化。

**依赖项安装**

确保你已经安装了以下 Python 库：

```bash
pip install matplotlib tqdm



#Simple Wine Classifier

This project is a simple wine classifier that categorizes wines into different classes based on their chemical attributes.

## Dataset

The dataset is located at `data/wine.data` and includes the following features:

- 'class'
- 'Alcohol'
- 'Malic acid'
- 'Ash'
- 'Alcalinity of ash'
- 'Magnesium'
- 'Total phenols'
- 'Flavanoids'
- 'Nonflavanoid phenols'
- 'Proanthocyanins'
- 'Color intensity'
- 'Hue'
- 'OD280/OD315 of diluted wines'
- 'Proline'

## Code Overview

### Data Loading and Processing

- `load_data(file_path)`: Loads the dataset.
- `split_data(data, split_ratio)`: Splits the dataset into training and testing sets based on a specified ratio.
- `separate_by_class(data)`: Separates data by class.

### Model Training

- `summarize(data)`: Calculates mean and standard deviation.
- `summarize_by_class(data)`: Summarizes data for each class, computing mean and standard deviation for features.

### Model Prediction

- `calculate_probability(x, mean, stdev)`: Calculates Gaussian probability density function.
- `calculate_class_probabilities(summaries, input_vector)`: Computes prior probabilities for different classes given an input vector.
- `predict(summaries, input_vector)`: Predicts the class of an input vector.

### Model Evaluation and Visualization

- `get_accuracy(test_set, summaries)`: Computes model accuracy.
- `find_best_split_ratio()`: Finds the best split ratio between training and testing sets.
- `summarize_by_class_with_plot(data, summaries)`: Visualizes feature data for each class.

## Usage

1. **Data Preparation**: Ensure the dataset path is correct and modify it if needed.
2. **Model Training**: Run `find_best_split_ratio()` to discover the best split ratio between the training and testing sets.
3. **Model Evaluation**: Run the model using the found best split ratio and observe accuracy and feature data visualization.

**Dependencies Installation**

Make sure you have installed the following Python libraries:

```bash
pip install matplotlib tqdm
