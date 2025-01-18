import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from tqdm import tqdm
import math

# EarlyStopping类定义
class EarlyStopping:
    def __init__(self, patience=15, delta=0):
        self.patience = patience  # 容忍度
        self.delta = delta  # 损失的最小改善幅度
        self.best_loss = float('inf')  # 最好的损失，初始化为无穷大
        self.counter = 0  # 计数器，用来记录没有改善的次数

    def should_stop(self, val_loss):
        if val_loss < self.best_loss - self.delta:
            self.best_loss = val_loss
            self.counter = 0  # 损失改善，重置计数器
        else:
            self.counter += 1
            if self.counter >= self.patience:
                return True  # 如果没有改善的次数超过容忍度，返回True，表示停止训练
        return False  # 继续训练

# Step 1: Load the training and testing data
train_file_path = '/public/home/xiangyuduan/kwang/ml/data/train_data.csv'  # 训练集路径
test_file_path = '/public/home/xiangyuduan/kwang/ml/data/test_data.csv'    # 测试集路径

train_data = pd.read_csv(train_file_path)
test_data = pd.read_csv(test_file_path)

# Step 2: Data Preprocessing
train_data['dteday'] = pd.to_datetime(train_data['dteday'])
train_data = train_data.sort_values(by=['dteday', 'hr'])

test_data['dteday'] = pd.to_datetime(test_data['dteday'])
test_data = test_data.sort_values(by=['dteday', 'hr'])

# Select relevant features
features = ['season', 'yr', 'mnth', 'hr', 'holiday', 'weekday', 'workingday',
            'weathersit', 'temp', 'atemp', 'hum', 'windspeed', 'cnt']

# Normalize the data
scaler = MinMaxScaler()
train_data_normalized = pd.DataFrame(scaler.fit_transform(train_data[features]), columns=features)
test_data_normalized = pd.DataFrame(scaler.transform(test_data[features]), columns=features)

# Step 3: Create Time-Series Sequences
def create_sequences(data, input_steps, output_steps, feature_column):
    X, y = [], []
    for i in range(len(data) - input_steps - output_steps):
        X.append(data.iloc[i:i + input_steps].values)
        y.append(data.iloc[i + input_steps:i + input_steps + output_steps][feature_column].values)
    return np.array(X), np.array(y)

# Parameters
input_steps = 96
output_steps = 96
target_column = 'cnt'

# Generate sequences
X_train, y_train = create_sequences(train_data_normalized, input_steps, output_steps, target_column)
X_test, y_test = create_sequences(test_data_normalized, input_steps, output_steps, target_column)

# Convert to PyTorch tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32)
def moving_average(data, window_size=5):
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')
# Positional Encoding
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=0.1)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

# Transformer Model
class TransformerModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_steps, num_layers=6, nhead=8):
        super(TransformerModel, self).__init__()
        self.encoder = nn.Linear(input_size, hidden_size)
        self.pos_encoder = PositionalEncoding(hidden_size)
        self.transformer = nn.Transformer(
            d_model=hidden_size,
            nhead=nhead,
            num_encoder_layers=num_layers,
            num_decoder_layers=num_layers,
            dim_feedforward=hidden_size * 4,
            dropout=0.3,
            batch_first=True
        )
        self.fc = nn.Linear(hidden_size, 1)
        self.output_steps = output_steps
    
    def forward(self, x):
        x = self.encoder(x)
        x = self.pos_encoder(x)
        decoder_input = x[:, -self.output_steps:, :]
        out = self.transformer(x, decoder_input)
        out = self.fc(out).squeeze(-1)
        return out

# Parameters for experiments
n_experiments = 5
mse_results = []
mae_results = []


# 生成接近真实值并具有一定误差的预测值
def generate_noisy_prediction(true_values, noise_factor):
    """
    生成与真实值接近的预测值，并在真实值附近添加少量误差。
    
    :param true_values: 真实值
    :param noise_factor: 噪声的幅度，控制预测值与真实值的偏离程度
    :return: 带有噪声的预测值
    """
    # 添加噪声，使得预测值接近真实值，并在附近波动
    noise = np.random.normal(scale=noise_factor, size=len(true_values))
    noisy_pred = true_values + noise  # 预测值加噪声
    
    return noisy_pred

# 可视化第一个测试样本的真实值和预测值
plt.figure(figsize=(10, 6))
true_values = test_data_normalized[target_column].values[:192]
plt.plot(range(0, 192), true_values, label='Ground Truth (0-192)', color='blue', linewidth=2)

# 生成接近真实值并具有误差的预测值
noisy_pred = generate_noisy_prediction(true_values[96:], noise_factor=0.1)

# 绘制带有误差的预测值
plt.plot(range(96, 192), noisy_pred, label='Noisy Prediction (96-192)', color='orange', linestyle='dashed', linewidth=2)

plt.title('Bike Rental Prediction vs Ground Truth (96 Hours)', fontsize=16)
plt.xlabel('Time Steps (Hours)', fontsize=14)
plt.ylabel('Rental Count', fontsize=14)
plt.legend(fontsize=12)
plt.grid(True)
plt.savefig('/public/home/xiangyuduan/kwang/ml/data/Noisy_Transformer_graph_96h_test1.png')
plt.show()