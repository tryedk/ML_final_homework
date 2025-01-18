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

# EarlyStopping实例化
early_stopping = EarlyStopping(patience=15, delta=0.001)

for i in range(n_experiments):
    print(f"Experiment {i + 1}/{n_experiments}...")

    # Convert to PyTorch DataLoader
    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    # Build the model
    input_size = X_train.shape[2]
    hidden_size = 256
    model = TransformerModel(input_size, hidden_size, output_steps)

    # Move model to GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # Define loss and optimizer
    criterion = nn.HuberLoss(delta=1.0)
    optimizer = optim.Adam(model.parameters(), lr=0.0005, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=15, verbose=True)

    # Train the model
    train_losses = []
    val_losses = []
    for epoch in tqdm(range(50), desc="Training Epochs", leave=False):
        model.train()
        epoch_loss = 0
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        train_losses.append(epoch_loss / len(train_loader))

        # 验证集损失
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch_X, batch_y in test_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                outputs = model(batch_X)
                val_loss += criterion(outputs, batch_y).item()
        val_losses.append(val_loss / len(test_loader))

        # 检查是否需要早停
        if early_stopping.should_stop(val_loss / len(test_loader)):
            print(f"Early stopping at epoch {epoch + 1}")
            break

        scheduler.step(val_loss)  # 根据验证集损失调整学习率

    # 绘制训练和验证损失曲线
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss')
    plt.savefig('/public/home/xiangyuduan/kwang/ml/data/Transformer_graph_96h_Training and Validation Loss.png')
    plt.show()

    # Evaluate the model on the test set
    model.eval()
    y_pred_list = []
    y_true_list = []
    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            outputs = model(batch_X)
            y_pred_list.append(outputs.cpu().numpy())
            y_true_list.append(batch_y.cpu().numpy())

    y_pred = np.concatenate(y_pred_list)
    y_true = np.concatenate(y_true_list)

    mse = mean_squared_error(y_true.flatten(), y_pred.flatten())
    mae = mean_absolute_error(y_true.flatten(), y_pred.flatten())

    mse_results.append(mse)
    mae_results.append(mae)

# Calculate mean and standard deviation for MSE and MAE
mse_mean = np.mean(mse_results)
mse_std = np.std(mse_results)
mae_mean = np.mean(mae_results)
mae_std = np.std(mae_results)

# Display results
print(f"\nEvaluation Results (5 Rounds):")
print(f"MSE: Mean = {mse_mean:.4f}, Std = {mse_std:.4f}")
print(f"MAE: Mean = {mae_mean:.4f}, Std = {mae_std:.4f}")

# 可视化第一个测试样本的真实值和预测值
plt.figure(figsize=(10, 6))
true_values = test_data_normalized[target_column].values[:192]
plt.plot(range(0, 192), true_values, label='Ground Truth (0-192)', color='blue', linewidth=2)
plt.plot(range(96, 192), y_pred[0], label='Prediction (96-192)', color='orange', linestyle='dashed', linewidth=2)
plt.title('Bike Rental Prediction vs Ground Truth (96 Hours)', fontsize=16)
plt.xlabel('Time Steps (Hours)', fontsize=14)
plt.ylabel('Rental Count', fontsize=14)
plt.legend(fontsize=12)
plt.grid(True)
plt.savefig('/public/home/xiangyuduan/kwang/ml/data/Transformer_graph_96h_test.png')
plt.show()
