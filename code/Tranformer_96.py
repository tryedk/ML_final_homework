import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt

# Step 1: Load the training and testing data
train_file_path = '/data/kangw/work/ML/data/train_data.csv'  # 训练集路径
test_file_path = '/data/kangw/work/ML/data/test_data.csv'    # 测试集路径

train_data = pd.read_csv(train_file_path)
test_data = pd.read_csv(test_file_path)

# Step 2: Data Preprocessing
# Convert 'dteday' to datetime format and sort by date and hour
train_data['dteday'] = pd.to_datetime(train_data['dteday'])
train_data = train_data.sort_values(by=['dteday', 'hr'])

test_data['dteday'] = pd.to_datetime(test_data['dteday'])
test_data = test_data.sort_values(by=['dteday', 'hr'])

# Select relevant features
features = ['season', 'yr', 'mnth', 'hr', 'holiday', 'weekday', 'workingday',
            'weathersit', 'temp', 'atemp', 'hum', 'windspeed', 'cnt']

# Normalize the data using the training data's scaler
scaler = MinMaxScaler()
train_data_normalized = pd.DataFrame(scaler.fit_transform(train_data[features]), columns=features)
test_data_normalized = pd.DataFrame(scaler.transform(test_data[features]), columns=features)

# Step 3: Create Time-Series Sequences
def create_sequences(data, input_steps, output_steps, feature_column):
    X, y = [], []
    for i in range(len(data) - input_steps - output_steps):
        X.append(data.iloc[i:i + input_steps].values)  # Input sequence
        y.append(data.iloc[i + input_steps:i + input_steps + output_steps][feature_column].values)  # Output sequence
    return np.array(X), np.array(y)

# Parameters
input_steps = 96  # Past 96 hours
output_steps = 96  # Short-term prediction (future 96 hours)
target_column = 'cnt'

# Generate sequences for training and testing
X_train, y_train = create_sequences(train_data_normalized, input_steps, output_steps, target_column)
X_test, y_test = create_sequences(test_data_normalized, input_steps, output_steps, target_column)

# Convert to PyTorch tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32)

class TransformerModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_steps, num_layers=4, nhead=4):
        super(TransformerModel, self).__init__()
        self.encoder = nn.Linear(input_size, hidden_size)
        self.transformer = nn.Transformer(
            d_model=hidden_size,
            nhead=nhead,
            num_encoder_layers=num_layers,
            num_decoder_layers=num_layers,
            dim_feedforward=hidden_size * 4,
            dropout=0.1,
            batch_first=True
        )
        self.fc = nn.Linear(hidden_size, 1)
        self.output_steps = output_steps
    
    def forward(self, x):
        x = self.encoder(x)
        decoder_input = x[:, -self.output_steps:, :]  # 使用输入序列的最后一部分作为解码器输入
        out = self.transformer(x, decoder_input)
        out = self.fc(out).squeeze(-1)
        return out

# Parameters for experiments
n_experiments = 5
mse_results = []
mae_results = []

for i in range(n_experiments):
    print(f"Experiment {i + 1}/{n_experiments}...")
    
    # Convert to PyTorch DataLoader
    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    
    # Build the model
    input_size = X_train.shape[2]
    hidden_size = 128
    model = TransformerModel(input_size, hidden_size, output_steps)
    
    # Move model to GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    # Define loss and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    
    # Train the model
    for epoch in range(50):  # 增加训练轮数
        model.train()
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
    
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

# 提取完整的真实值（0-192 小时）
true_values = test_data_normalized[target_column].values[:192]

# 绘制真实值（0-192）
plt.plot(range(0, 192), true_values, label='Ground Truth (0-192)', color='blue', linewidth=2)

# 绘制预测值（96-192）
plt.plot(range(96, 192), y_pred[0], label='Prediction (96-192)', color='orange', linestyle='dashed', linewidth=2)

plt.title('Bike Rental Prediction vs Ground Truth (96 Hours)', fontsize=16)
plt.xlabel('Time Steps (Hours)', fontsize=14)
plt.ylabel('Rental Count', fontsize=14)
plt.legend(fontsize=12)
plt.grid(True)
plt.savefig('/data/kangw/work/ML/data/Transformer_graph_96h_test.png')
plt.show()