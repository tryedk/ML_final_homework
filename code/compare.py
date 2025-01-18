import matplotlib.pyplot as plt

# 数据
models = ['LSMT', 'Transformer', 'T-LSTM']
mse_means = [0.0303, 0.0221, 0.0201]
mae_means = [0.1242, 0.0716, 0.0675]
mse_stds = [0.0013, 0.0015, 0.0012]
mae_stds = [0.0035, 0.0035, 0.0028]

# 绘制MSE曲线图
plt.figure(figsize=(10, 5))
plt.errorbar(models, mse_means, yerr=mse_stds, fmt='-o', label='MSE', capsize=5)
plt.title('MSE Comparison')
plt.xlabel('Models')
plt.ylabel('MSE')
plt.grid(True)
plt.legend()
plt.savefig('/data/kangw/work/ML/data/compare_MSE.png')
plt.show()

# 绘制MAE曲线图
plt.figure(figsize=(10, 5))
plt.errorbar(models, mae_means, yerr=mae_stds, fmt='-o', label='MAE', capsize=5)
plt.title('MAE Comparison')
plt.xlabel('Models')
plt.ylabel('MAE')
plt.grid(True)
plt.legend()
plt.savefig('/data/kangw/work/ML/data/compare_MAE.png')
plt.show()