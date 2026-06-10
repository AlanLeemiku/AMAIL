import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("/home/miku/Code/DRAIL/avg_ep_found_goal/wandb_export_2025-05-27T20_40_40.805+08_00.csv")  # 改成你的文件名
col_name = 'star-520-FPEC-1-NI-drail-noise1 - avg_ep_found_goal'
mean_goal = df[col_name].mean()
print("平均 avg_ep_found_goal:", mean_goal)

print("平均 avg_ep_found_goal:", mean_goal)







# 设置列名
step_col = 'Step'
goal_col = 'star-520-FPEC-1-NI-drail-noise1 - avg_ep_found_goal'

# 计算 cumulative average
df['cumulative_avg'] = df[goal_col].cumsum() / df[step_col]

# 画图
plt.figure(figsize=(10, 6))
plt.plot(df[step_col], df['cumulative_avg'], label='Cumulative Average of avg_ep_found_goal')
plt.xlabel('Step')
plt.ylabel('Cumulative Average')
plt.title('avg_ep_found_goal Cumulative Average over Steps')
plt.grid(True)
plt.legend()
plt.tight_layout()

# 保存图像为 PNG 文件
plt.savefig("/home/miku/Code/DRAIL/png/cumulative_avg_ep_found_goal.png", dpi=300)

# 显示图像
plt.show()