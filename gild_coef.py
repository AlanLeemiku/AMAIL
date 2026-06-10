import math
import matplotlib.pyplot as plt

end_update = 1000
k = 75
sigmoid_coefs = []
linear_coefs = []

for j in range(end_update + 1):
    ratio = j / end_update

    # 分段平滑下降：前段线性从1降到0.8，后段sigmoid下降到0
    if ratio <= 0.06:
        sigmoid_coef = 1.0 - (1.0 - 0.95) * (ratio / 0.05)  # 线性从1降到0.8
    else:
        # 后段 sigmoid，从 0.8 平滑下降到 0
        progress = (ratio - 0.05) / 0.95  # 归一化到 [0, 1]
        x = (progress - 0.07) * k  # sigmoid centered
        sigmoid_part = 1 - 1 / (1 + math.exp(-x))  # 输出范围 [0,1]
        sigmoid_coef = 0.95 * sigmoid_part  # 缩放到 [0,0.8]

    sigmoid_coef = max(sigmoid_coef, 0.0)
    sigmoid_coefs.append(sigmoid_coef)

    # 线性下降曲线（对比用）
    linear_coef = 1.0 - ratio
    linear_coefs.append(linear_coef)

# 绘图
plt.plot(range(end_update + 1), sigmoid_coefs, label="Piecewise Sigmoid Decay", linewidth=2)
plt.plot(range(end_update + 1), linear_coefs, label="Linear Decay", linestyle='--')
plt.title("GILD Coefficient: Piecewise Sigmoid (1→0.8→0)")
plt.xlabel("Training Step (j)")
plt.ylabel("gild_coef")
plt.legend()
plt.grid(True)
plt.show()



        #     ratio = j / end_update
        #     k = 30
        #     # 分段平滑下降：前段线性从1降到0.8，后段gild下降到0
        #     if ratio <= 0.15:
        #         gild_coef = 1.0 - (1.0 - 0.85) * (ratio / 0.15)  # 线性从1降到0.8
        #     else:
        # # 后段 gild，从 0.8 平滑下降到 0
        #         progress = (ratio - 0.15) / 0.85  # 归一化到 [0, 1]
        #         x = (progress - 0.15) * k  # gild centered
        #         gild_part = 1 - 1 / (1 + math.exp(-x))  # 输出范围 [0,1]
        #         gild_coef = 0.85 * gild_part  # 缩放到 [0,0.8]
        #         gild_coef = max(gild_coef, 0.0)
        
        
        
        
# import math
# import matplotlib.pyplot as plt

# end_update = 1000
# y0 = 0.92  # 提高前1/5结束时的值
# k = 4 * (1 - y0) / (0.2 * y0)

# gild_coefs = []

# for j in range(end_update + 1):
#     ratio = j / end_update

#     if ratio <= 0.2:
#         # 前1/5更缓慢下降
#         gild_coef = 1.0 - (1.0 - y0) * (ratio / 0.2) ** 0.8
#     else:
#         progress = (ratio - 0.2) / 0.8
#         x = progress * k
#         sigmoid_part = 1 / (1 + math.exp(-x))
#         gild_coef = y0 * (2 * (1 - sigmoid_part))

#     gild_coef = max(gild_coef, 0.0)
#     gild_coefs.append(gild_coef)

# plt.plot(range(end_update + 1), gild_coefs, label="更缓慢的前1/5", linewidth=2)
# plt.title("GILD Coefficient: 前1/5更缓慢")
# plt.xlabel("Training Step (j)")
# plt.ylabel("gild_coef")
# plt.legend()
# plt.grid(True)
# plt.show()