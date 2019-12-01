import matplotlib.pyplot as plt
import numpy as np
import matplotlib

import pandas as pd
# pd.set_option("display.max_colwidth",100)
# data = pd.read_csv(r'/Users/allmight/PycharmProjects/Mountain/src/data/data3/case1train.csv')
file = open("/Users/allmight/PycharmProjects/Mountain/src/data/data3/predict.txt")
talls = []
i = 1
for line in file.readlines():
     line=line.strip('\n')
     tall = float(line)
     talls.append(tall)
     # if tall < 700:
     #     print(str(tall))
file.close()
# for i in range(0, 1825):
#     string = str(data.loc[i])
#     splited = string.split("    ")
#     line = splited[1].split("\n")[0]
#     print(line)
#     fname_mname_tall = line.split(";")
#     tall = float(fname_mname_tall[2])
#     talls.append(tall)

# 设置matplotlib正常显示中文和负号
matplotlib.rcParams['axes.unicode_minus']=False     # 正常显示负号
# 随机生成（10000,）服从正态分布的数据
"""
绘制直方图
data:必选参数，绘图数据
bins:直方图的长条形数目，可选项，默认为10
normed:是否将得到的直方图向量归一化，可选项，默认为0，代表不归一化，显示频数。normed=1，表示归一化，显示频率。
facecolor:长条形的颜色
edgecolor:长条形边框的颜色
alpha:透明度
"""
(counts, bins, patch) = plt.hist(talls, bins=2, normed=0, facecolor="blue", edgecolor="black", alpha=0.7)
print(bins)
print(counts)
# 显示横轴标签
plt.xlabel("Altitude")
# 显示纵轴标签
plt.ylabel("count")
# 显示图标题
plt.title("p2")
plt.show()
