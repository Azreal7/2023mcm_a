import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import function as f
import constance as c

df = pd.read_excel(r"附件.xlsx")

x = df['x坐标 (m)'].tolist()
y = df['y坐标 (m)'].tolist()

# plt.scatter(x, y)
# plt.show()

d_HR, eta_at = f.get_d_HR(x, y, 84, 4, len(x))

# 此部分用于区分x,y中不同的圆，以计算圆之间的平均距离
distance = []
for i in range(len(x)):
    distance.append(int(f.distance_2d((0, 0), (x[i], y[i]))))
distance = sorted(list(set(distance)))
average_distance = 0 # 圆环之间的平均距离
for i in range(1, len(distance)):
    average_distance += distance[i] - distance[i-1]
average_distance /= len(distance) - 1

# 计算阴影遮挡效率
loss, t = f.get_shadow_effi(x, y, c.cos_delta, c.cos_alpha, 4, 80, 6, average_distance)
# print("loss:" + loss)

# 计算平均阴影遮挡效率
average_effi = f.get_average_mni(loss, len(x))
print( average_effi)

# 计算集热器截断效率
effi_trunc = f.get_trunc_effi(8, 7, 6, 6, len(x), d_HR)
print(np.mean(effi_trunc))

# 计算余弦效率
effi_cos = f.get_effi_cos(t, len(x))
average_cos = f.get_average_mni(effi_cos, len(x))
print(average_cos)

# 求光学效率，镜面反射率默认0.92
light_effi = f.get_light_effi(loss, effi_cos, eta_at, effi_trunc, len(x))
average_light = f.get_average_mni(light_effi, len(x))
print(average_light)

# print(average_light)
E_field = f.get_heat_W(light_effi, 6*6, c.DNI, len(x), loss)
density = []
for i in range(12):
    E = 0
    for time in range(5):
        E += E_field[i][time]
    density.append(E/(36*len(x)*5))
print(density)