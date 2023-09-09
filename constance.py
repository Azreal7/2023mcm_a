# 此文件用于存储常量
import numpy as np
import math

def cosToSin(x):
    return np.sqrt(1-x**2)

lat = np.deg2rad(39.4) # 维度
lot = np.deg2rad(98.5) # 经度
H = 3 # 海拔高度，单位km
G_0 = 1.366 # 太阳常数
ST = [9, 10.5, 12, 13.5, 15] # 当地时间
omega = [] # 太阳时角
D = [306, 337, 0, 31, 61, 92, 122, 153, 184, 214, 245, 275] # 从春分第0天起算的天数，默认一年365天，春分为3月21日
sin_delta = [] # 太阳赤纬角的sin值
cos_delta = [] # 太阳赤纬角的cos值
sin_alpha = np.zeros((12, 5)) # 太阳高度角的sin值
cos_alpha = np.zeros((12, 5)) # 太阳高度角的cos值，如何确定cos正负？
DNI = np.zeros((12, 5)) # 法向直接辐射辐照度
solar_angle = [] # 太阳入射角
cos_theta = np.zeros((12, 5)) # 太阳入射角
eta_ref = 0.92 # 镜面反射率

for _ in range(5):
    omega.append(np.pi/12*(ST[_]-12))

for _ in range(12):
    sin_delta.append(np.sin(2*np.pi*D[_]/365)*np.sin(2*np.pi*23.45/360))
    cos_delta.append(cosToSin(sin_delta[_]))

for month in range(12):
    for time in range(5):
        sin_alpha[month][time] = cos_delta[month]*math.cos(lat)*math.cos(omega[time]) + sin_delta[month]*math.sin(lat)
        cos_alpha[month][time] = cosToSin(sin_alpha[month][time])

cos_gamma = np.zeros((12, 5))
for month in range(12):
    for time in range(5):
        cos_gamma[month][time] = (sin_delta[month]-sin_alpha[month][time]*math.sin(lat))/(cos_alpha[month][time]*math.cos(lat))

a = 0.4237 - 0.00821*(6-H)**2
b = 0.5055 + 0.00595*(6.5-H)**2
c = 0.2711 + 0.01858*(2.5-H)**2

for month in range(12):
    for time in range(5):
        DNI[month][time] = G_0*(a+b*math.exp(-c/sin_alpha[month][time]))


for month in range(12):
        for time in range(5):
            cos_theta[month][time] = cos_delta[month]*cos_alpha[month][time] # 太阳入射角

max_circle = 25