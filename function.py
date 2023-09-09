import numpy as np
import math

def distance_3d(a:tuple, b:tuple):
    return np.sqrt((a[0]-b[0])**2+(a[1]-b[1])**2+(a[2]-b[2])**2)

def cosToSin(x):
    return np.sqrt(1-x**2)

def distance_2d(a, b):
    return np.sqrt((a[0]-b[0])**2+(a[1]-b[1])**2)

# 适用于month*time*i的三维array求每月平均值
def get_average_mni(input, len):
    average_mni = []
    for month in range(12):
        losses = 0
        for time in range(5):
            for i in range(len):
                losses += input[month][time][i]
        average_mni.append(losses/(5*len))
    return average_mni

# 计算阴影遮挡损失, 输入为：所有定日镜的坐标,cos太阳方位角,cos太阳高度角,镜面高度，吸收塔高度,镜面长度,镜子间距
def get_shadow_effi(x, y, cos_delta, cos_alpha, h1, h2, len1, average_distance):
    cos_theta = np.zeros((12, 5)) # 太阳入射角
    al = []
    t = np.zeros((12, 5, len(x)))
    loss = np.zeros((12, 5, len(x)))
    for i in range(len(x)):
        al.append(distance_2d((0, 0), (x[i], y[i])))
        al[i] = np.arctan(h2-h1/al[i]) # 塔与定日镜tan, a1
    for month in range(12):
        for time in range(5):
            cos_theta[month][time] = cos_delta[month]*cos_alpha[month][time] # 太阳入射角
            for i in range(len(x)):
                t[month][time][i] = (np.pi/2-np.arccos(cos_theta[month][time])-al[i])/2 # a2
                x1 = np.pi/2-(t[month][time][i]+al[i]) # a3
                a4 = np.pi/2+np.arccos(cos_theta[month][time])-x1
                loss[month][time][i] = average_distance*np.sin(np.pi/2-np.arccos(cos_theta[month][time]))/(len1*np.sin(a4))
                if loss[month][time][i] > 1 or int(distance_2d((0, 0), (x[i], y[i]))) == 107:
                    loss[month][time][i] = 1
    return loss, t

# 计算集热器截断效率，输入为集热器的高,集热器的直径,定日镜长,定日镜宽,数据组长度,集热器中心与定日镜中心距离
def get_trunc_effi(h_hotter, L_hotter, h_m, w_m, len, d_HR):
    effi_trunc = np.zeros(len) # 集热器截断效率
    for i in range(len):
        effi_trunc[i] = h_hotter/(h_m+2*d_HR[i]*np.tan(4.65e-3))*L_hotter/(w_m+2*d_HR[i]*np.tan(4.65e-3))
        if effi_trunc[i] > 1:
           effi_trunc[i] = 1
    return effi_trunc

# 计算集热器中心与定日镜中心距离和大气透射率，输入为坐标,集热器重心高度，定日镜高度，数据长度
def get_d_HR(x, y, h1, h2, len):
    eta_at = []
    d_HR = []
    for i in range(len):
        d_HR.append(distance_3d((0, 0, h1), (x[i], y[i], h2)))
        eta_at.append(0.99321-0.0001176*d_HR[i]+1.97e-8*d_HR[i]**2)
    return d_HR, eta_at

# 计算光学效率，输入为阴影遮挡效率，余弦效率，大气透射率，集热器截断效率,数组长度和镜面反射率
def get_light_effi(loss, effi_cos, eta_at, effi_trunc, len, eta_ref=0.92):
    light_effi = np.zeros((12, 5, len)) # 光学效率
    for month in range(12):
        for time in range(5):
            for i in range(len):
                light_effi[month][time][i] = (loss[month][time][i])*effi_cos[month][time][i]*eta_at[i]*effi_trunc[i]*eta_ref
    return light_effi

# 计算余弦效率，输入为t和长度
def get_effi_cos(t, len):
    effi_cos = np.zeros((12, 5, len))
    for month in range(12):
        for time in range(5):
            for i in range(len):
                effi_cos[month][time][i] = math.cos(t[month][time][i])
    return effi_cos

def get_heat_W(light_effi, S, DNI, len, shadow_loss):
    E_field = np.zeros((12, 5))
    for month in range(12):
        for time in range(5):
            for i in range(len):
                E_field[month][time] += S*(shadow_loss[month][time][i])*light_effi[month][time][i]
            E_field[month][time] *= DNI[month][time]
    return E_field

# 根据数量和间距得到每个半径分布数量
def get_nums(l, w, h, number, distance):
    min_dis = 5+w # 镜面之间最短距离
    r = 100 # 圆的半径
    nums = []
    while number > 0:
        C_circle = 2*np.pi*r
        num1 = int(C_circle/min_dis)
        if number >= num1:
            nums.append(num1)
            number -= num1
            r += distance
        else:
            nums.append(number)
            number = 0
    return nums

# 根据数量获取坐标系
def get_xy(nums, distance):
    r = 100
    nums_len = len(nums)
    x = []
    y = []
    for i in range(nums_len):
        deg = 0
        deg_plus = 2*math.pi*r/nums[i]
        for _ in range(nums[i]):
            x.append(np.cos(deg)*r)
            y.append(np.sin(deg)*r)
            deg += deg_plus
        r += distance
    return x, y

# 以下函数为第二题特供优化版
def new_get_shadow_loss(len, cos_delta, cos_alpha, average_distance, h1, h2, len1):
    cos_theta = np.zeros((12, 5)) # 太阳入射角
    al = []
    r = 100
    t = np.zeros((12, 5, len))
    loss = np.zeros((12, 5, len))
    for i in range(len):
        al.append(r)
        al[i] = np.arctan(h2-h1/al[i]) # 塔与定日镜tan, a1
        r += average_distance
    for month in range(12):
        for time in range(5):
            cos_theta[month][time] = cos_delta[month]*cos_alpha[month][time] # 太阳入射角
            for i in range(len):
                t[month][time][i] = (np.pi/2-np.arccos(cos_theta[month][time])-al[i])/2 # a2
                x1 = np.pi/2-(t[month][time][i]+al[i]) # a3
                a4 = np.pi/2+np.arccos(cos_theta[month][time])-x1
                loss[month][time][i] = average_distance*np.sin(np.pi/2-np.arccos(cos_theta[month][time]))/(len1*np.sin(a4))
                if loss[month][time][i] > 1 or i == 0:
                    loss[month][time][i] = 1
    return loss, t

def new_get_d_HR(h1, h2, len, average_distance):
    r = 100
    eta_at = []
    d_HR = []
    for i in range(len):
        d_HR.append(distance_3d((0, 0, h1), (r, 0, h2)))
        eta_at.append(0.99321-0.0001176*d_HR[i]+1.97e-8*d_HR[i]**2)
        r += average_distance
    return d_HR, eta_at

def new_get_heat_W(light_effi, S, DNI, len, shadow_loss, nums):
    E_field = np.zeros((12, 5))
    for month in range(12):
        for time in range(5):
            for i in range(len):
                E_field[month][time] += S*(shadow_loss[month][time][i])*light_effi[month][time][i]*nums[i]
            E_field[month][time] *= DNI[month][time]
    return E_field
#