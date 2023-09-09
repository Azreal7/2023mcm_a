import numpy as np
import math
import constance as c

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
    al = []
    t = np.zeros((12, 5, len(x)))
    loss = np.zeros((12, 5, len(x)))
    for i in range(len(x)):
        al.append(distance_2d((0, 0), (x[i], y[i])))
        al[i] = np.arctan(h2-h1/al[i]) # 塔与定日镜tan, a1
    for month in range(12):
        for time in range(5):
            for i in range(len(x)):
                t[month][time][i] = (np.pi/2-np.arccos(c.cos_theta[month][time])-al[i])/2 # a2
                x1 = np.pi/2-(t[month][time][i]+al[i]) # a3
                a4 = np.pi/2+np.arccos(c.cos_theta[month][time])-x1
                loss[month][time][i] = average_distance*np.sin(np.pi/2-np.arccos(c.cos_theta[month][time]))/(len1*np.sin(a4))
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
def get_nums(w, number, distance):
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
def new_get_shadow_loss(len, average_distance, h1, len1):
    al = []
    r = 100
    t = np.zeros((12, 5, len))
    loss = np.zeros((12, 5, len))
    for i in range(len):
        al.append(r)
        al[i] = np.arctan(84-h1/al[i]) # 塔与定日镜tan, a1
        r += average_distance
    for month in range(12):
        for time in range(5):
            for i in range(len):
                t[month][time][i] = (np.pi/2-np.arccos(c.cos_theta[month][time])-al[i])/2 # a2
                x1 = np.pi/2-(t[month][time][i]+al[i]) # a3
                a4 = np.pi/2+np.arccos(c.cos_theta[month][time])-x1
                loss[month][time][i] = average_distance*np.sin(np.pi/2-np.arccos(c.cos_theta[month][time]))/(len1*np.sin(a4))
                if loss[month][time][i] > 1 or i == 0:
                    loss[month][time][i] = 1
    return loss, t

def new_get_d_HR(h2, len, average_distance):
    r = 100
    eta_at = []
    d_HR = []
    for i in range(len):
        d_HR.append(distance_3d((0, 0, 84), (r, 0, h2)))
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

# len:圈数,cos_delta:cos太阳方位角,cos_alpha:cos太阳高度角,a_d:圈间距,h_mirror:镜子高度,length:镜子长度,width:镜子宽度, height_diff:两镜高度差
def q3_get_shadow_loss(distance, h_mirror, length, width, height, len=c.max_circle):
    r = 100
    al = []
    t = np.zeros((12, 5, len))
    loss = np.zeros((12, 5, len))
    for i in range(len):
        al.append(r)
        al[i] = np.arctan(84-h_mirror[i]/al[i]) # alpha
        r += distance[i]
    for month in range(12):
        for time in range(5):
            for i in range(len):
                t[month][time][i] = (np.pi/2-np.arccos(c.cos_theta[month][time])-al[i])/2
                if i == 0:
                    loss[month][time][i] = 1
                    continue
                h1 = distance[i-1]*np.tan(al[i])
                c0 = h1 - (height[i]-height[i-1])
                b = length[i]*np.sin(al[i])+length[i]*np.cos(al[i])*np.tan(np.arccos(c.cos_theta[month][time]))
                x_a = (b+c0)/(np.tan(al[i])+np.tan(np.arccos(c.cos_theta[month][time])))*-1
                y_a = np.tan(np.arccos(c.cos_theta[month][time]))*x_a+b
                x_b = -1*length[i]*np.cos(al[i])-distance[i-1]+length[i-1]*np.cos(al[i])
                y_b = -1*np.tan(al[i])*length[i-1]-c0
                D_ab = np.sqrt((x_a-x_b)**2+(y_a-y_b)**2)
                if width[i] > width[i-1]:
                    loss[month][time][i] = D_ab*width[i]/(length[i-1]*width[i-1])
                else:
                    loss[month][time][i] = D_ab/length[i-1]
                if loss[month][time][i] > 1:
                    loss[month][time][i] = 1
    return loss, t

# distance:间距, number:数量, width:宽度, 
def q3_get_new_nums(distances, number, width):
    r = 100
    nums = np.zeros(c.max_circle)
    for i in range(c.max_circle):
        min_dis = width[i] + 5
        C_circle = 2*np.pi*r
        if number == 0:
            break
        nums[i] = int(C_circle/min_dis)
        if nums[i] > number:
            nums[i] = number
            number = 0
            break
        number -= nums[i]
        r += distances[i]
    return nums

def q3_get_d_HR(height, distances, len=c.max_circle):
    r = 100
    eta_at = []
    d_HR = []
    for i in range(len):
        d_HR.append(distance_3d((0, 0, 84), (r, 0, height[i])))
        eta_at.append(0.99321-0.0001176*d_HR[i]+1.97e-8*d_HR[i]**2)
        r += distances[i]
    return d_HR, eta_at

def q3_get_xy(nums, distances):
    r = 100
    nums_len = len(nums)
    x = np.zeros(c.max_circle)
    y = np.zeros(c.max_circle)
    for i in range(nums_len):
        deg = 0
        if nums[i] == 0:
            break
        deg_plus = 2*math.pi*r/nums[i]
        for _ in range(int(nums[i])):
            x[i] = np.cos(deg)*r
            y[i] = np.sin(deg)*r
            deg += deg_plus
        r += distances[i]
    return x, y

def q3_get_trunc_effi(h_hotter, L_hotter, h_m, w_m, d_HR, len=c.max_circle):
    effi_trunc = np.zeros(len) # 集热器截断效率
    for i in range(len):
        effi_trunc[i] = h_hotter/(h_m[i]+2*d_HR[i]*np.tan(4.65e-3))*L_hotter/(w_m[i]+2*d_HR[i]*np.tan(4.65e-3))
        if effi_trunc[i] > 1:
           effi_trunc[i] = 1
    return effi_trunc

def q3_get_heat_W(light_effi, S, DNI, len, shadow_loss, nums):
    E_field = np.zeros((12, 5))
    for month in range(12):
        for time in range(5):
            for i in range(len):
                E_field[month][time] += S[i]*(shadow_loss[month][time][i])*light_effi[month][time][i]*nums[i]
            E_field[month][time] *= DNI[month][time]
    return E_field