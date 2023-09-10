import function as f
import constance as c
import numpy as np
import math
import time
import random


class sa_tsp:
    # 接受的输入参数：起始温度start_temp, 结束温度end_temp, 内循环次数num
    def __init__(self, s_t, e_t, num):
        self.start_temp = s_t
        self.end_temp = e_t
        self.num = num

    # 获取当前状态的功率密度
    def get_power_density(self, state):
        power, num = self.get_power(state)
        area = 0
        for i in range(len(num)):
            area += state[1][i] * state[2][i] * num[i]
        return power / area

    # 获取当前状态的功率值和摆放的圈数数组
    # state:数量 长度 宽度 高度 距离
    def get_power(self, state):
        nums = f.q3_get_new_nums(state[4], state[0], state[2])
        S = np.zeros(c.max_circle)
        for i in range(c.max_circle):
            S[i] = state[1][i] * state[2][i]
        x, y = f.q3_get_xy(nums, state[4])
        loss, t = f.q3_get_shadow_loss(state[4], state[3], state[1], state[2], state[3])
        d_HR, eta_at = f.q3_get_d_HR(state[3], state[4])
        effi_trunc = f.q3_get_trunc_effi(8, 7, state[1], state[2], d_HR)
        effi_cos = f.get_effi_cos(t, c.max_circle)
        light_effi = f.get_light_effi(loss, effi_cos, eta_at, effi_trunc, c.max_circle)
        E_field = f.q3_get_heat_W(light_effi, S, c.DNI, c.max_circle, loss, nums)
        return np.mean(E_field), nums


    # 获取下一个状态
    # 如果符合约束条件，则返回新状态，否则放回原状态
    def get_next_state(self, state):
        next_state = [state[0]]
        for i in range(1, 5):
            next_state.append(state[i].copy())
        vary_list = [100, 4, 4, 2, 10]
        vary_list = [0,0,0,0,0]
        flag = random.choice([-1, 1])
        vary_num = random.random()
        select_num = random.random()
        if select_num >= 0.5:
            next_state[0] = next_state[0] + flag * vary_list[i] * vary_num
        row_select = random.randint(0, c.max_circle - 1)
        flag = random.choice([-1, 1])
        vary_num = random.random()
        select_num = random.random()
        c_select = random.randint(1, 4)
        next_state[c_select][row_select] = next_state[c_select][row_select] + flag * vary_list[c_select] * vary_num
        # for i in range(c.max_circle):       # 最大圈数
        #     for j in range(1, 5):
        #         flag = random.choice([-1, 1])
        #         vary_num = random.random()
        #         select_num = random.random()
        #         if select_num >= 0.5:
        #             next_state[j][i] = next_state[j][i] + flag * vary_list[j] * vary_num
        next_state[0] = int(next_state[0])      # 数目只能是整数
        power, num = self.get_power(next_state)
        if self.is_valid(next_state, power, num):
            return next_state
        else:
            return state

    # 判断是否满足约束条件
    def is_valid(self, state, power, num):
        valid = True
        for i in range(len(num)):
            if num[i] == 0:
                break
            valid = valid and 2 <= state[1][i] <= 8 and 2 <= state[2][i] <= 8 and 2 <= state[3][i] <= 6 \
                    and state[1][i] <= state[3][i] * 2 and state[1][i] <= state[2][i] \
                    and state[4][i] >= state[2][i] + 5
        r = 100
        for i in range(len(num)-1):
            if num[i] == 0:
                r -= state[4][i-1]
                break
            r += state[4][i]
        return valid and r <= 350 and power >= 60000

    # 降温函数
    def update_temp(self, temp, num):
        # return self.start_temp / math.log(1+num)
        return 0.99 * temp

    # 模拟退火算法
    def simulated_annealing(self, initial_state):
        state = initial_state

        best_power_density, best_state = self.get_power_density(initial_state), initial_state
        cur_temp = self.start_temp
        change_num = 0

        # 外层循环，这里的终止条件采取的是设置温度的阈值
        while cur_temp > self.end_temp:
            # 内层循环，这里的终止条件采取的是设置内循环数目
            loop_num = self.num
            for i in range(loop_num):
                new_state = self.get_next_state(state)
                if state == new_state:
                    continue
                initial_power_density, new_power_density = self.get_power_density(state), self.get_power_density(new_state)
                delta = initial_power_density - new_power_density
                rec_p = math.exp(-(delta / cur_temp))
                if rec_p > random.uniform(0, 1):
                    state = new_state

                # 判断是否更优
                if new_power_density > best_power_density:
                    best_state = new_state
                    best_power_density = new_power_density
                    # print("temp:", cur_temp, ", best_power_density:", best_power_density, "state:", best_state)

            print("temp:", cur_temp, ", best_power_density:", best_power_density, "state:", best_state)
            # 更新温度
            cur_temp = self.update_temp(cur_temp, change_num)

        return best_power_density, best_state


if __name__ == '__main__':

    start_time = time.time()

    initial_state = [2900, [5.5] * c.max_circle, [5.5] * c.max_circle, [4] * c.max_circle, [10.5] * c.max_circle]

    # 模拟退火
    test = sa_tsp(10, 0.1, 100)
    best_power_density, best_state = test.simulated_annealing(initial_state)

    end_time = time.time()

    print(best_power_density)
    print(best_state)
    print("time:", end_time-start_time, "s")
