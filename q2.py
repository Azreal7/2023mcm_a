import random
import time
import math
import function as f
import numpy as np
import constance as c


class sa_tsp:
    # 接受的输入参数：起始温度start_temp, 结束温度end_temp, 内循环次数num
    def __init__(self, s_t, e_t, num):
        self.start_temp = s_t
        self.end_temp = e_t
        self.num = num

    # 获取当前状态的功率密度
    def get_power_density(self, state):
        power, _ = self.get_power(state)
        return power / (state[0] * state[1] * state[3])

    # 获取当前状态的功率值和摆放的圈数
    def get_power(self, state):
        nums = f.get_nums(state[0], state[1], state[2], state[3], state[4])
        S = state[0]*state[1]
        x, y = f.get_xy(nums, state[4])
        loss, t = f.new_get_shadow_loss(len(nums), c.cos_delta, c.cos_alpha, state[4], state[1], 84, state[2])
        d_HR, eta_at = f.new_get_d_HR(80, state[2], len(nums), state[4])
        effi_trunc = f.get_trunc_effi(8, 7, state[0], state[1], len(nums), d_HR)
        effi_cos = f.get_effi_cos(t, len(nums))
        light_effi = f.get_light_effi(loss, effi_cos, eta_at, effi_trunc, len(nums))
        E_field = f.new_get_heat_W(light_effi, S, c.DNI, len(nums), loss, nums)
        return np.mean(E_field), len(nums)


    # 获取下一个状态
    # 如果符合约束条件，则返回新状态，否则放回原状态
    def get_next_state(self, state):
        next_state = state.copy()
        select = random.randint(0, 4)  # 生成0到4之间的随机整数
        flag = random.choice([-1, 1])
        vary_list = [4, 4, 2, 100, 10]
        vary_num = random.random()
        next_state[select] = next_state[select] + flag * vary_list[select] * vary_num
        next_state[3] = int(next_state[3])      # 数目只能是整数
        power, num = self.get_power(next_state)
        if self.is_valid(next_state, power, num):
            return next_state
        else:
            return state

    # 判断是否满足约束条件
    def is_valid(self, state, power, num):
        return 2 <= state[0] <= 8 and 2 <= state[1] <= 8 and 2 <= state[2] <= 6 and state[1] <= 2 * state[2] \
               and state[0] >= state[1] and state[4] >= state[0] + 5 and power >= 60000 and 100 + (num-1) * state[4] <= 350

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

                # print("state:", state, "pd:", new_power_density)

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

    initial_state = [7, 7, 4, 2300, 13]
    initial_state = [7, 5.145833647521261, 4, 2360, 12.271304748613044]

    # 模拟退火
    test = sa_tsp(10, 0.1, 100)
    best_power_density, best_state = test.simulated_annealing(initial_state)

    end_time = time.time()

    print(best_power_density)
    print(best_state)
    print("time:", end_time-start_time, "s")



