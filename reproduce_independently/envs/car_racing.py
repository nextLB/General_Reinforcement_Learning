"""
    游戏 Car Racing
"""


import gymnasium as gym

class CarRacingEnv:

    def __init__(self):
        self.version = 'V1.0'
        self.env = None

    def make_env(self):
        # lap_complete_percent=0.95 完成一圈的百分比
        #       赛车可能因为轻微偏离赛道而无法100%准确经过所有点
        #       95%是一个合理的容差，确保AI能获得正奖励
        # domain_randomize=False 域随机化
        #       域随机化是提升模型泛化能力的技术
        # continuous=False 动作连续性
        #       这个参数控制动作空间是连续的还是离散的
        self.env = gym.make(
            "CarRacing-v3",
            render_mode="rgb_array",
            lap_complete_percent=0.95,
            domain_randomize=False,
            continuous=False)

    def reset_env(self):
        self.env.reset()

    def get_action_space(self):
        return self.env.action_space

    def step(self, action):
        state, reward, terminated, truncated, info = self.env.step(action)

        return state, reward, terminated, truncated, info

    def experience_buffer(self):
        pass




if __name__ == '__main__':
    CarRacingEnvInstance = CarRacingEnv()






# # lap_complete_percent=0.95 完成一圈的百分比
# #       赛车可能因为轻微偏离赛道而无法100%准确经过所有点
# #       95%是一个合理的容差，确保AI能获得正奖励
# # domain_randomize=False 域随机化
# #       域随机化是提升模型泛化能力的技术
# # continuous=False 动作连续性
# #       这个参数控制动作空间是连续的还是离散的
# env = gym.make(
#     "CarRacing-v3",
#     render_mode="rgb_array",
#     lap_complete_percent=0.95,
#     domain_randomize=False,
#     continuous=False)
#
# print(env.action_space)
#
# # 开启交互模式
# plt.ion()   # 重要：开启交互模式，避免阻塞
#
# # 创建图形窗口
# fig, ax = plt.subplots(figsize=(6, 6))
# imgDisplay = ax.imshow(np.zeros((96, 96, 3), dtype=np.uint8))
# ax.axis('off')  # 隐藏坐标轴
# plt.title("Car Racing - Real-time Visualization")
#
#
# env.reset()
# for i in range(100):
#     action = i % 5
#     # 执行一步，获取图像
#     obs, reward, terminated, truncated, info = env.step(action)
#
#     # 更新图像显示
#     imgDisplay.set_data(obs)
#
#     # 刷新显示
#     fig.canvas.draw()
#     fig.canvas.flush_events()
#
#     # 短暂暂停，控制频率（单位:秒）
#     plt.pause(0.01)     # 大约100 FPS
#
#
#     print(terminated)
#     print(truncated)
#
#
# # 关闭环境
# env.close()
#
# # 关闭交互模式，显示最终窗口
# plt.ioff()
# plt.show()



