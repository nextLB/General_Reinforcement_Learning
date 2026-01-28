
#
# # 关于Mujoco环境中的Pusher游戏的动作空间的说明
#
# #       行动      控制分钟       控制最大值        名称(在相应的XML文件中)      关节      类型(单位)
# # 0     肩部翻滚    -2          2                r_shoulder_pan_joint       铰链      扭矩(Nm)
# # 1     肩部关节旋转 -2          2                r_shoulder_lift_joint
# # 2     肩部滚动关节的旋转   -2      2              r_
# # 3     弯曲肘部的铰链接头旋转     -2      2       r_upper_arm_roll_joint
# # 4     前臂滚动的铰链旋转   -2      2            r_elbow_flex_joint
# # 5     弯曲手腕的旋转      -2      2            r_wrist_flex_joint
# # 6 转动手腕的旋转
# #
# #
# # -2
# #
# #
# # 2
# #
# #
# # r_wrist_roll_joint
#
#
#
# import gymnasium as gym
# import numpy as np
# import matplotlib.pyplot as plt
# import time
#
# # 创建Pusher-v5环境
# # 注意：xml_file参数需要替换为实际的MuJoCo XML文件路径
# # 如果没有自定义XML文件，可以省略此参数使用默认环境
# env = gym.make(
#     'Pusher-v4',
#     render_mode='rgb_array',  # 使用rgb_array模式以便获取图像数据
#     # xml_file='path/to/your/model.xml'  # 如果有自定义XML文件，取消注释并指定路径
# )
#
# # 初始化环境
# observation, info = env.reset()
# print(f"Observation shape: {observation.shape}")
# print(f"Action space: {env.action_space}")
#
# # 设置matplotlib实时显示
# plt.ion()  # 开启交互模式
# fig, ax = plt.subplots(figsize=(10, 8))
# img_display = ax.imshow(np.zeros((480, 480, 3), dtype=np.uint8))
# ax.set_title('Pusher-v5 Environment - Random Actions')
# ax.set_xlabel('X pixel')
# ax.set_ylabel('Y pixel')
# ax.grid(False)
#
# # 添加状态信息文本
# status_text = ax.text(10, 30, '', color='white', fontsize=12,
#                       bbox=dict(facecolor='black', alpha=0.7))
#
# # 添加步数计数器
# step_counter = 0
# episode_counter = 1
#
# # 主循环
# max_steps = 500
# try:
#     for step in range(max_steps):
#         # 生成随机动作
#         action = env.action_space.sample()
#
#
#
#         # 执行动作
#         observation, reward, terminated, truncated, info = env.step(action)
#
#         print(action, observation)
#
#         # 渲染环境并获取图像
#         frame = env.render()
#
#         # 更新显示
#         if frame is not None:
#             img_display.set_data(frame)
#
#             # 更新状态信息
#             status_info = f'Episode: {episode_counter}\nStep: {step_counter}\nReward: {reward:.3f}\nTerminated: {terminated}'
#             status_text.set_text(status_info)
#
#             # 重绘图
#             fig.canvas.draw()
#             fig.canvas.flush_events()
#
#         # 更新计数器
#         step_counter += 1
#
#         # 检查是否结束本回合
#         if terminated or truncated:
#             print(f"Episode {episode_counter} finished after {step_counter} steps")
#             observation, info = env.reset()
#             step_counter = 0
#             episode_counter += 1
#             time.sleep(1)  # 稍微暂停一下以便观察
#
#         # 控制帧率
#         time.sleep(0.05)
#
# except KeyboardInterrupt:
#     print("Visualization interrupted by user")
#
# finally:
#     # 清理资源
#     plt.ioff()
#     plt.close()
#     env.close()
#     print("Environment closed successfully")
#
# # 如果没有运行，显示最终图像
# if not plt.isinteractive():
#     plt.show()



import gymnasium as gym

class PusherV4Env:
    def __init__(self):
        self.environment = gym.make(
            'Pusher-v4',
            render_mode='rgb_array',  # 使用rgb_array模式以便获取图像数据
            # xml_file='path/to/your/model.xml'  # 如果有自定义XML文件，取消注释并指定路径
        )

    # 重置环境
    def reset(self):
        # info是空字典，这是Pusher-v4环境的默认设计 ------ 该环境在reset()和step()中暂未返回额外的辅助信息，仅返回字典作为占位符
        # observation 观测值   observation是一个长度为23的一维数组（形状：(23,)）,每个数值对应机器人(机械手)和目标物体的物理状态，具体含义按索引拆分如下:
        # 0-12  关节状态    机械手各关节的位置、速度、角度（包括肩关节、肘关节、腕关节等），其中前6个为位置/角度，后7个为角速度/速度
        # 13-15 末端执行器（抓手）位置     抓手在三维空间中的坐标（x, y, z）
        # 16-18 被推动的小球位置    小球在三维空间中的坐标（x, y, z）
        # 19-21 目标位置    小球需要被推到的目标点三维坐标（x, y, z）
        # 22    辅助计算值   末端执行器与小球的距离误差（或关节力矩补偿值，用于简化奖励计算）
        # 补充说明：所有坐标的单位均为米（m），角度单位为弧度（rad），速度单位为米/秒（m/s）或弧度/秒（rad/s）
        # 数值的正负代表方向（如x轴正方向/负方向、角度顺时针/逆时针）
        observation, info = self.environment.reset()

    # 执行动作
    def step(self, action):
        # reward (奖励值)  标量值     表示这一步动作的好坏
        # reward = reward_dist + reward_ctrl + reward_near
        #       a、目标距离奖励(目标距离奖励)
        #           含义:衡量小球与目标位置的距离，鼓励将小球推到目标点
        #               reward_dist = -reward_dist_weight * distance_ball_to_target
        #               # 小球与目标点之间的欧几里得距离
        #               distance_ball_to_target = sqrt((ball_x - target_x)**2 + (ball_y - target_y)**2 + (ball_z - target_z)**2)
        #           特点：1、总是负值（因为是距离的负值）2、距离越近，值越大（负得越少）3、距离越远，值越小（负得越多）
        #       b、reward_ctrl(控制代价惩罚)
        #           含义：惩罚过大的控制动作，鼓励使用更平稳、更高效的控制策略
        #               reward_ctrl = -reward_control_weight * sum(action**2)
        #               其中：action = 7维动作向量;sum(action**2)=动作向量的平方和;reward_control_weight = 控制代价权重
        #           特点：总是负值（因为是动作大小的负值）;动作幅度越大，惩罚越大;鼓励使用最小必要力矩完成任务
        #       c、reward_near(指尖接近奖励)
        #           含义：鼓励机械手末端(指尖)靠近小球，便于推动小球
        #               reward_near = -reward_near_weight * distance_fingertip_to_ball
        #           其中：distance_fingertip_to_all = 指尖与小球之间的欧几里得距离
        #               distance = sqrt((fingertip_x - ball_x)**2 + (fingertip_y - ball_y)**2 + (fingertip_z - ball_z)**2)
        #               reward_near_weight = 接近奖励权重
        #           特点：总是负值;指尖离小球越近，值越大(负得越少);帮助机械手学习如何接近并推动小球

        # terminated(终止标志)
        # 布尔值，表示episode是否正常结束
        #       触发terminated = True的条件:1、任务成功：小球到达目标区域2、任务失败：机械手超出安全范围;小球掉出工作空间;3、其他终止条件：特定任务条件满足;不可恢复的错误状态

        # truncated(截断标志)
        # 布尔值，表示episode是否被外部截断
        #       触发truncated = True的条件：步数限制：达到最大步数;时间限制：超过最大时间;其他外部限制：安全监控系统干预;用户手动终端;资源限制
        observation, reward, terminated, truncated, info = self.environment.step(action)


    def get_action_space(self):
        # Box(-2.0, 2.0, (7,), float32)
        # 现状 (7,) - 7维连续动作空间
        # 范围: 每个动作值在[-2.0, 2.0]之间
        # 类型：float32浮点数

        # 7个动作分量的具体含义
        # 索引0-2 肩部关节
        #   0: 肩部俯仰（pitch）
        #   1: 肩部横摆（yaw）
        #   2: 肩部寻转（roll）
        # 索引3-5 肘部关节
        #   3: 肘部俯仰（pitch）
        #   4: 肘部偏航（yaw）
        #   5: 肘部旋转（roll）
        # 索引6 腕部关节
        #   6: 腕部旋转（roll）或末端执行器旋转

        # 动作控制方式
        #       力矩控制：Pusher-v4环境通常使用力矩控制模式
        #       归一化范围：[-2, 2]是归一化的力矩值，实际执行时会根据关节的最大力矩进行缩放
        #       连续控制：每个时间步都可以平滑地改变关节力矩
        return self.environment.action_space





if __name__ == '__main__':
    PusherV4EnvInstance = PusherV4Env()
    PusherV4EnvInstance.reset()
    print(PusherV4EnvInstance.get_action_space())
