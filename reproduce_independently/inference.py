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
#     'Pusher-v5',
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
#         # 执行动作
#         observation, reward, terminated, truncated, info = env.step(action)
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
#
#



# 关于Mujoco环境中的Pusher游戏的动作空间的说明

#       行动      控制分钟       控制最大值        名称(在相应的XML文件中)      关节      类型(单位)
# 0     肩部翻滚    -2          2                r_shoulder_pan_joint       铰链      扭矩(Nm)
# 1     肩部关节旋转 -2          2                r_shoulder_lift_joint
# 2     肩部滚动关节的旋转   -2      2              r_
# 3     弯曲肘部的铰链接头旋转     -2      2       r_upper_arm_roll_joint
# 4     前臂滚动的铰链旋转   -2      2            r_elbow_flex_joint
# 5     弯曲手腕的旋转      -2      2            r_wrist_flex_joint
# 6 转动手腕的旋转
#
#
# -2
#
#
# 2
#
#
# r_wrist_roll_joint



import sys
sys.path.append('/home/next_lb/桌面/next/General_Reinforcement_Learning')
from reproduce_independently.train import Config
from reproduce_independently.envs.car_racing import CarRacingEnv
from reproduce_independently.network.DQN import DQNNetWork
import torch
import matplotlib.pyplot as plt



def CarRacingDQNInference():
    # 初始化环境类
    CarRacingEnvInstance = CarRacingEnv(Config.frameStacks)
    state, info = CarRacingEnvInstance.reset()
    # 灰度图通道形状
    grayImageShape = (state.shape[0], state.shape[1], Config.frameStacks)
    actionSpaceNumber = CarRacingEnvInstance.get_action_space()
    # 初始化网络模型
    DQNNetWorkInstance = DQNNetWork(grayImageShape, actionSpaceNumber)
    checkpointPath = '/home/next_lb/models/DQN_models/checkpoint_episode.pth'
    checkpoint = torch.load(checkpointPath, map_location=Config.device)

    # 加载策略网络权重
    DQNNetWorkInstance.load_state_dict(checkpoint['policy_network_state_dict'])
    DQNNetWorkInstance.to(Config.device)
    DQNNetWorkInstance.eval()

    # 初始化画布
    plt.ion()
    fig, ax = plt.subplots(figsize=(8, 6))

    done = False
    while not done:
        frameSates = CarRacingEnvInstance.stack_frames()
        grayList = []
        for i in range(len(frameSates)):
            grayList.append(CarRacingEnvInstance.preprocess_state_to_gray(frameSates[i]))
        grayStateTensors = torch.FloatTensor(grayList).to(Config.device).unsqueeze(0)
        # 获取其Q值
        qValues = DQNNetWorkInstance(grayStateTensors)
        # 选择最大Q值对应的动作
        action = qValues.max(1)[1].item()
        # 作用于环境并获取返回结果
        nextState, reward, terminated, truncated, info = CarRacingEnvInstance.step(action)

        # 判断本回合是否结束
        done = terminated or truncated

        # 检查是否有键盘事件
        if plt.get_fignums():  # 检查图形是否还存在
            try:
                # 检查是否按下了Q键
                if plt.waitforbuttonpress(0.001):
                    key = plt.gcf().canvas.key_press_event.key
                    if key == 'q':
                        print("Q pressed: Stopping episode")
                        break
            except:
                pass  # 忽略事件处理中的异常

        # 可视化图像
        im = ax.imshow(nextState)
        ax.set_title('State')
        ax.set_xlabel('Width')
        ax.set_ylabel('Height')
        ax.axis('off')
        plt.draw()
        plt.pause(0.00001)
        print(f'action: {action}')

    # 关闭图形窗口
    plt.close(fig)
    plt.ioff()

    print('Over!')


def main():
    CarRacingDQNInference()


if __name__ == '__main__':
    main()



