import gymnasium as gym
import time
env=gym.make('MsPacman-ram-v0',render_mode="human") #创建对应的游戏环境
env.seed(1) #可选，设置随机数，以便让过程重现

s=env.reset() #重新设置环境，并得到初始状态
while True: #每个步骤
    time.sleep(0.05)
    env.render() #展示环境
    a=env.action_space.sample() # 智能体随机选择一个动作
    s_,r,done,_,info=env.step(a) #环境返回执行动作a后的下一个状态、奖励值、是否终止以及其他信息
    # print("s_",s_)
    print("r",r)
    # print("done",done)
    # print("info",info)
    if done:
        break
