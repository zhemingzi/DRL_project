import gymnasium as gym

from stable_baselines3 import DQN
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy

from gymnasium.wrappers.monitoring import video_recorder
# 加载环境、video记录器、模型
env = gym.make("LunarLander-v2", render_mode="rgb_array")
video = video_recorder.VideoRecorder(env,'D:/py_project/DRL_learning/test_rl/LunarLander/video/'+'ppo_result.mp4')
# env = wrappers.   Moniter(env,'./videos/'+str(time())+'/')
model = PPO.load('D:/py_project/stable_baseline3/PPO_lunar',env=env)
env.reset()

# 7.Enjoy trained agent
vec_env = model.get_env()
obs = vec_env.reset()
for i in range(5000):
    action, _states = model.predict(obs, deterministic=True)
    obs, rewards, dones, info = vec_env.step(action)
    vec_env.render("human")
    video.capture_frame()

video.close()
env.close()