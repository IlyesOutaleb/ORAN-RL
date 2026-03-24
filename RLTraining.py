import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.evaluation import evaluate_policy

import trafficSteering as ts

def trainPPO(totalTimesteps):
    """
    Trains PPO (Proximal Policy Optimization) agent on the traffic steering evnrionment
    :param totalTimesteps: integer total number of training steps
    :return: trained PPO model
    """
    env = Monitor(ts.TraffciSteeringEnv())

    model = PPO("MlpPolicy", env, verbose=1, learning_rate=3e-4, n_steps=2048, batch_size=64, gamma=0.99,
                tensorboard_log="./logs/ppo")

    print(f"\n Training PPO for {totalTimesteps} timesteps")
    model.learn(total_timesteps=totalTimesteps)
    model.save("ppo_traffic_steeringSum")
    print(" PPO training complete.")

    env.close()
    return model

def evaluateModel(model, numEpisodes=100):
    """
    Evalutes an RL model over a given number of episodes
    :param model: trained model
    :param numEpisodes: interger representing after how much are results reported
    :return: list of episode returns
    """
    env = ts.TraffciSteeringEnv()
    episodeRewards = []

    for episode in range(numEpisodes):
        obs, info = env.reset()
        totalReward = 0
        truncated = False

        while not truncated:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            totalReward += reward

        episodeRewards.append(totalReward)

    env.close()

    return episodeRewards

def printSummary(rewards):
    """
    Prints a summary of episode returns for a given policy
    :param rewards:
    :return:
    """
    print(f"\n PPO")
    print(f" Mean return: {np.mean(rewards):.2f}")
    print(f" Std return: {np.std(rewards):.2f}")
    print(f" Min return: {np.min(rewards):.2f}")
    print(f" Max return: {np.max(rewards):.2f}")


if __name__ == "__main__":

    print("Open RAN TRAFFIC STEERING - RL TRAINING")
    # training
    ppoModel = trainPPO(totalTimesteps=500000)

    # evaluation
    print("\n Evaluating models every 100 episodes")
    ppoRewards = evaluateModel(ppoModel, numEpisodes = 100)

    # final output
    printSummary(ppoRewards)