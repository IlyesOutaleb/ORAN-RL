import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO, A2C
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.evaluation import evaluate_policy

import trafficSteering as ts


def trainA2C(totalTimesteps=500000):
    """
    Trains A2C (Actor to Critic) agent on the traffic steering evnrionment
    :param totalTimesteps: integer total number of training steps
    :return: trained A2C model
    """
    env = Monitor(ts.TraffciSteeringEnv())

    model = A2C("MlpPolicy", env, verbose=1, learning_rate=1e-3, gamma=0.99, n_steps=2000,
                tensorboard_log="./logs/a2c")

    print(f"\n Training A2C for {totalTimesteps} timesteps")
    model.learn(total_timesteps=totalTimesteps)
    model.save("a2x_traffic_steering")
    print(" A2C training complete.")

    env.close()
    return model


def trainPPO(totalTimesteps):
    """
    Trains PPO (Proximal Policy Optimization) agent on the traffic steering evnrionment
    :param totalTimesteps: integer total number of training steps
    :return: trained PPO model
    """
    env = Monitor(ts.TraffciSteeringEnv())

    model = PPO("MlpPolicy", env, verbose=1, learning_rate=1e-3, n_steps=1000, gamma=0.99,
                tensorboard_log="./logs/ppo")

    print(f"\n Training PPO for {totalTimesteps} timesteps")
    model.learn(total_timesteps=totalTimesteps)
    model.save("ppo_traffic_steeringDiffStdMoreHO0.0025500ksteps1kbatchSize200gam0.99lr1e-3_PPO39")
    print(" PPO training complete.")

    env.close()
    return model

def trainDQN(totalTimesteps):
    """
    Trains PPO (Proximal Policy Optimization) agent on the traffic steering evnrionment
    :param totalTimesteps: integer total number of training steps
    :return: trained PPO model
    """
    env = Monitor(ts.TraffciSteeringEnv())

    model = DQN("MlpPolicy", env, verbose=1, learning_rate=1e-3, n_steps=1000, batch_size=200, gamma=0.99,
                tensorboard_log="./logs/DQN")

    print(f"\n Training PPO for {totalTimesteps} timesteps")
    model.learn(total_timesteps=totalTimesteps)
    model.save("DQN")
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
    #ppoModel = trainPPO(totalTimesteps=500000)
    a2cModel = trainA2C(totalTimesteps=500000)
    #dqnmodel = trainDQN(totalTimesteps=500000)
    # evaluation
    print("\n Evaluating models every 100 episodes")
    #ppoRewards = evaluateModel(ppoModel, numEpisodes = 100)

    a2cRewards = evaluateModel(a2cModel, numEpisodes=100)
    #dqnRewards = evaluateModel(dqnmodel, numEpisodes=100)
    # final output
    #printSummary(ppoRewards)
    printSummary(a2cRewards)
    #printSummary(dqnRewards)