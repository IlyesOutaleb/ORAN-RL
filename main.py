import numpy as np
import trafficSteering as ts
from stable_baselines3 import PPO, A2C
import matplotlib.pyplot as plt

def runBaselineAgent(policy):
    """
    Runs a demo that would essentially similar to default traffic steering scenario handling in modern networks.
    """
    env = ts.TraffciSteeringEnv()
    obs, info = env.reset(seed=0)

    env.render()

    rewardHistory = []
    hoHistory = []
    stepCount = 0
    #policy = 1
    totalThroughput = 0

    if policy == 0:
        print(f"\n Running baseline policy:Random for 200 steps")
    elif policy == 1:
        print(f"\n Running baseline policy:Closest Distance for 200 steps")
    elif policy == 2:
        model = A2C.load("a2x_traffic_steering.zip")
        print(f"\n Running baseline policy:PPO for 200 steps")

    truncated = False

    while not truncated:
        # Select policy
        if policy == 0:
            action = env.action_space.sample()   # random policy
        elif policy == 1:
            action = ts.greedy_distance_policy(env) # closest distance
            #print(action)
        elif policy == 2:
            action, _ = model.predict(obs, deterministic=True) # deterministic = true does not output prob dist only max
        elif policy == 3:
            print("HI")
            """
            with open("optimalActions", "r") as file:
               for line in file:
                    content += line.strip()
            """
        obs, reward, terminated, truncated, info = env.step(action) # take a step
        #print(f"OBS:{obs}")
        #env.render()

        rewardHistory.append(reward)
        currentStepHo = info["num_handovers"]
        totalThroughput += env.getThroughputPerStep()
        hoHistory.append(currentStepHo)
        stepCount += 1

    env.close()

    print(f"==================== Episode Summary with {ts.getNumGnbs(env)}: GNBs & {ts.getNumUe(env)}: UEs ============================")
    print(f"\n  Steps completed  : {stepCount}")
    print(f"  Total reward     : {sum(rewardHistory):.2f}")
    print(f"  Mean reward/step : {np.mean(rewardHistory):.3f}")
    print(f"  Total handovers  : {sum(hoHistory)}")
    print(f"  Handovers/step   : {sum(hoHistory)/stepCount:.2f}")
    print(f" Total Throughput  : {totalThroughput}")

    steps = range(stepCount)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    fig.suptitle("O-RAN Traffic Steering — Episode Summary", fontsize=13)

    # --- Plot 1: Reward per step ---
    ax1.plot(steps, rewardHistory, color="#1f77b4", linewidth=1.5)
    ax1.axhline(np.mean(rewardHistory), color="orange", linewidth=1,
                linestyle="--", label=f"Mean: {np.mean(rewardHistory):.3f}")
    ax1.set_ylabel("Reward")
    ax1.set_title("Reward per Step")
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)

    # --- Plot 2: Handovers per step ---
    ax2.bar(steps, hoHistory, color="#d62728", width=1.0, label="# Handovers")
    ax2.set_ylabel("# Handovers")
    ax2.set_xlabel("Step")
    ax2.set_title("Handovers per Step")
    ax2.yaxis.set_major_locator(plt.MaxNLocator(integer=True))
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig("temp.png", dpi=150)
    plt.show()

if __name__ == "__main__":
    print("Open RAN TRAFFIC STEERING - GYM ENV")
    runBaselineAgent(2)

    """ To see if delta or sum is better
    env = ts.TraffciSteeringEnv()
    rewards_noop = []
    rewards_ho = []

    for seed in range(20):  # test 20 episodes
        obs, info = env.reset(seed=seed)

        # Step 1: force handover
        action_ho = ts.greedy_distance_policy(env)
        _, r_ho, _, _, _ = env.step(action_ho)
        rewards_ho.append(r_ho)

        # Step 2: NOOP from same state
        obs, info = env.reset(seed=seed)
        action_noop = np.full(env._num_ue, env._num_gnbs)
        _, r_noop, _, _, _ = env.step(action_noop)
        rewards_noop.append(r_noop)

    print(f"Mean HO reward:   {np.mean(rewards_ho):.6f}")
    print(f"Mean NOOP reward: {np.mean(rewards_noop):.6f}")
    print(f"HO better in {sum(h > n for h, n in zip(rewards_ho, rewards_noop))}/20 episodes")
    """
