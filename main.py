import numpy as np
import trafficSteering as ts

def runBaselineAgent():
    """
    Runs a demo that would essentially similar to default traffic steering scenario handling in modern networks.
    """
    env = ts.TraffciSteeringEnv()
    obs, info = env.reset(seed=0)

    #env.render()

    rewardHistory = []
    hoHistory = []
    stepCount = 0
    policy = 1

    if policy == 0:
        print(f"\n Running baseline policy:Random for 200 steps")
    elif policy == 1:
        print(f"\n Running baseline policy:Closest Distance for 200 steps")

    truncated = False

    while not truncated:
        # Select policy

        if policy == 0:
            action = env.action_space.sample()   # random policy
        elif policy == 1:
            action = ts.greedy_distance_policy(env) # closest distance

        obs, reward, terminated, truncated, info = env.step(action) # take a step
        #env.render()

        rewardHistory.append(reward)
        currentStepHo = info["num_handovers"]
        hoHistory.append(currentStepHo)
        stepCount += 1

    env.close()

    print(f"==================== Episode Summary with {ts.getNumGnbs(env)}: GNBs & {ts.getNumUe(env)}: UEs ============================")
    print(f"\n  Steps completed  : {stepCount}")
    print(f"  Total reward     : {sum(rewardHistory):.2f}")
    print(f"  Mean reward/step : {np.mean(rewardHistory):.3f}")
    print(f"  Total handovers  : {sum(hoHistory)}")
    print(f"  Handovers/step   : {sum(hoHistory)/stepCount:.2f}")


if __name__ == "__main__":
    print("Open RAN TRAFFIC STEERING - GYM ENV")
    runBaselineAgent()