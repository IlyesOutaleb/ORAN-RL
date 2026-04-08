 *RL for 5G ORAN Traffic Steering* 

This implementation has 12 UEs with 4 gNBs in a grid of 1000 by 1000 meters.

It is OpenAI GYM compliant. There are three components. These can be found visually inside the PythonCodeArchitecture.png

<img width="711" height="554" alt="image" src="https://github.com/user-attachments/assets/76839816-13b2-4aa2-b686-97f8bdda1bec" />


TrafficSteering.py is a simulated 5G network environment. It can be used to RL models for traffic steering purposes.

main.py is used for deployment/testing. It uses the environment and performs actions to then get rewards and evaluate the policy deterministic or stochastic.

RLTraining.py also uses the envrionemnt but for training. If one runs the RLTRaining.py there will be two files. One is the model it self in a zip file. This needs to be put in the PPO.load inside main.py to be tested. 
The second file is a set of logs that can be opened inside tensorboard. 

To do so go inside the logs and PPO agent you would like to view and type:
tensorboard --logdir=.

Tensorboard is GUI provided by stablebaselines3 (SB3) to visualize how well the agent is learning.

Here is the full SB3 documentation for more details:

https://stable-baselines3.readthedocs.io/en/master/
