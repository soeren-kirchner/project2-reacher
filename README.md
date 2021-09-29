## Environment

In this environment, a double-jointed arm can move to target locations. A reward of +0.1 is provided for each step that the agent's hand is in the goal location. Thus, the goal of your agent is to maintain its position at the target location for as many time steps as possible.

The observation space consists of 33 variables corresponding to position, rotation, velocity, and angular velocities of the arm. Each action is a vector with four numbers, corresponding to torque applicable to two joints. Every entry in the action vector should be a number between -1 and 1.

This version contains 20 identical agents, each with its own copy of the environment. 

The environment is considered solved, when the average (over 100 episodes) of the average scores is at least +30.

## Installation of the Unity Environment
* Clone or download the Deep Reinforcement Repocitory provided by udacity from GitHub. Navigate to the python directory and install the requirements. Please read the [detailed installation instructions](https://github.com/udacity/deep-reinforcement-learning/blob/master/README.md). (Hint: Create a python environment to install the requirements into it.)

```
git clone https://github.com/udacity/deep-reinforcement-learning.git
cd deep-reinforcement-learning/python
pip install .
```
* Download the Environment from one of the following sources depending on the requirement of your system.
    * [Mac OSX](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher.app.zip)
    * [Linux](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Linux.zip)
    * [Windows (32-bit)](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Windows_x86.zip)
    * [Windows (64-bit)](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Windows_x86_64.zip)
* Unzip the environment to the root path of this project.

## Run the Project
Open Continuous_Control.ipynb and run the cells!

## About this Solution
This solution tries and compares three different methods for adding noise for exploration. Ornstein-Uhlenbeck process action noise, normal noise and adaptive parameter noise.
More details you will find in this [article](https://medium.com/@soeren-kirchner/deep-deterministic-policy-gradient-ddpg-with-and-without-ornstein-uhlenbeck-process-e6d272adfc3).