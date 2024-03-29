# Banana - reinforcement learning DQN implementation
============

Banana is a deep Q learning implementation of [DQN paper](https://storage.googleapis.com/deepmind-media/dqn/DQNNaturePaper.pdf)

---

## Features
- Buffer size tuning
- Batch size tuning
- Gamma factor tuning
- TAU tuning for soft updates
- Update parameter according to the paper

---


## Screenshot

![Banana - solving environment](https://github.com/andreiliphd/banana-dqn/blob/master/pictures/video_gif.gif)

## Loss

![Banana - loss](https://github.com/andreiliphd/banana-dqn/blob/master/pictures/download.png)



---

## Setup
Clone this repo: 
```
git clone https://github.com/andreiliphd/carrot.git
```
Install all the dependencies.

---


## Installation

1. Download the environment from one of the links below.  You need only select the environment that matches your operating system:
    - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux.zip)
    - Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana.app.zip)
    - Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86.zip)
    - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86_64.zip)
    
    (_For Windows users_) Check out [this link](https://support.microsoft.com/en-us/help/827218/how-to-determine-whether-a-computer-is-running-a-32-bit-version-or-64) if you need help with determining if your computer is running a 32-bit version or 64-bit version of the Windows operating system.

    (_For AWS_) If you'd like to train the agent on AWS (and have not [enabled a virtual screen](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md)), then please use [this link](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux_NoVis.zip) to obtain the environment.

2. Place the file in the directory of GitHub repository files. 

## Usage

You will train an agent to collect bananas in a large, square world.  

A reward of +1 is provided for collecting a yellow banana, and a reward of -1 is provided for collecting a blue banana.  Thus, the goal of your agent is to collect as many yellow bananas as possible while avoiding blue bananas.  

The state space has 37 dimensions and contains the agent's velocity, along with ray-based perception of objects around agent's forward direction.  Given this information, the agent has to learn how to best select actions.  Four discrete actions are available, corresponding to:
- **`0`** - move forward.
- **`1`** - move backward.
- **`2`** - turn left.
- **`3`** - turn right.
Environement is considered solved if average of 100 episodes is more than 13.

Running a script is not a problem. Just execute it in sequential order.

---

## License
You can check out the full license in the LICENSE file.

This project is licensed under the terms of the **MIT** license.


