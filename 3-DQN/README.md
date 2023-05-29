# Deep Q-Learning - Lunar Lander

In this assignment, you will train an agent to land a lunar lander safely on a landing pad on the surface of the moon.

Reference: https://www.gymlibrary.dev/environments/box2d/lunar_lander/

# Outline
- [ 1 - Import Packages <img align="Right" src="./images/lunar_lander.gif" width = 60% >](#1)
- [ 2 - Hyperparameters](#2)
- [ 3 - The Lunar Lander Environment](#3)
  - [ 3.1 Action Space](#3.1)
  - [ 3.2 Observation Space](#3.2)
  - [ 3.3 Rewards](#3.3)
  - [ 3.4 Episode Termination](#3.4)
- [ 4 - Load the Environment](#4)
- [ 5 - Interacting with the Gym Environment](#5)
    - [ 5.1 Exploring the Environment's Dynamics](#5.1)
- [ 6 - Deep Q-Learning](#6)
  - [ 6.1 Target Network](#6.1)
    - [ Exercise 1](#ex01)
  - [ 6.2 Experience Replay](#6.2)
- [ 7 - Deep Q-Learning Algorithm with Experience Replay](#7)
  - [ Exercise 2](#ex02)
- [ 8 - Update the Network Weights](#8)
- [ 9 - Train the Agent](#9)
- [ 10 - See the Trained Agent In Action](#10)
- [ 11 - Congratulations!](#11)
- [ 12 - References](#12)

<a name="1"></a>
## 1 - Import Packages

We'll make use of the following packages:
- `numpy` is a package for scientific computing in python.
- `deque` will be our data structure for our memory buffer.
- `namedtuple` will be used to store the experience tuples.
- The `gym` toolkit is a collection of environments that can be used to test reinforcement learning algorithms.
- `PIL.Image` and `pyvirtualdisplay` are needed to render the Lunar Lander environment.
- We will use several modules from the `torch.nn` framework for building deep learning models.
- `utils` is a module that contains helper functions for this assignment. You do not need to modify the code in this file.

Run the cell below to import all the necessary packages.


```python
!apt-get install -y x11-utils xvfb python-opengl
```

    Reading package lists... Done
    Building dependency tree... Done
    Reading state information... Done
    E: Unable to locate package python-opengl



```python
!pip install pillow
!pip install imageio
!pip install matplotlib
!pip install pandas
!pip install statsmodels
```

    Requirement already satisfied: pillow in /opt/conda/lib/python3.10/site-packages (9.5.0)
    [33mWARNING: Running pip as the 'root' user can result in broken permissions and conflicting behaviour with the system package manager. It is recommended to use a virtual environment instead: https://pip.pypa.io/warnings/venv[0m[33m
    [0mRequirement already satisfied: imageio in /opt/conda/lib/python3.10/site-packages (2.28.1)
    Requirement already satisfied: numpy in /opt/conda/lib/python3.10/site-packages (from imageio) (1.23.5)
    Requirement already satisfied: pillow>=8.3.2 in /opt/conda/lib/python3.10/site-packages (from imageio) (9.5.0)
    [33mWARNING: Running pip as the 'root' user can result in broken permissions and conflicting behaviour with the system package manager. It is recommended to use a virtual environment instead: https://pip.pypa.io/warnings/venv[0m[33m
    [0mRequirement already satisfied: matplotlib in /opt/conda/lib/python3.10/site-packages (3.6.3)
    Requirement already satisfied: contourpy>=1.0.1 in /opt/conda/lib/python3.10/site-packages (from matplotlib) (1.0.7)
    Requirement already satisfied: cycler>=0.10 in /opt/conda/lib/python3.10/site-packages (from matplotlib) (0.11.0)
    Requirement already satisfied: fonttools>=4.22.0 in /opt/conda/lib/python3.10/site-packages (from matplotlib) (4.39.3)
    Requirement already satisfied: kiwisolver>=1.0.1 in /opt/conda/lib/python3.10/site-packages (from matplotlib) (1.4.4)
    Requirement already satisfied: numpy>=1.19 in /opt/conda/lib/python3.10/site-packages (from matplotlib) (1.23.5)
    Requirement already satisfied: packaging>=20.0 in /opt/conda/lib/python3.10/site-packages (from matplotlib) (21.3)
    Requirement already satisfied: pillow>=6.2.0 in /opt/conda/lib/python3.10/site-packages (from matplotlib) (9.5.0)
    Requirement already satisfied: pyparsing>=2.2.1 in /opt/conda/lib/python3.10/site-packages (from matplotlib) (3.0.9)
    Requirement already satisfied: python-dateutil>=2.7 in /opt/conda/lib/python3.10/site-packages (from matplotlib) (2.8.2)
    Requirement already satisfied: six>=1.5 in /opt/conda/lib/python3.10/site-packages (from python-dateutil>=2.7->matplotlib) (1.16.0)
    [33mWARNING: Running pip as the 'root' user can result in broken permissions and conflicting behaviour with the system package manager. It is recommended to use a virtual environment instead: https://pip.pypa.io/warnings/venv[0m[33m
    [0mRequirement already satisfied: pandas in /opt/conda/lib/python3.10/site-packages (1.5.3)
    Requirement already satisfied: python-dateutil>=2.8.1 in /opt/conda/lib/python3.10/site-packages (from pandas) (2.8.2)
    Requirement already satisfied: pytz>=2020.1 in /opt/conda/lib/python3.10/site-packages (from pandas) (2023.3)
    Requirement already satisfied: numpy>=1.21.0 in /opt/conda/lib/python3.10/site-packages (from pandas) (1.23.5)
    Requirement already satisfied: six>=1.5 in /opt/conda/lib/python3.10/site-packages (from python-dateutil>=2.8.1->pandas) (1.16.0)
    [33mWARNING: Running pip as the 'root' user can result in broken permissions and conflicting behaviour with the system package manager. It is recommended to use a virtual environment instead: https://pip.pypa.io/warnings/venv[0m[33m
    [0mRequirement already satisfied: statsmodels in /opt/conda/lib/python3.10/site-packages (0.13.5)
    Requirement already satisfied: pandas>=0.25 in /opt/conda/lib/python3.10/site-packages (from statsmodels) (1.5.3)
    Requirement already satisfied: patsy>=0.5.2 in /opt/conda/lib/python3.10/site-packages (from statsmodels) (0.5.3)
    Requirement already satisfied: packaging>=21.3 in /opt/conda/lib/python3.10/site-packages (from statsmodels) (21.3)
    Requirement already satisfied: scipy>=1.3 in /opt/conda/lib/python3.10/site-packages (from statsmodels) (1.10.1)
    Requirement already satisfied: numpy>=1.17 in /opt/conda/lib/python3.10/site-packages (from statsmodels) (1.23.5)
    Requirement already satisfied: pyparsing!=3.0.5,>=2.0.2 in /opt/conda/lib/python3.10/site-packages (from packaging>=21.3->statsmodels) (3.0.9)
    Requirement already satisfied: python-dateutil>=2.8.1 in /opt/conda/lib/python3.10/site-packages (from pandas>=0.25->statsmodels) (2.8.2)
    Requirement already satisfied: pytz>=2020.1 in /opt/conda/lib/python3.10/site-packages (from pandas>=0.25->statsmodels) (2023.3)
    Requirement already satisfied: six in /opt/conda/lib/python3.10/site-packages (from patsy>=0.5.2->statsmodels) (1.16.0)
    [33mWARNING: Running pip as the 'root' user can result in broken permissions and conflicting behaviour with the system package manager. It is recommended to use a virtual environment instead: https://pip.pypa.io/warnings/venv[0m[33m
    [0m


```python
!pip install swig
!pip install imageio[ffmpeg] 
!pip install gym pyvirtualdisplay pyglet 
!pip install gym[box2d]
```

    Collecting swig
      Downloading swig-4.1.1-py2.py3-none-manylinux_2_5_x86_64.manylinux1_x86_64.whl (1.8 MB)
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m1.8/1.8 MB[0m [31m30.6 MB/s[0m eta [36m0:00:00[0ma [36m0:00:01[0m
    [?25hInstalling collected packages: swig
    Successfully installed swig-4.1.1
    [33mWARNING: Running pip as the 'root' user can result in broken permissions and conflicting behaviour with the system package manager. It is recommended to use a virtual environment instead: https://pip.pypa.io/warnings/venv[0m[33m
    [0mRequirement already satisfied: imageio[ffmpeg] in /opt/conda/lib/python3.10/site-packages (2.28.1)
    Requirement already satisfied: numpy in /opt/conda/lib/python3.10/site-packages (from imageio[ffmpeg]) (1.23.5)
    Requirement already satisfied: pillow>=8.3.2 in /opt/conda/lib/python3.10/site-packages (from imageio[ffmpeg]) (9.5.0)
    Collecting imageio-ffmpeg (from imageio[ffmpeg])
      Downloading imageio_ffmpeg-0.4.8-py3-none-manylinux2010_x86_64.whl (26.9 MB)
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m26.9/26.9 MB[0m [31m48.4 MB/s[0m eta [36m0:00:00[0m00:01[0m00:01[0m
    [?25hRequirement already satisfied: psutil in /opt/conda/lib/python3.10/site-packages (from imageio[ffmpeg]) (5.9.3)
    Installing collected packages: imageio-ffmpeg
    Successfully installed imageio-ffmpeg-0.4.8
    [33mWARNING: Running pip as the 'root' user can result in broken permissions and conflicting behaviour with the system package manager. It is recommended to use a virtual environment instead: https://pip.pypa.io/warnings/venv[0m[33m
    [0mRequirement already satisfied: gym in /opt/conda/lib/python3.10/site-packages (0.26.2)
    Collecting pyvirtualdisplay
      Downloading PyVirtualDisplay-3.0-py3-none-any.whl (15 kB)
    Collecting pyglet
      Downloading pyglet-2.0.7-py3-none-any.whl (841 kB)
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m841.0/841.0 kB[0m [31m19.7 MB/s[0m eta [36m0:00:00[0m00:01[0m
    [?25hRequirement already satisfied: numpy>=1.18.0 in /opt/conda/lib/python3.10/site-packages (from gym) (1.23.5)
    Requirement already satisfied: cloudpickle>=1.2.0 in /opt/conda/lib/python3.10/site-packages (from gym) (2.2.1)
    Requirement already satisfied: gym-notices>=0.0.4 in /opt/conda/lib/python3.10/site-packages (from gym) (0.0.8)
    Installing collected packages: pyvirtualdisplay, pyglet
    Successfully installed pyglet-2.0.7 pyvirtualdisplay-3.0
    [33mWARNING: Running pip as the 'root' user can result in broken permissions and conflicting behaviour with the system package manager. It is recommended to use a virtual environment instead: https://pip.pypa.io/warnings/venv[0m[33m
    [0mRequirement already satisfied: gym[box2d] in /opt/conda/lib/python3.10/site-packages (0.26.2)
    Requirement already satisfied: numpy>=1.18.0 in /opt/conda/lib/python3.10/site-packages (from gym[box2d]) (1.23.5)
    Requirement already satisfied: cloudpickle>=1.2.0 in /opt/conda/lib/python3.10/site-packages (from gym[box2d]) (2.2.1)
    Requirement already satisfied: gym-notices>=0.0.4 in /opt/conda/lib/python3.10/site-packages (from gym[box2d]) (0.0.8)
    Collecting box2d-py==2.3.5 (from gym[box2d])
      Downloading box2d-py-2.3.5.tar.gz (374 kB)
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m374.4/374.4 kB[0m [31m11.5 MB/s[0m eta [36m0:00:00[0m
    [?25h  Preparing metadata (setup.py) ... [?25ldone
    [?25hCollecting pygame==2.1.0 (from gym[box2d])
      Downloading pygame-2.1.0-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (18.3 MB)
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m18.3/18.3 MB[0m [31m54.2 MB/s[0m eta [36m0:00:00[0m00:01[0m00:01[0m
    [?25hRequirement already satisfied: swig==4.* in /opt/conda/lib/python3.10/site-packages (from gym[box2d]) (4.1.1)
    Building wheels for collected packages: box2d-py
      Building wheel for box2d-py (setup.py) ... [?25ldone
    [?25h  Created wheel for box2d-py: filename=box2d_py-2.3.5-cp310-cp310-linux_x86_64.whl size=495297 sha256=7d15bd91da1f1b3c3325a224882cd076affce9f2463ddf06326f7343c1a31447
      Stored in directory: /root/.cache/pip/wheels/db/8f/6a/eaaadf056fba10a98d986f6dce954e6201ba3126926fc5ad9e
    Successfully built box2d-py
    Installing collected packages: box2d-py, pygame
    Successfully installed box2d-py-2.3.5 pygame-2.1.0
    [33mWARNING: Running pip as the 'root' user can result in broken permissions and conflicting behaviour with the system package manager. It is recommended to use a virtual environment instead: https://pip.pypa.io/warnings/venv[0m[33m
    [0m


```python
import time
from collections import deque, namedtuple

import gym
import numpy as np
import PIL.Image

from pyvirtualdisplay import Display
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import base64
import random
from itertools import zip_longest

import imageio
import IPython
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import torch
from statsmodels.iolib.table import SimpleTable

SEED = 0  # Seed for the pseudo-random number generator.
MINIBATCH_SIZE = 64  # Mini-batch size.
TAU = 1e-3  # Soft update parameter.
E_DECAY = 0.995  # ε-decay rate for the ε-greedy policy.
E_MIN = 0.01  # Minimum ε value for the ε-greedy policy.

random.seed(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def check_update_conditions(t, num_steps_upd, memory_buffer):
    """
    Determines if the conditions are met to perform a learning update.

    Checks if the current time step t is a multiple of num_steps_upd and if the
    memory_buffer has enough experience tuples to fill a mini-batch (for example, if the
    mini-batch size is 64, then the memory buffer should have more than 64 experience
    tuples in order to perform a learning update).
    
    Args:
        t (int):
            The current time step.
        num_steps_upd (int):
            The number of time steps used to determine how often to perform a learning
            update. A learning update is only performed every num_steps_upd time steps.
        memory_buffer (deque):
            A deque containing experiences. The experiences are stored in the memory
            buffer as namedtuples: namedtuple("Experience", field_names=["state",
            "action", "reward", "next_state", "done"]).

    Returns:
       A boolean that will be True if conditions are met and False otherwise. 
    """

    if (t + 1) % num_steps_upd == 0 and len(memory_buffer) > MINIBATCH_SIZE:
        return True
    else:
        return False


def get_new_eps(epsilon):
    """
    Updates the epsilon value for the ε-greedy policy.
    
    Gradually decreases the value of epsilon towards a minimum value (E_MIN) using the
    given ε-decay rate (E_DECAY).

    Args:
        epsilon (float):
            The current value of epsilon.

    Returns:
       A float with the updated value of epsilon.
    """

    return max(E_MIN, E_DECAY * epsilon)



def plot_history(point_history, **kwargs):
    """
    Plots the total number of points received by the agent after each episode together
    with the moving average (rolling mean). 

    Args:
        point_history (list):
            A list containing the total number of points the agent received after each
            episode.
        **kwargs: optional
            window_size (int):
                Size of the window used to calculate the moving average (rolling mean).
                This integer determines the fixed number of data points used for each
                window. The default window size is set to 10% of the total number of
                data points in point_history, i.e. if point_history has 200 data points
                the default window size will be 20.
            lower_limit (int):
                The lower limit of the x-axis in data coordinates. Default value is 0.
            upper_limit (int):
                The upper limit of the x-axis in data coordinates. Default value is
                len(point_history).
            plot_rolling_mean_only (bool):
                If True, only plots the moving average (rolling mean) without the point
                history. Default value is False.
            plot_data_only (bool):
                If True, only plots the point history without the moving average.
                Default value is False.
    """

    lower_limit = 0
    upper_limit = len(point_history)

    window_size = (upper_limit * 10) // 100

    plot_rolling_mean_only = False
    plot_data_only = False

    if kwargs:
        if "window_size" in kwargs:
            window_size = kwargs["window_size"]

        if "lower_limit" in kwargs:
            lower_limit = kwargs["lower_limit"]

        if "upper_limit" in kwargs:
            upper_limit = kwargs["upper_limit"]

        if "plot_rolling_mean_only" in kwargs:
            plot_rolling_mean_only = kwargs["plot_rolling_mean_only"]

        if "plot_data_only" in kwargs:
            plot_data_only = kwargs["plot_data_only"]

    points = point_history[lower_limit:upper_limit]

    # Generate x-axis for plotting.
    episode_num = [x for x in range(lower_limit, upper_limit)]

    # Use Pandas to calculate the rolling mean (moving average).
    rolling_mean = pd.DataFrame(points).rolling(window_size).mean()

    plt.figure(figsize=(10, 7), facecolor="white")

    if plot_data_only:
        plt.plot(episode_num, points, linewidth=1, color="cyan")
    elif plot_rolling_mean_only:
        plt.plot(episode_num, rolling_mean, linewidth=2, color="magenta")
    else:
        plt.plot(episode_num, points, linewidth=1, color="cyan")
        plt.plot(episode_num, rolling_mean, linewidth=2, color="magenta")

    text_color = "black"

    ax = plt.gca()
    ax.set_facecolor("black")
    plt.grid()
    plt.xlabel("Episode", color=text_color, fontsize=30)
    plt.ylabel("Total Points", color=text_color, fontsize=30)
    yNumFmt = mticker.StrMethodFormatter("{x:,}")
    ax.yaxis.set_major_formatter(yNumFmt)
    ax.tick_params(axis="x", colors=text_color)
    ax.tick_params(axis="y", colors=text_color)
    plt.show()


def display_table(initial_state, action, next_state, reward, done):
    """
    Displays a table containing the initial state, action, next state, reward, and done
    values from Gym's Lunar Lander environment.

    All floating point numbers in the table are displayed rounded to 3 decimal places
    and actions are displayed using their labels instead of their numerical value (i.e
    if action = 0, the action will be printed as "Do nothing" instead of "0").

    Args:
        initial_state (numpy.ndarray):
            The initial state vector returned when resetting the Lunar Lander
            environment, i.e the value returned by the env.reset() method.
        action (int):
            The action taken by the agent. In the Lunar Lander environment, actions are
            represented by integers in the closed interval [0,3] corresponding to:
                - Do nothing = 0
                - Fire right engine = 1
                - Fire main engine = 2
                - Fire left engine = 3
        next_state (numpy.ndarray):
            The state vector returned by the Lunar Lander environment after the agent
            takes an action, i.e the observation returned after running a single time
            step of the environment's dynamics using env.step(action).
        reward (numpy.float64):
            The reward returned by the Lunar Lander environment after the agent takes an
            action, i.e the reward returned after running a single time step of the
            environment's dynamics using env.step(action).
        done (bool):
            The done value returned by the Lunar Lander environment after the agent
            takes an action, i.e the done value returned after running a single time
            step of the environment's dynamics using env.step(action).
    
    Returns:
        table (statsmodels.iolib.table.SimpleTable):
            A table object containing the initial_state, action, next_state, reward,
            and done values. This will result in the table being displayed in the
            Jupyter Notebook.
    """

    action_labels = [
        "Do nothing",
        "Fire right engine",
        "Fire main engine",
        "Fire left engine",
    ]

    # Do not use column headers.
    column_headers = None

    # Display all floating point numbers rounded to 3 decimal places.
    with np.printoptions(formatter={"float": "{:.3f}".format}):
        table_info = [
            ("Initial State:", [f"{initial_state}"]),
            ("Action:", [f"{action_labels[action]}"]),
            ("Next State:", [f"{next_state}"]),
            ("Reward Received:", [f"{reward:.3f}"]),
            ("Episode Terminated:", [f"{done}"]),
        ]

    # Generate table.
    row_labels, data = zip_longest(*table_info)
    table = SimpleTable(data, column_headers, row_labels)

    return table


def embed_mp4(filename):
    """
    Embeds an MP4 video file in a Jupyter notebook.
    
    Args:
        filename (string):
            The path to the the MP4 video file that will be embedded (i.e.
            "./videos/lunar_lander.mp4").
    
    Returns:
        Returns a display object from the given video file. This will result in the
        video being displayed in the Jupyter Notebook.
    """

    video = open(filename, "rb").read()
    b64 = base64.b64encode(video)
    tag = """
    <video width="840" height="480" controls>
    <source src="data:video/mp4;base64,{0}" type="video/mp4">
    Your browser does not support the video tag.
    </video>""".format(
        b64.decode()
    )

    return IPython.display.HTML(tag)


def create_video(filename, env, q_network, fps=30):
    """
    Creates a video of an agent interacting with a Gym environment.

    The agent will interact with the given env environment using the q_network to map
    states to Q values and using a greedy policy to choose its actions (i.e it will
    choose the actions that yield the maximum Q values).
    
    The video will be saved to a file with the given filename. The video format must be
    specified in the filename by providing a file extension (.mp4, .gif, etc..). If you 
    want to embed the video in a Jupyter notebook using the embed_mp4 function, then the
    video must be saved as an MP4 file. 
    
    Args:
        filename (string):
            The path to the file to which the video will be saved. The video format will
            be selected based on the filename. Therefore, the video format must be
            specified in the filename by providing a file extension (i.e.
            "./videos/lunar_lander.mp4"). To see a list of supported formats see the
            imageio documentation: https://imageio.readthedocs.io/en/v2.8.0/formats.html
        env (Gym Environment): 
            The Gym environment the agent will interact with.
        q_network (torch.nn.Sequential):
            A Torch Sequential model that maps states to Q values.
        fps (int):
            The number of frames per second. Specifies the frame rate of the output
            video. The default frame rate is 30 frames per second.  
    """

    with imageio.get_writer(filename, fps=fps) as video:
        done = False
        state, info = env.reset()
        frame = env.render()
        video.append_data(frame)
        while not done:
            
            state_qn = torch.from_numpy(np.expand_dims(state, axis=0))  # state needs to be the right shape for the q_network
            state_qn = state_qn.to(device)
        
            q_values = q_network(state_qn)
            action = torch.argmax(q_values.detach()).item()
            state, _, done, _, _ = env.step(action)
            frame = env.render()
            video.append_data(frame)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```


```python
# Set up a virtual display to render the Lunar Lander environment.
# run on linux
Display(visible=0, size=(840, 480)).start();

import random
random.seed(0)
# Set the random seed for Torch
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
np.random.seed(0)
```

<a name="2"></a>
## 2 - Hyperparameters

Run the cell below to set the hyperparameters.


```python
MEMORY_SIZE = 100_000     # size of memory buffer
GAMMA = 0.995             # discount factor
ALPHA = 1e-3              # learning rate
NUM_STEPS_FOR_UPDATE = 4  # perform a learning update every C time steps
```

<a name="3"></a>
## 3 - The Lunar Lander Environment

In this notebook we will be using [OpenAI's Gym Library](https://www.gymlibrary.dev/). The Gym library provides a wide variety of environments for reinforcement learning. To put it simply, an environment represents a problem or task to be solved. In this notebook, we will try to solve the Lunar Lander environment using reinforcement learning.

The goal of the Lunar Lander environment is to land the lunar lander safely on the landing pad on the surface of the moon. The landing pad is designated by two flag poles and it is always at coordinates `(0,0)` but the lander is also allowed to land outside of the landing pad. The lander starts at the top center of the environment with a random initial force applied to its center of mass and has infinite fuel. The environment is considered solved if you get `200` points. 

<br>
<br>
<figure>
  <img src = "images/lunar_lander.gif" width = 40%>
      <figcaption style = "text-align: center; font-style: italic">Fig 1. Lunar Lander Environment.</figcaption>
</figure>



<a name="3.1"></a>
### 3.1 Action Space

The agent has four discrete actions available:

* Do nothing.
* Fire right engine.
* Fire main engine.
* Fire left engine.

Each action has a corresponding numerical value:

```python
Do nothing = 0
Fire right engine = 1
Fire main engine = 2
Fire left engine = 3
```

<a name="3.2"></a>
### 3.2 Observation Space

The agent's observation space consists of a state vector with 8 variables:

* Its $(x,y)$ coordinates. The landing pad is always at coordinates $(0,0)$.
* Its linear velocities $(\dot x,\dot y)$.
* Its angle $\theta$.
* Its angular velocity $\dot \theta$.
* Two booleans, $l$ and $r$, that represent whether each leg is in contact with the ground or not.

<a name="3.3"></a>
### 3.3 Rewards

The Lunar Lander environment has the following reward system:

* Landing on the landing pad and coming to rest is about 100-140 points.
* If the lander moves away from the landing pad, it loses reward. 
* If the lander crashes, it receives -100 points.
* If the lander comes to rest, it receives +100 points.
* Each leg with ground contact is +10 points.
* Firing the main engine is -0.3 points each frame.
* Firing the side engine is -0.03 points each frame.

<a name="3.4"></a>
### 3.4 Episode Termination

An episode ends (i.e the environment enters a terminal state) if:

* The lunar lander crashes (i.e if the body of the lunar lander comes in contact with the surface of the moon).

* The absolute value of the lander's $x$-coordinate is greater than 1 (i.e. it goes beyond the left or right border)

You can check out the [Open AI Gym documentation](https://www.gymlibrary.dev/environments/box2d/lunar_lander/) for a full description of the environment.

<a name="4"></a>
## 4 - Load the Environment

We start by loading the `LunarLander-v2` environment from the `gym` library by using the `.make()` method. `LunarLander-v2` is the latest version of the Lunar Lander environment and you can read about its version history in the [Open AI Gym documentation](https://www.gymlibrary.dev/environments/box2d/lunar_lander/#version-history).


```python
env = gym.make('LunarLander-v2', render_mode='rgb_array')
```

Once we load the environment we use the `.reset()` method to reset the environment to the initial state. The lander starts at the top center of the environment and we can render the first frame of the environment by using the `.render()` method.

<a name="5"></a>
## 5 - Interacting with the Gym Environment

The Gym library implements the standard “agent-environment loop” formalism:

<br>
<center>
<video src = "./videos/rl_formalism.m4v" width="840" height="480" controls autoplay loop poster="./images/rl_formalism.png"> </video>
<figcaption style = "text-align:center; font-style:italic">Fig 2. Agent-environment Loop Formalism.</figcaption>
</center>
<br>

In the standard “agent-environment loop” formalism, an agent interacts with the environment in discrete time steps $t=0,1,2,...$. At each time step $t$, the agent uses a policy $\pi$ to select an action $A_t$ based on its observation of the environment's state $S_t$. The agent receives a numerical reward $R_t$ and on the next time step, moves to a new state $S_{t+1}$.

<a name="5.1"></a>
### 5.1 Exploring the Environment's Dynamics

In Open AI's Gym environments, we use the `.step()` method to run a single time step of the environment's dynamics. In the version of `gym` that we are using the `.step()` method accepts an action and returns four values:

* `observation` (**object**): an environment-specific object representing your observation of the environment. In the Lunar Lander environment this corresponds to a numpy array containing the positions and velocities of the lander as described in section [3.2 Observation Space](#3.2).


* `reward` (**float**): amount of reward returned as a result of taking the given action. In the Lunar Lander environment this corresponds to a float of type `numpy.float64` as described in section [3.3 Rewards](#3.3).


* `done` (**boolean**): When done is `True`, it indicates the episode has terminated and it’s time to reset the environment. 


* `info` (**dictionary**): diagnostic information useful for debugging. We won't be using this variable in this notebook but it is shown here for completeness.

To begin an episode, we need to reset the environment to an initial state. We do this by using the `.reset()` method.


```python
env.reset()
PIL.Image.fromarray(env.render())
```




    
![png](output_14_0.png)
    



In order to build our neural network later on we need to know the size of the state vector and the number of valid actions. We can get this information from our environment by using the `.observation_space.shape` and `action_space.n` methods, respectively.


```python
state_size = env.observation_space.shape
num_actions = env.action_space.n

print('State Shape:', state_size)
print('Number of actions:', num_actions)
```

    State Shape: (8,)
    Number of actions: 4



```python
# Reset the environment and get the initial state.
initial_state = env.reset()
```

Once the environment is reset, the agent can start taking actions in the environment by using the `.step()` method. Note that the agent can only take one action per time step. 

In the cell below you can select different actions and see how the returned values change depending on the action taken. Remember that in this environment the agent has four discrete actions available and we specify them in code by using their corresponding numerical value:

```python
Do nothing = 0
Fire right engine = 1
Fire main engine = 2
Fire left engine = 3
```


```python
# Select an action
action = 0

# Run a single time step of the environment's dynamics with the given action.
next_state, reward, done, truncated, info = env.step(action)

# Display table with values. All values are displayed to 3 decimal places.
display_table(initial_state, action, next_state, reward, done)
```




<table class="simpletable">
<tr>
  <th>Initial State:</th>      <td>(array([0.002, 1.412, 0.199, 0.049, -0.002, -0.045, 0.000, 0.000],
      dtype=float32), {})</td>
</tr>
<tr>
  <th>Action:</th>                                                      <td>Do nothing</td>                                         
</tr>
<tr>
  <th>Next State:</th>                              <td>[0.004 1.413 0.198 0.023 -0.004 -0.044 0.000 0.000]</td>                    
</tr>
<tr>
  <th>Reward Received:</th>                                                <td>0.211</td>                                           
</tr>
<tr>
  <th>Episode Terminated:</th>                                             <td>False</td>                                           
</tr>
</table>



In practice, when we train the agent we use a loop to allow the agent to take many consecutive actions during an episode.

<a name="6"></a>
## 6 - Deep Q-Learning

In cases where both the state and action space are discrete we can estimate the action-value function iteratively by using the Bellman equation:

$$
Q_{i+1}(s,a) = R + \gamma \max_{a'}Q_i(s',a')
$$

This iterative method converges to the optimal action-value function $Q^*(s,a)$ as $i\to\infty$. This means that the agent just needs to gradually explore the state-action space and keep updating the estimate of $Q(s,a)$ until it converges to the optimal action-value function $Q^*(s,a)$. However, in cases where the state space is continuous it becomes practically impossible to explore the entire state-action space. Consequently, this also makes it practically impossible to gradually estimate $Q(s,a)$ until it converges to $Q^*(s,a)$.

In the Deep $Q$-Learning, we solve this problem by using a neural network to estimate the action-value function $Q(s,a)\approx Q^*(s,a)$. We call this neural network a $Q$-Network and it can be trained by adjusting its weights at each iteration to minimize the mean-squared error in the Bellman equation.

Unfortunately, using neural networks in reinforcement learning to estimate action-value functions has proven to be highly unstable. Luckily, there's a couple of techniques that can be employed to avoid instabilities. These techniques consist of using a ***Target Network*** and ***Experience Replay***. We will explore these two techniques in the following sections.

<a name="6.1"></a>
### 6.1 Target Network

We can train the $Q$-Network by adjusting it's weights at each iteration to minimize the mean-squared error in the Bellman equation, where the target values are given by:

$$
y = R + \gamma \max_{a'}Q(s',a';w)
$$

where $w$ are the weights of the $Q$-Network. This means that we are adjusting the weights $w$ at each iteration to minimize the following error:

$$
\overbrace{\underbrace{R + \gamma \max_{a'}Q(s',a'; w)}_{\rm {y~target}} - Q(s,a;w)}^{\rm {Error}}
$$

Notice that this forms a problem because the $y$ target is changing on every iteration. Having a constantly moving target can lead to oscillations and instabilities. To avoid this, we can create
a separate neural network for generating the $y$ targets. We call this separate neural network the **target $\hat Q$-Network** and it will have the same architecture as the original $Q$-Network. By using the target $\hat Q$-Network, the above error becomes:

$$
\overbrace{\underbrace{R + \gamma \max_{a'}\hat{Q}(s',a'; w^-)}_{\rm {y~target}} - Q(s,a;w)}^{\rm {Error}}
$$

where $w^-$ and $w$ are the weights the target $\hat Q$-Network and $Q$-Network, respectively.

In practice, we will use the following algorithm: every $C$ time steps we will use the $\hat Q$-Network to generate the $y$ targets and update the weights of the target $\hat Q$-Network using the weights of the $Q$-Network. We will update the weights $w^-$ of the the target $\hat Q$-Network using a **soft update**. This means that we will update the weights $w^-$ using the following rule:
 
$$
w^-\leftarrow \tau w + (1 - \tau) w^-
$$

where $\tau\ll 1$. By using the soft update, we are ensuring that the target values, $y$, change slowly, which greatly improves the stability of our learning algorithm.

<a name="ex01"></a>
### Exercise 1

In this exercise you will create the $Q$ and target $\hat Q$ networks and set the optimizer. Remember that the Deep $Q$-Network (DQN) is a neural network that approximates the action-value function $Q(s,a)\approx Q^*(s,a)$. It does this by learning how to map states to $Q$ values.

To solve the Lunar Lander environment, we are going to employ a DQN with the following architecture:



* A `Linear` layer with `state_size[0]` input units, `64` output units

* A `relu` activation function layer.

* A `Linear` layer with `64` input units, `64` output units

* A `relu` activation function layer.

* A `Linear` layer with `64` input units, `num_actions` output units


In the cell below you should create the $Q$-Network and the target $\hat Q$-Network using the model architecture described above. Remember that both the $Q$-Network and the target $\hat Q$-Network have the same architecture.

Lastly, you should set `Adam` as the optimizer with a learning rate equal to `ALPHA`. Recall that `ALPHA` was defined in the [Hyperparameters](#2) section. We should note that for this exercise you should use the already imported packages:


```python
# Create the Q-Network
q_network = nn.Sequential(
    ### START CODE HERE ### 
    nn.Linear(state_size[0], 64),
    nn.ReLU(),
    nn.Linear(64, 64),
    nn.ReLU(),
    nn.Linear(64, num_actions)
    ### END CODE HERE ### 
    )

# Create the target Q^-Network
target_q_network = nn.Sequential(
    ### START CODE HERE ### 
    nn.Linear(state_size[0], 64),
    nn.ReLU(),
    nn.Linear(64, 64),
    nn.ReLU(),
    nn.Linear(64, num_actions)
    ### END CODE HERE ### 
    )

### START CODE HERE ### 
optimizer = optim.Adam(q_network.parameters(), lr=ALPHA)
### END CODE HERE ###
```


```python
# Make them have the same initial parameters.
for target_param, param in zip(target_q_network.parameters(), q_network.parameters()):
    target_param.data.copy_(param.data)
```

<a name="6.2"></a>
### 6.2 Experience Replay

When an agent interacts with the environment, the states, actions, and rewards the agent experiences are sequential by nature. If the agent tries to learn from these consecutive experiences it can run into problems due to the strong correlations between them. To avoid this, we employ a technique known as **Experience Replay** to generate uncorrelated experiences for training our agent. Experience replay consists of storing the agent's experiences (i.e the states, actions, and rewards the agent receives) in a memory buffer and then sampling a random mini-batch of experiences from the buffer to do the learning. The experience tuples $(S_t, A_t, R_t, S_{t+1})$ will be added to the memory buffer at each time step as the agent interacts with the environment.

For convenience, we will store the experiences as named tuples.


```python
# Store experiences as named tuples
experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
```

By using experience replay we avoid problematic correlations, oscillations and instabilities. In addition, experience replay also allows the agent to potentially use the same experience in multiple weight updates, which increases data efficiency.

<a name="7"></a>
## 7 - Deep Q-Learning Algorithm with Experience Replay

Now that we know all the techniques that we are going to use, we can put them together to arrive at the Deep Q-Learning Algorithm With Experience Replay.
<br>
<br>
<figure>
  <img src = "images/deep_q_algorithm.png" width = 90% style = "border: thin silver solid; padding: 0px">
      <figcaption style = "text-align: center; font-style: italic">Fig 3. Deep Q-Learning with Experience Replay.</figcaption>
</figure>

<a name="ex02"></a>
### Exercise 2

In this exercise you will implement line ***12*** of the algorithm outlined in *Fig 3* above and you will also compute the loss between the $y$ targets and the $Q(s,a)$ values. In the cell below, complete the `compute_loss` function by setting the $y$ targets equal to:

$$
\begin{equation}
    y_j =
    \begin{cases}
      R_j & \text{if episode terminates at step  } j+1\\
      R_j + \gamma \max_{a'}\hat{Q}(s_{j+1},a') & \text{otherwise}\\
    \end{cases}       
\end{equation}
$$

Here are a couple of things to note:

* The `compute_loss` function takes in a mini-batch of experience tuples. This mini-batch of experience tuples is unpacked to extract the `states`, `actions`, `rewards`, `next_states`, and `done_vals`. You should keep in mind that these variables are *Pytorch Tensors* whose size will depend on the mini-batch size. For example, if the mini-batch size is `64` then both `rewards` and `done_vals` will be Pytorch Tensors with `64` elements.


* Using `if/else` statements to set the $y$ targets will not work when the variables are tensors with many elements. However, notice that you can use the `done_vals` to implement the above in a single line of code. To do this, recall that the `done` variable is a Boolean variable that takes the value `True` when an episode terminates at step $j+1$ and it is `False` otherwise. Taking into account that a Boolean value of `True` has the numerical value of `1` and a Boolean value of `False` has the numerical value of `0`, you can use the factor `(1 - done_vals)` to implement the above in a single line of code. Here's a hint: notice that `(1 - done_vals)` has a value of `0` when `done_vals` is `True` and a value of `1` when `done_vals` is `False`. 

Lastly, compute the loss by calculating the Mean-Squared Error (`MSE`) between the `y_targets` and the `q_values`. To calculate the mean-squared error you should use `F.mse_loss`:


```python
def compute_loss(experiences, gamma, q_network, target_q_network):
    """ 
    Calculates the loss.
    
    Args:
      experiences: (tuple) tuple of ["state", "action", "reward", "next_state", "done"] namedtuples
      gamma: (float) The discount factor.
      q_network: (torch.nn.Sequential) PyTorch model for predicting the q_values
      target_q_network: (torch.nn.Sequential) PyTorch model for predicting the targets
          
    Returns:
      loss: (PyTorch Tensor) the Mean-Squared Error between the y targets and the Q(s,a) values.
    """

    # Unpack the mini-batch of experience tuples
    states, actions, rewards, next_states, done_vals = experiences

    ### START CODE HERE ### 
    # Compute max Q^(s,a) using torch.max and target_q_network, pay attendion to the `dim` parameter.
    max_qsa = torch.max(target_q_network(next_states), dim=-1)[0]

    # Set y = R if episode terminates(done_vals are the boolean values indicating if the episode ended)
    # otherwise set y = R + γ max Q^(s,a).
    y_targets = rewards + (gamma * max_qsa * (1 - done_vals))

    # Get the q_values from q_network
    q_values = q_network(states)

    # Comprehend what `gather` does in the following line.
    q_values = q_values.gather(1, actions.long().unsqueeze(1))
    
    # Compute the loss using F.mse_loss between q_values and y_targets. You may need to use `unsqueeze` on y_targets.
    loss = F.mse_loss(q_values, y_targets.unsqueeze(1))
    ### END CODE HERE ### 

    return loss
```

<a name="8"></a>
## 8 - Update the Network Weights

We will use the `agent_learn` function below to implement lines ***12 -14*** of the algorithm outlined in [Fig 3](#7). The `agent_learn` function will update the weights of the $Q$ and target $\hat Q$ networks using a custom training loop.

The last line of this function updates the weights of the target $\hat Q$-Network using a [soft update](#6.1). If you want to know how this is implemented in code we encourage you to take a look at the `utils.update_target_network` function in the `utils` module.


```python
def update_target_network(q_network, target_q_network):
    """
    Updates the weights of the target Q-Network using a soft update.
    
    The weights of the target_q_network are updated using the soft update rule:
    
                    w_target = (TAU * w) + (1 - TAU) * w_target
    
    where w_target are the weights of the target_q_network, TAU is the soft update
    parameter, and w are the weights of the q_network.
    
    Args:
        q_network (torch.nn.Module): 
            The Q-Network. 
        target_q_network (torch.nn.Module):
            The Target Q-Network.
    """
    TAU = 1e-3  # Soft update parameter.
    for target_param, param in zip(target_q_network.parameters(), q_network.parameters()):
        target_param.data.copy_(TAU * param.data + (1.0 - TAU) * target_param.data)
```


```python
MINIBATCH_SIZE = 64  # Mini-batch size.

def get_experiences(memory_buffer):
    """
    Returns a random sample of experience tuples drawn from the memory buffer.

    Retrieves a random sample of experience tuples from the given memory_buffer and
    returns them as PyTorch Tensors. The size of the random sample is determined by
    the mini-batch size (MINIBATCH_SIZE). 
    
    Args:
        memory_buffer (deque):
            A deque containing experiences. The experiences are stored in the memory
            buffer as namedtuples: namedtuple("Experience", field_names=["state",
            "action", "reward", "next_state", "done"]).

    Returns:
        A tuple (states, actions, rewards, next_states, done_vals) where:

            - states are the starting states of the agent.
            - actions are the actions taken by the agent from the starting states.
            - rewards are the rewards received by the agent after taking the actions.
            - next_states are the new states of the agent after taking the actions.
            - done_vals are the boolean values indicating if the episode ended.

        All tuple elements are PyTorch Tensors whose shape is determined by the
        mini-batch size and the given Gym environment. For the Lunar Lander environment
        the states and next_states will have a shape of [MINIBATCH_SIZE, 8] while the
        actions, rewards, and done_vals will have a shape of [MINIBATCH_SIZE]. All
        PyTorch Tensors have elements with dtype=torch.float32.
    """

    experiences = random.sample(memory_buffer, k=MINIBATCH_SIZE)
    states = torch.tensor(
        [e.state for e in experiences if e is not None], dtype=torch.float32, device=device)
    actions = torch.tensor(
        [e.action for e in experiences if e is not None], dtype=torch.float32, device=device)
    rewards = torch.tensor(
        [e.reward for e in experiences if e is not None], dtype=torch.float32, device=device)
    next_states = torch.tensor(
        [e.next_state for e in experiences if e is not None], dtype=torch.float32, device=device)
    done_vals = torch.tensor(
        [e.done for e in experiences if e is not None], dtype=torch.float32, device=device)
    return (states, actions, rewards, next_states, done_vals)
```


```python
def agent_learn(experiences, gamma):
    """
    Updates the weights of the Q networks.
    
    Args:
      experiences: (tuple) tuple of ["state", "action", "reward", "next_state", "done"] namedtuples
      gamma: (float) The discount factor.
    
    """

    # Unpack the mini-batch of experience tuples
    states, actions, rewards, next_states, done_vals = experiences

    # Calculate the loss
    loss = compute_loss((states, actions, rewards, next_states, done_vals), gamma, q_network, target_q_network)

    # Compute gradients and update weights of q_network
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Update the weights of target_q_network
    update_target_network(q_network, target_q_network)
```

<a name="9"></a>
## 9 - Train the Agent

We are now ready to train our agent to solve the Lunar Lander environment. In the cell below we will implement the algorithm in [Fig 3](#7) line by line (please note that we have included the same algorithm below for easy reference. This will prevent you from scrolling up and down the notebook):

* **Line 1**: We initialize the `memory_buffer` with a capacity of $N =$ `MEMORY_SIZE`. Notice that we are using a `deque` as the data structure for our `memory_buffer`.


* **Line 2**: We skip this line since we already initialized the `q_network` in [Exercise 1](#ex01).


* **Line 3**: We initialize the `target_q_network` by setting its weights to be equal to those of the `q_network`.


* **Line 4**: We start the outer loop. Notice that we have set $M =$ `num_episodes = 2000`. This number is reasonable because the agent should be able to solve the Lunar Lander environment in less than `2000` episodes using this notebook's default parameters.


* **Line 5**: We use the `.reset()` method to reset the environment to the initial state and get the initial state.


* **Line 6**: We start the inner loop. Notice that we have set $T =$ `max_num_timesteps = 1000`. This means that the episode will automatically terminate if the episode hasn't terminated after `1000` time steps.


* **Line 7**: The agent observes the current `state` and chooses an `action` using an $\epsilon$-greedy policy. Our agent starts out using a value of $\epsilon =$ `epsilon = 1` which yields an $\epsilon$-greedy policy that is equivalent to the equiprobable random policy. This means that at the beginning of our training, the agent is just going to take random actions regardless of the observed `state`. As training progresses we will decrease the value of $\epsilon$ slowly towards a minimum value using a given $\epsilon$-decay rate. We want this minimum value to be close to zero because a value of $\epsilon = 0$ will yield an $\epsilon$-greedy policy that is equivalent to the greedy policy. This means that towards the end of training, the agent will lean towards selecting the `action` that it believes (based on its past experiences) will maximize $Q(s,a)$. We will set the minimum $\epsilon$ value to be `0.01` and not exactly 0 because we always want to keep a little bit of exploration during training. If you want to know how this is implemented in code we encourage you to take a look at the `utils.get_action` function in the `utils` module.


* **Line 8**: We use the `.step()` method to take the given `action` in the environment and get the `reward` and the `next_state`. 


* **Line 9**: We store the `experience(state, action, reward, next_state, done)` tuple in our `memory_buffer`. Notice that we also store the `done` variable so that we can keep track of when an episode terminates. This allowed us to set the $y$ targets in [Exercise 2](#ex02).


* **Line 10**: We check if the conditions are met to perform a learning update. We do this by using our custom `utils.check_update_conditions` function. This function checks if $C =$ `NUM_STEPS_FOR_UPDATE = 4` time steps have occured and if our `memory_buffer` has enough experience tuples to fill a mini-batch. For example, if the mini-batch size is `64`, then our `memory_buffer` should have more than `64` experience tuples in order to pass the latter condition. If the conditions are met, then the `utils.check_update_conditions` function will return a value of `True`, otherwise it will return a value of `False`.


* **Lines 11 - 14**: If the `update` variable is `True` then we perform a learning update. The learning update consists of sampling a random mini-batch of experience tuples from our `memory_buffer`, setting the $y$ targets, performing gradient descent, and updating the weights of the networks. We will use the `agent_learn` function we defined in [Section 8](#8) to perform the latter 3.


* **Line 15**: At the end of each iteration of the inner loop we set `next_state` as our new `state` so that the loop can start again from this new state. In addition, we check if the episode has reached a terminal state (i.e we check if `done = True`). If a terminal state has been reached, then we break out of the inner loop.


* **Line 16**: At the end of each iteration of the outer loop we update the value of $\epsilon$, and check if the environment has been solved. We consider that the environment has been solved if the agent receives an average of `200` points in the last `100` episodes. If the environment has not been solved we continue the outer loop and start a new episode.

Finally, we wanted to note that we have included some extra variables to keep track of the total number of points the agent received in each episode. This will help us determine if the agent has solved the environment and it will also allow us to see how our agent performed during training. We also use the `time` module to measure how long the training takes. 

<br>
<br>
<figure>
  <img src = "images/deep_q_algorithm.png" width = 90% style = "border: thin silver solid; padding: 0px">
      <figcaption style = "text-align: center; font-style: italic">Fig 4. Deep Q-Learning with Experience Replay.</figcaption>
</figure>
<br>


```python
def get_action(q_values, epsilon=0.0):
    """
    Returns an action using an ε-greedy policy.

    This function will return an action according to the following rules:
        - With probability epsilon, it will return an action chosen at random.
        - With probability (1 - epsilon), it will return the action that yields the
        maximum Q value in q_values.
    
    Args:
        q_values (torch.Tensor):
            The Q values returned by the Q-Network. For the Lunar Lander environment
            this PyTorch Tensor should have a shape of [1, 4] and its elements should
            have dtype=torch.float32. 
        epsilon (float):
            The current value of epsilon.

    Returns:
        An action (numpy.int64). For the Lunar Lander environment, actions are
        represented by integers in the closed interval [0,3].
    """

    if random.random() > epsilon:
        ### START CODE HERE ###
        return torch.argmax(q_values).item()
        ### END CODE HERE ### 
    else:
        ### START CODE HERE ### 
        return random.randint(0, 3)
        ### END CODE HERE ### 
```


```python
# %debug
start = time.time()

num_episodes = 2000
max_num_timesteps = 1000

total_point_history = []

num_p_av = 100    # number of total points to use for averaging
epsilon = 1.0     # initial ε value for ε-greedy policy

# Create a memory buffer D with capacity N
memory_buffer = deque(maxlen=MEMORY_SIZE)
q_network = q_network.to(device)
target_q_network = target_q_network.to(device)
# Set the target network weights equal to the Q-Network weights
target_q_network.load_state_dict(q_network.state_dict())

for i in range(num_episodes):

    # Reset the environment to the initial state and get the initial state
    state, info = env.reset()
    total_points = 0

    for t in range(max_num_timesteps):

        # From the current state S choose an action A using an ε-greedy policy
        state_qn = torch.from_numpy(np.expand_dims(state, axis=0))  # state needs to be the right shape for the q_network
        state_qn = state_qn.to(device)
        q_values = q_network(state_qn)
        action = get_action(q_values, epsilon)

        # Take action A and receive reward R and the next state S'
        next_state, reward, done, truncated, info = env.step(action)

        # Store experience tuple (S,A,R,S') in the memory buffer.
        # We store the done variable as well for convenience.
        memory_buffer.append(experience(state, action, reward, next_state, done))

        # Only update the network every NUM_STEPS_FOR_UPDATE time steps.
        update = check_update_conditions(t, NUM_STEPS_FOR_UPDATE, memory_buffer)

        if update:
            # Sample random mini-batch of experience tuples (S,A,R,S') from D
            experiences = get_experiences(memory_buffer)

            # Set the y targets, perform a gradient descent step,
            # and update the network weights.
            agent_learn(experiences, GAMMA)

        state = next_state.copy()
        total_points += reward

        if done:
            break

    total_point_history.append(total_points)
    av_latest_points = np.mean(total_point_history[-num_p_av:])

    # Update the ε value
    epsilon = get_new_eps(epsilon)

    print(f"\rEpisode {i+1} | Total point average of the last {num_p_av} episodes: {av_latest_points:.2f}", end="")

    if (i+1) % num_p_av == 0:
        print(f"\rEpisode {i+1} | Total point average of the last {num_p_av} episodes: {av_latest_points:.2f}")

    # We will consider that the environment is solved if we get an
    # average of 200 points in the last 100 episodes.
    if av_latest_points >= 200.0:
        print(f"\n\nEnvironment solved in {i+1} episodes!")
        torch.save(q_network, 'lunar_lander_model.pt')
        break

tot_time = time.time() - start

print(f"\nTotal Runtime: {tot_time:.2f} s ({(tot_time/60):.2f} min)")
```

    Episode 100 | Total point average of the last 100 episodes: -160.81
    Episode 200 | Total point average of the last 100 episodes: -94.379
    Episode 300 | Total point average of the last 100 episodes: -66.97
    Episode 400 | Total point average of the last 100 episodes: -19.23
    Episode 500 | Total point average of the last 100 episodes: 135.43
    Episode 600 | Total point average of the last 100 episodes: 196.77
    Episode 607 | Total point average of the last 100 episodes: 200.16
    
    Environment solved in 607 episodes!
    
    Total Runtime: 775.11 s (12.92 min)


We can plot the total point history along with the moving average to see how our agent improved during training. If you want to know about the different plotting options available in the `utils.plot_history` function we encourage you to take a look at the `utils` module.


```python
# Plot the total point history along with the moving average
plot_history(total_point_history)
```


    
![png](output_40_0.png)
    


<a name="10"></a>
## 10 - See the Trained Agent In Action

Now that we have trained our agent, we can see it in action. We will use the `utils.create_video` function to create a video of our agent interacting with the environment using the trained $Q$-Network. The `utils.create_video` function uses the `imageio` library to create the video. This library produces some warnings that can be distracting, so, to suppress these warnings we run the code below.


```python
# Suppress warnings from imageio
import logging
logging.getLogger().setLevel(logging.ERROR)
```

In the cell below we create a video of our agent interacting with the Lunar Lander environment using the trained `q_network`. The video is saved to the `videos` folder with the given `filename`. We use the `utils.embed_mp4` function to embed the video in the Jupyter Notebook so that we can see it here directly without having to download it.

We should note that since the lunar lander starts with a random initial force applied to its center of mass, every time you run the cell below you will see a different video. If the agent was trained properly, it should be able to land the lunar lander in the landing pad every time, regardless of the initial force applied to its center of mass.


```python
filename = "lunar_lander.mp4"

create_video(filename, env, q_network)
embed_mp4(filename)
```

    [swscaler @ 0x6840600] Warning: data is not aligned! This can lead to a speed loss






<video width="840" height="480" controls>
<source src="data:video/mp4;base64,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AQAACgAAAAABAAAEAAAAAAEAAAAAAAAAAQAAAgAAAAABAAAKAAAAAAEAAAQAAAAAAQAAAAAAAAABAAACAAAAAAEAAAoAAAAAAQAABAAAAAABAAAAAAAAAAEAAAIAAAAAAQAACgAAAAABAAAEAAAAAAEAAAAAAAAAAQAAAgAAAAABAAAKAAAAAAEAAAQAAAAAAQAAAAAAAAABAAACAAAAAAEAAAoAAAAAAQAABAAAAAABAAAAAAAAAAEAAAIAAAAAAQAACgAAAAABAAAEAAAAAAEAAAAAAAAAAQAAAgAAAAABAAAKAAAAAAEAAAQAAAAAAQAAAAAAAAABAAACAAAAAAEAAAoAAAAAAQAABAAAAAABAAAAAAAAAAEAAAIAAAAAAQAACgAAAAABAAAEAAAAAAEAAAAAAAAAAQAAAgAAAAABAAAKAAAAAAEAAAQAAAAAAQAAAAAAAAABAAACAAAAAAEAAAoAAAAAAQAABAAAAAABAAAAAAAAAAEAAAIAAAAAAQAACAAAAAACAAACAAAAAAEAAAoAAAAAAQAABAAAAAABAAAAAAAAAAEAAAIAAAAAAQAACgAAAAABAAAEAAAAAAEAAAAAAAAAAQAAAgAAAAABAAAKAAAAAAEAAAQAAAAAAQAAAAAAAAABAAACAAAAAAEAAAoAAAAAAQAABAAAAAABAAAAAAAAAAEAAAIAAAAAAQAACgAAAAABAAAEAAAAAAEAAAAAAAAAAQAAAgAAAAABAAAKAAAAAAEAAAQAAAAAAQAAAAAAAAABAAACAAAAAAEAAAoAAAAAAQAABAAAAAABAAAAAAAAAAEAAAIAAAAAAQAACgAAAAABAAAEAAAAAAEAAAAAAAAAAQAAAgAAAAABAAAKAAAAAAEAAAQAAAAAAQAAAAAAAAABAAACAAAAAAEAAAoAAAAAAQAABAAAAAABAAAAAAAAAAEAAAIAAAAAAQAACgAAAAABAAAEAAAAAAEAAAAAAAAAAQAAAgAAAAABAAAKAAAAAAEAAAQAAAAAAQAAAAAAAAABAAACAAAAAAEAAAoAAAAAAQAABAAAAAABAAAAAAAAAAEAAAIAAAAAAQAACgAAAAABAAAEAAAAAAEAAAAAAAAAAQAAAgAAAAABAAAKAAAAAAEAAAQAAAAAAQAAAAAAAAABAAACAAAAAAEAAAoAAAAAAQAABAAAAAABAAAAAAAAAAEAAAIAAAAAAQAACgAAAAABAAAEAAAAAAEAAAAAAAAAAQAAAgAAAAABAAAKAAAAAAEAAAQAAAAAAQAAAAAAAAABAAACAAAAAAEAAAoAAAAAAQAABAAAAAABAAAAAAAAAAEAAAIAAAAAAQAACgAAAAABAAAEAAAAAAEAAAAAAAAAAQAAAgAAAAABAAAKAAAAAAEAAAQAAAAAAQAAAAAAAAABAAACAAAAAAEAAAoAAAAAAQAABAAAAAABAAAAAAAAAAEAAAIAAAAAAQAACgAAAAABAAAEAAAAAAEAAAAAAAAAAQAAAgAAAAABAAAKAAAAAAEAAAQAAAAAAQAAAAAAAAABAAACAAAAAAEAAAoAAAAAAQAABAAAAAABAAAAAAAAAAEAAAIAAAAAAQAABAAAAAABAAAKAAAAAAEAAAQAAAAAAQAAAAAAAAABAAACAAAAAAEAAAoAAAAAAQAABAAAAAABAAAAAAAAAAEAAAIAAAAAAQAACAAAAAACAAACAAAAAAEAAAoAAAAAAQAABAAAAAABAAAAAAAAAAEAAAIAAAAAAQAACgAAAAABAAAEAAAAAAEAAAAAAAAAAQAAAgAAAAABAAAKAAAAAAEAAAQAAAAAAQAAAAAAAAABAAACAAAAAAEAAAoAAAAAAQAABAAAAAABAAAAAAAAAAEAAAIAAAAAAQAACgAAAAABAAAEAAAAAAEAAAAAAAAAAQAAAgAAAAABAAAKAAAAAAEAAAQAAAAAAQAAAAAAAAABAAACAAAAAAEAAAoAAAAAAQAABAAAAAABAAAAAAAAAAEAAAIAAAAAAQAACgAAAAABAAAEAAAAAAEAAAAAAAAAAQAAAgAAAAABAAAKAAAAAAEAAAQAAAAAAQAAAAAAAAABAAACAAAAAAEAAAoAAAAAAQAABAAAAAABAAAAAAAAAAEAAAIAAAAAAQAACgAAAAABAAAEAAAAAAEAAAAAAAAAAQAAAgAAAAABAAAKAAAAAAEAAAQAAAAAAQAAAAAAAAABAAACAAAAAAEAAAoAAAAAAQAABAAAAAABAAAAAAAAAAEAAAIAAAAAAQAACAAAAAACAAACAAAAAAEAAAoAAAAAAQAABAAAAAABAAAAAAAAAAEAAAIAAAAAAQAACgAAAAABAAAEAAAAAAEAAAAAAAAAAQAAAgAAAAABAAAKAAAAAAEAAAQAAAAAAQAAAAAAAAABAAACAAAAAAEAAAoAAAAAAQAABAAAAAABAAAAAAAAAAEAAAIAAAAAAQAACgAAAAABAAAEAAAAAAEAAAAAAAAAAQAAAgAAAAABAAAKAAAAAAEAAAQAAAAAAQAAAAAAAAABAAACAAAAAAEAAAoAAAAAAQAABAAAAAABAAAAAAAAAAEAAAIAAAAAAQAACgAAAAABAAAEAAAAAAEAAAAAAAAAAQAAAgAAAAABAAAKAAAAAAEAAAQAAAAAAQAAAAAAAAABAAACAAAAAAEAAAoAAAAAAQAABAAAAAABAAAAAAAAAAEAAAIAAAAAAQAACgAAAAABAAAEAAAAAAEAAAAAAAAAAQAAAgAAAAABAAAKAAAAAAEAAAQAAAAAAQAAAAAAAAABAAACAAAAAAEAAAoAAAAAAQAABAAAAAABAAAAAAAAAAEAAAIAAAAAAQAACgAAAAABAAAEAAAAAAEAAAAAAAAAAQAAAgAAAAABAAAKAAAAAAEAAAQAAAAAAQAAAAAAAAABAAACAAAAAAEAAAoAAAAAAQAABAAAAAABAAAAAAAAAAEAAAIAAAAAAQAACgAAAAABAAAEAAAAAAEAAAAAAAAAAQAAAgAAAAABAAAKAAAAAAEAAAQAAAAAAQAAAAAAAAABAAACAAAAAAEAAAoAAAAAAQAABAAAAAABAAAAAAAAAAEAAAIAAAAAAQAACgAAAAABAAAEAAAAAAEAAAAAAAAAAQAAAgAAAAABAAAKAAAAAAEAAAQAAAAAAQAAAAAAAAABAAACAAAAAAEAAAoAAAAAAQAABAAAAAABAAAAAAAAAAEAAAIAAAAAAQAACgAAAAABAAAEAAAAAAEAAAAAAAAAAQAAAgAAAAABAAAIAAAAAAIAAAIAAAAAAQAACgAAAAABAAAEAAAAAAEAAAAAAAAAAQAAAgAAAAABAAAKAAAAAAEAAAQAAAAAAQAAAAAAAAABAAACAAAAAAEAAAoAAAAAAQAABAAAAAABAAAAAAAAAAEAAAIAAAAAAQAACgAAAAABAAAEAAAAAAEAAAAAAAAAAQAAAgAAAAABAAAKAAAAAAEAAAQAAAAAAQAAAAAAAAABAAACAAAAAAEAAAoAAAAAAQAABAAAAAABAAAAAAAAAAEAAAIAAAAAAQAACgAAAAABAAAEAAAAAAEAAAAAAAAAAQAAAgAAAAABAAAKAAAAAAEAAAQAAAAAAQAAAAAAAAABAAACAAAAAAEAAAoAAAAAAQAABAAAAAABAAAAAAAAAAEAAAIAAAAAAQAACgAAAAABAAAEAAAAAAEAAAAAAAAAAQAAAgAAAAABAAAKAAAAAAEAAAQAAAAAAQAAAAAAAAABAAACAAAAAAEAAAoAAAAAAQAABAAAAAABAAAAAAAAAAEAAAIAAAAAAQAACgAAAAABAAAEAAAAAAEAAAAAAAAAAQAAAgAAAAABAAAKAAAAAAEAAAQAAAAAAQAAAAAAAAABAAACAAAAAAEAAAoAAAAAAQAABAAAAAABAAAAAAAAAAEAAAIAAAAAAQAACgAAAAABAAAEAAAAAAEAAAAAAAAAAQAAAgAAAAABAAAKAAAAAAEAAAQAAAAAAQAAAAAAAAABAAACAAAAAAEAAAoAAAAAAQAABAAAAAABAAAAAAAAAAEAAAIAAAAAAQAACgAAAAABAAAEAAAAAAEAAAAAAAAAAQAAAgAAAAABAAAKAAAAAAEAAAQAAAAAAQAAAAAAAAABAAACAAAAAAEAAAoAAAAAAQAABAAAAAABAAAAAAAAAAEAAAIAAAAAAQAABAAAAAABAAAIAAAAAAIAAAIAAAAAAQAABAAAAAABAAAKAAAAAAEAAAQAAAAAAQAAAAAAAAABAAACAAAAAAEAAAoAAAAAAQAABAAAAAABAAAAAAAAAAEAAAIAAAAAAQAABgAAAAABAAACAAAAAAEAAAoAAAAAAQAABAAAAAABAAAAAAAAAAEAAAIAAAAAAQAACgAAAAABAAAEAAAAAAEAAAAAAAAAAQAAAgAAAAABAAAKAAAAAAEAAAQAAAAAAQAAAAAAAAABAAACAAAAAAEAAAoAAAAAAQAABAAAAAABAAAAAAAAAAEAAAIAAAAAAQAACgAAAAABAAAEAAAAAAEAAAAAAAAAAQAAAgAAAAABAAAKAAAAAAEAAAQAAAAAAQAAAAAAAAABAAACAAAAAAEAAAoAAAAAAQAABAAAAAABAAAAAAAAAAEAAAIAAAAAAQAACgAAAAABAAAEAAAAAAEAAAAAAAAAAQAAAgAAAAABAAAGAAAAAAEAAAIAAAAAAQAACgAAAAABAAAEAAAAAAEAAAAAAAAAAQAAAgAAAAABAAAKAAAAAAEAAAQAAAAAAQAAAAAAAAABAAACAAAAAAEAAAoAAAAAAQAABAAAAAABAAAAAAAAAAEAAAIAAAAAAQAACgAAAAABAAAEAAAAAAEAAAAAAAAAAQAAAgAAAAABAAAGAAAAAAEAAAIAAAAAAQAACgAAAAABAAAEAAAAAAEAAAAAAAAAAQAAAgAAAAABAAAIAAAAAAIAAAIAAAAAAQAACgAAAAABAAAEAAAAAAEAAAAAAAAAAQAAAgAAAAABAAAKAAAAAAEAAAQAAAAAAQAAAAAAAAABAAACAAAAAAEAAAoAAAAAAQAABAAAAAABAAAAAAAAAAEAAAIAAAAAAQAACgAAAAABAAAEAAAAAAEAAAAAAAAAAQAAAgAAAAABAAAKAAAAAAEAAAQAAAAAAQAAAAAAAAABAAACAAAAAAEAAAoAAAAAAQAABAAAAAABAAAAAAAAAAEAAAIAAAAAAQAACgAAAAABAAAEAAAAAAEAAAAAAAAAAQAAAgAAAAABAAAKAAAAAAEAAAQAAAAAAQAAAAAAAAABAAACAAAAAAEAAAoAAAAAAQAABAAAAAABAAAAAAAAAAEAAAIAAAAAAQAACgAAAAABAAAEAAAAAAEAAAAAAAAAAQAAAgAAAAABAAAKAAAAAAEAAAQAAAAAAQAAAAAAAAABAAACAAAAAAEAAAoAAAAAAQAABAAAAAABAAAAAAAAAAEAAAIAAAAAAQAACgAAAAABAAAEAAAAAAEAAAAAAAAAAQAAAgAAAAABAAAKAAAAAAEAAAQAAAAAAQAAAAAAAAABAAACAAAAAAEAAAoAAAAAAQAABAAAAAABAAAAAAAAAAEAAAIAAAAAAQAACgAAAAABAAAEAAAAAAEAAAAAAAAAAQAAAgAAAAABAAAKAAAAAAEAAAQAAAAAAQAAAAAAAAABAAACAAAAAAEAAAoAAAAAAQAABAAAAAABAAAAAAAAAAEAAAIAAAAAAQAABgAAAAABAAACAAAAABxzdHNjAAAAAAAAAAEAAAABAAAChAAAAAEAAAokc3RzegAAAAAAAAAAAAAChAAACucAAAD8AAAAUgAAAD4AAAA9AAAAnAAAAEcAAAAtAAAALAAAAKQAAABCAAAALwAAAC8AAADVAAAARwAAAD4AAAA9AAAA7wAAAEwAAAA1AAAAQAAAAOsAAABnAAAAOgAAAFIAAAC5AAAARgAAAa8AAACFAAAAUwAAAIkAAAFLAAAAyAAAAWkAAAC5AAAA0AAAALIAAAFiAAAArgAAAK0AAAC7AAABWwAAAK8AAAExAAAA3wAAANoAAADTAAABHQAAAM8AAACsAAAA1wAAAUwAAADWAAAA2wAAAMAAAAFlAAAA+AAAANQAAADaAAABZQAAAMYAAADIAAAAwgAAAZAAAAC4AAAArwAAAMcAAAETAAAA2AAAAKYAAAC2AAABSwAAAM0AAADlAAAApQAAAQ4AAADnAAAAvgAAAPEAAAElAAAA9AAAALEAAADfAAABSQAAANwAAADaAAABAwAAAWIAAAEJAAAA+wAAAOIAAAFfAAAAwwAAAN0AAADhAAABVgAAAMYAAADUAAAAogAAARMAAADIAAAAvwAAALUAAAEjAAAA6wAAAMAAAAC4AAABjQAAAOkAAAC5AAAAxQAAAR0AAACrAAAAywAAAKsAAAFDAAAA0gAAAH4AAAC3AAABHwAAAOIAAAD0AAAA0wAAASIAAAD4AAAA6QAAAMwAAAEbAAAAzwAAAN0AAACNAAAA1AAAALAAAACgAAAAtQAAANoAAADKAAAAngAAAJsAAAFSAAAAwwAAAJ8AAAC9AAABBgAAAO0AAACyAAAAmAAAAPwAAACrAAAAnAAAAS0AAAC3AAAAtwAAAH8AAAE3AAAAkwAAAHQAAAC3AAAA6wAAAMkAAAB9AAAAuwAAAPkAAACrAAAAhgAAAJUAAADfAAAAqwAAAJ8AAACZAAABFAAAAKsAAACiAAAAuAAAAQkAAACvAAAAyAAAAKwAAAEYAAAAvAAAAK0AAADOAAABJAAAAP0AAADCAAAArgAAASgAAADXAAAA1QAAAMgAAAEkAAABFwAAAKgAAADmAAABBAAAAKQAAAC+AAAA7AAAAYoAAADVAAAAyAAAAL0AAAETAAAA+AAAANMAAADFAAABJwAAAO0AAADQAAAAvgAAARcAAADJAAAAtQAAAKoAAAEiAAAAywAAAJgAAACAAAABBgAAAKkAAACBAAAAlQAAAO4AAAC3AAAAqQAAAKMAAADbAAAAhwAAAIAAAACFAAAA5AAAAHUAAACEAAAAaQAAAPUAAACFAAAAfQAAAGsAAACcAAAAgQAAAG4AAAB4AAAA/AAAAIgAAABpAAAAoAAAAQIAAACdAAAAkQAAAIgAAAqxAAABIAAAAH0AAABOAAAAUQAAAQYAAACKAAAAWgAAAGYAAAD0AAAARwAAAF4AAADRAAAASAAAAHkAAACAAAAA+AAAAGoAAABXAAAASwAAAQwAAABQAAAAVwAAAGoAAAEWAAAAnAAAAFkAAABXAAAAcgAAAGoAAABgAAAAQwAAAJUAAABUAAAAaQAAAEkAAADpAAAATAAAAEcAAABDAAABOgAAAGcAAABxAAAAcgAAAO0AAABcAAAAUQAAAFsAAADIAAAAVAAAAGQAAABSAAABNAAAAI0AAABoAAAAXQAAATYAAAB3AAAATQAAAG4AAAEhAAAAdQAAAE4AAABkAAAA9wAAAFcAAABIAAABRQAAAFcAAABFAAAAdwAAANIAAABwAAAAagAAAHIAAAD6AAAArgAAAI8AAACtAAABEwAAALEAAACyAAAAmQAAAUUAAACZAAAAiQAAAKkAAAEyAAAAzAAAAKcAAAC/AAABMgAAAIoAAACEAAAAfwAAAUcAAAB6AAAAVgAAAGcAAAEzAAAAkQAAAHUAAABrAAABEgAAAJIAAAB7AAAAbAAAAPcAAAB2AAAAfAAAAFEAAACdAAAAeQAAAGcAAABBAAAA8QAAAFQAAABLAAAAQgAAATYAAABcAAAARgAAAHgAAAEJAAAAcwAAAH8AAABdAAAArgAAAIcAAABsAAAAiQAAANMAAAB8AAAAhAAAAHgAAAEvAAAAmAAAAGsAAABUAAABUQAAAIgAAABpAAAAkgAAATcAAACJAAAAeQAAAGoAAAFSAAAAbQAAAGYAAABiAAABQAAAAIUAAACEAAAAeQAAARAAAACRAAAAbgAAAIgAAAD1AAAAgQAAAJcAAAD6AAAAZgAAAIkAAACDAAABHgAAAKwAAACpAAAAqgAAAXQAAACCAAAAsAAAAIoAAAFqAAAAxQAAAIQAAACqAAABPAAAAIwAAAB3AAAAZgAAATYAAACZAAAAggAAAFEAAAEeAAAAbQAAAKsAAACUAAAA9gAAAGkAAAB+AAAASwAAAPcAAABfAAAAZAAAAF4AAADTAAAAUQAAAD8AAABJAAAAugAAAHIAAABKAAAAXwAAAJgAAAA5AAAARQAAADsAAAC2AAAAZwAAAEgAAABGAAAA3wAAAFgAAABnAAAAXwAAATIAAACOAAAAdgAAAI0AAAElAAAAhQAAAHoAAACNAAABIAAAAHEAAAB2AAAAYAAAAP8AAACUAAAAawAAAGUAAAEIAAAAZAAAAEwAAABQAAAA3wAAAFgAAABVAAAAQwAAAMkAAABgAAAALwAAAFAAAAC/AAAAYwAAAE4AAABEAAAKEQAAASwAAAB6AAAAXgAAAHcAAAEUAAAAgAAAAGIAAABYAAAA4wAAAEkAAAEnAAAAcAAAAFoAAABRAAABFwAAAIgAAABpAAAAUgAAATgAAACCAAAAdwAAAHYAAAEEAAAAXgAAAGcAAABnAAAA5QAAAGwAAAB2AAAAagAAAPkAAABNAAAAdgAAAE0AAADSAAAAUAAAAGQAAABHAAABGgAAAFgAAABRAAAAcQAAANYAAACDAAAAowAAAGMAAAB2AAAAYQAAAMUAAABFAAAARAAAACQAAAEbAAAAUwAAAE4AAAA3AAAAxAAAAFcAAABfAAAAWQAAAHEAAABMAAAA/wAAAH8AAABIAAAAXAAAANcAAABFAAAARQAAAMMAAABzAAAAPgAAAFwAAADCAAAAOQAAAEoAAABJAAAAuwAAAFAAAAA6AAAARgAAAMYAAABJAAAAOgAAADAAAADXAAAAYQAAADMAAABDAAABCwAAAEwAAABNAAAANgAAAMEAAABnAAAATgAAAF0AAAEZAAAAUgAAAFIAAABHAAABHQAAADcAAAAnAAAAHwAAANUAAAA4AAAAIwAAAB0AAACyAAAAKgAAAB4AAAAcAAAAfQAAACEAAAAbAAAAHAAAAB4AAAAgAAAAGwAAABoAAAAcAAAAHgAAABkAAAAaAAAAHAAAAB4AAAAZAAAAGgAAABwAAAAeAAAAGQAAABoAAAAbAAAAHgAAABkAAAAaAAAAGwAAAB4AAAAZAAAAGgAAABwAAAAcAAAAFHN0Y28AAAAAAAAAAQAAADAAAABidWR0YQAAAFptZXRhAAAAAAAAACFoZGxyAAAAAAAAAABtZGlyYXBwbAAAAAAAAAAAAAAAAC1pbHN0AAAAJal0b28AAAAdZGF0YQAAAAEAAAAATGF2ZjU4LjI5LjEwMA==" type="video/mp4">
Your browser does not support the video tag.
</video>



<a name="11"></a>
## 11 - Congratulations!

You have successfully used Deep Q-Learning with Experience Replay to train an agent to land a lunar lander safely on a landing pad on the surface of the moon. Congratulations!

<a name="12"></a>
## 12 - References

If you would like to learn more about Deep Q-Learning, we recommend you check out the following papers.


* Mnih, V., Kavukcuoglu, K., Silver, D. et al. Human-level control through deep reinforcement learning. Nature 518, 529–533 (2015).


* Lillicrap, T. P., Hunt, J. J., Pritzel, A., et al. Continuous Control with Deep Reinforcement Learning. ICLR (2016).


* Mnih, V., Kavukcuoglu, K., Silver, D. et al. Playing Atari with Deep Reinforcement Learning. arXiv e-prints.  arXiv:1312.5602 (2013).


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```
