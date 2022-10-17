# SMA-NBO: A Sequential Multi-Agent Target Tracking Simulator

We'd like to share our simulator of multi-sensor multi-target tracking (MMT), and our most recent research in cooperative multi-UAV target tracking.

| ![doc/Behavior%20of%20SMA.gif](https://github.com/TianqiLi7398/SMA_NBO/blob/main/doc/Behavior%20of%20SMA.gif) | 
|:--:| 
| *Multi-UAV multi-target tracking* |

**Scenario**: a multi-UAV team works in an area of interest (AOI) to track the state of our object of interest (OOI). 
There are *occlusions*(blue circles) over the AOI, which is regarded as semantic map information in planning algorithm.
Every UAV has its field of view (FOV), like the rectangle projected in RGB camera.
Every OOI's estimation is the distribution being updated via Bayesian principle, i.e. Kalman Filter.

The **key** point we are showing here is a distributed motion planning of this fleet of UAVs for a cooperative behavior in the MMT task.

For our proposed algorithm SMA-NBO, please refer to our paper:

>T. Li, L. W. Krakow and S. Gopalswamy, "[SMA-NBO: A Sequential Multi-Agent Planning with Nominal Belief-State Optimization in Target Tracking
](https://arxiv.org/abs/2203.01507)" (To be Appear) in Proceedings of IEEE IROS 2022

## Functionality

This code base of MMT simualtione contains following functionality

### Prerequisite & Dependencies

Please check the [prerequisite readme](https://github.com/TianqiLi7398/SMA_NBO/blob/main/data/env/prerequisite.md), which includes the preparation of sensors' parameters, maps' configuration and trajectories of targets.

All scripts are generated base on Python 3.6+, you can resolve the library dependency via 

`$ pip install -r requirement.txt`

### main file

All parameters are defined in [`main.py`](https://github.com/TianqiLi7398/SMA_NBO/blob/main/main.py). 
One example of parameter configuration is [`start.sh`](https://github.com/TianqiLi7398/SMA_NBO/blob/main/start.sh) file.

The following funcations can be realized via changing the `--task=` in [`start.sh`](https://github.com/TianqiLi7398/SMA_NBO/blob/main/start.sh#L3).


### 1. Simulation of Multi-UAV tracking

`--task=run`

As explained in our paper, this simulation contains 3 multi-agent decision making schemes {SMA, dec-POMDP, Centralized} and 2 receding horizon planning methods {Nominal Belief Optimization (NBO), Monte Carlo Rollout (MCR)}.
o trigger one simulation, change the 
To trigger one simulation, change the

Our simulation contains the following cases

|     |       SMA          |       dec-POMDP    |     Centralized   |
|-----|--------------------|--------------------|-------------------|
| NBO | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark:|
| MCR | :heavy_check_mark: |                    | :heavy_check_mark:|


### 2. Simulation Visualization

`--task=visualize`

Once simulation is completed, a `.json` file will be generated as the record. The simulation can be visualized based on this record as the MTT scenario above.


### 3. Evaluation

`--task=freq_analysis`

Plot the CDF of OSPA metrics ([Beard et. al, 2017](https://ieeexplore.ieee.org/document/8217598)) in MTT over all time steps in simulations configed by `--r-list, --horizon-list, --deci-Schema-list, --horizon-list, --repeated-times, --iteration`.

`--task=time_series_ospa`

Plot the mean and variance of OSPA metrics in MTT at every time steps in simulations configed by `--r-list, --horizon-list, --deci-Schema-list, --horizon-list, --repeated-times, --iteration`.
