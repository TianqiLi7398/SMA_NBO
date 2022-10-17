# Prerequisite

To successfully run a multi-target multi-sensor target tracking simulation, we need to define the following specifications

## Sensor Parameters

One good example is file [`parkingSensorPara.json`](https://github.com/TianqiLi7398/SMA_NBO/blob/main/data/env/parkingSensorPara.json).
Sensors parameter is contained in subdict `sensors`

```
"shape": [shape, size] of sensor's FoV
"dm": distance of heurisitic base policy
"r0": minimal distance in noise difference
"v": maximal speed of robot
"color": color in visualization 
"position": initial position [x, y, theta]
"r": 0.1
"quality": quality scalar used in observation noise covariance matrix
"dc": maximum distance of connectivity with other sensors
"d0": maximun distance to match 2 trajectories from 2 sensors estimatio;,
"ConfirmationThreshold": [2, 3] 2 out of 3 observations to confirm a new target
"DeletionThreshold": [5, 5] 5 missings out of 5 observations to cinfirm a target deletion
```

## Trajectory Generation

You can use [`createTraj.py`](https://github.com/TianqiLi7398/SMA_NBO/blob/main/createTraj.py) to generate trajectories of the targets. 
The example traj used is [`ParkingTraj2.json`](https://github.com/TianqiLi7398/SMA_NBO/blob/main/data/env/ParkingTraj2.json) in our simulations.

A pic of what trajectory looks like.

## Occlusion Map Generation

You can use [`poisson_map.py`](https://github.com/TianqiLi7398/SMA_NBO/blob/main/poisson_map.py) to generate your own map. 
One example file for map generation is in folder [`data/env/poisson`](https://github.com/TianqiLi7398/SMA_NBO/tree/main/data/env/poisson), which contains random generated occlusion maps with parameter of circular occlusion $(\lambda, R)$.


- $\lambda$: density of the occlusion, # of uncertainty per $m^2$
- $R$: radius of the occlusion, unit in $m$

