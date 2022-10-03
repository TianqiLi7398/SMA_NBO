# SMA-NBO: A Sequential Multi-Agent Target Tracking Simulator

We'd like to share our simulator of multi-sensor target tracking, and our most recent research in cooperative multi-UAV target tracking.

For our proposed algorithm SMA-NBO, please refer to our paper:

>T. Li, L. W. Krakow and S. Gopalswamy, "[SMA-NBO: A Sequential Multi-Agent Planning with Nominal Belief-State Optimization in Target Tracking
](https://arxiv.org/abs/2203.01507)" (To be Appear) in Proceedings of IEEE IROS 2022

## Functionality

This code base contains the following functionality

### Prerequisite

Please check the [prerequisite readme](https://github.com/TianqiLi7398/SMA_NBO/blob/main/data/env/prerequisite.md) for details.

### 1. Simulation of Multi-UAV tracking

**Monte Carlo - Multi-Agent Rollout (MCR)**


**Sequential Multi-Agent - Nominal Belief Optimization (SMA-NBO)**

```
isParallel=False, IsDistriOpt=True, 
```

**Centralized - NBO**

```
isParallel=False, IsDistriOpt=False,
```

**Decentralized POMDP (Dec-POMDP)**
```
isParallel=True, IsDistriOpt=False,
```

**PMA-NBO**
```
isParallel=True, IsDistriOpt=True, 
```

need to write this out

ok, this is from dev repo

### 2. Visualization

`$ python3 visual.py`

### 3. Evaluation

## Dependencies



## Multiagent-Target-Planning

To run this package, please use python3 version

`$ python3 main.py` 
