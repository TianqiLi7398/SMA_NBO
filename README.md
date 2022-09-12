SMA-NBO: A Sequential Multi-Agent Target Tracking Method

Please check the official implementation of the Paper:
T. Li, L. Krakow and S. Gopalswamy, "[SMA-NBO: A Sequential Multi-Agent Planning with Nominal Belief-State Optimization in Target Tracking
](https://arxiv.org/abs/2203.01507)" (To be Appear) in Proceedings of IEEE IROS 2022

# Multiagent-Target-Planning

To run this package, please use python3 version

`$ python3 main.py` 

## Agent Dropout

The dropout of agent defined as

| dropout_pattern | explanation                                                                                                               | p                            |
|-----------------|---------------------------------------------------------------------------------------------------------------------------|------------------------------|
| fail_publisher  | every agent has the p prob to fail sending its decision  at one time-step, but still receive other agents plan temporally | p: prob of publisher failure |
| fail_channel    | every channel (i, j) has p prob to fail sending  their decision to each other temporally                                  | p: prob of channel failure   |
| breakdown       | One agent will breakdown and make no contribution  to the network after a time t_bd                                       | t_bd                         |