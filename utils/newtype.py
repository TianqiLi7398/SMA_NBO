'''
The Nominal Belief Optimization (NBO) method for multisensor target tracking with discrete action space

[1]     Miller, Scott A., Zachary A. Harris, and Edwin KP Chong. "A POMDP framework for coordinated 
        guidance of autonomous UAVs for multitarget tracking." EURASIP Journal on Advances in 
        Signal Processing 2009 (2009): 1-17.
'''
import copy
import numpy as np
from collections import namedtuple
from utils.nbo import nbo_agent