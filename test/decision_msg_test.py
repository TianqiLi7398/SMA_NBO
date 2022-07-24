import pytest
import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.msg import Planning_msg
from test_agent import test_agent

def test_sequential_planning_msg():
    sensor_num = 3
    sensor_list = []
    for i in range(sensor_num):
        sensor_list.append(test_agent(sensor_num, i))

    # simulate sequential control
    plan_msg = Planning_msg(sensor_num)

    for t in range(2):
        for i, agent_i in enumerate(sensor_list):
            agent_i.t = t
            agent_i.update_group_decision(plan_msg)
            u_i = [[i, t]] * 3
            plan_msg.push(u_i, t, i)
    
    # output at the end of policy exchange should be 
    intent_dim = {0: [2, 2, 2],
        1: [3, 2, 2],
        2: [3, 3, 2]}
    errors = []
    for i, agent_i in enumerate(sensor_list):
        print(intent_dim[i])
        if agent_i.report_intents() != intent_dim[i]:
            errors.append(i)
    
    assert not errors, "errors occured in agent \n{}".format("\n".join(errors))