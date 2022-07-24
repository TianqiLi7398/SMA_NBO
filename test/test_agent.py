import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.msg import Planning_msg
class test_agent:
    def __init__(self, sensor_num: int, id_: int):
        self.t = 0
        self.id = id_
        self.sensor_num = sensor_num
        self.policy_stack = [[]] * sensor_num
    
    def update_group_decision(self, plan_msg: Planning_msg):
        '''
        input: u with samesize of the output of self.standardNBO(), this function stores the future actions 
        in self.policy_stack for the calculation of Estimation Reward To Go
        '''
        
        miss_list = plan_msg.miss()
        future_policy = []
        for i in range(self.sensor_num):
            if i in miss_list:
                # if i miss, still utlize the previous stack value if we have it
                try:
                    agent_i_policy = self.policy_stack[i][1:]
                except: 
                    agent_i_policy = [] 
            else:
                if plan_msg.u[i].time < self.t:
                    # previous intention
                    agent_i_policy = plan_msg.u[i].action_vector[1:]
                else:
                    agent_i_policy = plan_msg.u[i].action_vector
            future_policy.append(agent_i_policy)
        self.policy_stack = future_policy
    
    def report_intents(self) -> list:
        a_list = []
        for i in range(self.sensor_num):
            a_list.append(len(self.policy_stack[i]))
        return a_list