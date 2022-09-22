from collections import namedtuple

class Single_track:
    def __init__(self):
        self.track_id = 0

        # mean, which is x
        # float64[4] 
        self.x = []


        # information vector, which is q = Sigma*x
        # float64[4] 
        self.q = []

        # P matrix 
        # float64[16] 
        self.P = []

        # Sigma matrix, which is P^-1
        # float64[16] 
        self.Sigma = []

        # bool 
        self.isObs = False
    
    def __str__(self):
        return 'track ID: {id}, x: {mean}, P: {cov}, isObs: {isObs}'.format(id=self.track_id, 
            mean=self.x, cov=list(self.P[i] for i in [0, 5, 10, 15]), isObs=self.isObs)


class Info_sense:
    def __init__(self):
        # int8 
        self.id = -1

        # iteration time
        # int8 
        self.l = 0

        # Single_track[] 
        self.tracks = []
    
    def __str__(self):
        a = 'id: {id}, iter: {iter}, '.format(id=self.id, iter=self.l)
        for track in self.tracks:
            b = track.__str__()+'\n'
            a += b
        return a

class Agent_basic:
    def __init__(self):
        # int8 
        self.id = -1
        # float64 
        self.x = 0.0
        # float64 
        self.y = 0.0
        # float64
        self.theta = 0.0

class Planning_msg:
    def __init__(self, agent_number: int):
        # init a hashmap to save all agents plans over horizon
        self.Intention = namedtuple('Intention', ['time', 'action_vector'])
        self.u = {}
        for i in range(agent_number):
            non_intention = self.Intention(-1, [])
            self.u[i] = non_intention
    
    def miss(self) -> list:
        # return the id of agents who does not share the policy
        miss_list = []
        for key in self.u.keys():
            if len(self.u[key]) == 0:
                miss_list.append(key)
        return miss_list

    def clean(self):
        # we simulate the communication process, thus we assume the 
        # channel only saves the temporal info
        for key in self.u.keys():
            self.u[key] = []
    
    def push(self, action_vector: list, t: float, agent_id: int):
        new_intent = self.Intention(t, action_vector)
        self.u[agent_id] = new_intent