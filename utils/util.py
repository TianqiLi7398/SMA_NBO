import numpy as np
from numpy.linalg import inv
import copy

class LinearKF:
    # here actually a standard Linear KF is implemented 
    def __init__(self, _F,_H,_x,_P,_Q,_R, q0 = 1):
        
        self.F=_F
        self.H=_H
        
        self.x_k_k_min=_x
        self.P_k_k_min=_P
        self.Q=_Q
        self.R=_R
        
        self.x_k_k=_x
        self.P_k_k=_P
        
        self.x_dim = _x.shape[0]
        self.z_dim = _H.shape[1]
        
        self.S_k = self.R

        # for Monte Carlo, it needs a feature 
        self.isUpdated = False
        self.Q0 = np.diag([q0, q0, q0, q0])
        self.init = True
    
    def getCurrentState(self) -> np.ndarray:
        return self.x_k_k_min
    
    def getEst(self) -> np.ndarray:
        return np.dot(self.H, self.x_k_k)
    
    def predict(self):
        # KF prediction step 
        self.x_k_k_min = np.dot(self.F, self.x_k_k)
        if not self.init:
            self.P_k_k_min = np.dot(self.F, np.dot(self.P_k_k,self.F.T)) + self.Q
        else:
            self.P_k_k_min = np.dot(self.F, np.dot(self.P_k_k,self.F.T)) + self.Q0
            self.init = False
        
    def update(self, z: list, r: float =-1):
        # posterior update in KF
        if r > 0:
            self.R = np.diag([r, r])
        z = np.matrix(z).T

        self.z_bar = np.dot(self.H, self.x_k_k_min)
        self.z_res = z - self.z_bar
        self.S_k = np.dot(np.dot(self.H, self.P_k_k_min), self.H.T) + self.R
        # Cholesky's method for inverse
        # c = np.linalg.inv(np.linalg.cholesky(self.S_k))
        # inv_S = np.dot(c.T,c)
        inv_S = inv(self.S_k)
        K_k = np.dot(self.P_k_k_min, self.H.T) * inv_S
        self.x_k_k = self.x_k_k_min + np.dot(K_k, self.z_res)
        self.P_k_k = np.dot(np.eye(self.x_dim) - np.dot(K_k, self.H), self.P_k_k_min)

        self.isUpdated = True

class dynamic_kf(LinearKF):

    def __init__(self, _F,_H,_x,_P,_Q,_R, r0, quality = 1.0):
        LinearKF.__init__(self, _F,_H,_x,_P,_Q,_R)
        # this alpha factor is a scalar for R matrix for sepcific sensor
        # which can be regarded as sensing quality
        self.alpha = quality
        self.base_R = copy.deepcopy(self.R)
        self.r0 = r0

    def cal_R(self, z, xs):
        # xs: sensor pos, = [x, y, theta]
        # z: measurement, [x, y] 
        dx = z[0] - xs[0]
        dy = z[1] - xs[1]
        theta = np.arctan2(dy, dx) - xs[2]
        r = max(self.r0, np.sqrt(dx**2 + dy**2))
        G = np.matrix([[np.cos(theta), -np.sin(theta)], 
                        [np.sin(theta), np.cos(theta)]])
        M = np.diag([0.1 * r, 0.1 * np.pi * r])
        R = self.alpha * np.dot(np.dot(G, M), G.T)
        
        # if np.trace(R) < np.trace(self.base_R):
        #     R = self.base_R
        
        return R

    def update(self, z, xs, alpha):
        self.alpha = alpha  # TODO adapt centralized and remove this step
        self.R = self.cal_R(z, xs)
        z = np.matrix(z).T
        self.z_bar = np.dot(self.H, self.x_k_k_min)
        self.z_res = z - self.z_bar
        self.S_k = np.dot(np.dot(self.H, self.P_k_k_min), self.H.T) + self.R
        # Cholesky's method for inverse
        # c = np.linalg.inv(np.linalg.cholesky(self.S_k))
        # inv_S = np.dot(c.T,c)
        inv_S = inv(self.S_k)
        K_k = np.dot(self.P_k_k_min, self.H.T) * inv_S
        self.x_k_k = self.x_k_k_min + np.dot(K_k, self.z_res)
        self.P_k_k = np.dot(np.eye(self.x_dim) - np.dot(K_k, self.H), self.P_k_k_min)

        self.isUpdated = True

class measurement:
    def __init__(self, z, id_):
        self.value = z
        self.track = []
        self.g_ij = []
        self.table = {"track": self.track, "g_ij": self.g_ij}
        self.id = id_
        
    def inside_track(self, track_id):
        self.track.append(track_id)

class track:
    """track should have the function of initilization, confirmation, deletion, save all tracks"""
    #     initalized -> confirmed -> deleted
    #        |
    #        V
    #     abandoned
    def __init__(self, t0, id_, kf, DeletionThreshold, ConfirmationThreshold,
        memory_len = 10,
        isForeign = False):
        self.id = id_
        self.measurement = []
        self.confirmed = False
        self.deleted = False     # if deleted after confirmed, this deleted is true
        self.kf = kf
        self.t0 = t0
        self.history = [1]
        self.record = {"points": [], "time_interval": [t0]}
        self.abandoned = False   # if not get confirmed, this abandoned is true 
        self.S = self.kf.S_k
        self.pred_z = self.kf.x_k_k
        self.ConfirmationThreshold = ConfirmationThreshold
        self.DeletionThreshold = DeletionThreshold
        self.bb_box_size = [10, 10] # default size for bounding box
        self.agent_id = -1
        self.neighbor = []
        self.memory = {}
        self.memory_len = memory_len
        self.isForeign = isForeign
        
    def track_memory(self, agent_id, sub_id):
        # accumulate memory with fiex horizon
        # TODO
        # Q: if we need to save memory for non-matching siuation
        # A: I don't like that.... that's just trival memories
        try:
            if len(self.memory[agent_id]) == self.memory_len:
                self.memory[agent_id].pop(0)
                self.memory[agent_id].append(sub_id)
            else:
                self.memory[agent_id].append(sub_id)
        except:
            self.memory[agent_id] = [sub_id]

    def update(self, t, kf, isObs):
        
        self.kf = kf
        self.kf.predict()
        
        self.record["points"].append(self.kf.getEst())
        
        if self.confirmed:
            # check if there is observation inside the gate
            if isObs:
                self.history.append(1)
                
            else:
                self.history.append(0)
            
            
            if len(self.history) > self.DeletionThreshold[1]:
                self.history.pop(0)
                    
            if len(self.history) == self.DeletionThreshold[1] and sum(self.history) < self.DeletionThreshold[1] - self.DeletionThreshold[0]:
                # reaches the threshold to delete: self.DeletionThreshold[0] is number of missings allowed
                self.deletion(t)
        else:
            # check if there is observation inside the gate
            if isObs:
                self.history.append(1)
                
            else:
                self.history.append(0)
                
            # when enough observation is done, check the N/M
            if len(self.history) == self.ConfirmationThreshold[1]:
                if sum(self.history) < self.ConfirmationThreshold[0]:
                    # reaches the threshold to abandon, ow, confirm
                    self.abandoned = True
                else:
                    self.confirmation()
                    # print(self.record["points"], self.history, self.id)
    
    def confirmation(self):
        self.confirmed = True
        
    
    def deletion(self, t):
        self.deleted = True
        self.record["time_interval"] = [self.t0, t] 
        # if t > 15:
        #     print(self.DeletionThreshold)
        #     raise RuntimeError("deleted early")
    
    def get_measurement(self, a_list):
        # here a_list is np.ndarray data type from np.where function
        self.measurement = a_list.tolist()
        return

class vertex:

    def __init__(self, track_id):
        # self.agent_id = agent_id
        self.track_id = track_id
        self.neighbor = []
        self.memory = []