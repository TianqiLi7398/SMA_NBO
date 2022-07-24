import numpy as np 
from scipy.optimize import linear_sum_assignment   # Hungarian alg, minimun bipartite matching


class ospa:
    @staticmethod
    def dist(x, y):
        dx = x[0] - y[0]
        dy = x[1] - y[1]
        return np.sqrt(dx**2 + dy**2)

    @staticmethod
    def pairing(traj_k, est_k):
        # find the set with minimun track number
        if len(traj_k) <= len(est_k):
            X, Y = traj_k, est_k
            
        else:
            X, Y = est_k, traj_k
        # m <= n
        m, n = len(X), len(Y)
        matrix = np.full((m, n), 0.0)
        for i in range(m):
            x = X[i]
            for j in range(n):
                y = Y[j]
                
                matrix[i,j] = ospa.dist(x,y)
        
        result = linear_sum_assignment(matrix)
        return X, Y, result

    @staticmethod
    def metrics(traj_k, est_k, c, p):
        # ospa metric for multi target tracking from 
        # Schuhmacher, Dominic, Ba-Tuong Vo, and Ba-Ngu Vo. "A consistent metric for performance evaluation of multi-object filters." 
        # IEEE transactions on signal processing 56.8 (2008): 3447-3457.
        X, Y, result = ospa.pairing(traj_k, est_k)
        row_ind = result[1]
        sum_ = 0.0
        m, n = len(X), len(Y)
        np.testing.assert_array_almost_equal(m, len(row_ind))
        
        for i in range(m):
            x = X[i]
            y = Y[row_ind[i]]
            sum_ += min(c, ospa.dist(x, y))**p
        
        sum_ += c**p * (n-m)

        return (sum_ / n)**(1./p)
    
    @staticmethod
    def metric_sep(traj_k, est_k, c, p):
        # ospa metric for multi target tracking from 
        # Schuhmacher, Dominic, Ba-Tuong Vo, and Ba-Ngu Vo. "A consistent metric for performance evaluation of multi-object filters." 
        # IEEE transactions on signal processing 56.8 (2008): 3447-3457.
        X, Y, result = ospa.pairing(traj_k, est_k)
        row_ind = result[1]
        sum_ = 0.0
        m, n = len(X), len(Y)
        np.testing.assert_array_almost_equal(m, len(row_ind))
        
        for i in range(m):
            x = X[i]
            y = Y[row_ind[i]]
            sum_ += min(c, ospa.dist(x, y))**p
        
        car_penalty =  (n-m)

        return sum_/n, car_penalty
    

    @staticmethod
    def metric_counting_missing(traj_k, est_k, c, p):
        # ospa metric for multi target tracking from 
        # Schuhmacher, Dominic, Ba-Tuong Vo, and Ba-Ngu Vo. "A consistent metric for performance evaluation of multi-object filters." 
        # IEEE transactions on signal processing 56.8 (2008): 3447-3457.
        X, Y, result = ospa.pairing(traj_k, est_k)
        row_ind = result[1]
        missing_num = 0
        
        sum_ = 0.0
        m, n = len(X), len(Y)           # n >= m
        # np.testing.assert_array_almost_equal(m, len(row_ind))
        errors_k = [c] * len(traj_k)
        if len(traj_k) <= len(est_k):
            # x is traj
        
            for i in range(m):
                x = X[i]
                y = Y[row_ind[i]]
                error = ospa.dist(x, y)
                if c < error:
                    missing_num += 1
                    sum_ += c**p
                else:
                    sum_ += error**p
                    errors_k[i] = error
        
        else:
            # y is traj_k
            for i in range(m):
                x = X[i]
                y = Y[row_ind[i]]
                error = ospa.dist(x, y)
                if c < error:
                    missing_num += 1
                    sum_ += c**p
                else:
                    sum_ += error**p
                    errors_k[row_ind[i]] = error
        # penalty of extra trajectories
        sum_ += c**p * (n-m)
        car_penalty =  n-m
        
        # print(n, m, sum_, (sum_ / n), (sum_ / n)**(1./ p))
        return  (sum_ / n)**(1./p), car_penalty, errors_k

        
    @staticmethod
    def sort_tracks(traj_k, est_k, c):
        # ospa metric for multi target tracking from 
        # Schuhmacher, Dominic, Ba-Tuong Vo, and Ba-Ngu Vo. "A consistent metric for performance evaluation of multi-object filters." 
        # IEEE transactions on signal processing 56.8 (2008): 3447-3457.
        X, Y, result = ospa.pairing(traj_k, est_k)
        row_ind = result[1]
        col_ind = result[0]

        m, n = len(X), len(Y)
        errors_k = [c] * len(traj_k)
        
        if len(traj_k) <= len(est_k):
            # x is true value
        
            for i in range(m):
                x = X[i]
                y = Y[row_ind[i]]
                error = ospa.dist(x, y)
                if error < c:
                    errors_k[i] = error
        
        else:
            # y is true value
            for i in range(m):
                x = X[i]
                y = Y[row_ind[i]]
                error = ospa.dist(x, y)
                if error < c:
                    
                    errors_k[row_ind[i]] = error
        
        return  errors_k

