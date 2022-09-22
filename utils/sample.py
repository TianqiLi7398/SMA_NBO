import numpy as np 
from scipy.stats import levy
from scipy.optimize import fsolve

class sampling:
    @staticmethod
    def levy_walk(n: int, minstep: float, maxstep: float) -> np.array:
        # sample a levy distributed trajectory

        min_array, max_array = np.ones(n)*minstep, np.ones(n)*maxstep
        r = levy.rvs( size=n )/10
        r = np.maximum(np.minimum(max_array,r), min_array)
        return r

    @staticmethod
    def f_list(x: list, coeff: list) -> list:
        # generate a list of polynomial trajectories based on input x
        y_ = []
        for xj in x:
            y_.append(sampling.poly(xj, coeff))
        return y_

    @staticmethod
    def poly(x: float, coeff: list) -> float:
        # polynomial function
        yj = 0
        for i in range(len(coeff)):
            yj += coeff[i]*x**i
        return yj

    @staticmethod
    def f2solve(z: list, coeff: list, r: float, z0: list) -> list:
        x, y = z
        x0, y0 = z0
        f1 = y - sampling.poly(x, coeff)
        f2 = (x-x0)**2 + (y-y0)**2 - r**2
        return [f1, f2]

    @staticmethod
    def discretize_levy(coeff, x0, y0, n, minstep, maxstep, direction):
        x_, y_ = [x0], [y0]
        r_ = sampling.levy_walk(n, minstep, maxstep)
        for r in r_:
            z0 = [x0 + direction*0.1, y0 + direction*0.1]
            x, y = fsolve(sampling.f2solve, z0, args=(coeff, r, [x0, y0]))
            x_.append(x)
            y_.append(y)
            x0, y0 = x, y
        return x_, y_, r_
    
    @staticmethod
    def euclidan_dist(p1: list, p2: list) -> float:
        return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
    
    @staticmethod
    def circles_no_overlap(
            a: float, 
            b: float, 
            x0: float, 
            y0: float, 
            num: int, 
            r: float
        ) -> tuple:
        
        x_list, y_list = [], []
        
        while num > 0:
            # find circle
            x = np.random.uniform(-0.5 * a + x0, 0.5 * a + x0)
            y = np.random.uniform(-0.5 * b + y0, 0.5 * b + y0)
            isgood = True

            for i in range(len(x_list)):
                if sampling.euclidan_dist([x, y], [x_list[i], y_list[i]]) < 2 * r:
                    isgood = False
                    break
            
            if isgood:
                x_list.append(x)
                y_list.append(y)
                num -= 1
        
        return x_list, y_list


