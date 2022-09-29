import numpy as np
import matplotlib.pyplot as plt #for plotting
import os, json
from utils.occupancy import occupancy
from utils.sample import sampling

def generate_xy(a, b, x0, y0, lambda0, r):
 

    areaTotal= a*b
    # circles without overlaps
    x_list, y_list = sampling.circles_no_overlap(a, b, x0, y0, lambda0*areaTotal, r)

    return x_list, y_list

def main():
    repeat_time = 50
    radius = 5
    lambda0 = 3e-3
    dx = 0.1
    location = os.path.join(os.getcwd(), 'data', 'env', 'poisson')
    canvas = [[-40, 100], [-20, 80]]
    x0, y0 = 0.5 * sum(canvas[0]), 0.5 * sum(canvas[1])
    a, b = canvas[0][1] - canvas[0][0], canvas[1][1] - canvas[1][0]
    

    # load trajectories of targets
    filename = os.path.join(os.getcwd(), "data", 'env', "ParkingTraj2.json")
    # filename = os.path.join(path, "data", 'env', "ParkingTraj_straight.json")
    with open(filename) as json_file:
        traj = json.load(json_file)
    
    # convert trajectories to plot points
    traj_occupancy = []
    for target in range(len(traj[0])):
        target_traj = [[], []]
        for i in range(len(traj[0][target])):
            row, col = occupancy.xy2rowcol(traj[0][target][i], traj[1][target][i], canvas, dx)
            target_traj[1].append(row)
            target_traj[0].append(col)
        traj_occupancy.append(target_traj)

    for i in range(repeat_time):
        filename = os.path.join(location, 'r_' + str(radius) + '_lambda_' + str(lambda0) + '_' + str(i) + '.json')
        x_list, y_list = generate_xy(a, b, x0, y0, lambda0, radius)
        discrete_map = occupancy.occupancy_map(canvas, x_list, y_list, dx, radius).tolist()
        a_map = {"canvas": canvas, 
                "radius": radius,
                "lambda": lambda0,
                "centers": [x_list, y_list],
                "occupancy": discrete_map}
        with open(filename, 'w') as outfiles:
            json.dump(a_map, outfiles, indent=4)
        # generate fig
        fig = plt.figure()
        ax = fig.add_subplot(111, aspect='equal')
        ax.matshow(discrete_map, cmap=plt.cm.Blues)
        # add the trajectories
        for target in range(len(traj[0])):
            ax.plot(traj_occupancy[target][0], traj_occupancy[target][1])
        plt.savefig(os.path.join(os.getcwd(), 'pics', 'poisson_map', 'r_' + str(radius) + '_lambda_' + str(lambda0) + '_' + str(i) + '.png'))

if __name__ == "__main__":
    main()
