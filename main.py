import os
from utils.simulator import simulator
import time
import json
import utils.analysis
import utils.mrs_analysis
import argparse
import run_sim, analysis

def args_def() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="MTT simulation")

    parser.add_argument("--task", 
        help="define the task in ['run_trials', 'simulation', '']")

    # general parameters
    parser.add_argument("--horizon", type=int, default=5, 
        help="Horizon steps in non-myopic planning")
    parser.add_argument("--deci-Schema", type=str, default='sma', 
        help="Multi-agent decision schema in [cen, sma, decPOMDP]")
    parser.add_argument("--repeat-time", type=int, default=1,
        help="Times of repeatition over same map")

    # simulation map parameter
    parser.add_argument("--case", type=str, default='poission', 
        help="the map type in simulation")
    parser.add_argument("--useSemantic", type=bool, default=True,
        help="if use map information in planning")
    
    
    parser.add_argument("--lambda0", type=float, default=5e-3,
        help="the density of occlusions, unit in num/m^2")
    parser.add_argument("--r", type=float, default=5.0,
        help="radius of the occlusion, unit in m")
    
    # trajectory type
    parser.add_argument("--traj-type", type=str, default="normal",
        help="trajectory type, in set ['straight', 'normal'. 'static']")
    
    # parameters in objective function
    parser.add_argument("--info-gain", type=str, default="trace_sum",
        help="the objective in objective function, in set\
            ['trace_sum', 'info_gain', 'info_acc', 'log_det', 'cov_gain']")

    # parameters in belief state
    parser.add_argument("--domain", type=str, default='nbo', 
        help="Belief state approximation in [nbo, MonteCarlo]")
    parser.add_argument("--MCSnum", type=int, default=50, 
        help="Monte Carlo sample number")

    parser.add_argument("--ckf", type=bool, default=True,
        help="whether to use centralized KF in planning horizons")
    parser.add_argument("--wtp", type=bool, default=False,
        help="whether to add the multiple weighted trace penalty term at end of horizon")
    
    # optimization parameters in PSO
    parser.add_argument("--ftol", type=float, default=5e-4,
        help="ftol in PSO, 5e-4 for decentralized, 2.5e-4 for centralized")
    parser.add_argument("--gtol", type=int, default=50,
        help="gtol in PSO, 50 for decentralized, 75 for centralized")
    parser.add_argument("--optmethod", type=str, default="pso",
        help="optimization methods in planning")
    
    # specific for rollout schema
    parser.add_argument("--opt-step", type=int, default=1,
        help="actual optimization step in rollout algorithm")
    
    # for repeated experiments over same map
    parser.add_argument("--lambda0-list", nargs='+', type=float,
        help="list of lambda0 if want to run simulation in multiple maps")
    
    parser.add_argument("--r-list", nargs='+', type=float,
        help="list of r if want to run simulation in multiple maps")
    
    parser.add_argument("--horizon-list", nargs='+', type=int,
        help="list of horizon if want to run simulation in multiple maps")

    '''
    repeated: run repeated times on same map
    iteration: change the index of random possion map
    '''
    
    parser.add_argument("--iteration", type=int, default=0,
        help="the index of random maps in possion map case")
    
    parser.add_argument("--repeated", type=int, default=-1,
        help="num of repeatition over same map")
    
    # arguments for analysis
    
    parser.add_argument("--deci-Schema-list", type=float, 
        help="different deci-Schema to check")
    
    parser.add_argument("--repeated-times", type=int, default=1,
        help="total num of repeatition over same map to analyze")
    

    args = parser.parse_args()
    return args

def main(args: argparse.Namespace):
    

    if task == 'run':
        run_sim.run(args)
            
    elif task == 'freq_analysis':
        analysis.freq_analysis(args)
    elif task == 'time_series_ospa':
        analysis.time_series_ospa(args)

    
    else:
        print("please indicate the task")
    
    '''
    rename files https://www.networkworld.com/article/3433865/how-to-rename-a-group-of-files-on-linux.html
    '''



           
if __name__ == "__main__":
    # test_email()
    task = 'ospa_analysis'  # 'ospa_analysis', 'run', 'freq_analysis', 'time_series_ospa'
    main(task)