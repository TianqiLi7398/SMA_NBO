from utils.simulator import simulator
import argparse
import time

def run(args: argparse.Namespace):

    horizon_list, lambda0_list, r_list = [], [], []
    
    horizon_list = args.horizon_list if args.horizon_list else [args.horizon]
    r_list = args.r_list if args.r_list else [args.r]
    lambda0_list = args.lambda0_list if args.lambda0_list else [args.lambda0]

    for horizon, lambda0, r in [(horizon, lambda0, r) for horizon in horizon_list\
            for lambda0 in lambda0_list for r in r_list]:
        run_trials(args, horizon, lambda0, r)


def run_trials(
        args: argparse.Namespace, 
        horizon: int,
        lambda0: float, 
        r: float, 
        dropout_pattern: bool = None, 
        dropout_prob: float = 0.0, 
    ):

    '''
    Trigger the simulation based on the hyperparameters
    '''
    
    
    start_time = time.time()
    if args.domain == 'nbo':
        if args.deci_Schema == 'cen':
            simulator.NBO_central(args.iteration, horizon, args.ftol, args.gtol, args.wtp, args.case, args.useSemantic, central_kf=args.ckf, 
                    optmethod=args.optmethod, lambda0 = lambda0, r = r, traj_type = args.traj_type, coverfile = False, 
                    info_gain = args.info_gain, repeated = args.repeated)
            
        elif args.deci_Schema == 'sma':
            
            simulator.SMA_NBO(args.iteration, horizon, args.ftol, args.gtol, args.wtp, args.case, args.useSemantic, central_kf=args.ckf, 
                    optmethod=args.optmethod, lambda0 = lambda0, r = r, traj_type = args.traj_type, coverfile = True, 
                    info_gain = args.info_gain, repeated = args.repeated)
            
        elif args.deci_Schema == 'pma':
            if args.optmethod == 'pso':
                simulator.PMA_NBO(args.iteration, horizon, args.ftol, args.gtol, args.wtp, args.case, args.useSemantic, central_kf=args.ckf, 
                        optmethod=args.optmethod, lambda0 = lambda0, r = r, traj_type = args.traj_type, coverfile = False, 
                        info_gain = args.info_gain, repeated = args.repeated, dropout_pattern=dropout_pattern, 
                        dropout_prob=dropout_prob)
            else: raise RuntimeError("optmethod can only be pso so far")
        elif args.deci_Schema == 'decPOMDP':
            simulator.decPOMDP_NBO(args.iteration, horizon, args.ftol, args.gtol, args.wtp, args.case, args.useSemantic, central_kf=args.ckf, 
                    optmethod=args.optmethod, lambda0 = lambda0, r = r, traj_type = args.traj_type, coverfile = True, 
                    info_gain = args.info_gain, repeated = args.repeated)
        elif args.deci_Schema == 'test':
            simulator.test_NBO(args.iteration, horizon, args.ftol, args.gtol, args.wtp, args.case, args.useSemantic, central_kf=args.ckf, 
                    optmethod=args.optmethod, lambda0 = lambda0, r = r, traj_type = args.traj_type, coverfile = True, 
                    info_gain = args.info_gain, repeated = args.repeated)
        else:
            raise RuntimeError('%s_NBO not defined!' % args.deci_Schema)


    elif args.domain == 'MonteCarloRollout':
        if args.deci_Schema == 'cen':
            simulator.MCRollout_central(args.iteration, horizon, args.ftol, args.gtol, args.MCSnum, args.wtp, central_kf=args.ckf, 
                    optmethod=args.optmethod, leader = 0, repeated = args.repeated)
        elif args.deci_Schema == 'sma':
            simulator.MCRollout_distributed(args.iteration, horizon, args.ftol, args.gtol, args.MCSnum, args.wtp, args.case, central_kf=args.ckf, 
                    optmethod=args.optmethod, lambda0 = lambda0, r = r, traj_type = args.traj_type, coverfile = False, 
                    info_gain = args.info_gain, repeated = args.repeated)
        else:
            raise RuntimeError('%s + MonteCarloRollout not defined!' % args.deci_Schema)
    else:
        print("Please indicate the domain parameter, nbo or mcrollout")
    print("Run time for %sth optimization: %s sec"%(args.iteration, time.time() - start_time))
    
            