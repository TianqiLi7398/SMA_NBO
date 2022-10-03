import utils.mrs_analysis
import argparse


def freq_analysis(args: argparse.Namespace, agentid: int = 0):
    '''
    return with picture and csv files of different algorithms, different 
    map parameters tracking performance (OSPA), you go through every random map
    once (by indexing args.iteration, and args.repeated_num = 0)
    '''

    print("sim number is %s, deci_Schema = %s" % (args.iteration, args.deci_Schema))
    
    horizon_list = args.horizon_list
    lambda0_list = args.lambda0_list
    cen_list = args.deci_Schema_list
    r_list = args.r_list
    paralist = []

    for deci_Schema, horizon, lambda0, r in [(deci_Schema, horizon, lambda0, r) \
        for deci_Schema in cen_list for horizon in horizon_list for lambda0 in lambda0_list \
        for r in r_list]:
            ele = {
                "deci_Schema": deci_Schema, 
                "wtp": args.wtp, 
                "horizon": horizon, 
                "ftol": args.ftol,
                "gtol": args.gtol,
                "optmethod": args.optmethod,
                "MCSnum": args.MCSnum, 
                "env": args.case, 
                "domain": args.domain,
                "lambda0": lambda0, 
                "r": r, 
                "traj_type": args.traj_type, 
                "info_gain": args.info_gain
                }
            paralist.append(ele)
    
    utils.mrs_analysis.error_frequency(paralist, agentid, args.iteration, c=50.0, \
            p=2, start_index = 0 if args.deci_Schema == 'test' else 10)

def time_series_ospa(args: argparse.Namespace, agentid: int = 0):
    
    print("sim number is %s, deci_Schema = %s" % (args.repeated_times, args.deci_Schema))
    
    """ 
    plot the OSPA mean and error bar over the time with repeated_time for a list of algs
    you run simulation over the same map (map index is args.iteration) multiple times (args.repeated_times)
    """

    horizon_list = args.horizon_list
    lambda0_list = args.lambda0_list
    cen_list = args.deci_Schema_list
    r_list = args.r_list

    # fix horzion, lambda as the parameter, compare different algirthms
    for horizon, lambda0, r in [(horizon, lambda0, r) \
        for horizon in horizon_list\
        for lambda0 in lambda0_list \
        for r in r_list]:
            paralist = []
            for deci_Schema in cen_list:
                ele = {
                    "deci_Schema": deci_Schema, 
                    "wtp": args.wtp, 
                    "horizon": horizon, 
                    "ftol": args.ftol,
                    "gtol": args.gtol,
                    "optmethod": args.optmethod,
                    "MCSnum": args.MCSnum, 
                    "env": args.case, 
                    "domain": args.domain,
                    "lambda0": lambda0, 
                    "r": r, 
                    "traj_type": args.traj_type, 
                    "info_gain": args.info_gain
                    }
                paralist.append(ele)
                utils.mrs_analysis.time_series_analysis(paralist, \
                    agentid, args.repeated_times, c=50.0, p=2,
                    start_index = 0 if args.deci_Schema == 'test' else 10)



