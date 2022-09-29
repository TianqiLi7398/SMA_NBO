import argparse
import main

def test_argparse():
    args = main.args_def()
    print(args.lambda0_list)

if __name__ == '__main__':
    test_argparse()