#!/usr/bin/env python3
import argparse
import numpy as np
import os
import os.path as osp
import sys

from env_utils import *

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path1', type=str, dest='path1')
    parser.add_argument('--path2', type=str, dest='path2')
    parser.add_argument('--to', type=str, dest='output_path', default=None)
    args = parser.parse_args()

    return args

if __name__ == '__main__':
    args = parse_args()

    output_path = args.output_path

    make_dir(output_path)

    for i in range(8000):
        path1 = osp.join(args.path1, '%06d.txt' % i)
        path2 = osp.join(args.path2, '%06d.txt' % i)
        opath = osp.join(output_path, '%06d.txt' % i)
        if not os.path.exists(path1): continue
        print(i)
        with open(path1) as f:
            obj_str = f.read()
        with open(path2) as f:
            obj_str2 = f.read()

        with open(opath, 'w') as f:
            f.write(obj_str)
            f.write(obj_str2)
