import os
import os.path as osp
import shutil
import sys
import numpy as np
from datetime import datetime
from glob import glob
from itertools import chain
import gc
import torch

def mem_info():
    import subprocess
    dev = subprocess.check_output(
        "nvidia-smi | grep MiB | awk -F '|' '{print $3}' | awk -F '/' '{print $1}' | grep -Eo '[0-9]{1,10}'",
        shell=True)
    dev = dev.decode()
    dev_mem = list(map(lambda x: int(x), dev.split('\n')[:-1]))
    return dev_mem

def random_int(obj=None):
    return (id(obj) + os.getpid() + int(datetime.now().strftime("%Y%m%d%H%M%S%f"))) % 4294967295

def cmd(command):
    import subprocess
    output = subprocess.check_output(command, shell=True)
    output = output.decode()
    return output

def reset_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True

