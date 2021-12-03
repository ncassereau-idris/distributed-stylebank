# /usr/bin/env python3
# -*- coding: utf-8 -*-

from datetime import timedelta
import os
from pathlib import Path
import hostlist
import logging
import fcntl
import numpy as np

# from http://www.idris.fr/jean-zay/gpu/jean-zay-gpu-torch-multi.html

# get SLURM variables
rank = int(os.environ['SLURM_PROCID'])
local_rank = int(os.environ['SLURM_LOCALID'])
size = int(os.environ['SLURM_NTASKS'])
cpus_per_task = int(os.environ['SLURM_CPUS_PER_TASK'])

# get node list from slurm
hostnames = hostlist.expand_hostlist(os.environ['SLURM_JOB_NODELIST'])

# get IDs of reserved GPU
gpu_ids = os.environ['SLURM_STEP_GPUS'].split(",")

# define MASTER_ADD & MASTER_PORT
os.environ['MASTER_ADDR'] = hostnames[0]

# to avoid port conflict on the same node
os.environ['MASTER_PORT'] = str(16785 + int(min(gpu_ids)))

# os.environ["TMPDIR"] = os.environ["JOBSCRATCH"]
# tmpdir = Path(os.environ["JOBSCRATCH"])


log = logging.getLogger(__name__)


class Lock:
    # This lock is system-wide and does not need to be communicated to
    # other MPI processes. They only need to agree on the index
    # associated with the lock, which could be a predefined sequence of
    # integers or a random sequence with the same seed for each process.
    def __init__(self, idx):
        self.lockfile = f"/tmp/monet_{idx}.lock"
        os.system(f"touch {self.lockfile}")

    def __enter__ (self):
        self.fp = open(self.lockfile)
        fcntl.flock(self.fp.fileno(), fcntl.LOCK_EX)

    def __exit__ (self, type, value, traceback):
        fcntl.flock(self.fp.fileno(), fcntl.LOCK_UN)
        self.fp.close()


def format_duration(seconds):
    return str(timedelta(seconds=int(seconds)))


def mkdir(*args):
    path = os.path.join(*args)
    try:
        os.makedirs(path)
    except FileExistsError:  # folder already exists
        pass
    else:
        log.info(f"Subfolder {path} created!")
    finally:
        return Path(path)


def prepare_imgs(imgs):
    imgs = imgs.float().clamp(min=0, max=1)
    imgs = imgs.cpu().numpy().transpose(0, 2, 3, 1)
    imgs *= 255
    return list(imgs.astype(np.uint8))
