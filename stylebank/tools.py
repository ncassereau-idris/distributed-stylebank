# /usr/bin/env python3
# -*- coding: utf-8 -*-

from datetime import timedelta
import os
# from pathlib import Path
import hostlist

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

def format_duration(seconds):
    return str(timedelta(seconds=int(seconds)))