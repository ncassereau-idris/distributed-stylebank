#!/bin/bash
#SBATCH --job-name=monet
#SBATCH --output=slurm.out
#SBATCH --error=slurm.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --time=00:10:00
#SBATCH -C v100-32g
#SBATCH --gres=gpu:4
#SBATCH --account=sos@gpu
#SBATCH --qos=qos_gpu-dev
#SBATCH --cpus-per-task=10
#SBATCH --hint=nomultithread

cd ${SLURM_SUBMIT_DIR}

module purge
module load pytorch-gpu/py3/1.10.0

set -x

echo "START TIME: $(date)"

srun python3 main.py hydra/launcher=submitit_slurm

echo "END TIME: $(date)"
