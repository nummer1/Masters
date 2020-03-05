#!/bin/sh
#SBATCH --partition=WORKQ
#SBATCH --account=ie-idi
#SBATCH --time=00:20:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --mem=8000
#SBATCH --gres=gpu:0
#SBATCH --job-name="install venv"
#SBATCH --output=debug.out
#SBATCH --mail-user=johanag@ntnu.no
#SBATCH --mail-type=ALL

WORKDIR=${SLURM_SUBMIT_DIR}
cd ${WORKDIR}

module purge
module load fosscuda/2018b

module load GCCcore/.8.3.0
module load Python/3.7.4

module load GCC/8.3.0
module load CUDA/10.1.243
module load OpenMPI/3.1.4
module load TensorFlow/2.1.0-Python-3.7.4

mpirun hostname

# python -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install gym
pip install pandas
pip install requests
pip install ray
pip install ray[tune]
pip install ray[rllib]
pip install procgen

pip freeze
