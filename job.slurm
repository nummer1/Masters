#!/bin/sh
#SBATCH --partition=EPIC2
#SBATCH --account=ie-idi
#SBATCH --time=00:01:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --mem=2000
#SBATCH --gres=gpu:1
#SBATCH --job-name="test"
#SBATCH --output=test-srun.out
#SBATCH --mail-user=johanag@ntnu.no
#SBATCH --mail-type=ALL
  
WORKDIR=${SLURM_SUBMIT_DIR}
cd ${WORKDIR}
 
module purge
module load fosscuda/2018b

# module load GCCcore/.8.3.0
# module load Python/3.7.4

# module load GCC/8.3.0
# module load CUDA/10.1.243
# module load OpenMPI/3.1.4
# module load TensorFlow/2.1.0-Python-3.7.4

module load GCCcore/.8.2.0
module load Python/3.7.2

module load GCC/8.2.0-2.31.1
module load CUDA/10.1.105
module load OpenMPI/3.1.3
module load TensorFlow/1.13.1-Python-3.7.2

mpirun hostname

python test.py
