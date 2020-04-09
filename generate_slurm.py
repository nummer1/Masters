with open("email.txt", "r") as email:
    email = email.read().strip('\n')

with open("experiments.txt", "r") as experiments:
    for line in experiments:
        if line == "\n":
            continue

        line = line.split()
        name = [line[0], line[1], line[2], "singletask" if line[3] == 't' else "multitask", line[5]]
        if line[3] == 't':
            name.append(line[4])
        job_name = '_'.join(name)
        output = job_name + "_srun.out"
        run = ' '.join(line)
        run = "python src/trainer.py " + run

        lines = ["#!/bin/sh",
        "#SBATCH --partition=GPUQ",
        "#SBATCH --account=ie-idi",
        "#SBATCH --time=40:00:00",
        "#SBATCH --nodes=1",
        "#SBATCH --ntasks-per-node=6",
        "#SBATCH --mem=90000",
        "#SBATCH --gres=gpu:V100:1",
        '#SBATCH --job-name="' + job_name + '"',
        "#SBATCH --output=" + output,
        "#SBATCH --mail-user=" + email,
        "#SBATCH --mail-type=ALL",
        "",
        "WORKDIR=${SLURM_SUBMIT_DIR}",
        "cd ${WORKDIR}",
        "",
        "module purge",
        "module load fosscuda/2018b",
        "",
        "module load GCCcore/.8.3.0",
        "module load Python/3.7.4",
        "",
        "module load GCC/8.3.0",
        "module load CUDA/10.1.243",
        "module load OpenMPI/3.1.4",
        "module load TensorFlow/2.1.0-Python-3.7.4",
        "",
        "mpirun hostname",
        "",
        "source venv/bin/activate",
        run,
        ""]

        lines = '\n'.join(lines)
        with open("tests/" + job_name + '.slurm', 'w') as out:
            out.write(lines)
