
with open("base.slurm", "r") as base:
    base = base.read()

with open("experiments.txt", "r") as experiments:
    for line in experiments:
        line = line.split()
        name = [line[0], line[1], "singletask" if line[2] == 't' else "multitask", line[4]]
        if line[2] == 't':
            name.append(line[3])
        job_name = '-'.join(name)
        output = job_name + "-srun.out"
        run = ' '.join(line)
        run = "src/trainer.py " + run

        lines = base.split('\n')
        for i, line in enumerate(lines):
            if line == "#SBATCH --job-name=":
                # print(line)
                # print(lines[i])
                # print('"' + job_name + '"')
                lines[i] += ('"' + job_name + '"')
            if line == "#SBATCH --output=":
                lines[i] += output
            if line == "python ":
                lines[i] += run
        lines = '\n'.join(lines)
        with open(job_name + '.slurm', 'w') as out:
            out.write(lines)
