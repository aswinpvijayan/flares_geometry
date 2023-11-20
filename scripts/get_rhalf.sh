#!/bin/bash
#SBATCH --ntasks 3
#SBATCH -A dp004
#SBATCH -p cosma6
#SBATCH --job-name=get_rhalf
#SBATCH -t 0-24:00
#SBATCH --cpus-per-task=5
#SBATCH --ntasks-per-node=3
#SBATCH --array=6-9
#SBATCH --mem-per-cpu=7G
#SBATCH -o logs/std_output.%J
#SBATCH -e logs/std_error.%J

module purge
module load gnu_comp/7.3.0 openmpi/3.0.1 hdf5/1.10.3 python/3.6.5


source ../flares_pipeline/venv_fl/bin/activate

mpiexec -n 3 python3 r_half_orient.py $SLURM_ARRAY_TASK_ID 010_z005p000


echo "Job done, info follows..."
sacct -j $SLURM_JOBID --format=JobID,JobName,Partition,MaxRSS,Elapsed,ExitCode
exit
