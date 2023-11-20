#!/bin/bash --login
#SBATCH -A dp004
#SBATCH -p cosma7
#SBATCH --job-name=get_mollview_plots
#SBATCH -t 0-1:30
#SBATCH --ntasks 1
#SBATCH --cpus-per-task=1
###SBATCH --ntasks-per-node=14
#SBATCH --array=0-1   ###0-39
#SBATCH -o logs/std_los_output.%J
#SBATCH -e logs/std_los_error.%J

module purge
module load rockport-settings
module load gnu_comp/7.3.0 openmpi/3.0.1 hdf5/1.10.3 python/3.10.1
source /cosma7/data/dp004/dc-payy1/my_files/flares_inclination/venv_inc/bin/activate
export PYTHONPATH=$PYTHONPATH:/cosma7/data/dp004/dc-payy1/my_files/flares_pipeline/

### Tags for FLARES galaxies, change as required
# array=(010_z005p000 009_z006p000 008_z007p000 007_z008p000 006_z009p000 005_z010p000)
array=(010_z005p000 009_z006p000 008_z007p000 007_z008p000 006_z009p000 005_z010p000)
array=(010_z005p000)

# N_JOB=$SLURM_NTASKS                # create as many jobs as tasks
# ini=$((${SLURM_ARRAY_TASK_ID}*${N_JOB}))
#
# for((i=0;i<$N_JOB;i++))
# do
#   export region=$((${SLURM_ARRAY_TASK_ID}*${N_JOB}+$i))
#
#   echo "Running job for region ${region} with N_JOB=${i}"
#
#   # for ii in ${array[@]}
#   #   do
#   #     # python3 calc_los_for_orientations.py $region $ii FLARES
#   #     # python3 lum_orient.py $SLURM_ARRAY_TASK_ID ${array[1]} FLARES
#   #     python3 calc_luminosity.py $region $ii 1
#   # done &
#   #
#   # wait -n
#   sh exec.sh -n $region -f 1 &
#
# done
#
# #Wait for all
# wait
#
# echo "Regions $ini - $region done..."
for ii in ${array[@]}
  do
    # python3 calc_luminosity.py $SLURM_ARRAY_TASK_ID $ii 0
    python3 mollview_att.py $SLURM_ARRAY_TASK_ID $ii
  done

#For shm queues
# for ii in ${array[@]}
#   do
#     # python3 calc_luminosity.py $SLURM_ARRAY_TASK_ID $ii 0
#     mpiexec $RP_OPENMPI_ARGS -n $SLURM_NTASKS python3 calc_luminosity.py $SLURM_ARRAY_TASK_ID $ii 0
#     # mpiexec $RP_OPENMPI_ARGS -n 28 python3 calc_los_for_orientations.py $SLURM_ARRAY_TASK_ID $ii FLARES
# done

echo "Job done, info follows..."
sacct -j $SLURM_JOBID --format=JobID,JobName,Partition,MaxRSS,Elapsed,ExitCode
exit
