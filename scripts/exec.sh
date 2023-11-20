#!/bin/bash
while getopts n:f: flag
do
    case "${flag}" in
        n) num=${OPTARG};;
        f) opt=${OPTARG};;
    esac
done


array=(010_z005p000 009_z006p000 008_z007p000 007_z008p000 006_z009p000 005_z010p000)
for tag in ${array[@]}
  do

    # mpiexec -n 16 python3 calc_los_for_orientations.py $SLURM_ARRAY_TASK_ID $ii FLARES
    # python3 calc_rhalf.py $SLURM_ARRAY_TASK_ID $ii
    # python3 calc_luminosity.py $SLURM_ARRAY_TASK_ID $ii 1
    python3 calc_luminosity.py $num $tag $opt

done
