#!/bin/bash
#
#
### comment lines start with ## or #+space
### slurm option lines start with #SBATCH


#SBATCH --job-name=baseline  	## job name
#SBATCH --time=0-01:00:00       ## days-hours:minutes:seconds
#SBATCH --mem=8000             ##   3GB ram (hardware ratio is < 4GB/core)

### SBATCH --output=job.out	## standard out file
#SBATCH --ntasks=1            ## Ntasks.  default is 1.
#SBATCH --cpus-per-task=4	## Ncores per task.  Use greater than 1 for multi-threaded jobs.  default is 1.
###BATCH --partition=volta  ##  can specify partition here, but it is pre-empted by what module is loaded
#SBATCH --account=iict-sp2.volk.cl.uzh    ## only need to specify if you belong to multiple tenants on ScienceCluster
#SBATCH --gres gpu:1
###SBATCH --gres gpu:Tesla-V100-32GB:1

module load nvidia/cuda10.2-cudnn7.6.5
module load anaconda3
source activate align
stdbuf -o0 -e0 srun --unbuffered $@
