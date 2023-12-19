#!/usr/bin/env bash
#PBS -N Apptainer_Job
#PBS -l select=1:scratch_local=10gb:ncpus=1:mem=8gb
#PBS -l walltime=1:00:00

# define variables
SING_IMAGE="/cvmfs/singularity.metacentrum.cz/NGC/PyTorch\:23.08-py3.SIF"
HOMEDIR=/storage/praha1/home/nexuso1/ # substitute username and path to to your real username and path

# test if scratch directory is set
# if scratch directory is not set, issue error message and exit
test -n "$SCRATCHDIR" || { echo >&2 "Variable SCRATCHDIR is not set!"; exit 1; }

#set SINGULARITY variables for runtime data
export SINGULARITY_CACHEDIR=$HOMEDIR
export SINGULARITY_LOCALCACHEDIR=$SCRATCHDIR
export SINGULARITY_TMPDIR=$SCRATCHDIR

singularity exec --bind /storage/,/cvmfs/ \
$SING_IMAGE bash -c "python3 $HOMEDIR/bio-research/train_script.py"
