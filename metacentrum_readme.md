# Metacentrum README

## Links
- Metacentrum has its own documentation, available here: https://docs.metacentrum.cz/
- Registration: https://metavo.metacentrum.cz/osobniv3/wayf/proxy.jsp?locale=en&target=https%3A%2F%2Fsignup.e-infra.cz%2Ffed%2Fregistrar%2F%3Fvo%3Dmeta%26locale%3Den

## Connecting to metacentrum
- Choose a frontend (list of fronteds : https://docs.metacentrum.cz/access/log-in/), i.e. `tarkil.metacentrum.cz`
- login using this command
```
ssh <username>@tarkil.metacentrum.cz
```
- You will be placed in your home directory
- You can check the path via command ``pwd``

## Running jobs
- Source: https://docs.metacentrum.cz/computing/
- If you do not submit a job, your computation will run on the slow frontend, which is meant for very light tasks.
- To run a job, you have to submit it via `qsub`
- There are two types of jobs: interactive and batch
    - interactive jobs need an open terminal session, and you can use the terminal normally
    - batch jobs run a script/execute commands that you submit. These do not require an active session.
- You can customize the resource you want to reserve for this job.

```
qsub -I -l select=1:ncpus=1:mem=8gb:scratch_local=10gb -l walltime=2:00:00
```
- `-I `= specifies that this is an interactive job
- `ncpus` = number of cpus
- `mem` = available RAM
- `scratch` = scratch memory for temporary files
- `walltime` = maximum lifetime of the job. It will be killed automatically after this time period

- more about various arguments: https://docs.metacentrum.cz/computing/pbs-resources/

- how to extend a job: https://docs.metacentrum.cz/computing/extend-walltime/

- how to check output of a job while it's running: https://docs.metacentrum.cz/computing/job-tracking/#output-of-running-jobs
## GPU enabled jobs
- To use a gpu, you have to specify it when submitting a job, via argument `-q gpu`
- This will put the job into the gpu queue
- Example command

```
qsub -I -q gpu -l select=1:ncpus=1:mem=16gb:scratch_local=16gb:gpu_mem=6gb:gpu_cap=cuda6 -l walltime=2:00:00
```

- `gpu_mem` = minimum gpu memory required
- `gpu_cap` = minimum CUDA compute capability required. `cuda60` means we need at least CUDA 6.0

## Containers
- To make use of Tensorflow, PyTorch etc., we need to use containers. There are many images available in the `/cvmfs/` folder that come with prepared computational enviroments. The most used container is Singularity, and you can find the images for Singularity in `/cvmfs/singularity.metacentrum.cz/`

- Images for TF and PyTorch are in the `/cvmfs/singularity.metacentrum.cz/NGC/` folder

- Singularity uses a few environment variables. It is good to set them before launching an image. These variables are `CACHEDIR` `LOCALCACHEDIR` `SCRATCHDIR`. Read more about them here https://docs.metacentrum.cz/software/containers/. 

- Generally run this commands after running a job with a scratch memory
```
export CACHEDIR=$SCRATCHDIR
export LOCALCACHEDIR=$SCRATCHDIR
```

- to launch a container, use the `singularity`

```
singularity shell <image_path>
```
- this will open a shell inside the singularity image. You can now for example install additional libraries using pip
```
pip install datasets transformers 
```
- instead of `shell`, you can use
    - `exec` - executes the given command
    ```
    singularity exec <image_path> bash -c "<command>"
    ```
    - `run` - runs the container as a program itself, i.e. it runs the default action

## GPU enabled Singularity image
To run computations on a gpu inside an image, you have to
1. Run a gpu job
2. Execute the singularity container with a `--nv` argument

Example - runs a shell inside a PyTorch image 
```
singularity shell --nv /cvmfs/singularity.metacentrum.cz/NGC/PyTorch\:20.09-py3.SIF
```

## Example scripts
Tensorflow GPU shell
```bash
#!/bin/bash
# Path to home directory
HOMEDIR=/storage/praha1/home/<username>/

# Path to singularity image we want to use
IMAGE=/cvmfs/singularity.metacentrum.cz/NGC/TensorFlow\:23.08-tf2-py3.SIF

export SINGULARITY_CACHEDIR=$HOMEDIR
export SINGULARITY_LOCALCACHEDIR=$SCRATCHDIR # Scratchdir should be set already
export SINGULARITY_TMPDIR=$SCRATCHDIR

singularity shell --nv $IMAGE
cd $HOMEDIR
```

GPU job script
```bash
#!/bin/bash
#PBS -N gpu_train
#PBS -q gpu
#PBS -l select=1:ncpus=1:mem=32gb:scratch_ssd=32gb:ngpus=1:gpu_mem=12gb:gpu_cap=cuda70
#PBS -l walltime=12:00:00
#PBS -m ae

# Path to home directory
HOMEDIR=/storage/praha1/home/nexuso1/

# Path to singularity image we want to use
IMAGE=/cvmfs/singularity.metacentrum.cz/NGC/TensorFlow\:23.08-tf2-py3.SIF

export SINGULARITY_CACHEDIR=$HOMEDIR
export SINGULARITY_LOCALCACHEDIR=$SCRATCHDIR # Scratchdir should be set already
export SINGULARITY_TMPDIR=$SCRATCHDIR

singularity exec --nv $IMAGE bash -c "<path to a script or a command>"
```
This script can be submitted as

```
qsub job_script.sh
```

and it will run

