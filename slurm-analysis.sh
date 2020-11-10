# in the general partition
#SBATCH --partition=general

# job name is my_job
#SBATCH --job-name=covid

# enable gpu support
#SBATCH --gres=gpu:GPUS

# load environment
conda activate ml

# launch job scripts
python analysis.py
python ml-analysis.py
