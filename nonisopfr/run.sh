#!/bin/bash
#SBATCH --job-name=nonisopfr             # Job name
#SBATCH --nodes=1                           # Node count
#SBATCH --ntasks=1                          # Total number of tasks across all nodes
#SBATCH --cpus-per-task=1                   # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --gres=gpu:1                        # Total number of gpus per node
#SBATCH --time=48:00:00                     # Maximum run time (HH:MM:SS)
#SBATCH --mem=16G                            # Request total 4GB CPU memory
#SBATCH --mail-type=begin                   # Send email when job begins
#SBATCH --mail-type=end                     # Send email when job ends
#SBATCH --mail-user=hv0085@princeton.edu

module purge
module load anaconda3/2024.6
conda activate PINN

# Run the Python script
python main.py
python check.py