#!/bin/bash

#SBATCH --time=10:59:00
#SBATCH --mem=4000
#SBATCH -c 8
#SBATCH --exclude=compute-1-[1-28]



srun python classical_spins_python2.py