#!/usr/bin/env bash
#SBATCH --nodes=4
#SBATCH --ntasks=4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --exclusive
#SBATCH --mem=30gb
#SBATCH --job-name slurm
#SBATCH --output=slurm.out
# source scl_source enable gcc-toolset-11
# module load hpcx-2.7.0/hpcx-ompi
# module load openmpi/4.1.5
src="hello-mpi"
out="$HOME/Logs/$src$1.log"
module load hpcx-2.7.0/hpcx-ompi
ulimit -s unlimited
printf "" > "$out"

# Download program
if [[ "$DOWNLOAD" != "0" ]]; then
  rm -rf $src
  git clone https://github.com/puzzlef/$src
fi
cd $src

# Fixed config
: "${TYPE:=double}"
: "${MAX_THREADS:=32}"
: "${REPEAT_BATCH:=5}"
: "${REPEAT_METHOD:=1}"
# SLURM specific parameters
: "${SLURM_NTASKS:=4}"
# Define macros (dont forget to add here)
DEFINES=(""
"-DTYPE=$TYPE"
"-DMAX_THREADS=$MAX_THREADS"
"-DREPEAT_BATCH=$REPEAT_BATCH"
"-DREPEAT_METHOD=$REPEAT_METHOD"
)

# Run
MPIRUN="mpirun -np $SLURM_NTASKS --map-by node --bind-to none --report-bindings"
mpic++ ${DEFINES[*]} -std=c++17 -O3 -fopenmp main.cxx -o "a$1.out"
stdbuf --output=L $MPIRUN ./"a$1.out" 2>&1 | tee -a "$out"

# Signal completion
curl -X POST "https://maker.ifttt.com/trigger/puzzlef/with/key/${IFTTT_KEY}?value1=$src$1"
