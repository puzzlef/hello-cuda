#!/usr/bin/env bash
src="hello-cuda"
out="$HOME/Logs/$src$1.log"
ulimit -s unlimited
printf "" > "$out"

# Download program
if [[ "$DOWNLOAD" != "0" ]]; then
  rm -rf $src
  git clone https://github.com/puzzlef/$src
fi
cd $src

# Fixed config
: "${MAX_THREADS:=32}"
: "${REPEAT_METHOD:=5}"
# Define macros (dont forget to add here)
DEFINES=(""
"-DMAX_THREADS=$MAX_THREADS"
"-DREPEAT_METHOD=$REPEAT_METHOD"
)

# Run
nvcc ${DEFINES[*]} -std=c++17 -arch=sm_60 -O3 -Xcompiler -fopenmp main.cu -o "a$1.out"
stdbuf --output=L ./"a$1.out" 2>&1 | tee -a "$out"

# Signal completion
curl -X POST "https://maker.ifttt.com/trigger/puzzlef/with/key/${IFTTT_KEY}?value1=$src$1"
