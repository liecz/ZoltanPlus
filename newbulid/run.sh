#!/bin/bash
rm -f example/C/simplePHG.exe
rm -f example/C/simplePHG-simplePHG.o
rm -f src/phg.o
rm -f src/phg_build.o
rm -f src/phg_rdivide.o
rm -f src/phg_build_calls.o
rm -f src/phg_hypergraph.o
rm -f src/phg_Vcycle.o
rm -f src/phg_gather.o
rm -f src/phg_coarse.o
rm -f src/phg_match.o
rm -f src/phg_serialpartition.o
rm -f example/C/core

#export CFLAGS="$-Wall -g"
#export CXXFLAGS="$-Wall -g"
#export CPPFLAGS="$-Wall -g"
#../configure
make everything

cd example/C

ulimit -c unlimited
mpirun -n 12 simplePHG.exe "hypergraph.txt" 1
