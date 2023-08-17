A basic example to output text to console from nodes over a network using [MPI].

A cluster at [IIIT] has four [SLURM] nodes. We want to run one process on each
node, and run `32` threads using [OpenMP]. In future, such a setup would allow
us to run distributed algorithms that utilize each node's memory efficiently and
minimize communication cost (within the same node). Output is saved in [gist].
Technical help from [Semparithi Aravindan].

<br>

```bash
$ scl enable gcc-toolset-11 bash
$ sbatch main.sh

# ==========================================
# SLURM_JOB_ID = 3373
# SLURM_NODELIST = node[01-04]
# SLURM_JOB_GPUS =
# ==========================================
# Cloning into 'hello-mpi'...
# [node01.local:2180262] MCW rank 0 is not bound (or bound to all available processors)
# [node02.local:3790641] MCW rank 1 is not bound (or bound to all available processors)
# [node04.local:3758212] MCW rank 3 is not bound (or bound to all available processors)
# [node03.local:3287974] MCW rank 2 is not bound (or bound to all available processors)
# P00: NAME=node01.local
# P00: OMP_NUM_THREADS=32
# P02: NAME=node03.local
# P02: OMP_NUM_THREADS=32
# P03: NAME=node04.local
# P03: OMP_NUM_THREADS=32
# P01: NAME=node02.local
# P01: OMP_NUM_THREADS=32
# P00.T00: Hello MPI
# P00.T24: Hello MPI
# P00.T16: Hello MPI
# P00.T26: Hello MPI
# P00.T05: Hello MPI
# P00.T29: Hello MPI
# P00.T22: Hello MPI
# P00.T06: Hello MPI
# P00.T17: Hello MPI
# P00.T23: Hello MPI
# P00.T25: Hello MPI
# P00.T13: Hello MPI
# P00.T01: Hello MPI
# P00.T09: Hello MPI
# P00.T03: Hello MPI
# P00.T02: Hello MPI
# P00.T31: Hello MPI
# P03.T00: Hello MPI
# P03.T24: Hello MPI
# P03.T05: Hello MPI
# P03.T21: Hello MPI
# P03.T04: Hello MPI
# ...
```

<br>
<br>


## References

- [MPI Basics : Tom Nurkkala](https://www.youtube.com/watch?v=c0C9mQaxsD4)
- [OpenMPI tutorial coding in Fortran 90 - 01 Hello World! : yinjianz](https://www.youtube.com/watch?v=wTlu971fXkE)
- [Mod-09 Lec-40 MPI programming : Prof. Matthew Jacob](https://www.youtube.com/watch?v=mzfVimVbguQ)
- [MPI/OpenMP Hybrid Programming : Neil Stringfellow](https://www.youtube.com/watch?v=TiQRPMBBmDs)
- [Introduction to MPI Programming, part 1 : Hristo Iliev](https://www.youtube.com/watch?v=LBgx_S5ougk)
- [Hybrid MPI+OpenMP programming : Dr. Jussi Enkovaara](https://www.youtube.com/watch?v=1Inj6hdSnG0)
- [Running an MPI Cluster within a LAN : Dwaraka Nath](https://mpitutorial.com/tutorials/running-an-mpi-cluster-within-a-lan/)
- [Return values of MPI calls : RIP Tutorial](https://riptutorial.com/mpi/example/16808/return-values-of-mpi-calls)
- [MPI Error Handling : Dartmouth College](https://www.dartmouth.edu/~rc/classes/intro_mpi/mpi_error_functions.html)
- [Does storing mpi rank enhance the performance : Cosmin Ioniță](https://stackoverflow.com/a/49873583/1413259)
- [MPI error handler not getting called when exception occurs : Hristo Iliev](https://stackoverflow.com/a/20067763/1413259)
- [Assert function for MPI Programs : Gilles Gouaillardet](https://stackoverflow.com/a/50519696/1413259)
- [In MPI, how to make the following program Wait till all calculations are completed : Gilles](https://stackoverflow.com/a/39609017/1413259)
- [MPI_Abort() vs exit() : R.. GitHub STOP HELPING ICE](https://stackoverflow.com/a/54844676/1413259)
- [MPI_Datatype : RookieHPC](https://rookiehpc.org/mpi/docs/mpi_datatype/index.html)
- [MPI_Error_string : DeinoMPI](https://mpi.deino.net/mpi_functions/MPI_Error_string.html)
- [MPI_Error_string : MPICH](https://www.mpich.org/static/docs/v3.3/www3/MPI_Error_string.html)
- [MPI_Comm_size : MPICH](https://www.mpich.org/static/docs/v3.3/www3/MPI_Comm_size.html)
- [MPI_Comm_rank : MPICH](https://www.mpich.org/static/docs/v3.3/www3/MPI_Comm_rank.html)
- [MPI_Get_processor_name : MPICH](https://www.mpich.org/static/docs/v3.2/www3/MPI_Get_processor_name.html)

<br>
<br>

[![](https://img.youtube.com/vi/c0C9mQaxsD4/maxresdefault.jpg)](https://www.youtube.com/watch?v=c0C9mQaxsD4)<br>
[![ORG](https://img.shields.io/badge/org-puzzlef-green?logo=Org)](https://puzzlef.github.io)


[Semparithi Aravindan]: https://www.iiit.ac.in/people/faculty/Semparithi.Aravindan/
[IIIT]:   https://www.iiit.ac.in
[MPI]:    https://en.wikipedia.org/wiki/Message_Passing_Interface
[SLURM]:  https://en.wikipedia.org/wiki/Slurm_Workload_Manager
[OpenMP]: https://en.wikipedia.org/wiki/OpenMP
[gist]:   https://gist.github.com/wolfram77/41114570e75f5c0d0ffeb9fd73ec252b
