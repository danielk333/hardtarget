============
Parallellism
============

GMF Analysis supports parallel execution, but leaves it to developers to select
a particular mechanism. This way, performance acceleration can be achieved in
different ways, for example using `mpi` (multiprocess support on Linux systems),
`cuda` (GPU-based hardware acceleration), or any other multiprocessing
framework.


Job Division
--------------------

Support for parallelism is acheived through the concept of `job`. A job
specifies a logical part of a larger task. A particular job
is defined by the tuple ``(index,N)``, where ``index`` identifies a particular
job out of a total of ``N`` jobs. ``index`` and ``N`` are both non-negative
integers ``0 <= index < N``. For instance, job ``(3,10)`` means the 4'th job out
of 10, whereas ``(0,1)`` means the first job out of 1, which effectively means
no subdivision -- the one job is equal to the larger task.


GMF Analysis with Job
------------------------------------

GMF Analysis accepts an optional parameter job, restricting processing to that
particular job. In the case of GMF Analysis, this corresponds to processing a
subset of the data, i.e. the Hardtarget DRF product.

Job objects are represented as a dictionary.

.. code:: python

    job = {"idx": 3, "N": 10}

If no job is specified, the default is to process the entire product as one job.

.. code:: python

    job = {"idx": 0, "N": 1}

The implementation guarantees that jobs cover all input data, without any
overlap, and that different jobs do not write results to the same file, thus avoiding any
race conditions.


Parallelism with Jobs
---------------------

Support for job division allow for different execution strategies. For example,
a trivial sequential execution can be achieved simply by processing all tasks in
sequence, or skip subdivision altogether.

.. code:: python

    from hardtarget.analysis import compute_gmf

    // alt 1 - no subdivision
    compute_gmf(*args, job={"idx":0, "N":1}, **kwargs)

    // alt 2 - subdivision, 
    for index in range(N):
        compute_gmf(*args, job={"idx":index, "N":N}, **kwargs)


Parallelism can then be achieved by distributing jobs across different threads,
processes and hosts. The CLI interface uses this mechanism to support parallell processing
with `mpi` as well as to partitioning the work between `cuda` kernels. For example, this
runs GMF Analysis in paralell on 4 CPU's

.. code:: shell

    mpirun -np 4 hardtarget -v analyze /data/drf uhf --config cfg.ini --progress -o /data/gmf


Here is the relevant code within the `hardtarget analyze` script. 

.. code:: python
    
    from hardtarget.analysis import compute_gmf
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    compute_gmf(*args, job={"idx": comm.rank, "N": comm.size}, **kwargs)



