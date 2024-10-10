============
Parallellism
============

GMF Analysis supports parallel execution, but leaves it to developers to select a
particular mechanism. This way, performance acceleration can be achieved in
different ways, for example using `mpi` (multiprocess support on
Linux systems), `cuda` (GPU-based hardware acceleration), or be used
with any other multiprocessing framework.


Logical Job Division
--------------------

Support for parallelism is acheived through the concept of `job`. A job
specifies one logical (and independent) part of a larger task. A particular job
is specified as a tuple ``(index,N)``, where ``index`` identifies a particular
job out of a total of ``N`` jobs. ``index`` and ``N`` are both non-negative
integers ``0 <= index < N``. For instance, job ``(3,10)`` means the 4'th job out
of 10, whereas ``(0,1)`` means the first job out of 1, which effectively means
no partitioning, the job is equal to the larger task.

Logical Job Division in GMF Analysis
------------------------------------

By supplying a job to the GMF analysis function as a parameter, the function
will only process a given part of the larger task. In the case of GMF Analysis,
the larger task is defined by the data input, i.e. the Hardtarget DRF product.
So, a single job corresponds to processing some subset of the input file.
Moreover, the implementation ensures that jobs cover all input data, without any
overlap, and that jobs write results to different files. This avoids any race
conditions due to parallel processing.

GMF analysis expect jobs represented as a dictionary.

.. code:: python3

    job = {"idx": 3, "N": 10}


Parallelism with Jobs
---------------------

Support for logical job division allow for different execution strategies. A trivial sequential
execution can be achieved simply by processing all tasks in sequence, or skip subdivision altogether.

.. code:: python3

    // no subdivision
    process(job={"idx":0, "N":1}))

    // subdivision, sequential execution
    for index in range(N):
        process(job={"idx":index, "N":N})


In general, parallelism can be achieved by distributing jobs across different
process. For example, the CLI interface uses this mechanism to support parallell
processing with `mpi` or to partition the work between `cuda` kernels. 

.. code:: python3

    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    process(job={"idx": comm.rank, "N": comm.size})

