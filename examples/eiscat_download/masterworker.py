"""
Master Worker
"""
import queue
import multiprocessing
import signal
from taskmanager import TaskManager, TaskState
from enum import Enum
import logging


class WorkerState(Enum):
    IDLE = "idle"
    BUSY = "busy"
    DEAD = "dead"


class Worker:

    def __init__(self, worker_id, worker_func, job_queue, result_queue, stop_event, logger=None):
        self.id = worker_id
        self.worker_func = worker_func
        self.job_queue = job_queue
        self.result_queue = result_queue
        self.stop_event = stop_event
        self.logger = logger
        if self.logger is None:
            self.logger = logging.getLogger(__name__)

    def run(self):

        # Ignore SIGINT in the worker processes
        signal.signal(signal.SIGINT, signal.SIG_IGN)

        while not self.stop_event.is_set():
            try:
                # Get job from the job queue with a timeout
                task = self.job_queue.get(timeout=1)
            except queue.Empty:
                continue
            if task is None:
                continue
            try:
                self.logger.info(f"Worker {self.id}: {task[0]}")

                # Process the job (you can replace this with your actual job processing logic)
                ok, result = self.worker_func(self, task[2])
                # Put the result into the result queue
                self.result_queue.put((task, ok, result))
            except Exception as e:
                self.result_queue.put((task, False, e))


class Master:

    def __init__(self, num_workers, worker_func, task_file, logger=None):
        self.job_queue = multiprocessing.Queue()
        self.result_queue = multiprocessing.Queue()
        self.stop_event = multiprocessing.Event()
        self.tm = TaskManager(task_file)
        self.logger = logger
        if self.logger is None:
            self.logger = logging.getLogger(__name__)

        def make_worker(worker_id):
            worker = Worker(worker_id,
                            worker_func,
                            self.job_queue,
                            self.result_queue,
                            self.stop_event,
                            logger=self.logger
                            )
            return {
                "state": WorkerState.IDLE,
                "worker": worker,
                "process": multiprocessing.Process(target=worker.run)
            }
        self.workers = {worker_id: make_worker(worker_id) for worker_id in range(num_workers)}

    def get_batch(self):
        # find ready tasks
        tasks = self.tm.list_tasks()
        init_tasks = [tup for tup in tasks if tup[1] == TaskState.INIT]
        exec_tasks = [tup for tup in tasks if tup[1] == TaskState.EXEC]

        if len(init_tasks) + len(exec_tasks) == 0:
            # done
            return False, []

        # apply a batch of init tasks
        count_workers = len(self.workers)
        exec_workers = len(exec_tasks)
        return True, init_tasks[:count_workers-exec_workers]        

    def start_workers(self):
        # start all workers
        for worker in self.workers.values():
            worker["process"].start()

    def stop_workers(self):
        # stop all workers
        self.stop_event.set()

    def join_workers(self):
        # Wait for all workers to finish
        for worker in self.workers.values():
            worker["process"].join()

    def run_once(self):
        if self.stop_event.is_set():
            return False

        # try to dispatch a batch
        ok, batch = self.get_batch()
        if ok:
            for task in batch:
                self.tm.update_task_state(task[0], TaskState.EXEC)
                self.job_queue.put(task)
            self.logger.info(f"Master: put {[tup[0] for tup in batch]}")
        else:
            # no more to do
            self.logger.info("Nothing more to do")
            return False

        # process received results
        # want to block on queue.get for up to one seconde
        # but not on repeated queue.get because 
        # after at least one message have been received,
        # there is a need to empty the queue and 
        # break out as quickly as possible 
        timeout = 1
        while True:
            try:
                # Get msg from the result queue with a timeout
                msg = self.result_queue.get(timeout=timeout)
                if msg is None:
                    continue
                timeout = 0
                task, ok, result = msg                
                self.logger.info(f"Master get {task[0]}")
                if not ok:
                    self.tm.update_task_state(task[0], TaskState.FAIL)
                else:
                    self.tm.update_task_state(task[0], TaskState.DONE)
            except queue.Empty:
                break

        return True

    def run(self):
        while not self.stop_event.is_set():
            ok = self.run_once()
            if not ok:
                break
        self.stop_workers()
        self.join_workers()


if __name__ == "__main__":

    def worker_func(worker, task):
        return True, "result"

    task_file = "tasks.txt"

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    ch = logging.StreamHandler()
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    master = Master(2, worker_func, task_file, logger=logger)
    master.start_workers()
    try:
        master.run()
    except KeyboardInterrupt:
        master.stop_workers()
        master.join_workers()
        print()
