"""
Master Worker
"""
import queue
import multiprocessing
import signal
from taskmanager import TaskManager, TaskState
from enum import Enum


class WorkerState(Enum):
    IDLE = "idle"
    BUSY = "busy"
    DEAD = "dead"


class Worker:

    def __init__(self, worker_id, worker_func, job_queue, result_queue, stop_event):
        self.id = worker_id
        self.worker_func = worker_func
        self.job_queue = job_queue
        self.result_queue = result_queue
        self.stop_event = stop_event

    def run(self):

        # Ignore SIGINT in the worker processes
        signal.signal(signal.SIGINT, signal.SIG_IGN)

        while not self.stop_event.is_set():
            try:
                # Get job from the job queue with a timeout
                job = self.job_queue.get(timeout=1)
            except queue.Empty:
                continue
            if job is None:
                continue
            try:
                # Process the job (you can replace this with your actual job processing logic)
                ok, result = self.worker_func(self, job)
                # Put the result into the result queue
                self.result_queue.put((job, ok, result))
            except Exception as e:
                self.result_queue.put((job, False, e))


class Master:

    def __init__(self, num_workers, worker_func, task_file):
        self.job_queue = multiprocessing.Queue()
        self.result_queue = multiprocessing.Queue()
        self.stop_event = multiprocessing.Event()
        self.tm = TaskManager(task_file)

        def make_worker(worker_id):
            worker = Worker(worker_id,
                            worker_func,
                            self.job_queue,
                            self.result_queue,
                            self.stop_event)
            return {
                "state": WorkerState.IDLE,
                "worker": worker,
                "process": multiprocessing.Process(target=worker.run)
            }
        self.workers = {worker_id: make_worker(worker_id) for worker_id in range(num_workers)}

    def apply_jobs(self):
        
        # find ready tasks
        tasks = self.tm.list_tasks()
        ready_tasks = [task for task, state in tasks if state == TaskState.READY]
        running_tasks = [task for task, state in tasks if state == TaskState.RUNNING]

        if len(ready_tasks) + len(running_tasks) == 0:
            # done
            return -1

        # apply a batch from ready tasks
        count_workers = len(self.workers)
        batch = ready_tasks[:count_workers]
        
        for task in batch:
            self.tm.update_task_state(task, TaskState.RUNNING)
            self.job_queue.put(task)

        # return number of ready tasks which were applied 
        return len(batch)

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

        count = self.apply_jobs()
        if count == -1:
            # no more to do
            return False

        while True:
            try:
                # Get msg from the result queue with a timeout
                msg = self.result_queue.get(timeout=1)
                if msg is None:
                    continue
                task, ok, result = msg
                if not ok:
                    self.tm.update_task_state(task, TaskState.FAILED)
                    print("job failed", task)
                else:
                    self.tm.update_task_state(task, TaskState.COMPLETED)
                    print("job ok", task, result)
            except queue.Empty:
                break
        return True

    def run(self):
        while not self.stop_event.is_set():
            ok = self.run_once()
            if not ok:
                break


if __name__ == "__main__":

    def worker_func(worker, job):
        print("Worker", worker.id, job)
        return True, "result"

    task_file = "tasks.txt"

    master = Master(2, worker_func, task_file)
    master.start_workers()
    try:
        master.run()
    except KeyboardInterrupt:
        master.stop_workers()
    master.join_workers()
    print()
