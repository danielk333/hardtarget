"""
Master Worker
"""
import multiprocessing
from multiprocessing import Queue


class Worker (multiprocessing.Process):

    def __init__(self, job_queue, result_queue, stop_event):
        super(Worker, self).__init__()
        self.job_queue = job_queue
        self.result_queue = result_queue
        self.stop_event = stop_event

    def run(self):
        while not self.stop_event.is_set():
            try:
                # Get job from the job queue with a timeout
                job = self.job_queue.get(timeout=1)
            except Queue.Empty:
                continue

            # Check for a termination signal
            if job is None:
                break

            # Process the job (you can replace this with your actual job processing logic)
            result = self.process_job(job)

            # Put the result into the result queue
            self.result_queue.put(result)

    def process_job(self, job):
        # Replace this method with your actual job processing logic
        # This is just a placeholder
        raise NotImplementedError



class Master:
    def __init__(self, num_workers):
        self.job_queue = multiprocessing.Queue()
        self.result_queue = multiprocessing.Queue()
        self.stop_event = multiprocessing.Event()
        self.workers = [Worker(self.job_queue, self.result_queue, self.stop_event) for _ in range(num_workers)]

    def start_workers(self):
        for worker in self.workers:
            worker.start()

    def stop_workers(self):
        # Add termination signals to the job queue
        for _ in self.workers:
            self.job_queue.put(None)

        # Wait for all workers to finish
        for worker in self.workers:
            worker.join()

    def listen_for_responses(self):
        results = []
        while not self.result_queue.empty():
            result = self.result_queue.get()
            results.append(result)

        return results

    def cleanup(self):
        self.stop_event.set()



if __name__ == "__main__":

    pass