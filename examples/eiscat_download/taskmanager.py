"""
Thread-safe Task Manager
"""
from enum import Enum
from threading import Lock


class TaskState(Enum):
    READY = "ready"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class TaskManager:
    def __init__(self, file_path='tasks.txt'):
        self.file_path = file_path
        self.lock = Lock()
        self.create_file()

    def create_file(self):
        with self.lock:
            with open(self.file_path, 'a'):  # Create the file if it doesn't exist
                pass

    def add_task(self, task_identifier):
        with self.lock:
            with open(self.file_path, 'a') as file:
                file.write(f"{task_identifier} {TaskState.READY.value}\n")

    def get_task_state(self, task_identifier):
        with self.lock:
            with open(self.file_path, 'r') as file:
                for line in file:
                    name, state = line.strip().split()
                    if name == task_identifier:
                        return TaskState(state)
                return None

    def update_task_state(self, task_identifier, new_state):
        with self.lock:
            if not isinstance(new_state, TaskState):
                print(f"Invalid state: {new_state}")
                return

            with open(self.file_path, 'r') as file:
                lines = file.readlines()

            with open(self.file_path, 'w') as file:
                for line in lines:
                    name, state = line.strip().split()
                    if name == task_identifier:
                        file.write(f"{task_identifier} {new_state.value}\n")
                    else:
                        file.write(line)

    def list_tasks(self):
        tasks = []
        with self.lock:
            with open(self.file_path, 'r') as file:
                for line in file:
                    name, state = line.strip().split()
                    tasks.append((name, TaskState(state)))
        return tasks

    def clear_tasks(self):
        with self.lock:
            with open(self.file_path, 'w'):
                pass


if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser(description="Manage product tasks.")
    parser.add_argument("command", choices=["get", "list", "clear"], help="Command to execute")
    parser.add_argument("--file", default="tasks.txt", help="Path to the tasks text file")
    parser.add_argument("--task", help="Task identifier")

    args = parser.parse_args()

    tm = TaskManager(args.file)

    if args.command == "get":
        state = tm.get_task_state(args.task)
        print(f"{args.task}: {state.value}")
    elif args.command == "clear":
        tm.clear_tasks()
    elif args.command == "list":
        for task_identifier, state in tm.list_tasks():
            print(f"{task_identifier}: {state.value}")
