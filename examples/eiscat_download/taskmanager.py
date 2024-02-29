"""
Thread-safe Task Manager
"""
from enum import Enum
from threading import Lock
import json

class TaskState(Enum):
    INIT = "init"
    EXEC = "exec"
    DONE = "done"
    FAIL = "fail"

def dump_line(task_id, state, task):
    return f"{task_id} {state} {task}\n"

def load_line(line):
    parts = line.split()
    task_id = parts[0]
    state = parts[1]
    task = ' '.join(parts[2:])
    return task_id, state, task


class TaskManager:
    def __init__(self, file_path='tasks.txt'):
        self.file_path = file_path
        self.lock = Lock()
        self.create_file()

    def create_file(self):
        with self.lock:
            with open(self.file_path, 'a'):  # Create the file if it doesn't exist
                pass

    def init(self, tasks_json):
        tm.clear_tasks()
        with open(tasks_json, "r") as f:
            data = json.load(f)
        with self.lock:
            with open(self.file_path, 'a') as file:
                for idx, item in enumerate(data):
                    task_id = item.get("id", str(idx))
                    task_json = json.dumps(item)
                    line = dump_line(task_id, TaskState.INIT.value, task_json)
                    file.write(line)

    def get_task_state(self, task_id):
        assert isinstance(task_id, str)
        with self.lock:
            with open(self.file_path, 'r') as file:
                for line in file:
                    _task_id, _state, _ = load_line(line)
                    if _task_id == task_id:
                        return TaskState(_state)
                return None

    def update_task_state(self, task_id, new_state):
        assert isinstance(task_id, str)
        with self.lock:
            if not isinstance(new_state, TaskState):
                print(f"Invalid state: {new_state}")
                return

            with open(self.file_path, 'r') as file:
                lines = file.readlines()

            with open(self.file_path, 'w') as file:
                for line in lines:
                    _task_id, _, task_json = load_line(line)
                    if _task_id == task_id:
                        line = dump_line(task_id, new_state.value, task_json)
                        file.write(line)
                    else:
                        file.write(line)

    def list_tasks(self):
        tasks = []
        with self.lock:
            with open(self.file_path, 'r') as file:
                for line in file:
                    _task_id, state, task_json = load_line(line)
                    task = json.loads(task_json)
                    tasks.append((_task_id, TaskState(state), task))
        return tasks

    def clear_tasks(self):
        with self.lock:
            with open(self.file_path, 'w'):
                pass


if __name__ == "__main__":

    import argparse


    parser = argparse.ArgumentParser(description="Manage product tasks.")
    parser.add_argument("command", choices=["init", "get", "list", "clear"], help="Command to execute")
    parser.add_argument("--file", default="tasks.txt", help="Path to the tasks text file")
    parser.add_argument("--task", help="Task identifier")
    parser.add_argument("--tasks", help="Path to JSON file with task")

    args = parser.parse_args()

    tm = TaskManager(args.file)

    if args.command == "get":
        state = tm.get_task_state(args.task)
        print(f"{args.task}: {state.value}")
    elif args.command == "clear":
        tm.clear_tasks()
    elif args.command == "list":
        for task_id, state, task in tm.list_tasks():
            print(f"{task_id}: {state.value} {task}")
    elif args.command == "init":
        tm.init(args.tasks)
        

