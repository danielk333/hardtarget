import json
from taskmanager import TaskManager

"""
Commandline tool for manual management of eiscat tasks
"""

if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser(description="Manage product tasks.")
    parser.add_argument("command", choices=["init", "get", "list", "clear"], help="Command to execute")
    parser.add_argument("--file", default="tasks.txt", help="Path to the tasks text file")
    parser.add_argument("--products", default="products.json", help="Path to the products JSON file")
    parser.add_argument("--task", help="Task identifier")

    args = parser.parse_args()

    tm = TaskManager(args.file)

    if args.command == "init":
        tm.clear_tasks()

        def get_task_identifier(item):
            day = item["date"]
            instrument = item["experiment"]
            chnl = item["type"]
            return f"{day}-{instrument}-{chnl}"

        with open(args.products, "r") as f:
            data = json.load(f)
            for item in data:
                task_identifier = get_task_identifier(item)
                tm.add_task(task_identifier)

    elif args.command == "get":
        state = tm.get_task_state(args.task)
        print(f"{args.task}: {state.value}")
    elif args.command == "clear":
        tm.clear_tasks()
    elif args.command == "list":
        for task_identifier, state in tm.list_tasks():
            print(f"{task_identifier}: {state.value}")
