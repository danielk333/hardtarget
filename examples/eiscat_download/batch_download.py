"""
Testing multiprocessing
"""
import json
import  multiprocessing
import logging
from hardtarget.radars.eiscat import download
from pathlib import Path


def load_products(jsonfile, index, size):
    products = []
    with open(jsonfile, "r") as f:
        data = json.load(f)
        products = list(enumerate(data))[index::size]
    return products


def worker_function(index, size, result_queue, datadir, tmpdir):
    print(f"Process {index} started.")

    # load json products
    products = load_products(PRODUCTS_JSON, index, size)

    # mock products
    products = [(44, {'date': '20220408', "experiment": 'leo_bpark_2.1u_NO', "type": 'uhf'})]

    # download
    completed = []
    for product in products:

        idx, item = product
        day = item["date"]
        instrument = item["experiment"]
        chnl = item["type"]
        targetdir = Path(datadir) / f"{day}_{instrument}_{chnl}"
        targetdir.mkdir(exist_ok=True, parents=True)
        # ok, result = download(day, instrument, chnl, targetdir,
        #                  keep_zip=False, update=False,
        #                  tmpdir=TMPDIR, logger=logger, progress=False)
        ok = True
        if ok:
            completed.append(product)

    result_queue.put(completed)


if __name__ == "__main__":

    N_PROCESSES = 1
    PRODUCTS_JSON = "batch.json"
    DATADIR = "data/"
    TMPDIR = "/tmp"
    # logger = logging.getLogger(__name__)
    # logger.setLevel(logging.INFO)

    processes = []
    result_queue = multiprocessing.Queue()

    # Start 9 processes, passing an argument to each
    for index in range(0, N_PROCESSES):
        args = (index, N_PROCESSES, result_queue, DATADIR, TMPDIR)
        process = multiprocessing.Process(target=worker_function, args=args)
        processes.append(process)
        process.start()

    # Wait for all processes to complete
    for process in processes:
        process.join()

    results = []
    while not result_queue.empty():
        results.extend(result_queue.get())

    # sort by index

    print("All processes have completed.")
    print(results)