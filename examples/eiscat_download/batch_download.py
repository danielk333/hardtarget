"""
Testing multiprocessing
"""
from pathlib import Path
from hardtarget.radars.eiscat import download


def process(worker, task):
    """
    Wrapper function for Eiscat download.
    """
    day = task["date"]
    instrument = task["experiment"]
    chnl = task["type"]
    dst_dir = worker.config["dst"]
    tmp_dir = worker.config["tmp"]

    if not Path(dst_dir).is_dir():
        return False, f"config['dst']> is not directory {dst_dir}"

    if not Path(tmp_dir).exists():
        return False, f"config['tmp']> is not directory {tmp_dir}"

    target_dir = Path(dst_dir) / f"{day}-{instrument}-{chnl}"
    target_dir.mkdir(exist_ok=True, parents=True)

    # Download
    return download(day, instrument, chnl, target_dir,
                    keep_zip=False,
                    update=False,
                    tmpdir=tmp_dir,
                    logger=worker.logger,
                    progress=True)


if __name__ == "__main__":

    import logging
    from masterworker import Master

    PROJECT_T_DISK = "/NORCE/Data/600/60090/106119 - Space Debris radar characterization"

    cfg = {
        # "tmp": f"{PROJECT_T_DISK}/scratch",
        "tmp": "/tmp",
        "dst": PROJECT_T_DISK
    }

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    #formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    #ch = logging.StreamHandler()
    #ch.setFormatter(formatter)
    #logger.addHandler(ch)

    WORKERS = 1
    TASKS = "single_tasks.txt"

    master = Master(WORKERS, process,
                    config=cfg,
                    tasks=TASKS,
                    logger=logger)

    master.start_workers()
    try:
        master.run()
    except KeyboardInterrupt:
        master.stop_workers()
        master.join_workers()
        print()
