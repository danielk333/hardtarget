from pathlib import Path
import logging
from hardtarget.analysis import compute_gmf

if __name__ == '__main__':

    # test
    P_DRF = "/cluster/projects/p106119-SpaceDebrisRadarCharacterization/drf"
    product = Path(P_DRF) / "leo_bpark_2.1u_NO-20220408-UHF"
    dst = "/tmp/ht"
    gmf_config = "/cluster/projects/p106119-SpaceDebrisRadarCharacterization/config/gmf_config.ini"
    gmf_method = "fdpt"
    gmf_implementation = "numpy"
    chnl = product.name.split('-')[-1].lower()
    rx = (str(product), chnl)
    tx = (str(product), chnl)
    job = {"idx": 0, "N": 1}  
    progress = True
    clobber = True

    # logging
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)


    logger.info(f"Analyze {product.name}")
    try:
        results = compute_gmf(
            rx,
            tx,
            config=gmf_config,
            output=dst,
            job=job,
            progress=progress,
            gmf_method=gmf_method,
            gmf_implementation=gmf_implementation,
            clobber=clobber,
            # start_time=args.start_time,
            # end_time=args.end_time,
            # relative_time=args.relative_time,
            progress_position=job["idx"],
            logger=logger
        )
    except KeyboardInterrupt:
        if logger:
            logger.warning("Analyze interrupted")

