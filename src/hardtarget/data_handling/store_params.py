"""Store Analysed, Experiment, Configuration and Process parameters in a unified way"""

import logging
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Iterable

import h5py

from hardtarget.types import CfgParams, DataItem, ExpDef, GenericCfg, GenericPro, ProParams

# Python StrEnum has default lowercase for auto() but is only available from py 3.11
try:
    from enum import StrEnum
except ImportError:
    from strenum import (  # type: ignore[assignment, no-redef, unused-ignore, import-not-found]
        LowercaseStrEnum as StrEnum,  # type: ignore[import-not-found,no-redef, unused-ignore]
    )


logger = logging.getLogger(__name__)


def dump_params_to_file(
    data_items: Iterable[tuple[str, DataItem]],
    exp_params: ExpDef,
    cfg_params: CfgParams,
    pro_params: ProParams,
    outfile: Path,
    clobber: bool = False,
    mode: str = "w",
    include_params: bool = True,
) -> None:
    """
    Stores any Analysed output, ExpDef, CfgParams and ProParams and any class inherited from these in
    a standardized h5 format.

    ```
    H5 Format:
        ├ method
        ├ method_lib
        ├ OutArgs
        |      ├ data item 1
        |      └ data item 2
        ├ <Class<ExpDef>.__name__
        |      ├ parameter 1
        |      └ parameter 2
        ├ <Class<CfgParams>.__name__
        |      ├ parameter 1
        |      └ parameter 2
        └ <Class<ProParams>.__name__
               ├ parameter 1
               └ parameter 2
    ```

    Args:
        data_items: Iterable of data items to be stored in the OutArgs section. This section contains more
                    detailed information for each parameter.
        exp_params: Experiment parameters
        cfg_params: Configuration parameters related to the specific process, bound to CfgParams
        pro_params: Process parameters related to the specific process, bound to ProParams
        outfile: Path to file
        clobber (optional): Overwrite old data, default is False,
        mode (optional): File mode, default is "w"
        include_params (optional): Include parameters, if true exp, cfg and pro params will be saved,
                                   default True
    """

    with h5py.File(outfile, mode) as file:
        # Save analysis method
        method = f"{ProParams.method=}".split("=")[0].split(".")[1]
        if method not in file:
            file.create_dataset(
                f"{ProParams.method=}".split("=")[0].split(".")[1], data=pro_params.method.value
            )
        elif clobber:
            file[method] = pro_params.method.value

        # Save method library used
        if pro_params.method_lib is not None:
            method_lib = f"{ProParams.method_lib=}".split("=")[0].split(".")[1]
            if method_lib not in file:
                file.create_dataset(
                    f"{ProParams.method_lib=}".split("=")[0].split(".")[1], data=pro_params.method_lib.value
                )
            elif clobber:
                file[method_lib] = pro_params.method_lib.value

        grp_name = "OutArgs"  # TODO: Fix this to a set value
        if grp_name not in file:
            target_grp = file.create_group(grp_name)
        else:
            target_grp = file[grp_name]

        for key, item in data_items:
            if clobber and key in target_grp:
                del target_grp[key]

            if key not in target_grp:
                ds = target_grp.create_dataset(key, data=item.data)

                ds.attrs["long_name"] = item.long_name
                if item.units is not None:
                    ds.attrs["units"] = item.units
                if item.scale is not None:
                    ds.make_scale(key)
                if item.dims is not None:
                    for idx, (scale_key, label) in enumerate(item.dims):
                        scale = target_grp[scale_key]
                        ds.dims[idx].attach_scale(scale)
                        ds.dims[idx].label = label

        if include_params:

            def dump_params(file: h5py.File, param: GenericCfg | ExpDef | GenericPro) -> None:
                grp_name = type(param).__name__
                if grp_name not in file:
                    grp = file.create_group(grp_name)
                else:
                    grp = file[grp_name]
                if is_dataclass(param):
                    d = asdict(param).items()
                else:
                    d = param._asdict().items()

                for key, val in d:
                    if val is not None:
                        # h5py cant handle StrEnum so get the string by .value
                        if isinstance(val, StrEnum):
                            val = val.value
                        grp.create_dataset(key, data=val)

            dump_params(file, exp_params)
            dump_params(file, cfg_params)
            dump_params(file, pro_params)
