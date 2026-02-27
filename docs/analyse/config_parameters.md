# Configuration Parameters
---

For any analysis a configuration needs to be provided, the configuration is the base of the process to
determine coherent integration lengths, range_gates, decimation and more. It will determine the results
quality and analysis time. There are two ways of defining the configuration parameters.


## Configuration **.ini** file
---
The first option is with a **.ini** file, an example can be seen below. Here you can configure any parameters
wanted and several different process specific sections at onece. The proccessing section which is the main
section can be empty but must atleast be present, the configuration will then be based on the default values.
The default values and desciption of the configuration parameters can be found at
[CfgParams](../reference/hardtarget/types/types.md#hardtarget.types.types.CfgParams).

```ini
    [processing]
        node_gpus = 1
        n_ipp = 10
        ipp_offset = 0
        samp_offset= 0
        min_range_gate = 6800
        max_range_gate = 7280
        range_gate_step = 1
        tx_amp_limit = 1.0
        num_cohints_per_file = 10
    [section...]
        ...
        ...
```

As seen above there is a another section after the processing section, this section is for the specific
method that will be used. Some available can be seen below, even if all are defined in the .ini file only the
section relevant to the method running will be extracted.

```ini
    [target_estimation]
        range_gate_sub_resolution = 5
        frequency_decimation = 16
        clutter_length = 0
        min_acceleration = -200.0
        max_acceleration = 200.0
        optimization = False
    [dpt]
        ipp_delay_parameter = 10
    [gmf]
        acceleration_steps = 10

```

A real example of a configuration file for a target estimation gmf analysis could then look as:

```ini
  [processing]
        n_ipp=10
        ipp_offset=0
        samp_offset=3
        min_range_gate=6660
        max_range_gate=6680
        range_gate_step=1
        num_cohints_per_file=10
        node_gpus=1
    [target_estimation]
        min_acceleration=-300.0
        max_acceleration=300.0
        frequency_decimation=1
    [gmf]
        acceleration_steps = 10

```

Note that a sample offset has been added, this is due to that in the measurement file the file start is not
at the ipp start. This config could then be used for the analysis as:

```python
from hardtarget import analyse
from pathlib import Path

result = analyse(
        path=Path("../some/measurement/"),
        config=Path("path/to/config.ini"),
        method = "target_estimation",
        method_lib = "fgmf",
        start_time=0,
        end_time=100000,
        relative_time=True,
    )
```

## Process specific CfgParams

When running the analysis in a python script or similar then the **.ini** file is not always the goto method.
If you want to define the configuration parameters in place you could use the **Process specific CfgParams**,
this is any derived class that has [CfgParams](../reference/hardtarget/types/types.md#hardtarget.types.types.CfgParams)
as base class and is connected to a specific [Process](../reference/hardtarget/process/process.md).
In this case it is of much importance that the derived class is connected to the analysis process running,
otherwise it will fail. An example for a gmf process configuration could be:

```python
from hardtarget.matched_filter.gmf.types import GMFCfgParams

cfg = GMFCfgParams(
    n_ipp=1,
    ipp_offset=0,
    samp_offset=3,
    min_range_gate=4000,
    max_range_gate=8000,
    min_acceleration=0,
    max_acceleration=0,
    range_gate_step=1,
    frequency_decimation=1,
    num_cohints_per_file=10,
    node_gpus=1,
    acceleration_steps=1,
)
```
The acceleration_steps parameter is unique for the [GMFCfgParams](../reference/hardtarget/matched_filter/gmf/types.md#hardtarget.matched_filter.gmf.types.GMFCfgParams) and not a part of the base class [CfgParams](../reference/hardtarget/types/types.md#hardtarget.types.types.CfgParams). Examining the  [GMF](../reference/hardtarget/matched_filter/gmf/gmf_process.md) process it can be seen that it is aswell connected to the process, thus they are compatible.
The configuration object above could then be used in a analysis as:

```python
from hardtarget import analyse
from hardtarget.types.constants import TargetEstimationMethod, AnalysisMethod

result = analyse(
        path=Path("../some/measurement/"),
        config=cfg,
        method=AnalysisMethod.target_estimation,
        method_lib=TargetEstimationMethod.fgmf, # Note: method is gmf as the cfg param
        start_time=0,
        end_time=100000,
        relative_time=True,
    )
```