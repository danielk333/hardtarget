# Hardtarget matched filter analysis output
-------------
This describes **Hardtarget_MF**,the data format produced by [analyse api](../analyse/api_mf.md) for matched
filter processes. Each analysis will return a [AnalysedResult](../reference/hardtarget/types/types.md#hardtarget.types.types.AnalysedResult)
object, but the content of this object will vary based on a output path has been declared or not.

::: hardtarget.types.types.AnalysedResult

## Save to path

If a output path is declared the results from the [analysis](../analyse/api_mf.md) will be stored at given path
in folders, named after the hour of the measurement analysed. Within each folder, there is a set of
*HDF5* files, each including relevant datasets, and named by epoch timestamp in seconds. The parent directory
of the stored that is available in the output [AnalysedResult](../reference/hardtarget/types/types.md#hardtarget.types.types.AnalysedResult)`["dir"]`
and the path to each file can be found in [AnalysedResult](../reference/hardtarget/types/types.md#hardtarget.types.types.AnalysedResult)`["files"]`
```bash

    2021-04-12T11-00-00/
    └── mf-1618228774000000.h5
    └── mf-1618228776000000.h5
    └── ...
```
For more detailed **.h5** filestructure see [store_data](../reference/hardtarget/data_handling/store_params.md)

### Data stored
---
In the **.h5** file the [Experiment parameters](../reference/hardtarget/types/types.md) and the process
specific[Configuration parameters](../reference/hardtarget/types/types.md#hardtarget.types.types.CfgParams) and
[Process parameters](../reference/hardtarget/types/types.md#hardtarget.types.types.ProParams) are stored to be able to replicate and
analyse the analysis. Furthermore the analysis output is stored, the output is not in the same
format as during the analysis, each object in the output has had further attributes added to
it, following the [DataItem](../reference/hardtarget/types/types.md#hardtarget.types.types.DataItem) structure.
Depending on the process (e.g [GMF](../analyse/gmf_process.md)/[DPT](../analyse/dpt_process.md)/
[Optimize](../analyse/optimization_process.md)) the data will be different as the output will vary.

::: hardtarget.types.types.DataItem


## Store in RAM

If running a small analysis and there is no need to store the output the data can be stored in RAM.
By not declaring any output path to the analysis the output can be retrived from [AnalysedResult](../reference/hardtarget/types/types.md#hardtarget.types.types.AnalysedResult)`["data"]`, this will be a dict containing the sample start as key and contains the output,
experiment parameters, configuration parameters and process parameters


```Python
out, exp, cfg, pro = analysed_result["data"][00001]
```
