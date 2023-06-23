## Use Case - Analysis

### Step 1 - Get Data

Raw measurement data can be downloaded from Eiscat.

Sample dataset available locally at NORCE

/processed/projects/106119_DebrisSize/beampark_raw/leo_bpark_2.1u_NO@uhf



### Step 2 - Convert to DRF data

Assuming raw data is availble on _~/Data/leo_bpark_2.1u_NO@uhf_

Use script iescat2drf.py for conversion

```bash
eiscat2drf ~/Data/leo_bpark_2.1u_NO@uhf/
```

By default, this creates a folder with data inside the directory ~/Data/leo_bpark_2.1u_NO@uhf/drf
Alternatively, use the -o option to specify a different output directory.


### Step 3 - Run GMF Analysis

Assuming rdf data is available at _~/Data/leo_bpark_2.1u_NO@uhf/drf_

Assuming also that a config file for the analyis is available at _~/Git/hard_target/cfg/test.ini_

Assuming also that an output folder exists at _tmp/_

Then use script analyzegmf to perform the analysis. Processing might be time-consuming.

```bash
analyzegmf ~/Data/leo_bpark_2.1u_NO@uhf/drf/ ~/Git/hard_target/cfg/test.ini -o tmp/ --log-level INFO
```