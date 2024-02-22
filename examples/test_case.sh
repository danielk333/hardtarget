# Convert from EISCAT standard raw data format to Hardtarget DRF
hardtarget convert eiscat ~/data/spade/beamparks_raw/leo_bpark_2.1u_NO@uhf \
    -o ~/data/spade/beamparks_raw/leo_bpark_2.1u_NO@uhf_drf

# check which range gates to use to measure 240 km -> 2500 km
hardtarget check range-gates -s 240 -e 2500 ~/data/spade/beamparks_raw/leo_bpark_2.1u_NO@uhf_drf/

# To analyze the DRF
hardtarget gmf ~/data/spade/beamparks_raw/leo_bpark_2.1u_NO@uhf_drf/ uhf \
    --config ./examples/cfg/example_uhf_analysis.ini \
    -o ~/data/spade/beamparks_analyzed/leo_bpark_2.1u_NO@uhf/ \
    --progress -G cuda

# To re-analyze the DRF portion with the echo
hardtarget gmf ~/data/spade/beamparks_raw/leo_bpark_2.1u_NO@uhf_drf/ uhf \
    --config ./examples/cfg/example_uhf_analysis.ini \
    -o ~/data/spade/beamparks_analyzed/leo_bpark_2.1u_NO@uhf_cuda/ \
    --progress -G cuda --clobber \
    -s "2021-04-12T12:15:40" -e "2021-04-12T12:16:10"

mpirun -np 2 hardtarget gmf ~/data/spade/beamparks_raw/leo_bpark_2.1u_NO@uhf_drf/ uhf \
    --config ./examples/cfg/example_uhf_analysis.ini \
    -o ~/data/spade/beamparks_analyzed/leo_bpark_2.1u_NO@uhf_numpy_daf/ \
    --progress -G numpy_daf --clobber \
    -s "2021-04-12T12:15:40" -e "2021-04-12T12:16:10"


hardtarget plot drf ~/data/spade/beamparks_raw/leo_bpark_2.1u_NO@uhf_drf/ \
    -s "2021-04-12T12:15:40" -e "2021-04-12T12:16:10" --axis_units\
    --monostatic --start-range 360 --unit km --log

hardtarget plot gmf ~/data/spade/beamparks_analyzed/leo_bpark_2.1u_NO@uhf/ \
    -s "2021-04-12T12:15:40" -e "2021-04-12T12:16:10"
