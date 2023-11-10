# Convert from EISCAT standard raw data format to Hardtarget DRF
hardtarget convert eiscat ~/data/spade/beamparks_raw/leo_bpark_2.1u_NO@uhf \
    -o ~/data/spade/beamparks_raw/leo_bpark_2.1u_NO@uhf_drf

# To analyze the DRF
hardtarget gmf ~/data/spade/beamparks_raw/leo_bpark_2.1u_NO@uhf_drf/ uhf \
    --config ./examples/cfg/test.ini \
    -o ~/data/spade/beamparks_analyzed/leo_bpark_2.1u_NO@uhf/ \
    --progress -g cuda

# To re-analyze the DRF portion with the echo
hardtarget gmf ~/data/spade/beamparks_raw/leo_bpark_2.1u_NO@uhf_drf/ uhf \
    --config ./examples/cfg/test.ini \
    -o ~/data/spade/beamparks_analyzed/leo_bpark_2.1u_NO@uhf/ \
    --progress -g cuda --clobber \
    -s "2021-04-12T12:15:40" -e "2021-04-12T12:16:10"


# Since the experiment records the tx signal, to remove clutter we must choose a clutter removal that
# includes the tx time and the desired removal region
hardtarget plot_drf ~/data/spade/beamparks_raw/leo_bpark_2.1u_NO@uhf_drf/ \
    -s "2021-04-12T12:15:40" -e "2021-04-12T12:16:10" \
    --clutter_removal 2300.0e-6 --axis_units

