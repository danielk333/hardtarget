hardtarget convert eiscat ~/data/spade/beamparks_raw/leo_bpark_2.1u_NO@uhf -o ~/data/spade/beamparks_raw/leo_bpark_2.1u_NO@uhf_drf
hardtarget gmf ~/data/spade/beamparks_raw/leo_bpark_2.1u_NO@uhf_drf/ uhf --config ./examples/cfg/test.ini -o ~/data/spade/beamparks_analyzed/leo_bpark_2.1u_NO@uhf/ --progress -g cuda
# Since the experiment records the tx signal, to remove clutter we must choose a clutter removal that
# includes the tx time and the desired removal region
hardtarget plot_drf ~/data/spade/beamparks_raw/leo_bpark_2.1u_NO@uhf_drf/ -s "2021-04-12T12:15:40" -e "2021-04-12T12:16:10" --clutter_removal 2300.0 --axis_units

