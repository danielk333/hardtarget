import analyze_gmf as ag
import sim_raw as sr
import gmf_opts as go
import analyze_fine as af
import numpy as n
import os

if __name__ == "__main__":
    # configuration options
    sr_mhz=1
    r0=1000e3
    v0=2e3
    a0=80.0
    # after signal processing
    snr=10000.0
    tx_len=2000*sr_mhz
    ipp=10000*sr_mhz
    bit_len=100*sr_mhz
    n_ipp=10
    snr_per_sample_lin=snr/tx_len/n_ipp
    snr_per_sample = 10.0*n.log10(snr_per_sample_lin)
    noise_pwr=(1.0/snr_per_sample_lin)*tx_len*n_ipp
    print("snr %1.2f (dB) noise_amp %1.2f"%(snr_per_sample,n.sqrt(noise_pwr)))
    
    # let's make this self contained by generating the configuration file
    # using this script
    cfg ="""
    [config]
    n_ipp=10
    data_dirs=["/media/j/ebd77b41-7efd-4238-b6f8-2b17bc33c84c/debsim"]
    n_range_gates=1000
    ground_clutter_length=0
    min_acceleration=-200.0
    max_acceleration=200.0
    acceleration_resolution=0.01
    save_parameters=true
    doppler_sign=1.0
    rx_channel="ch000"
    tx_channel="tx"
    radar_frequency=230e6
    output_dir="./spade_det"
    debug_plot=false
    debug_plot_acc=false
    debug_print=false
    round_trip_range=false
    reanalyze=true
    num_cohints_per_file=1
    use_gpu=false
    snr_thresh=10.0
"""
    
    with open("cfg/sim.ini","w") as f:
        f.writelines(cfg)
        # sample-rate specific configuration options
        f.write("ipp=%d\n"%(ipp))
        f.write("tx_pulse_length=%d\n"%(tx_len))
        f.write("sample_rate=%d\n"%(sr_mhz*1000000))
        f.write("range_gate_0=%d\n"%(100*sr_mhz))
        f.write("range_gate_step=%d\n"%(10*sr_mhz))
        f.write("frequency_decimation=%d\n"%(25*sr_mhz))

    # simulate a measurement with range, range-rate and acceleration.
    sr.simple_sim(dirname="/media/j/ebd77b41-7efd-4238-b6f8-2b17bc33c84c/debsim",
                  r0=r0,     # m
                  v0=v0,        # m/s
                  a0=a0,       # m/s^2
                  ipp=ipp,    # samples
                  tx_len=tx_len,  # samples
                  bit_len=bit_len,  # samples
                  n_ipp=n_ipp+1,  # pad one ipp
                  freq=230e6,
                  sr=1000000*sr_mhz,
                  snr=snr_per_sample)     # one sample snr

    conf=go.gmf_opts("cfg/sim.ini")
    # remove old analysis
    os.system("rm spade_det/*/*.h5")

    
    ag.analyze_gmf(conf)

    res=af.fine_tune(conf,noise_pwr=noise_pwr)
    print(res)
