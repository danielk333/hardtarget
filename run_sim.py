
# First stage course GMF grid search
# Numpy, C, CUDA
import analyze_gmf as ag
# Simulate raw voltage measurement
import sim_raw as sr
# Options class
import gmf_opts as go
import analyze_fine as af
import numpy as n
import os
import time
import h5py
import scipy.constants as c


def one_cohint(sr_mhz=1,
               r0=1000e3,
               v0=2e3,
               a0=80.0,
               snr=1000.0,
               tx_len_us=2000,
               ipp_us=10000,
               bit_len_us=100,
               max_doppler_vel=10e3,
               radar_frequency=230e6,
               n_ipp=10):

    dirname="/scratch/data/juha/debsim"

    # after signal processing
    tx_len=tx_len_us*sr_mhz
    ipp=ipp_us*sr_mhz
    bit_len=bit_len_us*sr_mhz

    max_dop_shift = n.abs(2.0*radar_frequency*max_doppler_vel/c.c)
    freq_dec = n.round(sr_mhz*1e6/max_dop_shift/2.0)
    print("freq_dec %d"%(freq_dec))
    
    # noise calculate
    snr_per_sample_lin=snr/tx_len/n_ipp
    snr_per_sample = 10.0*n.log10(snr_per_sample_lin)
    noise_pwr=(1.0/snr_per_sample_lin)*tx_len*n_ipp
    print("snr %1.2f (dB) noise_amp %1.2f"%(snr_per_sample,n.sqrt(noise_pwr)))
    
    # configuration for course search
    cfg ="""
    [config]
    n_range_gates=2000
    ground_clutter_length=0
    min_acceleration=0.0
    max_acceleration=200.0
    acceleration_resolution=0.2
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
    use_gpu=true
    snr_thresh=10.0
    """
    with open("cfg/sim.ini","w") as f:
        f.writelines(cfg)
        # sample-rate specific configuration options
        f.write("data_dirs=[\"%s\"]\n"%(dirname))
        f.write("ipp=%d\n"%(ipp))
        f.write("tx_pulse_length=%d\n"%(tx_len))
        f.write("sample_rate=%d\n"%(sr_mhz*1000000))
        f.write("range_gate_0=%d\n"%(100*sr_mhz))
        f.write("range_gate_step=%d\n"%(5*sr_mhz))
        f.write("frequency_decimation=%d\n"%(freq_dec))
        f.write("n_ipp=%d"%(n_ipp))

    # another configuration for fine-tuning the result
    fine_tune_cfg ="""
    [config]
    n_range_gates=100
    ground_clutter_length=0
    min_acceleration=-200.0
    max_acceleration=200.0
    acceleration_resolution=0.02
    save_parameters=true
    doppler_sign=1.0
    rx_channel="ch000"
    tx_channel="tx"
    radar_frequency=230e6
    output_dir="./spade_fine"
    debug_plot=false
    debug_plot_acc=false
    debug_print=false
    round_trip_range=false
    reanalyze=true
    num_cohints_per_file=1
    use_gpu=true
    snr_thresh=10.0
    range_gate_step=1
    """
    
    with open("cfg/sim_fine.ini","w") as f:
        f.writelines(fine_tune_cfg)
        # sample-rate specific configuration options
        f.write("data_dirs=[\"%s\"]\n"%(dirname))
        f.write("ipp=%d\n"%(ipp))
        f.write("tx_pulse_length=%d\n"%(tx_len))
        f.write("sample_rate=%d\n"%(sr_mhz*1000000))
        f.write("range_gate_0=%d\n"%(100*sr_mhz))
        f.write("frequency_decimation=%d\n"%(freq_dec))
        f.write("n_ipp=%d"%(n_ipp))
        
    # simulate a measurement with range, range-rate and acceleration.
    sr.simple_sim(dirname=dirname,
                  r0=r0,            # m
                  v0=v0,            # m/s
                  a0=a0,            # m/s^2
                  ipp=ipp,          # samples
                  tx_len=tx_len,    # samples
                  bit_len=bit_len,  # samples
                  n_ipp=n_ipp+1,    # pad one ipp
                  freq=230e6,
                  sr=1000000*sr_mhz,
                  snr=snr_per_sample)     # one sample snr

    conf=go.gmf_opts("cfg/sim.ini")
    conf_fine=go.gmf_opts("cfg/sim_fine.ini")    
    # remove old analysis
    os.system("rm spade_det/*/*.h5")

    t0 = time.time()
    # run through data using coarse GMF, one cohint only
    ag.analyze_gmf(conf,n_ints=1)
    t1 = time.time()
    gmf_cpu_time = t1-t0

    t0 = time.time()    
    # fine tune parameters using non-linear least squares
    res=af.fine_tune(conf, conf_fine,noise_pwr=noise_pwr)
    t1 = time.time()
    fine_tune_cpu_time = t1-t0
    
    print("Analysis result")
    print(res)
    print("Should be")
    print("[%1.2f, %1.2f, %1.2f, %1.2f]"%(snr,r0,v0,a0))
    return([res[0],res[1],res[2],res[3],gmf_cpu_time,fine_tune_cpu_time])

def snr_sweep():
    r0=1000e3
    v0=2e3
    a0=80.0
    sr_mhz=5
    # sweeps
    # snr sweep
    snrs = [30.0,40.0,50.0]
    n_reps=10
    snr_results=[]
    for snri,snr in enumerate(snrs):
        for ri in range(n_reps):
            dt=0.1
            print("snr %1.2f"%(snr))
        
            res=one_cohint(sr_mhz=sr_mhz,
                           r0=r0,
                           v0=v0,
                           a0=a0,
                           snr=10**(snr/10.0),
                           tx_len_us=2000,
                           ipp_us=10000,
                           bit_len_us=100,
                           n_ipp=10)
            
            snr_results.append([snr-10.0*n.log10(res[0]),r0-res[1],v0-res[2],a0-res[3],res[4]/dt,res[5]/dt])
        
    snr_results=n.array(snr_results)
    print(snr_results.shape)
    print(snr_results)
    ho=h5py.File("snr_sweep-%d.h5"%(int(time.time())),"w")
    ho["snrs"]=snrs
    ho["snr_err"]=snr_results[:,0]
    ho["range_err"]=snr_results[:,1]
    ho["vel_err"]=snr_results[:,2]
    ho["acc_err"]=snr_results[:,3]
    ho["gmf_time"]=snr_results[:,4]
    ho["finetune_time"]=snr_results[:,5]
    ho.close()


def n_ipp_sweep():
    r0=1000e3
    v0=2e3
    a0=80.0
    # sweeps
    # snr sweep
    n_ipps = [5,10,15,20,25,30,35,40]
    n_ipp_results=[]
    snr=30.0
    for ippi,n_ipp in enumerate(n_ipps):
        
        dt=0.01*n_ipp
        print("n_ipp %d"%(n_ipp))
        
        res=one_cohint(sr_mhz=20,
                       r0=r0,
                       v0=v0,
                       a0=a0,
                       snr=10**(snr/10.0),
                       tx_len_us=2000,
                       ipp_us=10000,
                       bit_len_us=100,
                       n_ipp=n_ipp)
        
        n_ipp_results.append([snr-10.0*n.log10(res[0]),r0-res[1],v0-res[2],a0-res[3],res[4]/dt,res[5]/dt])
        
    n_ipp_results=n.array(n_ipp_results)
    print(n_ipp_results.shape)
    print(n_ipp_results)
    ho=h5py.File("n_ipp_sweep.h5","w")
    ho["n_ipps"]=n_ipps
    ho["snr_err"]=n_ipp_results[:,0]
    ho["range_err"]=n_ipp_results[:,1]
    ho["vel_err"]=n_ipp_results[:,2]
    ho["acc_err"]=n_ipp_results[:,3]
    ho["gmf_time"]=n_ipp_results[:,4]
    ho["finetune_time"]=n_ipp_results[:,5]
    ho.close()

if __name__ == "__main__":
    snr_sweep()        
#    n_ipp_sweep()    
    
