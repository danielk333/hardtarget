# First stage course GMF grid search
# Numpy, C, CUDA
import analyze_gmf as ag
# Simulate raw voltage measurement
import sim_raw as sr
# Options class
import gmf_opts as go
import numpy as n
import os
import time
import h5py
import scipy.constants as c
import digital_rf as drf
import gmf as g

class sim_conf:
    def __init__(self,
                 dirname="/scratch/data/juha/debsim",
                 sr_mhz=1,
                 tx_len_us=2000,
                 ipp_us=10000,
                 bit_len_us=100,
                 max_doppler_vel=10e3,
                 radar_frequency=230e6,
                 use_gpu=False,
                 n_ipp=10):
        """
        Get configuration files for coarse and fine grained GMF grid search
        """
        
        # after signal processing
        self.tx_len=tx_len_us*sr_mhz
        self.ipp=ipp_us*sr_mhz
        self.bit_len=bit_len_us*sr_mhz
        self.n_ipp=n_ipp
        self.radar_frequency=radar_frequency
        self.dirname=dirname
        self.max_dop_shift = n.abs(2.0*self.radar_frequency*max_doppler_vel/c.c)
        self.sr_mhz=sr_mhz
        self.freq_dec = n.round(self.sr_mhz*1e6/self.max_dop_shift/2.0)
        print("freq_dec %d"%(self.freq_dec))
        
        
        # configuration for course search
        cfg ="""
        [config]
        n_range_gates=2000
        ground_clutter_length=0
        min_acceleration=0.0
        max_acceleration=200.0
        acceleration_resolution=0.2
        save_parameters=false
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
        snr_thresh=10.0
        """
        with open("cfg/sim.ini","w") as f:
            f.writelines(cfg)
            # sample-rate specific configuration options
            f.write("data_dirs=[\"%s\"]\n"%(self.dirname))
            f.write("ipp=%d\n"%(self.ipp))
            f.write("tx_pulse_length=%d\n"%(self.tx_len))
            f.write("sample_rate=%d\n"%(self.sr_mhz*1000000))
            f.write("range_gate_0=%d\n"%(100*self.sr_mhz))
            f.write("range_gate_step=%d\n"%(5*self.sr_mhz))
            f.write("frequency_decimation=%d\n"%(self.freq_dec))
            f.write("n_ipp=%d\n"%(n_ipp))
            if use_gpu:
                f.write("use_gpu=true")
            else:
                f.write("use_gpu=false")
                
            
        # another configuration for fine-tuning the result
        fine_tune_cfg ="""
        [config]
        n_range_gates=100
        ground_clutter_length=0
        min_acceleration=0.0
        max_acceleration=200.0
        acceleration_resolution=0.02
        save_parameters=false
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
        snr_thresh=10.0
        range_gate_step=1
        """
        
        with open("cfg/sim_fine.ini","w") as f:
            f.writelines(fine_tune_cfg)
            # sample-rate specific configuration options
            f.write("data_dirs=[\"%s\"]\n"%(self.dirname))
            f.write("ipp=%d\n"%(self.ipp))
            f.write("tx_pulse_length=%d\n"%(self.tx_len))
            f.write("sample_rate=%d\n"%(self.sr_mhz*1000000))
            f.write("range_gate_0=%d\n"%(100*self.sr_mhz))
            f.write("frequency_decimation=%d\n"%(self.freq_dec))
            f.write("n_ipp=%d\n"%(self.n_ipp))
            if use_gpu:
                f.write("use_gpu=true")
            else:
                f.write("use_gpu=false")
            
        
        self.conf=go.gmf_opts("cfg/sim.ini")
        self.conf_fine=go.gmf_opts("cfg/sim_fine.ini")

def run_cohint(d,conf,i0,r0):
    """
    run gmf with a finer resolution. reset range gate setting with existing high res configuration
    """
    n_rg=conf.n_range_gates
    ri0 = int(n.round((r0/c.c)*conf.sample_rate)) - int(n_rg/2)
    conf.set_n_ranges(ri0,n_rg)
    
    gmf_max, gmf_dc, gmf_v, gmf_a, gmf_txp = g.analyze_ipps(d,i0,conf)

    mi=n.argmax(gmf_max)
    gmf = gmf_max[mi]
    rgi = ri0+mi
    r00 = c.c*rgi/conf.sample_rate
    v00 = gmf_v[mi]
    a00 = gmf_a[mi]
    
    return(gmf,r00,v00,a00)


def one_cohint(conf,
               r0=1000e3,
               v0=2e3,
               a0=80.0,
               snr=1000.0):

    """
    simulate measurement of one coherent integration. 
    @sr_mhz sample-rate
    @r0 two-way range (m)
    @v0 two-wave range-rate (m/s)
    @snr signal-to-noise ratio
    @tx_len_us transmit pulse length (us)
    @ipp_us    interpulse-period (us)
    @bit_len_us code bit length (us)
    @radar frequency (Hz)
    @n_ipp number of ipps to coherently integrate
    """
    # noise power calculation
    snr_per_sample_lin=snr/conf.tx_len/conf.n_ipp
    snr_per_sample = 10.0*n.log10(snr_per_sample_lin)
    noise_pwr=(1.0/snr_per_sample_lin)*conf.tx_len*conf.n_ipp
    print("snr %1.2f (dB) noise_amp %1.2f"%(snr_per_sample,n.sqrt(noise_pwr)))

    

    # simulate a measurement with range, range-rate and acceleration.
    sr.simple_sim(dirname=conf.dirname,
                  r0=r0,            # m
                  v0=v0,            # m/s
                  a0=a0,            # m/s^2
                  ipp=conf.ipp,          # samples
                  tx_len=conf.tx_len,    # samples
                  bit_len=conf.bit_len,  # samples
                  n_ipp=conf.n_ipp+1,    # pad one ipp
                  freq=conf.radar_frequency,
                  sr=1000000*conf.sr_mhz,
                  snr=snr_per_sample)     # one sample snr

    d=drf.DigitalRFReader(conf.dirname)

    # coarse grained first step analysis
    t0 = time.time()    
    i0=0
    gmf, gmf_dc, gmf_v, gmf_a, gmf_txp = g.analyze_ipps(d,i0,conf.conf)
    t1 = time.time()
    gmf_cpu_time = t1-t0


    # fine tuned analysis
    t0 = time.time()

    mi=n.argmax(gmf)
    rgi = conf.conf.range_gate_0+conf.conf.range_gate_step*mi
    r00 = c.c*rgi/conf.conf.sample_rate
    v00 = gmf_v[mi]
    
    snr00=(gmf[mi]-n.sqrt(noise_pwr))**2.0/noise_pwr

    # refine guess
    gmf01,r01,v01,a01=run_cohint(d,
                                 conf.conf_fine,
                                 i0,
                                 r00)

    snr01=(gmf01-n.sqrt(noise_pwr))**2.0/noise_pwr
            
    print("Found range %1.5f,%1.5f (km) vel %1.5f,%1.5f (km/s) a %1.5f snr %1.2f,%1.2f (dB)"%(r00/1e3,
                                                                                              r01/1e3,
                                                                                              v00/1e3,
                                                                                              v01/1e3,
                                                                                              a01,
                                                                                              10.0*n.log10(snr00),
                                                                                              10.0*n.log10(snr01)))
    res=n.array([snr01,r01,v01,a01])
    t1 = time.time()
    fine_tune_cpu_time = t1-t0
    
    print("Analysis result")
    print(res)
    print("Should be")
    print("[%1.2f, %1.2f, %1.2f, %1.2f]"%(snr,r0,v0,a0))
    return([res[0],res[1],res[2],res[3],gmf_cpu_time,fine_tune_cpu_time])


    
    

def snr_sweep():

    sconf=sim_conf(dirname="/scratch/data/juha/debsim",
                  sr_mhz=1,
                  tx_len_us=2000,
                  ipp_us=10000,
                  bit_len_us=100,
                  max_doppler_vel=10e3,
                  radar_frequency=230e6,
                  n_ipp=10)

    r0=1000e3
    v0=1e3
    a0=80.0

    # snr sweep
    snrs = [30.0,40.0,50.0]
    n_reps=10
    snr_results=[]
    for snri,snr in enumerate(snrs):
        for ri in range(n_reps):
            dt=0.1
            print("snr %1.2f"%(snr))
        
            res=one_cohint(sconf,
                           r0=r0,
                           v0=v0,
                           a0=a0,
                           snr=10**(snr/10.0))
            
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
    # ipp sweep
    r0=1000e3
    v0=1e3
    a0=80.0
    
    n_ipps = [5,10,15,20,25,30,35,40]
    n_ipp_results=[]
    snr=30.0
    for ippi,n_ipp in enumerate(n_ipps):
        
        dt=0.01*n_ipp
        print("n_ipp %d"%(n_ipp))
        sconf=sim_conf(dirname="/scratch/data/juha/debsim",
                       sr_mhz=1,
                       tx_len_us=2000,
                       ipp_us=10000,
                       bit_len_us=100,
                       max_doppler_vel=10e3,
                       radar_frequency=230e6,
                       n_ipp=n_ipp)
        
        res=one_cohint(sconf,
                       r0=r0,
                       v0=v0,
                       a0=a0,
                       snr=10**(snr/10.0))
        
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
    
    sconf=sim_conf(dirname="/scratch/data/juha/debsim",
                  sr_mhz=10,
                  tx_len_us=2000,
                  ipp_us=10000,
                  bit_len_us=100,
                  max_doppler_vel=10e3,
                  radar_frequency=230e6,
                  n_ipp=10)
    
    # analyze one coherent integration period
    res=one_cohint(sconf,
                   r0=1000e3,
                   v0=2e3,
                   a0=80.0,
                   snr=1000.0)
    print(res)

    n_ipp_sweep()
    snr_sweep()
    

    
    
