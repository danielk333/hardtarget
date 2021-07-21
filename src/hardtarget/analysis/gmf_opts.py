from matplotlib.pyplot import cla
import numpy as np
import scipy.constants as sc
import os
from ..config import Config
try:
    import configparser
except ImportError as e:
    import configparser2 as configparser
    
import json
import os 

# configuration of hard target analysis
#
# doppler sign convention: range reduces with positive doppler velocity.
# 
class gmf_opts(Config):
    
    @classmethod
    def from_default(cls, data_dir, output_dir, rx_channel, tx_channel):
        #Set default paramaters in a dictionary
        default_params = {
                     "n_ipp":'5',
                     "data_dirs": data_dir,
                     "sample_rate":'1000000',
                     "n_range_gates":'10000',
                     "range_gate_0":'200',
                     "range_gate_step":'1',
                     "frequency_decimation":'25',
                     "ipp":'10000',                   # microseconds 
                     "tx_pulse_length":'445',  
                     "tx_bit_length":'20',
                     "ground_clutter_length":'1500',
                     "min_acceleration":'-400.0',
                     "max_acceleration":'400.0',
                     "acceleration_resolution":"0.2",
                     "snr_thresh":"10.0",                     
                     "save_parameters":'true',
                     "doppler_sign":'1.0',
                     "rx_channel": rx_channel,
                     "tx_channel": tx_channel,
                     "radar_frequency":'500e6',
                     "reanalyze":"true",
                     "output_dir": output_dir,
                     "debug_plot":'true',
                     "debug_plot_acc":'true',
                     "debug_print":'true',
                     "round_trip_range":'true',
                     "num_cohints_per_file":'100',
                     "use_gpu":'false',
                     "use_python":'false',
                     "use_cpu":'true'
                     }

        #Return object based on default params
        return cls(default_params)

    def __str__(self):
        out="Configuration\n"
        for e in dir(self):
            if not callable(getattr(self,e)) and not e.startswith("__"):
                out+="%s = %s\n"%(e,getattr(self,e))
        return(out)

    def set_n_ranges(self, range_gate_0, n_range_gates):
        """
        Reset the number of range-gates. This is useful when reanalyzing with better resolution
        to fine-tune the result
        """
        self.n_range_gates=n_range_gates
        self.range_gate_0=range_gate_0

        # range gates to search through
        self.rgs=np.arange(self.n_range_gates)*self.range_gate_step+self.range_gate_0
        self.rgs_float=np.array(self.rgs,dtype=np.float32)
        
        # total propagation range
        self.ranges=self.rgs*sc.c/1e3/self.sample_rate

    def set_values(self):
        self.t0=None
        self.t1=None

        #Try to bypass header title if ini file is used
        try:
            self._params = self._params["config"]
        except KeyError:
            pass

        self.n_ipp=int(self._params["n_ipp"])
        self.data_dirs=self._params["data_dirs"]
        print(self.data_dirs)
        self.sample_rate=float(self._params["sample_rate"])
        self.n_range_gates=int(self._params["n_range_gates"])
        self.range_gate_0=int(self._params["range_gate_0"])
        self.range_gate_step=int(self._params["range_gate_step"])
        self.frequency_decimation=int(self._params["frequency_decimation"])
        self.ipp=int(self._params["ipp"])
        self.tx_pulse_length=int(self._params["tx_pulse_length"])
        self.ground_clutter_length=int(self._params["ground_clutter_length"])
        self.min_acceleration=float(self._params["min_acceleration"])
        self.max_acceleration=float(self._params["max_acceleration"])
        self.acceleration_resolution=float(self._params["acceleration_resolution"])
        self.snr_thresh=float(self._params["snr_thresh"])
        self.save_parameters=bool(self._params["save_parameters"])
        self.doppler_sign=float(self._params["doppler_sign"])
        self.rx_channel=self._params["rx_channel"]
        self.tx_channel=self._params["tx_channel"]
        self.radar_frequency=float(self._params["radar_frequency"])
        self.output_dir=self._params["output_dir"]
        self.debug_plot=bool(self._params["debug_plot"])
        self.debug_plot_acc=bool(self._params["debug_plot_acc"])
        self.debug_print=bool(self._params["debug_print"])
        self.debug_plot_data_read=False
        self.num_cohints_per_file=int(self._params["num_cohints_per_file"])
        self.use_gpu=bool(self._params["use_gpu"])
        self.use_python=bool(self._params["use_python"])
        self.reanalyze=bool(self._params["reanalyze"])       
        self.round_trip_range=bool(self._params["round_trip_range"])
        self.debug_gmf_output=True

    def __init__(self,paramaters):

        super().__init__(paramaters)

        print(self.get_keys())
        self.set_values()
        if self.save_parameters:
            self.save_param('config',output_dir=self.output_dir, ini=self.values_as_strings)
        
        # length of coherent integration
        self.n_fft=self.n_ipp*self.ipp

        # frequency vector 
        self.fvec=np.fft.fftfreq(int(self.n_fft/self.frequency_decimation),d=self.frequency_decimation/self.sample_rate)
        
        self.set_n_ranges(self.range_gate_0, self.n_range_gates)
        
        # range-rate is doppler-shift in hertz multiplied with wavelength 
        self.wavelength = sc.c/self.radar_frequency
        self.range_rates=self.doppler_sign*self.wavelength*self.fvec
        
        # time vector 
        times=self.frequency_decimation*np.arange(int(self.n_fft/self.frequency_decimation))/self.sample_rate
        times2=times**2.0

        # radar frequency in radians per second
        om=2.0*np.pi*self.radar_frequency

        # these are the accelerations we'll try out
        tau = self.n_ipp*self.ipp/self.sample_rate
        
        # acceleration sampled with 0.2 radian steps at the end of the coherent integration window
        delta_a = self.max_acceleration - self.min_acceleration
        self.n_accelerations = int(np.ceil( delta_a*(np.pi/self.wavelength)*tau**2.0 / self.acceleration_resolution))
        
        self.accs=np.linspace(self.min_acceleration,self.max_acceleration,num=self.n_accelerations) # m/s**2
        self.acc_phasors=np.zeros([self.n_accelerations,int(self.n_fft/self.frequency_decimation)],dtype=np.complex64)

        # precalculate phasors corresponding to different accelerations
        for ai,a in enumerate(self.accs):
            self.acc_phasors[ai,:]=np.exp(-1j*2.0*np.pi*(self.doppler_sign*0.5*self.accs[ai]/self.wavelength)*times2)
            
        # how many extra ipps do we need to read for coherent integration
        self.n_extra=int(np.ceil(np.max(self.rgs)/self.ipp))+1

        # this stencil is used to block tx pulses and ground clutter
        self.read_length=self.n_fft+self.n_extra*self.ipp
        self.rx_stencil=np.ones(self.read_length,dtype=np.float32)
        # this stencil is used to select tx pulses
        self.tx_stencil=np.ones(self.read_length,dtype=np.float32)    

        # for each coherently integrated IPP, create stencils
        for k in range(self.n_ipp+self.n_extra):
            self.tx_stencil[(k*self.ipp+self.tx_pulse_length):(k*self.ipp+self.ipp)]=0.0        
            # pad zeros to rx
            self.rx_stencil[(k*self.ipp):(k*self.ipp+self.tx_pulse_length+self.ground_clutter_length)]=0.0

        if self.debug_plot_acc:
            import matplotlib.pyplot as plt
            plt.plot(self.accs,self.acc_phasors.real[:,int(self.n_fft/self.frequency_decimation)-1])
            plt.plot(self.accs,self.acc_phasors.imag[:,int(self.n_fft/self.frequency_decimation)-1])
            plt.plot(self.accs,self.acc_phasors.real[:,int(self.n_fft/self.frequency_decimation)-1],"*")
            plt.plot(self.accs,self.acc_phasors.imag[:,int(self.n_fft/self.frequency_decimation)-1],"*")
            plt.xlabel("Accelerations (m/s^2)")
            plt.title("Acceleration phasors at maximum coherent integration")
            plt.show()
            # plot acceleration phasors
            plt.subplot(121)
            plt.pcolormesh(times/1e-3,self.accs,self.acc_phasors.real)
            plt.ylabel("Acceleration ($m/s^2$)")
            plt.xlabel("Time (ms)")
            plt.colorbar()
            plt.title("Acceleration phasors (Real component)")
            plt.subplot(122)
            plt.pcolormesh(times/1e-3,self.accs,self.acc_phasors.imag)
            plt.ylabel("Acceleration ($m/s^2$)")
            plt.xlabel("Time (ms)")
            plt.colorbar()
            plt.title("Acceleration phasors (Im component)")
            plt.show()

            plt.plot(np.arange(self.read_length),self.tx_stencil,label="tx stencil")
            plt.plot(np.arange(self.read_length),self.rx_stencil,label="rx stencil")
            plt.title("TX and RX stencils")
            plt.legend()
            plt.xlabel("Samples")
            plt.show()

if __name__ == "__main__":
    o=gmf_opts("cfg/esr_2018.ini")
    print(o)
