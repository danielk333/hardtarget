#!/usr/bin/env python3

import numpy as n
import matplotlib.pyplot as plt
import pickle
import digital_rf as drf
import os
import scipy.interpolate as sint
import scipy.constants as c

class raw_sim:
    def __init__(self,freq=230e6,cohint_len=0.1,pulse_len=2000,bit_len=10,sr=1000000,ipp=10000):
        # radar configuration
        self.freq=freq
        self.pulse_len=int(pulse_len)
        self.bit_len=int(bit_len)
        self.sr=int(sr)
        self.ipp=int(ipp)
        self.n_bits=int(pulse_len/bit_len)
        self.wavelength=c.c/self.freq

    def raw_voltage(self,d,dirname="/media/j/ebd77b41-7efd-4238-b6f8-2b17bc33c84c/debris"):
        print("simulating raw voltage")


        
        os.system("mkdir -p %s"%(dirname))
        states=d["states"]
        n_objects = len(states)
        n_tx = len(d["observations"][0])
        n_rx = len(d["observations"][0][0])
    
        t = d["t"]

        min_t = n.min(t)
        max_t = n.max(t)
        delta_t=max_t-min_t

        # how many ipps do we simulate
        n_ipp = int(delta_t*self.sr/self.ipp)

        observations = d["observations"]
        n_rx=0
        passes=[]
        for oi in range(n_objects):
            for txi in range(n_tx):
                txo = observations[oi][txi]
                n_rx =len(txo)
                
                for rxi in range(n_rx):
                    rxo = txo[rxi]
                    n_passes=len(rxo)
                    
                    for pass_i in range(n_passes):
                        po=rxo[pass_i]
                        print(po.keys())
                        rfun=sint.interp1d(po["t"],po["range"])
                        rrfun=sint.interp1d(po["t"],po["range_rate"])
                        rrdt=n.diff(po["t"])[0]
                        rrrfun=sint.interp1d(po["t"],n.gradient(po["range_rate"],rrdt))
                        passes.append({"t0":n.min(po["t"]),"t1":n.max(po["t"]),"oid":oi,"rxi":rxi,"po":po,"phase":0.0,"rfun":rfun,"rrfun":rrfun,"rrrfun":rrrfun})

        print("n_objects %d n_rx %d"%(n_objects,n_rx))

        for p in passes:
            print("oid %d rxi %d t0 %1.2f t1 %1.2f"%(p["oid"],p["rxi"],p["t0"],p["t1"]))

        # go for it!
        dwos=[]
        arrs=[]

        i0=int(n.min(t)*self.sr)
        
        tx_chdir="%s/tx"%(dirname)
        os.system("rm -rf %s"%(tx_chdir))        
        os.system("mkdir -p %s"%(tx_chdir))
        tx_dwo = drf.DigitalRFWriter(tx_chdir,
                                     n.complex64,
                                     3600,
                                     1000,
                                     i0,
                                     self.sr,
                                     1,
                                     "fake_uuid",
                                     0,
                                     False,
                                     True,
                                     1,
                                     True,
                                     True)
        txz = n.zeros(self.ipp,dtype=n.complex64)
         
        for rxi in range(n_rx):
            chdir="%s/ch%03d"%(dirname,rxi)
            os.system("rm -rf %s"%(chdir))                    
            os.system("mkdir -p %s"%(chdir))
            dwo = drf.DigitalRFWriter(chdir,
                                      n.complex64,
                                      3600,
                                      1000,
                                      i0,
                                      self.sr,
                                      1,
                                      "fake_uuid",
                                      0,
                                      False,
                                      True,
                                      1,
                                      True,
                                      True)
            arr = n.zeros(self.ipp,dtype=n.complex64)
            dwos.append(dwo)
            arrs.append(arr)

        n_passes=len(passes)

        # time vector
        tipp=n.arange(self.ipp+1,dtype=n.float64)/float(self.sr)

        print("generating tx")
        for i in range(n_ipp):

            tnow = float(i)*float(self.ipp)/float(self.sr)
            
            txz[:]=0.0
            for bi in range(self.n_bits):
                txz[(bi*self.bit_len):(bi*self.bit_len+self.bit_len)]=n.complex64(n.sign(n.random.randn(1)))

            tx_dwo.rf_write(txz)
        tx_dwo.close()

        txd = drf.DigitalRFReader(dirname)
        txb=txd.get_bounds("tx")
        print(txb)
        
        print("Generating RXs %d"%(n_passes))
        
        for i in range(n_ipp):
            inow = i0+i*self.ipp
            tnow = inow/float(self.sr)

            # zero echo
            for ci in range(n_rx):
                arrs[ci][:]=0.0
                
            for pi in range(n_passes):
                po=passes[pi]
                if tnow > po["t0"] and tnow < po["t1"]:

                    rxi=po["rxi"]

                    rdelta = po["rrfun"](tnow)*tipp + 0.5*po["rrrfun"](tnow)*tipp**2.0
                    csin=n.exp(1j*2.0*n.pi*rdelta/self.wavelength)*n.exp(1j*po["phase"])
                    po["phase"]=n.angle(csin[-1])

                    # total propagation range in samples
                    rs = int(n.round((po["rfun"](tnow)/c.c)*self.sr))

                    print("pass visible! t=%1.2f rs %d rxi %d oid %d r %1.2f rr %1.2f rrr %1.2f"%(tnow,rs,po["rxi"],po["oid"],po["rfun"](tnow),po["rrfun"](tnow),po["rrrfun"](tnow)))                    
                    txi0 = inow - rs
#                    print(txi0)
#                    print("inow %d"%(inow))
                    if txi0 > txb[0] and (txi0+self.ipp) < txb[1]:
                        txz = txd.read_vector_c81d(txi0,self.ipp,"tx")
                        sim_echo = csin[0:self.ipp]*txz
                        arrs[rxi][:]=sim_echo
                        if False:
                            plt.plot(sim_echo.real)
                            plt.plot(sim_echo.imag)                    
                            plt.show()
            for ci in range(n_rx):
                dwo=dwos[rxi]
                z=arrs[rxi]
                dwo.rf_write(z)
            
            

        tx_dwo.close()
        for rxi in range(n_rx):
            dwos[rxi].close()



if __name__ == "__main__":
    
    sample_rate = 1000000
    

    # pulse length, number pulses integrated coherently, 
    d=pickle.load(open("data/e3d_htpl_end_to_end/tracking.pickle","rb"))

    # coherent integration period
    dt = 0.1
    # pulse length (microseconds)
    pulse_length=2000 
    ipp=20000
    
    rs=raw_sim()
    rs.raw_voltage(d)
    
#                    for ti in range(n_t):
 #                       print(pass_obj["metas"][ti]["pulse_length"]*1e6)
  #                      print(pass_obj["metas"][ti]["ipp"]*1e6)
  #                     print(pass_obj["metas"][ti]["n_ipp"])

            
            
        
        
    
