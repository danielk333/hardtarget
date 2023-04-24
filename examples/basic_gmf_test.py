import matplotlib.pyplot as plt
import numpy as np
import time

from hardtarget.analysis.gmf import GMF_LIBS

gmf = GMF_LIBS['c']

z_tx=np.zeros(10000,dtype=np.complex64)
z_rx=np.zeros(12000,dtype=np.complex64)    
for i in range(10):
    z_tx[(i*1000):(i*1000+20)]=1.0
    z_rx[(i*1000+500):(i*1000+(500+20))]=0.5 # simulated "echo"

dec=10
acc_phasors=np.zeros([20,1000],dtype=np.complex64)
acc_phasors[0,:]=1.0
rgs=np.zeros(1000,dtype=np.float32)#arange(700,dtype=np.int64)
for ri in range(len(rgs)):
    rgs[ri]=ri
n_r=len(rgs)
gmf_vec=np.zeros(n_r,dtype=np.float32);
gmf_dc_vec=np.zeros(n_r,dtype=np.float32);
v_vec=np.zeros(n_r,dtype=np.float32);
a_vec=np.zeros(n_r,dtype=np.float32);
cput0=time.time()
for i in range(20):
    gmf(z_tx,z_rx,acc_phasors,rgs,dec,gmf_vec,gmf_dc_vec,v_vec,a_vec)
cput1=time.time()
plt.plot(gmf_vec)
plt.show()
plt.plot(gmf_dc_vec)
plt.show()
plt.plot(v_vec)
plt.show()
plt.plot(a_vec)
plt.show()
print("Execution time %1.2f"%(cput1-cput0))
ri=np.argmax(gmf_vec)
print("Rmax %d gmf %1.2g v %1.2f a %1.2f"%(ri,gmf_vec[ri],v_vec[ri],a_vec[ri]))
