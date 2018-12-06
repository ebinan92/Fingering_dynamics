import time
import matplotlib.pyplot as plt
#import numpy as np
from numba import jit
import cupy as np
from cupy import testing
import numpy
import chainer.cuda
#Reference
'''https://aip.scitation.org/doi/10.1063/1.866726'''

M = 256  # number of transverse collocation points
N = 128  # number of vertical collocation points
R = 3  # viscosity ratio
A = 1  # maybe don't need
Pe = 500  # U(velocity)*M /D(dispersion)
DELTA_T = 0.1
MAX_T = 300

#@testing.gpu
class Compute:
    #@testing.for_all_dtypes()
    def __init__(self,dtype):
        self.c = np.hstack([np.random.rand(N, int(2 * M / 4),dtype=np.float32), np.zeros((N, int(2 * M / 4)))])
        self.x = np.array([(m / M) * A * Pe for m in range(M)], dtype=np.float32)
        self.y = np.array([(n / N) * Pe for n in range(N)],dtype=np.float32)
        a = np.asarray(range(0, int(M / 2)),dtype=np.float32) * 2 * np.pi / (A * Pe)
        b = np.asarray(range(int(-M / 2 + 1), 0),dtype=np.float32) * 2 * np.pi / (A * Pe)
        self.km = np.concatenate((a, np.concatenate((np.array([0]), b)))).astype(np.float32)
        c = np.asarray(range(0, int(N / 2)), dtype=np.float32) * 2 * np.pi / Pe
        d = np.asarray(range(int(-N / 2 + 1), 0),dtype=np.float32) * 2 * np.pi / Pe
        self.kn = np.concatenate((c, np.concatenate((np.array([0]), d)))).astype(np.float32)
        self.km_km = np.tile(self.km, (N, 1)).astype(np.float32)
        self.kn_kn = np.tile(self.kn, (M, 1)).astype(np.float32).T
        #print(self.kn_kn.shape)
        self.c_hat = np.fft.fft2(self.c).astype(np.complex64)
        self.phi_x = np.zeros((N, M),dtype=np.complex)
        self.phi_y = np.zeros((N, M),dtype=np.complex)
        self.c_x = np.fft.ifft2(self.c_hat * self.km_km * 1.0j).astype(np.complex64)
        self.c_y = np.fft.ifft2(self.c_hat * self.kn_kn * 1.0j).astype(np.complex64)
        self.km2kn2 = self.km_km ** 2 + self.kn_kn ** 2

    def getJ_hat(self):
        J = self.phi_y * self.c_x - self.phi_x * self.c_y
        J_hat = np.fft.fft2(J)
        return J_hat

    def getPhi_hat(self):
        # w_hat/0 = 0とする
        temp1 = R*self.c_hat*1.0j*self.kn_kn
        temp2 = 1.0j*R*(self.km_km*self.c_x+self.kn_kn*self.c_y) + self.km2kn2
        temp1 = chainer.cuda.to_cpu(temp1).astype(np.complex)
        temp2 = chainer.cuda.to_cpu(temp2).astype(np.complex)
        phi_hat = - numpy.divide(temp1, temp2, out=numpy.zeros_like(temp1), where=chainer.cuda.to_cpu(self.km2kn2) != 0)
        phi_hat = np.asarray(phi_hat)
        return phi_hat

    def updatePhi_xAndPhi_y(self, phi_hat):
        self.phi_x = np.fft.ifft2(phi_hat * self.km_km * 1.0j).real
        self.phi_y = np.fft.ifft2(phi_hat * self.kn_kn * 1.0j).real

    def getC_barbyAdams(self, J_hat, J_hat_before):
        c_tilda = DELTA_T * (- 1.5 * J_hat + 0.5*J_hat_before) + self.c_hat
        c_bar = c_tilda * np.exp(-self.km2kn2 * DELTA_T)
        return c_bar

    def getNextC(self, J_hat, J_bar, c_hat, c_bar):
        nextc_hat = c_hat - DELTA_T * ((J_bar + J_hat) / 2 + self.km2kn2 * (c_bar + c_hat) / 2)
        return nextc_hat

    def updateC(self, c):
        self.c = c
        self.c_hat = np.fft.fft2(self.c)
        self.c_x = (np.fft.ifft2(self.c_hat * self.km_km * 1.0j))
        self.c_y = (np.fft.ifft2(self.c_hat * self.kn_kn * 1.0j))

#@jit()

def main():
    t1 = time.time()
    cm = Compute()
    cc = np.array([cm.c],dtype=np.complex64)
    J_hat_before = np.array([]).astype(np.float32)
    t = 0
    while t < MAX_T:
        c_hat = cm.c_hat
        J_hat = cm.getJ_hat()
        if t == 0:
            t += DELTA_T
            J_hat_before = J_hat
            cc = np.concatenate((cc, np.array([cm.c], dtype=np.complex64)), axis=0)
            continue
        else:
            c_bar = cm.getC_barbyAdams(J_hat, J_hat_before)
        cm.updateC(np.fft.ifft2(c_bar))
        phi_bar = cm.getPhi_hat()
        cm.updatePhi_xAndPhi_y(phi_bar)
        J_bar = cm.getJ_hat()
        nextc_hat = cm.getNextC(J_hat, J_bar, c_hat, c_bar)
        nextc = (np.fft.ifft2(nextc_hat))
        cm.updateC(nextc)
        phi_hat = cm.getPhi_hat()
        cm.updatePhi_xAndPhi_y(phi_hat)
        cc = np.concatenate((cc, np.array([nextc],dtype=np.complex64)),axis=0)
        J_hat_before = J_hat
        t += DELTA_T
        print(t)
    t2 = time.time()
    print(t2 - t1)
    plt.figure()
    plt.pcolor(cm.x, cm.y, cc[int(MAX_T / DELTA_T)].real,label='R='+str(R)+','+'Pe='+str(Pe))
    plt.colorbar()
    plt.legend()
    plt.show()

# if __name__ == '__main__':
#     #np.seterr(all='raise')
#     # try:
#     #     main()
#     # except Warning as e:
#     #     print(e)
#     main()