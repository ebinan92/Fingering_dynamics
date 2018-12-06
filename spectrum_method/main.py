import time
import matplotlib.pyplot as plt
import numpy as np
import copy
import matplotlib.animation as animation
from numba import jit

# import cupy as np

# Reference
'''https://aip.scitation.org/doi/10.1063/1.866726'''

M = 256  # number of transverse collocation points
N = 128  # number of vertical collocation points
R = 0  # viscosity ratio
A = 2.0  # aspect ratio
Pe = 500  # U(velocity)*N /D(dispersion)
DELTA_T = 0.1
MAX_T = 500


class Compute:
    def __init__(self):
        self.c = np.hstack([np.random.rand(N, int(2 * M / 4)), np.zeros((N, int(2 * M / 4)))])
        self.x = np.array([(m / M) * A * Pe for m in range(M)])
        self.y = np.array([(n / N) * Pe for n in range(N)])
        a = np.asarray(range(0, int(M / 2))) * 2 * np.pi / (A * Pe)
        b = np.asarray(range(int(-M / 2 + 1), 0)) * 2 * np.pi / (A * Pe)
        self.km = np.append(a, np.append(0, b))
        c = np.asarray(range(0, int(N / 2))) * 2 * np.pi / Pe
        d = np.asarray(range(int(-N / 2 + 1), 0)) * 2 * np.pi / Pe
        self.kn = np.append(c, np.append(0, d))
        self.km_km = np.tile(self.km, (N, 1))
        self.kn_kn = np.tile(self.kn, (M, 1)).T
        self.c_hat = np.fft.fft2(self.c)
        self.phi_x = np.zeros((N, M))
        self.phi_y = np.zeros((N, M))
        self.c_x = np.fft.ifft2(self.c_hat * self.km_km * 1.0j)
        self.c_y = np.fft.ifft2(self.c_hat * self.kn_kn * 1.0j)
        self.km2kn2 = self.km_km ** 2 + self.kn_kn ** 2

    def getJ_hat(self):
        J = self.phi_y * self.c_x - self.phi_x * self.c_y
        J_hat = np.fft.fft2(J)
        return J_hat

    def getPhi_hat(self):
        # w_hat/0 = 0とする
        temp1 = R * self.c_hat * 1.0j * self.kn_kn
        temp2 = 1.0j * R * (self.km_km * self.c_x + self.kn_kn * self.c_y) + self.km2kn2
        phi_hat = - np.divide(temp1, temp2, out=np.zeros_like(temp1), where=self.km2kn2 != 0)
        return phi_hat

    def updatePhi_xAndPhi_y(self, phi_hat):
        self.phi_x = np.fft.ifft2(phi_hat * self.km_km * 1.0j).real
        self.phi_y = np.fft.ifft2(phi_hat * self.kn_kn * 1.0j).real

    def getC_barbyAdams(self, J_hat, J_hat_before):
        c_tilda = DELTA_T * (- 1.5 * J_hat + 0.5 * J_hat_before) + self.c_hat
        c_bar = c_tilda * np.exp(-self.km2kn2 * DELTA_T)
        return c_bar

    def updateC(self, c):
        self.c = c
        self.c_hat = np.fft.fft2(self.c)
        self.c_x = (np.fft.ifft2(self.c_hat * self.km_km * 1.0j))
        self.c_y = (np.fft.ifft2(self.c_hat * self.kn_kn * 1.0j))

    def updateC_hat(self, k):
        self.c_hat = self.c_hat + k
        self.c_x = np.fft.ifft2(self.c_hat * self.km_km * 1.0j)
        self.c_y = np.fft.ifft2(self.c_hat * self.kn_kn * 1.0j)

    def updateAndgetJ_hat(self, k):
        self.updateC_hat(k)
        phi_hat = self.getPhi_hat()
        self.updatePhi_xAndPhi_y(phi_hat)
        J_hat = self.getJ_hat()
        return J_hat

    def getC_barbyRunge(self):
        c_hat = copy.deepcopy(self.c_hat)
        phi_x = copy.deepcopy(self.phi_x)
        phi_y = copy.deepcopy(self.phi_y)
        k1 = -self.updateAndgetJ_hat(0) * DELTA_T
        # reset c
        self.c_hat = c_hat
        k2 = -self.updateAndgetJ_hat(1 / 2 * k1) * DELTA_T
        # reset c
        self.c_hat = c_hat
        k3 = -self.updateAndgetJ_hat(1 / 2 * k2) * DELTA_T
        # reset c
        self.c_hat = c_hat
        k4 = -self.updateAndgetJ_hat(k3) * DELTA_T
        # reset c and phi
        self.c_hat = c_hat
        self.phi_x = phi_x
        self.phi_y = phi_y
        c_tilda = c_hat + 1 / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        c_bar = c_tilda * np.exp(-self.km2kn2 * DELTA_T)
        # print(np.fft.ifft2(c_bar).real)
        return c_bar

    def getNextC(self, J_hat, J_bar, c_hat, c_bar):
        nextc_hat = c_hat - DELTA_T * ((J_bar + J_hat) / 2 + self.km2kn2 * (c_bar + c_hat) / 2)
        return nextc_hat


# @jit()
def update(i, x, y, cc):
    print(i)
    plt.cla()
    plt.pcolor(x, y, cc[i].real, label='R=' + str(R) + ',' + 'Pe=' + str(Pe), vmin=0, vmax=1)
    # plt.clim(0,1)
    plt.legend()


def getwavenum(cc, A):
    cc = cc[-1]
    cc = cc[:, int(M / 2 - (M // A)):int(M / 2 + (M // A))]
    x = np.array([(m / int(2 * (M // A))) * A * Pe for m in range(int((2 * (M // A))))])
    yy = []
    for c in cc:
        temp = np.array([i for i, x in enumerate(c) if 0.05 < x < 0.15])
        if len(temp) == 0:
            temp = np.array([0])
        print(int(temp.mean()))
        yy.append(x[int(temp.mean())])
    F = np.fft.fft(yy)
    freq = np.fft.fftfreq(N, d=0.01)
    Amp = np.abs(F / (N / 2))  # 振幅
    # plt.subplot(2,1,1)
    plt.plot(freq[1:int(N / 2)], Amp[1:int(N / 2)], label="Pe={},R={}".format(Pe, R))
    plt.xlabel("Freqency")
    plt.ylabel("Amplitude")
    plt.xlim(0, 30)
    plt.grid()
    # plt.show()
    # plt.figure()
    # plt.plot(y,yy)
    # plt.show()


def main():
    t1 = time.time()
    cm = Compute()
    cc = np.array([cm.c])
    J_hat_before = np.array([])
    t = 0
    while t < MAX_T:
        c_hat = cm.c_hat
        J_hat = cm.getJ_hat()
        if t == 0:
            # t += DELTA_T
            c_bar = cm.getC_barbyRunge()
            # J_hat_before = J_hat
            # cc = np.append(cc, np.array([cm.c]), axis=0)
            # continue
        else:
            c_bar = cm.getC_barbyAdams(J_hat, J_hat_before)
        cm.updateC(np.fft.ifft2(c_bar).real)
        phi_bar = cm.getPhi_hat()
        cm.updatePhi_xAndPhi_y(phi_bar)
        J_bar = cm.getJ_hat()
        # nextc_hat = c_bar
        nextc_hat = cm.getNextC(J_hat, J_bar, c_hat, c_bar)
        nextc = np.fft.ifft2(nextc_hat).real
        cm.updateC(nextc)
        phi_hat = cm.getPhi_hat()
        cm.updatePhi_xAndPhi_y(phi_hat)
        # if int(t * 10) % 60 == 0:
        #     cc = np.append(cc, np.array([nextc]), axis=0)
        J_hat_before = J_hat
        t += DELTA_T
        print("time:{}".format(round(t, 3)))
    cc = np.append(cc, np.array([nextc]), axis=0)
    print(cc.shape)
    #getwavenum(cc.real,10)
    # fig = plt.figure()
    # plt.colorbar(plt.pcolor(cm.x, cm.y, DELTA_Ccc[0].real, vmax=1.0, vmin=0))
    # ani = animation.FuncAnimation(fig, update, fargs=(cm.x, cm.y, cc), frames=int(len(cc)))
    # ani.save('./movies/R{}_Pe{}_MAX_T{}.mp4'.format(R, Pe,MAX_T), fps=8)
    # plt.subplot(2,1,2)
    plt.figure()
    plt.pcolor(cm.x, cm.y, cc[-1].real, label='R=' + str(R) + ',' + 'Pe=' + str(Pe))
    plt.show()
    t2 = time.time()
    print(t2 - t1)


if __name__ == '__main__':
    np.seterr(all='raise')
    try:
        main()
        # for i in range(3):
        #     #R = R + 1.0
        #     Pe = Pe + 200
        #     main()
        # plt.legend()
        # plt.savefig('./movies/Wavenum_P_MAXT:{}_adamsfix.png'.format(MAX_T))
        #plt.show()

    except Warning as e:
        print(e)
