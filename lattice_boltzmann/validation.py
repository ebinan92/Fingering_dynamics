import numpy as np
import sympy
import matplotlib.pyplot as plt
import copy
from multiprocessing import Pool
from scipy import optimize as opt
from scipy import interpolate
from multiprocessing import Process
import matplotlib.animation as animation
import math
import time

H = 200  # lattice dimensions
W = 250
MAX_T = 100
psi_wall = 0.0
Pe = 100  # peclet number
rho = 1.0
n_non = 1.4  # power-law parameter
M = 20.0  # Eta non_newtonian / Eta newtonian
Theta = np.pi / 4  # contact angle
tau = 1 / (3 - np.sqrt(3))
C_W = 1.0 * (10.0 ** (-5)) / W
C_rho = 10.0 ** 3
v0 = (tau - 0.5) / 3  # kinetic viscosity of newtonian
C_t = v0 / (10.0 ** (-6)) * (C_W ** 2)
DELTA_X = 1.0  # time step
DELTA_T = 1.0  # lattice spacing
sigma = 0.045 * (C_t ** 2) / (C_rho * (C_W ** 3))  # interfacial tension
u0 = C_t / C_W  # initial velocity
c = DELTA_X / DELTA_T  # particle streaming speed
cs = c / np.sqrt(3)
xi = 2.0 * DELTA_X
kappa = (3 / 4) * sigma * xi
a = 2 * kappa / (xi ** 2)
gamma = u0 * W / (a * Pe) / ((tau - 0.5) * DELTA_T)
Eta_n = 0.001 / (C_rho * (C_W ** 2) / C_t)  # non_dimentional Eta newtonian　
x = sympy.symbols('x')
x_array = np.arange(0.1, 1.0, 0.01)


class Compute:
    def __init__(self):
        self.e = np.array([np.array([0.0, 0.0]) for i in range(9)])
        self.w = np.array([0.0 for i in range(9)])
        for i in range(9):
            if i == 0:
                self.w[i] = 4.0 / 9.0
                self.e[i] = np.array([0.0, 0.0])
            elif i < 5:
                self.w[i] = 1.0 / 9.0
                self.e[i] = np.array([np.cos((i - 1) * np.pi / 2), np.sin((i - 1) * np.pi / 2)]) * c
            if i >= 5:
                self.w[i] = 1.0 / 36.0
                self.e[i] = np.array([np.cos((i - 5) * np.pi / 2 + np.pi / 4),
                                      np.sin(np.pi * ((i - 5.0) / 2 + 1 / 4))]) * c * np.sqrt(2)

            if abs(self.e[i][0]) < 0.1:
                self.e[i][0] = 0

            if abs(self.e[i][1]) < 0.1:
                self.e[i][1] = 0

            print(self.e[i], self.w[i])
        self.psi = np.full((H, W), -1.0).astype(float)
        circle = create_circle(W, 36).T
        # self.psi[int(H/2 - 20):int(H/2 + 20), int(W/2 - 20):int(W/2 + 20)] = 1.0
        circl = circle[:H, :]
        # print(circl.shape)
        # circl[0, int(W/2)] = 0
        # circl = np.roll(circl, -1, axis=0)
        self.psi[circl] = 1.0
        self.gamma = gamma
        # self.psi[int(H/2 - 30):int(H/2 + 30), int(W/2 - 30): int(W/2 + 30)] = 1.0
        # self.psi[int(H/2 - 20):int(H/2 + 20), int(W/2 - 20):int(W/2 + 20)] = 0.8
        # self.psi[:80, int(W/2 - 40):int(W/2+40)] = -1.0
        # self.psi_wall_list = self.getPsi_wall_list()
        self.psi_wall_list = np.full((1, W), psi_wall).astype(float)
        self.rho = np.ones((H, W), dtype=float) * rho  # macroscopic density
        self.ux = np.zeros((H, W), dtype=float)
        self.uy = np.zeros((H, W), dtype=float)
        self.f = np.array([np.zeros((H, W), dtype=float) for i in range(9)])
        self.g = np.array([np.zeros((H, W), dtype=float) for i in range(9)])
        self.feq = np.array([np.zeros((H, W), dtype=float) for i in range(9)])
        self.geq = np.array([np.zeros((H, W), dtype=float) for i in range(9)])
        self.mu = self.getMu()
        self.F = np.array([np.zeros((H, W), dtype=float) for i in range(9)])
        self.mix_tau = self.getMix_tau()
        self.p = self.getP()
        for i in range(9):
            if i == 0:
                self.A0 = self.getA0()
                self.f[i] = self.getfeq(i)
                self.B0 = self.getB0()
                self.g[i] = self.getgeq(i)
            elif i < 5:
                self.A1_8 = self.getA1_8()
                self.f[i] = self.getfeq(i)
                self.B1_8 = self.getB1_8()
                self.g[i] = self.getgeq(i)
            elif i >= 5:
                self.f[i] = self.getfeq(i)
                self.g[i] = self.getgeq(i)

    def getP(self):
        p = (cs ** 2) * self.rho + self.psi * self.mu
        # print("p:{}".format(p.mean()))
        return p

    def updateP(self):
        self.p = (cs ** 2) * self.rho + self.psi * self.mu
        # print("p:{}".format(self.p.mean()))

    def updateU(self):
        temp1 = np.zeros((H, W), dtype=float)
        temp2 = np.zeros((H, W), dtype=float)
        for i in range(9):
            temp1 += self.f[i] * self.e[i][0]
            temp2 += self.f[i] * self.e[i][1]
        self.ux = (temp1 + self.mu * self.getNabla_psix() * DELTA_T / 2) / self.rho
        self.uy = (temp2 + self.mu * self.getNabla_psiy() * DELTA_T / 2) / self.rho
        # print("ux:{}, uy:{}".format(self.ux.mean(), self.uy.mean()))

    def getMu(self):
        mu = a * self.psi * (self.psi ** 2 - 1) - kappa * self.getNabla_psi2()
        # print("mu:{}".format(mu.mean()))
        return mu

    def updateMu(self):
        self.mu = a * self.psi * (self.psi ** 2 - 1) - kappa * self.getNabla_psi2()
        # print("mu:{}".format(self.mu.mean()))

    def updateRho(self):
        self.rho = np.sum(self.f, axis=0)  # macroscopic density
        # print("rho:{}".format(self.rho.mean()))

    def getA0(self):
        a0 = (self.rho - (1.0 - self.w[0]) * self.p / (cs ** 2)) / self.w[0]
        # print("a0",a0.mean())
        return a0

    def getA1_8(self):
        a1_8 = self.p / (cs ** 2)
        # print("a1_8", a1_8.mean())
        return a1_8

    def getB0(self):
        b0 = (self.psi - (1.0 - self.w[0]) * self.gamma * self.mu / (cs ** 2)) / self.w[0]
        return b0

    def getB1_8(self):
        b1_8 = self.gamma * self.mu / (cs ** 2)
        return b1_8

    def getfeq(self, n):
        if n == 0:
            feq = self.w[n] * (self.getA0() + self.rho * (
                    3 * (self.e[n][0] * self.ux + self.e[n][1] * self.uy) / (c ** 2) + 4.5 * (
                    self.e[n][0] * self.ux + self.e[n][1] * self.uy) ** 2 / (c ** 4) - 1.5 * (
                                self.ux ** 2 + self.uy ** 2) / (c ** 2)))
        else:
            feq = self.w[n] * (self.getA1_8() + self.rho * (
                    3 * (self.e[n][0] * self.ux + self.e[n][1] * self.uy) / (c ** 2) + 4.5 * (
                    self.e[n][0] * self.ux + self.e[n][1] * self.uy) ** 2 / (c ** 4) - 1.5 * (
                                self.ux ** 2 + self.uy ** 2) / (c ** 2)))
        return feq

    def getgeq(self, n):
        if n == 0:
            geq = self.w[n] * (self.getB0() + self.psi * (
                    3 * (self.e[n][0] * self.ux + self.e[n][1] * self.uy) / (c ** 2) + 4.5 * (
                    self.e[n][0] * self.ux + self.e[n][1] * self.uy) ** 2 / (c ** 4) - 1.5 * (
                                self.ux ** 2 + self.uy ** 2) / (c ** 2)))
        else:
            geq = self.w[n] * (self.getB1_8() + self.psi * (
                    3 * (self.e[n][0] * self.ux + self.e[n][1] * self.uy) / (c ** 2) + 4.5 * (
                    self.e[n][0] * self.ux + self.e[n][1] * self.uy) ** 2 / (c ** 4) - 1.5 * (
                                self.ux ** 2 + self.uy ** 2) / (c ** 2)))
        return geq

    def getLarge_F(self, n):
        f = DELTA_T * self.mu * self.w[n] * (1 - 1 / (2 * self.mix_tau)) * ((
                                                                                    (self.e[n][0] - self.ux) / (
                                                                                        cs ** 2) + self.e[n][0] * (
                                                                                                self.e[n][0] * self.ux +
                                                                                                self.e[n][
                                                                                                    1] * self.uy) / (
                                                                                                cs ** 4)) *
                                                                            self.getNabla_psix() + (
                                                                                        (self.e[n][1] - self.uy) / (
                                                                                            cs ** 2)
                                                                                        + self.e[n][1] * (self.e[n][
                                                                                                              0] * self.ux +
                                                                                                          self.e[n][
                                                                                                              1] * self.uy) / (
                                                                                                    cs ** 4)) * self.getNabla_psiy())
        return f

    def updateMix_tau(self):
        temp1 = np.zeros((H, W), dtype=np.complex)
        for i in range(9):
            temp1 += (self.f[i] - self.feq[i]) * self.e[i][0] * self.e[i][1]
        temp2 = 3 * Eta_n * M / self.rho * (3 / (2 * self.rho) * temp1) ** (n_non - 1)
        t1 = time.time()
        p = Pool(6)
        temp3 = temp2.ravel()
        print(temp3.shape)
        tau2 = np.array(p.map(self.power_law, temp3)).reshape((H, W))
        print(tau2.mean())
        p.close()
        t2 = time.time()
        print("power_law:{}".format(t2 - t1))
        v2 = (c ** 2) * (tau2 - 0.5 * DELTA_T) / 3
        # v2 = v1 * M
        ones = np.ones((H, W))
        v1 = Eta_n / self.rho
        v2 = Eta_n * M / self.rho
        mix_v = np.divide(2 * v1 * v2, (v1 * (ones - self.psi) + v2 * (ones + self.psi)))
        mix_tau = 3 * mix_v + 0.5
        # print("mix_tau:{}".format(mix_tau.mean()))

    def power_law(self, temp2):
        # print(temp2)
        y_array = x_array - temp2 * x_array ** (1 - n_non) - 0.5 * DELTA_T
        tau2 = interpolate.interp1d(y_array.real, x_array, kind='nearest', fill_value='extrapolate')(0)
        # tau2 = sympy.solve(x - temp2 * x ** (1 - n_non) - 0.5 * DELTA_T)[0]
        # print(tau2)
        return tau2

    def getMix_tau(self):
        ones = np.ones((H, W))
        v1 = Eta_n / self.rho
        v2 = Eta_n * M / self.rho
        mix_v = np.divide(2 * v1 * v2, (v1 * (ones - self.psi) + v2 * (ones + self.psi)))
        mix_tau = 3 * mix_v + 0.5
        # print("mix_tau:{}".format(mix_tau.mean()/DELTA_T))
        return mix_tau

    def updatePsi(self):
        self.psi = np.sum(self.g, axis=0)
        # print("psi:{}".format(self.psi.mean()))

    def getNabla_psix(self):
        f = np.zeros((H, W), dtype=float)
        temp = np.vstack((self.psi_wall_list, np.vstack((self.psi, self.psi_wall_list))))
        for i in range(9):
            if i == 0 or i == 2 or i == 4:
                continue
            elif i == 1:
                f += 4 * np.roll(temp, -1, axis=1)[1:-1, :]
            elif i == 3:
                f += -4 * np.roll(temp, 1, axis=1)[1:-1, :]
            elif i == 5:
                f += np.roll(np.roll(temp, -1, axis=1), -1, axis=0)[1:-1, :]
            elif i == 6:
                f += - np.roll(np.roll(temp, 1, axis=1), -1, axis=0)[1:-1, :]
            elif i == 7:
                f += - np.roll(np.roll(temp, 1, axis=1), 1, axis=0)[1:-1, :]
            elif i == 8:
                f += np.roll(np.roll(temp, -1, axis=1), 1, axis=0)[1:-1, :]
        # print("psi_x:{}".format(f.mean()))
        return f / (12 * DELTA_X)

    def getNabla_psiy(self):
        f = np.zeros((H, W), dtype=float)
        temp = np.vstack((self.psi_wall_list, np.vstack((self.psi, self.psi_wall_list))))
        for i in range(9):
            if i == 0 or i == 1 or i == 3:
                continue
            elif i == 2:
                f += 4 * np.roll(temp, -1, axis=0)[1:-1, :]
            elif i == 4:
                f += - 4 * np.roll(temp, 1, axis=0)[1:-1, :]
            elif i == 5:
                f += np.roll(np.roll(temp, -1, axis=1), -1, axis=0)[1:-1, :]
            elif i == 6:
                f += np.roll(np.roll(temp, 1, axis=1), -1, axis=0)[1:-1, :]
            elif i == 7:
                f += - np.roll(np.roll(temp, 1, axis=1), 1, axis=0)[1:-1, :]
            elif i == 8:
                f += - np.roll(np.roll(temp, -1, axis=1), 1, axis=0)[1:-1, :]
        # print("psi_y:{}".format(f.mean()))
        return f / (12 * DELTA_X)

    def getNabla_psi2(self):
        f = np.zeros((H, W), dtype=float)
        temp = np.vstack((self.psi_wall_list, np.vstack((self.psi, self.psi_wall_list))))
        for i in range(9):
            if i == 0:
                f += -20 * temp[1:-1, :]
            elif i == 1:
                f += 4 * np.roll(temp, -1, axis=1)[1:-1, :]
            elif i == 2:
                f += 4 * np.roll(temp, -1, axis=0)[1:-1, :]
            elif i == 3:
                f += 4 * np.roll(temp, 1, axis=1)[1:-1, :]
            elif i == 4:
                f += 4 * np.roll(temp, 1, axis=0)[1:-1, :]
            elif i == 5:
                f += np.roll(np.roll(temp, -1, axis=1), -1, axis=0)[1:-1, :]
            elif i == 6:
                f += np.roll(np.roll(temp, 1, axis=1), -1, axis=0)[1:-1, :]
            elif i == 7:
                f += np.roll(np.roll(temp, 1, axis=1), 1, axis=0)[1:-1, :]
            elif i == 8:
                f += np.roll(np.roll(temp, -1, axis=1), 1, axis=0)[1:-1, :]
        # print("psi_2:{}".format(f.mean() / ( 6 * DELTA_X ** 2) * kappa))
        return f / (6 * (DELTA_X ** 2))

    def updateF(self):
        for i in range(9):
            self.f[i] = self.f[i] - 1 / self.mix_tau * (self.f[i] - self.feq[i]) + self.F[i]

    def updateG(self):
        for i in range(9):
            self.g[i] = self.g[i] - 1 / tau * (self.g[i] - self.geq[i])
            # print("g{}:{}".format(i, self.g[i].mean()))


def create_circle(n, r):
    y, x = np.ogrid[-int(W / 2): int(W / 2), -r: n - r]
    mask = x ** 2 + y ** 2 <= r ** 2
    return mask


def stream(f, g):
    for i in range(9):
        if i == 1:
            f[i] = np.roll(f[i], 1, axis=1)
            g[i] = np.roll(g[i], 1, axis=1)
        elif i == 2:
            f[i] = np.roll(f[i], 1, axis=0)
            g[i] = np.roll(g[i], 1, axis=0)
        elif i == 3:
            f[i] = np.roll(f[i], -1, axis=1)
            g[i] = np.roll(g[i], -1, axis=1)
        elif i == 4:
            f[i] = np.roll(f[i], -1, axis=0)
            g[i] = np.roll(g[i], -1, axis=0)
        elif i == 5:
            f[i] = np.roll(np.roll(f[i], 1, axis=1), 1, axis=0)
            g[i] = np.roll(np.roll(g[i], 1, axis=1), 1, axis=0)
        elif i == 6:
            f[i] = np.roll(np.roll(f[i], -1, axis=1), 1, axis=0)
            g[i] = np.roll(np.roll(g[i], -1, axis=1), 1, axis=0)
        elif i == 7:
            f[i] = np.roll(np.roll(f[i], -1, axis=1), -1, axis=0)
            g[i] = np.roll(np.roll(g[i], -1, axis=1), -1, axis=0)
        elif i == 8:
            f[i] = np.roll(np.roll(f[i], 1, axis=1), -1, axis=0)
            g[i] = np.roll(np.roll(g[i], 1, axis=1), -1, axis=0)


def halfway_bounceback(f_behind, g_behind, f, g):
    for i in range(9):
        if i == 2:
            f[i][0, :] = f_behind[4][0, :]
            g[i][0, :] = g_behind[4][0, :]
        elif i == 4:
            f[i][-1, :] = f_behind[2][-1, :]
            g[i][-1, :] = g_behind[2][-1, :]
        elif i == 5:
            f[i][0, :] = f_behind[7][0, :]
            g[i][0, :] = g_behind[7][0, :]
        elif i == 6:
            f[i][0, :] = f_behind[8][0, :]
            g[i][0, :] = g_behind[8][0, :]
        elif i == 7:
            f[i][-1, :] = f_behind[5][-1, :]
            g[i][-1, :] = g_behind[5][-1, :]
        elif i == 8:
            f[i][-1, :] = f_behind[6][-1, :]
            g[i][-1, :] = g_behind[6][-1, :]


def update(i, x, y, cc):
    print(i)
    plt.cla()
    plt.pcolor(x, y, cc[i], label="MAX_T:{}, Pe:{}, M:{}, wall{}".format(MAX_T, Pe, M, psi_wall))
    # plt.clim(0,1)
    plt.legend()


def main():
    # 15
    mabiki = MAX_T // 150
    print(mabiki)
    cm = Compute()
    cc = np.array([cm.psi])
    for i in range(MAX_T):
        f_behind = copy.deepcopy(cm.f)
        g_behind = copy.deepcopy(cm.g)
        stream(cm.f, cm.g)
        halfway_bounceback(f_behind, g_behind, cm.f, cm.g)
        cm.updateRho()
        cm.updatePsi()
        cm.updateMu()
        cm.updateU()
        cm.updateP()
        for j in range(9):
            cm.feq[j] = cm.getfeq(j)
            cm.geq[j] = cm.getgeq(j)
        cm.updateMix_tau()
        for j in range(9):
            cm.F[j] = cm.getLarge_F(j)
        cm.updateF()
        cm.updateG()
        # if i % mabiki == 0:
        #     print("HI")
        #     cc = np.append(cc, np.array([cm.psi]), axis=0)
        print("timestep:{}".format(i))
    y = [i for i in range(H)]
    x = [i for i in range(W)]
    # fig = plt.figure()
    # plt.colorbar(plt.pcolor(x, y, cc[0]))
    # ani = animation.FuncAnimation(fig, update, fargs=(x, y, cc), frames=int(len(cc)))
    # ani.save('./movies/lattice_boltzmann/MAX_T{}_Pe{}_M{}_wall{}.mp4'.format(MAX_T, Pe, M, psi_wall), fps=10)
    plt.figure()
    plt.pcolor(x, y, cm.psi, label="MAX_T:{}, Pe:{}, M:{}, wall{}".format(MAX_T, Pe, M, psi_wall))
    plt.colorbar()
    plt.legend()
    plt.gca().set_aspect('equal', adjustable='box')
    # plt.savefig('./images/droplet_MAXT:{}_wall:{}.png'.format(MAX_T, psi_wall))
    plt.show()


if __name__ == '__main__':
    np.seterr(all='raise')
    try:
        t1 = time.time()
        main()
        t2 = time.time()
        print((t2 - t1) / 60)
    except Warning as e:
        print(e)
