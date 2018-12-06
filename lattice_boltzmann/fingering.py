import numpy as np
import sympy
import matplotlib.pyplot as plt
import copy
import matplotlib.animation as animation

from scipy import optimize
from scipy.ndimage.filters import convolve
import math
import time

H = 200  # lattice dimensions
W = 300
DELTA_T = 1.0 * 10 ** (-11)  # time step
DELTA_X = 1.0 * 10 ** (-8)  # lattice spacing
MAX_T = 2000
psi_wall = 0.0
sigma = 0.045  # interfacial tension
Pe = 0.1  # peclet number
u0 = DELTA_X  # initial velocity
rho = 10.0 ** 3
n_non = 1.0  # power-law parameter
Eta_n = 0.023  # Eta newtonian
M = 10.0  # Eta non_newtonian / Eta newtonian
Theta = np.pi / 4  # contact angle
v1 = 10.0 ** (-6)  # kinetic viscosity of newtonian
c = DELTA_X / DELTA_T  # particle streaming speed
cs = c / np.sqrt(3.0)
xi = 2 * DELTA_X
kappa = (3 / 4) * sigma * xi
a = - 2 * kappa / (xi ** 2)
tau = 1 / (3 - np.sqrt(3))
gamma = u0 / (-a * Pe) / ((tau - 0.5) * DELTA_T)
# print(gamma)
# gamma = 1.0 * 10.0 ** (-10)
b = np.array([i for i in range(1, 10)]).reshape(3, 3)


# print(np.roll(b, 1, axis=0))


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
            # print(self.e[i], self.w[i])
        self.psi = np.full((H, W), -1.0).astype(float)
        self.psi[int(H / 2 - 10):int(H / 2 + 10), int(W / 4 - 20):int(W / 4)] = psi_wall
        # print(circl.shape)
        # circle[0, int(W/2)] = 0
        # circle = np.roll(circle, -1, axis=0)
        # self.psi[:, 30] = 1.0
        self.left_wall = np.full((H, 1), 1.0).astype(float)
        self.right_wall = np.full((H, 1), -1.0).astype(float)
        self.gamma = gamma
        # self.psi[int(H/2 - 30):int(H/2 + 30), int(W/2 - 30): int(W/2 + 30)] = 1.0
        # self.psi[int(H/2 - 20):int(H/2 + 20), int(W/2 - 20):int(W/2 + 20)] = 0.8
        # self.psi[:80, int(W/2 - 40):int(W/2+40)] = -1.0
        # self.psi_wall_list = self.getPsi_wall_list()
        # self.psi_wall_list = np.full((1, W), psi_wall).astype(float)
        self.rho = np.ones((H, W), dtype=float) * rho  # macroscopic density
        self.ux = np.zeros((H, W), dtype=float)
        self.uy = np.zeros((H, W), dtype=float)
        self.f = np.array([np.zeros((H, W), dtype=float) for i in range(9)])
        self.g = np.array([np.zeros((H, W), dtype=float) for i in range(9)])
        self.feq = np.array([np.zeros((H, W), dtype=float) for i in range(9)])
        self.geq = np.array([np.zeros((H, W), dtype=float) for i in range(9)])
        self.mu = self.getMu()
        self.updateU()
        self.F = np.array([np.zeros((H, W), dtype=float) for i in range(9)]).astype(float)
        self.mix_tau = self.getMix_tau()
        self.p = self.getP()
        for i in range(9):
            if i == 0:
                self.A0 = self.getA0()
                self.feq[i] = self.getfeq(i)
                self.f[i] = self.getfeq(i)
                self.B0 = self.getB0()
                self.geq[i] = self.getgeq(i)
                self.g[i] = self.getgeq(i)
            elif i < 5:
                self.A1_8 = self.getA1_8()
                self.f[i] = self.getfeq(i)
                self.feq[i] = self.getfeq(i)
                self.B1_8 = self.getB1_8()
                self.geq[i] = self.getgeq(i)
                self.g[i] = self.getgeq(i)
            if i >= 5:
                self.f[i] = self.getfeq(i)
                self.feq[i] = self.getfeq(i)
                self.geq[i] = self.getgeq(i)
                self.g[i] = self.getgeq(i)

    # def updateP(self):
    #     self.p = np.sum(self.f, axis=0) * (cs ** 2)
    #     print("p:{}".format(self.p.mean()))
    def getP(self):
        # p0 = self.rho * (cs ** 2) + a/2.0*self.psi**2 - 3.0/4.0*a*self.psi**4 - kappa*self.psi*self.getNabla_psi2()-kappa/2*(self.getNabla_psix()**2+self.getNabla_psiy()**2)
        # p = (cs ** 2) * self.rho + self.psi * self.mu
        return 1 / 3 * self.rho + self.psi * self.mu
        # return (cs ** 2) * self.rho + self.psi * self.mu

    def updateP(self):
        # p0 = self.rho * (cs ** 2) + a/2.0*self.psi**2 - 3.0/4.0*a*self.psi**4 - kappa*self.psi*self.getNabla_psi2()-kappa/2*(self.getNabla_psix()**2+self.getNabla_psiy()**2)
        self.p = 1 / 3 * self.rho + self.psi * self.mu
        # self.p = (cs ** 2) * self.rho + self.psi * self.mu

        # self.p = p0
        print("p:{}".format(self.p.mean()))

    def updateU(self):
        temp1 = np.zeros((H, W), dtype=float)
        temp2 = np.zeros((H, W), dtype=float)
        for i in range(9):
            temp1 += self.f[i] * self.e[i][0]
            temp2 += self.f[i] * self.e[i][1]
        self.ux = (temp1 + self.mu * self.getNabla_psix() * DELTA_T / 2) / self.rho
        self.uy = (temp2 + self.mu * self.getNabla_psiy() * DELTA_T / 2) / self.rho
        print("ux:{}, uy:{}".format(self.ux.mean(), self.uy.mean()))

    def getMu(self):
        mu = a * self.psi - a * (self.psi ** 3) - kappa * self.getNabla_psi2()
        return mu

    def updateMu(self):
        self.mu = a * self.psi - a * (self.psi ** 3) - kappa * self.getNabla_psi2()
        # print("mu:{}".format(self.mu.mean()))

    def updateRho(self):
        self.rho = np.sum(self.f, axis=0)  # macroscopic density
        # print("rho:{}".format(self.rho.mean()))

    def getA0(self):
        a0 = (self.rho - 3.0 * (1.0 - self.w[0]) * self.p / c ** 2) / self.w[0]
        return a0

    def getA1_8(self):
        a1_8 = 3 * self.p / c ** 2
        return a1_8

    def getB0(self):
        b0 = (self.psi - 3.0 * (1.0 - self.w[0]) * self.gamma * self.mu / (c ** 2)) / self.w[0]
        return b0

    def getB1_8(self):
        b1_8 = 3 * self.gamma * self.mu / (c ** 2)
        return b1_8

    def getfeq(self, n):
        if n == 0:
            feq = self.w[n] * (self.getA0() + self.rho * (
                    3 * (self.e[n][0] * self.ux + self.e[n][1] * self.uy) / (c ** 2) + 9 * (
                    self.e[n][0] * self.ux + self.e[n][1] * self.uy) ** 2 / (
                            2 * (c ** 4)) - 3 * (self.ux ** 2 + self.uy ** 2) / (2 * c ** 2)))
        else:
            feq = self.w[n] * (self.getA1_8() + self.rho * (
                    3 * (self.e[n][0] * self.ux + self.e[n][1] * self.uy) / c ** 2 + 9 * (
                    self.e[n][0] * self.ux + self.e[n][1] * self.uy) ** 2 / (
                            2 * c ** 4) - 3 * (self.ux ** 2 + self.uy ** 2) / (2 * c ** 2)))
        # print("feq{}:{}".format(n, feq.mean()))
        return feq

    def getgeq(self, n):
        if n == 0:
            geq = self.w[n] * (self.getB0() + self.psi * (
                    3 * (self.e[n][0] * self.ux + self.e[n][1] * self.uy) / (c ** 2) + 9 * (
                    self.e[n][0] * self.ux + self.e[n][1] * self.uy) ** 2 / (
                            2 * (c ** 4)) - 3 * (self.ux ** 2 + self.uy ** 2) / (2 * (c ** 2))))
        else:
            geq = self.w[n] * (self.getB1_8() + self.psi * (
                    3 * (self.e[n][0] * self.ux + self.e[n][1] * self.uy) / c ** 2 + 9 * (
                    self.e[n][0] * self.ux + self.e[n][1] * self.uy) ** 2 / (
                            2 * c ** 4) - 3 * (self.ux ** 2 + self.uy ** 2) / (2 * c ** 2)))
        # print("geq{}:{}".format(n, geq.mean()))
        return geq

    def getLarge_F(self, n):
        f = DELTA_T * self.mu * self.w[n] * (1 - 1 / (2 * self.mix_tau)) \
            * (((self.e[n][0] - self.ux) / (cs ** 2) + self.e[n][0] * (self.e[n][0] * self.ux + self.e[n][1] * self.uy)
                / (cs ** 4)) * self.getNabla_psix()
               + ((self.e[n][1] - self.uy) / (cs ** 2) + self.e[n][1] * (
                            self.e[n][0] * self.ux + self.e[n][1] * self.uy) / (
                          cs ** 4)) * self.getNabla_psiy())
        return f

    def updateMix_tau(self):
        # t1 = time.time()
        # temp1 = np.zeros((H+2, W), dtype=np.complex)
        # x = sympy.symbols('x')
        # for i in range(9):
        #     temp1 += self.f[i] * self.e[i][0] * self.e[i][1]
        # temp2 = 3 * Eta_n * 20 / self.rho / (c ** 2) * np.sign(temp1) * (np.abs(temp1) / (2 * self.rho * (c ** 2))) ** (n_non - 1)
        # tau2 = np.zeros((H+2, W))
        # for i in range(H+2):
        #     for j in range(W):
        #         tau2[i][j] = sympy.solve(x - temp2[i][j] * x ** (1 - n_non) - 0.5 * DELTA_T)[0]
        #         #print(i*W + j, tau2[i][j])
        # #optimize.newton(self.ss,0)
        # v2 = (c ** 2) * (tau2 - 0.5 * DELTA_T) / 3
        v2 = v1 * M
        mix_v = np.divide(2 * v1 * v2, (v1 * (1.0 - self.psi) + v2 * (1.0 + self.psi)))
        mix_tau = 3 * mix_v / (c ** 2) + 0.5 * DELTA_T
        self.mix_tau = mix_tau
        # print("mix_tau:{}".format(mix_tau.mean()))

    def getMix_tau(self):
        v2 = v1 * M
        mix_v = np.divide(2 * v1 * v2, (v2 * (1.0 - self.psi) + v1 * (1.0 + self.psi)))
        mix_tau = 3 * mix_v / (c ** 2) + 0.5 * DELTA_T
        return mix_tau

    def updatePsi(self):
        self.psi = np.sum(self.g, axis=0)
        self.psi[int(H / 2 - 10):int(H / 2 + 10), int(W / 4 - 20):int(W / 4)] = psi_wall
        # self.psi[:, 0] = 1.0

    def getNabla_psix(self):
        f = np.zeros((H, W), dtype=float)
        temp = np.hstack((self.left_wall, np.hstack((self.psi, self.right_wall))))
        for i in range(9):
            if i == 0 or i == 2 or i == 4:
                continue
            elif i == 1:
                f += 4 * np.roll(temp, -1, axis=1)[:, 1:-1]
            elif i == 3:
                f += -4 * np.roll(temp, 1, axis=1)[:, 1:-1]
            elif i == 5:
                f += np.roll(np.roll(temp, -1, axis=1), -1, axis=0)[:, 1:-1]
            elif i == 6:
                f += - np.roll(np.roll(temp, 1, axis=1), -1, axis=0)[:, 1:-1]
            elif i == 7:
                f += - np.roll(np.roll(temp, 1, axis=1), 1, axis=0)[:, 1:-1]
            elif i == 8:
                f += np.roll(np.roll(temp, -1, axis=1), 1, axis=0)[:, 1:-1]
        return f / (12 * DELTA_X)

    def getNabla_psiy(self):
        f = np.zeros((H, W), dtype=float)
        temp = np.hstack((self.left_wall, np.hstack((self.psi, self.right_wall))))
        for i in range(9):
            if i == 0 or i == 1 or i == 3:
                continue
            elif i == 2:
                f += 4 * np.roll(temp, -1, axis=0)[:, 1:-1]
            elif i == 4:
                f += -4 * np.roll(temp, 1, axis=0)[:, 1:-1]
            elif i == 5:
                f += np.roll(np.roll(temp, -1, axis=1), -1, axis=0)[:, 1:-1]
            elif i == 6:
                f += np.roll(np.roll(temp, 1, axis=1), -1, axis=0)[:, 1:-1]
            elif i == 7:
                f += - np.roll(np.roll(temp, 1, axis=1), 1, axis=0)[:, 1:-1]
            elif i == 8:
                f += - np.roll(np.roll(temp, -1, axis=1), 1, axis=0)[:, 1:-1]
        return f / (12 * DELTA_X)

    def getNabla_psi2(self):
        f = np.zeros((H, W), dtype=float)
        temp = np.hstack((self.left_wall, np.hstack((self.psi, self.right_wall))))
        for i in range(9):
            if i == 0:
                f += -20 * temp[:, 1:-1]
            elif i == 1:
                f += 4 * np.roll(temp, -1, axis=0)[:, 1:-1]
            elif i == 2:
                f += 4 * np.roll(temp, -1, axis=1)[:, 1:-1]
            elif i == 3:
                f += 4 * np.roll(temp, 1, axis=1)[:, 1:-1]
            elif i == 4:
                f += 4 * np.roll(temp, 1, axis=0)[:, 1:-1]
            elif i == 5:
                f += np.roll(np.roll(temp, -1, axis=1), -1, axis=0)[:, 1:-1]
            elif i == 6:
                f += np.roll(np.roll(temp, 1, axis=1), -1, axis=0)[:, 1:-1]
            elif i == 7:
                f += np.roll(np.roll(temp, 1, axis=1), 1, axis=0)[:, 1:-1]
            elif i == 8:
                f += np.roll(np.roll(temp, -1, axis=1), 1, axis=0)[:, 1:-1]
        return f / (6 * DELTA_X * DELTA_X)

    def getPsi_wall_list(self):
        psi_wall = sympy.symbols('psi')
        out = sympy.solve(1 / 2 * psi_wall ** 3 - 3 / 2 * psi_wall + np.cos(Theta))[0].as_real_imag()
        return np.full((1, W), out[0]).astype(float)

    def updateF(self):
        for i in range(9):
            self.f[i] = self.f[i] - DELTA_T / self.mix_tau * (self.f[i] - self.feq[i]) + self.F[i]

    def updateG(self):
        for i in range(9):
            self.g[i] = self.g[i] - 1 / tau * (self.g[i] - self.geq[i])
            # print("g{}:{}".format(i, self.g[i].mean()))

    """http://phelafel.technion.ac.il/~drorden/project/ZouHe.pdf"""

    def zou_he_boundary(self):
        ux = 0.01
        uy = 0.0
        rho_wall = 1 / (1 - uy) * (self.f[0][:, 0] + self.f[2][:, 0] + self.f[4][:, 0] + 2 * (
                self.f[3][:, 0] + self.f[6][:, 0] + self.f[7][:, 0] - 0.5 * self.mu * self.getNabla_psix() * DELTA_T ))
        # self.rho[:, 0] = rho_wall
        psi_in = 1.0 - (
                self.g[0][:, 0] + self.g[2][:, 0] + self.g[3][:, 0] + self.g[4][:, 0] + self.g[6][:, 0] + self.g[7][
                                                                                                          :, 0])
        # self.ux[:, 0] = ux
        # self.uy[:, 0] = uy
        for i in range(9):
            if i == 1:
                self.f[i][:, 0] = self.f[3][:, 0] + 1.5 * ux * rho_wall
                self.g[i][:, 0] = self.w[i] * psi_in / (self.w[1] + self.w[5] + self.w[8])
                # self.g[i][:, 0] = self.getgeq(i)[:, 0]
            if i == 5:
                self.f[i][:, 0] = self.f[7][:, 0] - 0.5 * (
                        self.f[2][:, 0] - self.f[4][:, 0]) + 1.0 / 6.0 * ux * rho_wall
                self.g[i][:, 0] = self.w[i] * psi_in / (self.w[1] + self.w[5] + self.w[8])
                # self.g[i][:, 0] = self.getgeq(i)[:, 0]
            if i == 8:
                self.f[i][:, 0] = self.f[6][:, 0] + 0.5 * (
                        self.f[2][:, 0] - self.f[4][:, 0]) + 1.0 / 6.0 * ux * rho_wall
                self.g[i][:, 0] = self.w[i] * psi_in / (self.w[1] + self.w[5] + self.w[8])
                # self.g[i][:, 0] = self.getgeq(i)[:, 0]


def create_circle(n, r):
    y, x = np.ogrid[-int(H / 2): int(H / 2), -r: n - r]
    mask = x ** 2 + y ** 2 <= r ** 2
    return mask


def stream(f, g):
    # with open_boundary on right side
    for i in range(9):
        if i == 0:
            continue
        elif i == 1:
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
    # North_barrier
    n_barrier = np.zeros((H, W), dtype=bool)
    s_barrier = np.zeros((H, W), dtype=bool)
    e_barrier = np.zeros((H, W), dtype=bool)
    w_barrier = np.zeros((H, W), dtype=bool)
    nw_corner = np.zeros((H, W), dtype=bool)
    ne_corner = np.zeros((H, W), dtype=bool)
    sw_corner = np.zeros((H, W), dtype=bool)
    se_corner = np.zeros((H, W), dtype=bool)
    n_barrier[int(H / 2 + 10), int(W / 4 - 19):int(W / 4 - 1)] = True
    s_barrier[int(H / 2 - 10), int(W / 4 - 19):int(W / 4 - 1)] = True
    w_barrier[int(H / 2 - 10 + 1):int(H / 2 + 10 - 1), int(W / 4)] = True
    e_barrier[int(H / 2 - 10 + 1):int(H / 2 + 10 - 1), int(W / 4 - 20)] = True
    nw_corner[int(H / 2 + 10), int(W / 4)] = True
    ne_corner[int(H / 2 + 10), int(W / 4 - 20)] = True
    sw_corner[int(H / 2 - 10), int(W / 4)] = True
    se_corner[int(H / 2 - 10), int(W / 4 - 20)] = True

    for i in range(9):
        if i == 0:
            continue
        elif i == 1:
            f[i][nw_corner] = f_behind[3][nw_corner]
            g[i][nw_corner] = g_behind[3][nw_corner]
            f[i][sw_corner] = f_behind[3][sw_corner]
            g[i][sw_corner] = g_behind[3][sw_corner]
            f[i][ne_corner] = 0.0
            g[i][ne_corner] = 0.0
            f[i][se_corner] = 0.0
            g[i][se_corner] = 0.0
            f[i][s_barrier] = 0.0
            g[i][s_barrier] = 0.0
            f[i][w_barrier] = f_behind[3][w_barrier]
            g[i][w_barrier] = g_behind[3][w_barrier]
            f[i][e_barrier] = 0.0
            g[i][e_barrier] = 0.0
            continue
        elif i == 2:
            f[i][nw_corner] = f_behind[4][nw_corner]
            g[i][nw_corner] = g_behind[4][nw_corner]
            f[i][sw_corner] = 0.0
            g[i][sw_corner] = 0.0
            f[i][ne_corner] = f_behind[4][ne_corner]
            g[i][ne_corner] = g_behind[4][ne_corner]
            f[i][se_corner] = 0.0
            g[i][se_corner] = 0.0
            f[i][n_barrier] = f_behind[4][n_barrier]
            g[i][n_barrier] = g_behind[4][n_barrier]
            f[i][s_barrier] = 0.0
            g[i][s_barrier] = 0.0
            f[i][w_barrier] = 0.0
            g[i][w_barrier] = 0.0
            f[i][e_barrier] = 0.0
            g[i][e_barrier] = 0.0

        elif i == 3:
            f[i][nw_corner] = 0.0
            g[i][nw_corner] = 0.0
            f[i][sw_corner] = 0.0
            g[i][sw_corner] = 0.0
            f[i][ne_corner] = f_behind[1][ne_corner]
            g[i][ne_corner] = g_behind[1][ne_corner]
            f[i][se_corner] = f_behind[1][se_corner]
            g[i][se_corner] = g_behind[1][se_corner]
            f[i][n_barrier] = 0.0
            g[i][n_barrier] = 0.0
            f[i][s_barrier] = 0.0
            g[i][s_barrier] = 0.0
            f[i][e_barrier] = f_behind[1][e_barrier]
            g[i][e_barrier] = g_behind[1][e_barrier]
            f[i][w_barrier] = 0.0
            g[i][w_barrier] = 0.0

        elif i == 4:
            f[i][nw_corner] = 0.0
            g[i][nw_corner] = 0.0
            f[i][sw_corner] = f_behind[2][sw_corner]
            g[i][sw_corner] = g_behind[2][sw_corner]
            f[i][ne_corner] = 0.0
            g[i][ne_corner] = 0.0
            f[i][se_corner] = f_behind[2][se_corner]
            g[i][se_corner] = g_behind[2][se_corner]
            f[i][n_barrier] = 0.0
            g[i][n_barrier] = 0.0
            f[i][s_barrier] = f_behind[2][s_barrier]
            g[i][s_barrier] = g_behind[2][s_barrier]
            f[i][w_barrier] = 0.0
            g[i][w_barrier] = 0.0
            f[i][e_barrier] = 0.0
            g[i][e_barrier] = 0.0
        elif i == 5:
            f[i][nw_corner] = f_behind[7][nw_corner]
            g[i][nw_corner] = g_behind[7][nw_corner]
            f[i][se_corner] = 0.0
            g[i][se_corner] = 0.0
            f[i][n_barrier] = f_behind[7][n_barrier]
            g[i][n_barrier] = g_behind[7][n_barrier]
            f[i][s_barrier] = 0.0
            g[i][s_barrier] = 0.0
            f[i][w_barrier] = f_behind[7][w_barrier]
            g[i][w_barrier] = g_behind[7][w_barrier]
            f[i][e_barrier] = 0.0
            g[i][e_barrier] = 0.0
        elif i == 6:
            f[i][ne_corner] = f_behind[8][ne_corner]
            g[i][ne_corner] = g_behind[8][ne_corner]
            f[i][sw_corner] = 0.0
            g[i][sw_corner] = 0.0
            f[i][n_barrier] = f_behind[8][n_barrier]
            g[i][n_barrier] = g_behind[8][n_barrier]
            f[i][s_barrier] = 0.0
            g[i][s_barrier] = 0.0
            f[i][w_barrier] = 0.0
            g[i][w_barrier] = 0.0
            f[i][e_barrier] = f_behind[8][e_barrier]
            g[i][e_barrier] = g_behind[8][e_barrier]
        elif i == 7:
            f[i][nw_corner] = 0.0
            g[i][nw_corner] = 0.0
            f[i][se_corner] = f_behind[5][se_corner]
            g[i][se_corner] = g_behind[5][se_corner]
            f[i][n_barrier] = 0.0
            g[i][n_barrier] = 0.0
            f[i][s_barrier] = f_behind[5][s_barrier]
            g[i][s_barrier] = g_behind[5][s_barrier]
            f[i][w_barrier] = 0.0
            g[i][w_barrier] = 0.0
            f[i][e_barrier] = f_behind[5][e_barrier]
            g[i][e_barrier] = g_behind[5][e_barrier]
        elif i == 8:
            f[i][ne_corner] = 0.0
            g[i][ne_corner] = 0.0
            f[i][sw_corner] = f_behind[6][sw_corner]
            g[i][sw_corner] = g_behind[6][sw_corner]
            f[i][n_barrier] = 0.0
            g[i][n_barrier] = 0.0
            f[i][s_barrier] = f_behind[6][s_barrier]
            g[i][s_barrier] = g_behind[6][s_barrier]
            f[i][w_barrier] = f_behind[6][w_barrier]
            g[i][w_barrier] = g_behind[6][w_barrier]
            f[i][e_barrier] = 0.0
            g[i][e_barrier] = 0.0


# """http://phelafel.technion.ac.il/~drorden/project/ZouHe.pdf"""
# def zou_he_boundary(f, g, fs):
#     ux = 1.0
#     uy = 0.0
#     rho_wall = 1 / (1 - uy) * (f[0][:, 0] + f[2][:, 0] + f[4][:, 0] + 2 * (f[3][:, 0] + f[6][:, 0] + f[7][:, 0]) + 0.5 * fs[:, 0] * DELTA_T)
#     for i in range(9):
#
#         if i == 1:
#             f[i][:, 0] = f[3][:, 0] + 1.5 * ux * rho_wall
#             g[i][:, 0] = g[3][:, 0] + 1.5 * ux
#         if i == 5:
#             f[i][:, 0] = f[7][:, 0] - 0.5 * (f[2][:, 0] - f[4][:, 0]) + 1.0 / 6.0 * ux * rho_wall
#             g[i][:, 0] = g[7][:, 0]
#         if i == 8:
#             f[i][:, 0] = f[6][:, 0] + 0.5 * (f[2][:, 0] - f[4][:, 0]) + 1.0 / 6.0 * ux * rho_wall
#             g[i][:, 0] = g[6][:, 0]

def open_boundary(f, g):
    for i in range(9):
        if i == 0:
            continue
        elif i == 3 or i == 6 or i == 7:
            f[i][:, -1] = 2 * f[i][:, -2] - f[i][:, -3]
            g[i][:, -1] = 2 * g[i][:, -2] - g[i][:, -3]


def block(f, g):
    n_barrier = np.zeros((H, W), dtype=bool)
    s_barrier = np.zeros((H, W), dtype=bool)
    e_barrier = np.zeros((H, W), dtype=bool)
    w_barrier = np.zeros((H, W), dtype=bool)
    n_barrier[int(H / 2 + 10), int(W / 4 + 1):int(W / 4 + 20 - 1)] = True
    s_barrier[int(H / 2 - 10), int(W / 4 + 1):int(W / 4 + 20 - 1)] = True
    w_barrier[int(H / 2 - 10 + 1):int(H / 2 + 10 - 1), int(W / 4 + 20)] = True
    e_barrier[int(H / 2 - 10 + 1):int(H / 2 + 10 - 1), int(W / 4)] = True


def main():
    cm = Compute()
    for i in range(MAX_T):
        f_behind = copy.deepcopy(cm.f)
        g_behind = copy.deepcopy(cm.g)
        stream(cm.f, cm.g)
        # cm.psi[:, 0] = 1.0
        halfway_bounceback(f_behind, g_behind, cm.f, cm.g)
        cm.zou_he_boundary()
        open_boundary(cm.f, cm.g)
        cm.updateRho()
        cm.updatePsi()
        cm.updateMu()
        cm.updateP()
        # print("Fs", cm.mu * cm.getNabla_psix().mean())
        cm.updateU()
        cm.updateMix_tau()
        for j in range(9):
            cm.F[j] = cm.getLarge_F(j)
            cm.feq[j] = cm.getfeq(j)
            cm.geq[j] = cm.getgeq(j)
        cm.updateF()
        cm.updateG()

        print("timestep:{}".format(i))
    y = [i for i in range(H)]
    x = [i for i in range(W)]
    plt.figure()
    plt.pcolor(x, y, cm.psi, label="MAX_T:{}, Pe:{}, M:{}".format(MAX_T, Pe, M))
    plt.colorbar()
    plt.legend()
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
