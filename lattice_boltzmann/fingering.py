import numpy as np
import sympy
import matplotlib.pyplot as plt
import copy
from create_block import Createblock
from bounce_back import Bounce_back
import matplotlib.animation as animation
# for server
# plt.switch_backend('agg')
import math
import time
import subprocess

'''fingering with top and bottom wall'''

H = 380  # lattice dimensions
W = 380
MAX_T = 1000
psi_wall = -1.0  # wettability on block and wall
Pe = 400  # Peclet numbernet
C_W = 1.5 * (10 ** (-7))  # conversion width
Ca = 3.0 * 7.33 * (10 ** (-3))  # Capillary number
M = 20.0  # Eta non_newtonian / Eta newtonian
Eta = 0.001
block_num = 10
# Eta = 0.001
R_Nu = Eta / 1000  # physical kinematic viscosity of newtonian
tau = 1 / (3.0 - math.sqrt(3))  # relaxation time
# tau = 0.85
rho0 = 1.0  # non-dimensional pressure
n_non = 1.0  # rho0 power-law parameter
R_sigma = 0.045  # physical interfacial tension
C_rho = 1000  # conversion pressure
v0 = (tau - 0.5) / 3  # non-dimensional kinematic viscosity of newtonian
C_t = (v0 / R_Nu) * (C_W ** 2)  # conversion time step
Eta_n = Eta * C_t / (C_rho * (C_W ** 2))  # non_dimentional Eta newtonian　
# x_array = np.arange(1.0, 1.7, 0.01)* 100
sigma = R_sigma * (C_t ** 2) / (C_rho * (C_W ** 3))  # interfacial tension
u0 = Ca * sigma / Eta_n  # inlet velocity
xi = 2.0  # interface thickness
kappa = 0.75 * xi * sigma
a = - 2.0 * kappa / (xi ** 2)
gamma = u0 * H / ((-a * Pe) * (tau - 0.5))
x = sympy.symbols('x')
print("u0:{}".format(u0))
print("C_t:{}".format(C_t))
print("R_u0:{}".format(Ca * R_sigma / Eta))
print("Ca:{}".format(u0 * Eta_n / sigma))
print("Re:{}".format(u0 * 20 / Eta_n))
Re = u0 * 20 / Eta_n


class Compute:
    def __init__(self, mask):
        self.mask = mask
        self.e = np.array([np.array([0, 0]) for i in range(9)])
        self.w = np.array([0.0 for i in range(9)])
        for i in range(9):
            if i == 0:
                self.w[i] = 4 / 9
                self.e[i][0] = 0
                self.e[i][1] = 0
            elif i == 1:
                self.w[i] = 1 / 9
                self.e[i][0] = 1
                self.e[i][1] = 0
            elif i == 2:
                self.w[i] = 1 / 9
                self.e[i][0] = 0
                self.e[i][1] = 1
            elif i == 3:
                self.w[i] = 1 / 9
                self.e[i][0] = -1
                self.e[i][1] = 0
            elif i == 4:
                self.w[i] = 1 / 9
                self.e[i][0] = 0
                self.e[i][1] = -1
            elif i == 5:
                self.w[i] = 1 / 36
                self.e[i][0] = 1
                self.e[i][1] = 1
            elif i == 6:
                self.w[i] = 1 / 36
                self.e[i][0] = -1
                self.e[i][1] = 1
            elif i == 7:
                self.w[i] = 1 / 36
                self.e[i][0] = -1
                self.e[i][1] = -1
            elif i == 8:
                self.w[i] = 1 / 36
                self.e[i][0] = 1
                self.e[i][1] = -1
        self.psi = np.full((H, W), -1.0)
        self.psi[:, :5] = 1.0
        self.block_mask = np.logical_not(mask)
        self.psi[self.block_mask] = psi_wall
        self.left_wall = np.full((H, 1), 1.0)
        self.right_wall = np.full((H, 1), -1.0)
        self.gamma = gamma
        self.top_bottom_wall = np.full((1, W + 2), psi_wall)
        self.nabla_psix = np.zeros((H, W))
        self.nabla_psiy = np.zeros((H, W))
        self.nabla_psi2 = np.zeros((H, W))
        my_number = np.random.randint(0, 1, size=(H, W)) * 2 - 1.0
        self.rho = np.ones((H, W))[mask] + np.random.rand(H, W)[mask] * 0.001 * my_number[mask]  # macroscopic density
        self.ux = np.zeros((H, W))[mask]
        self.uy = np.zeros((H, W))[mask]
        self.p = np.zeros((H, W))[mask]
        self.f = np.array([np.zeros((H, W)) for i in range(9)])
        self.g = np.array([np.zeros((H, W)) for i in range(9)])
        self.feq = np.array([np.zeros((H, W))[mask] for i in range(9)])
        self.geq = np.array([np.zeros((H, W))[mask] for i in range(9)])
        self.mu = np.zeros((H, W))[mask]
        self.mix_tau = np.zeros((H, W))[mask]
        self.F = np.array([np.zeros((H, W))[mask] for i in range(9)])
        self.nabla_psix = self.getNabla_psix()
        self.nabla_psiy = self.getNabla_psiy()
        self.nabla_psi2 = self.getNabla_psi2()
        self.p = self.getP()
        self.mu = self.getMu()
        self.uy = self.getUy()
        self.mix_tau = self.getMix_tau()
        for i in range(9):
            self.f[i][self.mask] = self.getfeq(i)
            self.g[i][self.mask] = self.getgeq(i)

    def getP(self):
        return 1 / 3 * self.rho + self.psi[self.mask] * self.mu

    def getUx(self):
        temp = np.zeros((H, W))[self.mask]
        for i in range(9):
            temp += self.f[i][self.mask] * self.e[i][0]
        ux = (temp + self.mu * self.nabla_psix[self.mask] / 2) / self.rho
        return ux
        # print("ux:{}, uy:{}".format(self.ux.mean(), self.uy.mean()))

    def getUy(self):
        temp = np.zeros((H, W))[self.mask]
        for i in range(9):
            temp += self.f[i][self.mask] * self.e[i][1]
        uy = (temp + self.mu * self.nabla_psiy[self.mask] / 2) / self.rho
        return uy

    def getMu(self):
        ones = np.ones((H, W))[self.mask]
        mu = a * self.psi[self.mask] * (ones - np.power(self.psi[self.mask], 2)) - kappa * self.nabla_psi2[self.mask]
        return mu

    def getMu_plain(self):
        ones = np.ones((H, W))
        mu = a * self.psi * (ones - np.power(self.psi, 2)) - kappa * self.nabla_psi2
        return mu

    def getRho(self):
        return np.sum(self.f, axis=0)[self.mask]  # macroscopic density
        # print("rho:{}".format(self.rho.mean()))

    def getA0(self):
        a0 = (self.rho - 3.0 * (1.0 - self.w[0]) * self.p) / self.w[0]
        return a0

    def getA1_8(self):
        a1_8 = 3 * self.p
        return a1_8

    def getB0(self):
        b0 = (self.psi[self.mask] - 3.0 * (1.0 - self.w[0]) * self.gamma * self.mu) / self.w[0]
        return b0

    def getB1_8(self):
        b1_8 = 3 * self.gamma * self.mu
        return b1_8

    def getfeq(self, n):
        if n == 0:
            feq = self.w[n] * (self.getA0() + self.rho * (
                    3 * (self.e[n][0] * self.ux + self.e[n][1] * self.uy) + 4.5 * (
                    self.e[n][0] * self.ux + self.e[n][1] * self.uy) ** 2 - 1.5 * (self.ux ** 2 + self.uy ** 2)))
        else:
            feq = self.w[n] * (self.getA1_8() + self.rho * (
                    3 * (self.e[n][0] * self.ux + self.e[n][1] * self.uy) + 4.5 * (
                    self.e[n][0] * self.ux + self.e[n][1] * self.uy) ** 2 - 1.5 * (self.ux ** 2 + self.uy ** 2)))
        # print("feq{}:{}".format(n, feq.mean()))
        return feq

    def getgeq(self, n):
        if n == 0:
            geq = self.w[n] * (self.getB0() + self.psi[self.mask] * (
                    3 * (self.e[n][0] * self.ux + self.e[n][1] * self.uy) + 4.5 * (
                    self.e[n][0] * self.ux + self.e[n][1] * self.uy) ** 2 - 1.5 * (self.ux ** 2 + self.uy ** 2)))
        else:
            geq = self.w[n] * (self.getB1_8() + self.psi[self.mask] * (
                    3 * (self.e[n][0] * self.ux + self.e[n][1] * self.uy) + 4.5 * (
                    self.e[n][0] * self.ux + self.e[n][1] * self.uy) ** 2 - 1.5 * (self.ux ** 2 + self.uy ** 2)))
        return geq

    def getLarge_F(self, n):
        f = self.mu * self.w[n] * (1 - 1 / (2 * self.mix_tau)) \
            * (((self.e[n][0] - self.ux) * 3 + self.e[n][0] * (self.e[n][0] * self.ux + self.e[n][1] * self.uy) * 9)
               * self.nabla_psix[self.mask] + ((self.e[n][1] - self.uy) * 3 + self.e[n][1] * (
                        self.e[n][0] * self.ux + self.e[n][1] * self.uy) * 9) * self.nabla_psiy[self.mask])
        return f

    def getMix_tau(self):
        ones = np.ones((H, W))[self.mask]
        v11 = Eta_n / self.rho
        v2 = Eta_n * M / self.rho
        mix_v = np.divide(2 * v11 * v2, (v11 * (ones - self.psi[self.mask]) + v2 * (ones + self.psi[self.mask])))
        mix_tau = 3 * mix_v + 0.5
        return mix_tau

    def udpatePsi(self):
        self.psi[self.mask] = np.sum(self.g, axis=0)[self.mask]
        self.psi[self.block_mask] = psi_wall

    def getNabla_psix(self):
        f = np.zeros((H, W))
        psi_with_block = copy.deepcopy(self.psi)
        psi_with_block[self.block_mask] = psi_wall
        temp = np.hstack((self.left_wall, np.hstack((psi_with_block, self.right_wall))))
        temp = np.vstack((self.top_bottom_wall, np.vstack((temp, self.top_bottom_wall))))
        for i in range(9):
            if i == 1:
                f += 4 * np.roll(temp, -1, axis=1)[1:-1, 1:-1]
            elif i == 3:
                f += -4 * np.roll(temp, 1, axis=1)[1:-1, 1:-1]
            elif i == 5:
                f += np.roll(np.roll(temp, -1, axis=1), -1, axis=0)[1:-1, 1:-1]
            elif i == 6:
                f += - np.roll(np.roll(temp, 1, axis=1), -1, axis=0)[1:-1, 1:-1]
            elif i == 7:
                f += - np.roll(np.roll(temp, 1, axis=1), 1, axis=0)[1:-1, 1:-1]
            elif i == 8:
                f += np.roll(np.roll(temp, -1, axis=1), 1, axis=0)[1:-1, 1:-1]
        return f / 12

    def getNabla_psiy(self):
        f = np.zeros((H, W))
        psi_with_block = copy.deepcopy(self.psi)
        psi_with_block[self.block_mask] = psi_wall
        temp = np.hstack((self.left_wall, np.hstack((psi_with_block, self.right_wall))))
        temp = np.vstack((self.top_bottom_wall, np.vstack((temp, self.top_bottom_wall))))
        for i in range(9):
            if i == 2:
                f += 4 * np.roll(temp, -1, axis=0)[1:-1, 1:-1]
            elif i == 4:
                f += -4 * np.roll(temp, 1, axis=0)[1:-1, 1:-1]
            elif i == 5:
                f += np.roll(np.roll(temp, -1, axis=1), -1, axis=0)[1:-1, 1:-1]
            elif i == 6:
                f += np.roll(np.roll(temp, 1, axis=1), -1, axis=0)[1:-1, 1:-1]
            elif i == 7:
                f += - np.roll(np.roll(temp, 1, axis=1), 1, axis=0)[1:-1, 1:-1]
            elif i == 8:
                f += - np.roll(np.roll(temp, -1, axis=1), 1, axis=0)[1:-1, 1:-1]
        return f / 12

    def getNabla_psi2(self):
        f = np.zeros((H, W))
        psi_with_block = copy.deepcopy(self.psi)
        psi_with_block[self.block_mask] = psi_wall
        temp = np.hstack((self.left_wall, np.hstack((psi_with_block, self.right_wall))))
        temp = np.vstack((self.top_bottom_wall, np.vstack((temp, self.top_bottom_wall))))
        for i in range(9):
            if i == 0:
                f += -20 * temp[1:-1, 1:-1]
            elif i == 1:
                f += 4 * np.roll(temp, -1, axis=0)[1:-1, 1:-1]
            elif i == 2:
                f += 4 * np.roll(temp, -1, axis=1)[1:-1, 1:-1]
            elif i == 3:
                f += 4 * np.roll(temp, 1, axis=1)[1:-1, 1:-1]
            elif i == 4:
                f += 4 * np.roll(temp, 1, axis=0)[1:-1, 1:-1]
            elif i == 5:
                f += np.roll(np.roll(temp, -1, axis=1), -1, axis=0)[1:-1, 1:-1]
            elif i == 6:
                f += np.roll(np.roll(temp, 1, axis=1), -1, axis=0)[1:-1, 1:-1]
            elif i == 7:
                f += np.roll(np.roll(temp, 1, axis=1), 1, axis=0)[1:-1, 1:-1]
            elif i == 8:
                f += np.roll(np.roll(temp, -1, axis=1), 1, axis=0)[1:-1, 1:-1]
        return f / 6

    def getF(self, i):
        f = self.f[i][self.mask] - 1.0 / self.mix_tau * (self.f[i][self.mask] - self.feq[i]) + self.F[i]
        return f

    def getG(self, i):
        g = self.g[i][self.mask] - 1.0 / tau * (self.g[i][self.mask] - self.geq[i])
        return g

    """http://phelafel.technion.ac.il/~drorden/project/ZouHe.pdf"""

    def zou_he_boundary_inlet(self):
        ux = u0
        psi_x = self.getNabla_psix()
        psi_y = self.getNabla_psiy()
        mu = self.getMu_plain()
        rho_inlet = 1 / (1 - ux) * (self.f[0][:, 0] + self.f[2][:, 0] + self.f[4][:, 0] + 2 * (
                self.f[3][:, 0] + self.f[6][:, 0] + self.f[7][:, 0]) - psi_x[:, 0] * mu[:, 0] / 2)
        psi_in = 1.0 - (
                self.g[0][:, 0] + self.g[2][:, 0] + self.g[3][:, 0] + self.g[4][:, 0] + self.g[6][:, 0] + self.g[7][
                                                                                                          :, 0])
        for i in range(9):
            if i == 1:
                self.f[i][1:-1, 0] = self.f[3][1:-1, 0] + (2 / 3) * ux * rho_inlet[1:-1] - psi_x[1:-1, 0] * mu[1:-1,
                                                                                                            0] / 6
                self.g[i][1:-1, 0] = self.w[i] * psi_in[1:-1] / (self.w[1] + self.w[5] + self.w[8])
            if i == 5:
                self.f[i][1:-1, 0] = self.f[7][1:-1, 0] - 0.5 * (
                        self.f[2][1:-1, 0] - self.f[4][1:-1, 0]) + 1.0 / 6.0 * ux * rho_inlet[1:-1] - psi_x[1:-1,
                                                                                                      0] * mu[1:-1,
                                                                                                           0] / 6 - psi_y[
                                                                                                                    1:-1,
                                                                                                                    0] * mu[
                                                                                                                         1:-1,
                                                                                                                         0] / 4
                self.g[i][1:-1, 0] = self.w[i] * psi_in[1:-1] / (self.w[1] + self.w[5] + self.w[8])
            if i == 8:
                self.f[i][1:-1, 0] = self.f[6][1:-1, 0] + 0.5 * (
                        self.f[2][1:-1, 0] - self.f[4][1:-1, 0]) + 1.0 / 6.0 * ux * rho_inlet[1:-1] - psi_x[1:-1,
                                                                                                      0] * mu[1:-1,
                                                                                                           0] / 6 + psi_y[
                                                                                                                    1:-1,
                                                                                                                    0] * mu[
                                                                                                                         1:-1,
                                                                                                                         0] / 4
                self.g[i][1:-1, 0] = self.w[i] * psi_in[1:-1] / (self.w[1] + self.w[5] + self.w[8])

        # left bottom corner node
        self.f[1][0, 0] = self.f[3][0, 0]
        self.g[1][0, 0] = self.g[3][0, 0]
        self.f[2][0, 0] = self.f[4][0, 0]
        self.g[2][0, 0] = self.g[4][0, 0]
        self.f[5][0, 0] = self.f[7][0, 0]
        self.g[5][0, 0] = self.g[7][0, 0]
        self.f[6][0, 0] = 0.5 * (rho_inlet[1] - (self.f[0][0, 0] + self.f[1][0, 0] + self.f[2][0, 0]
                                                 + self.f[3][0, 0] + self.f[4][0, 0] + self.f[5][0, 0] + self.f[7][
                                                     0, 0]))
        self.g[6][0, 0] = self.w[6] * (1.0 - (self.g[0][0, 0] + self.g[1][0, 0] + self.g[2][0, 0]
                                              + self.g[3][0, 0] + self.g[4][0, 0] + self.g[5][0, 0] + self.g[7][
                                                  0, 0])) / (self.w[6] + self.w[8])
        self.f[8][0, 0] = self.f[6][0, 0]
        self.g[8][0, 0] = self.g[6][0, 0]

        # left top corner node
        self.f[1][-1, 0] = self.f[3][-1, 0]
        self.g[1][-1, 0] = self.g[3][-1, 0]
        self.f[4][-1, 0] = self.f[2][-1, 0]
        self.g[4][-1, 0] = self.g[2][-1, 0]
        self.f[8][-1, 0] = self.f[6][-1, 0]
        self.g[8][-1, 0] = self.g[6][-1, 0]
        self.f[5][-1, 0] = 0.5 * (rho_inlet[-2] - (self.f[0][-1, 0] + self.f[1][-1, 0] + self.f[2][-1, 0]
                                                   + self.f[3][-1, 0] + self.f[4][-1, 0] + self.f[6][-1, 0] + self.f[8][
                                                       -1, 0]))
        self.g[5][-1, 0] = self.w[5] * (1.0 - (self.g[0][-1, 0] + self.g[1][-1, 0] + self.g[2][-1, 0]
                                               + self.g[3][-1, 0] + self.g[4][-1, 0] + self.g[6][-1, 0] + self.g[8][
                                                   -1, 0])) / (self.w[5] + self.w[7])
        self.f[7][-1, 0] = self.f[5][-1, 0]
        self.g[7][-1, 0] = self.g[5][-1, 0]

    """Lattice Boltzmann method: fundamentals and engineering applications with computer codes"""

    # open_boundary_with_force
    def zou_he_boundary_outlet(self):
        ux = u0
        psi_x = self.getNabla_psix()[:, -1]
        psi_y = self.getNabla_psiy()[:, -1]
        mu = self.getMu_plain()[:, -1]
        rho_outlet = 1 / (1 + ux) * (self.f[0][:, -1] + self.f[2][:, -1] + self.f[4][:, -1] + 2 * (
                self.f[1][:, -1] + self.f[5][:, -1] + self.f[8][:, -1]) + psi_x * mu / 2)
        psi_out = -1.0 - (self.g[0][:, -1] + self.g[1][:, -1] + self.g[2][:, -1] + self.g[4][:, -1] + self.g[5][:, -1] +
                          self.g[8][:, -1])
        self.f[3][:, -1] = self.f[1][:, -1] - 1.5 * ux * rho_outlet + psi_x * mu / 6
        self.g[3][:, -1] = self.w[3] * psi_out / (self.w[3] + self.w[6] + self.w[7])
        self.f[6][:, -1] = self.f[8][:, -1] - 0.5 * (
                self.f[2][:, -1] - self.f[4][:, -1]) - 1.0 / 6.0 * ux * rho_outlet + psi_y * mu / 4 + psi_x * mu / 6
        self.g[6][:, -1] = self.w[6] * psi_out / (self.w[3] + self.w[6] + self.w[7])
        self.f[7][:, -1] = self.f[5][:, -1] + 0.5 * (
                self.f[2][:, -1] - self.f[4][:, -1]) - 1.0 / 6.0 * ux * rho_outlet - psi_y * mu / 4 + psi_x * mu / 6
        self.g[7][:, -1] = self.w[7] * psi_out / (self.w[3] + self.w[6] + self.w[7])

        self.f[2][0, -1] = self.f[4][0, -1]
        self.g[2][0, -1] = self.g[4][0, -1]
        self.f[4][-1, -1] = self.f[2][-1, -1]
        self.g[4][-1, -1] = self.g[2][-1, -1]


def stream(f, g):
    # with open_boundary on right side
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


def update(i, x, y, cc):
    print(i)
    plt.cla()
    plt.pcolor(x, y, cc[i], label='MAX_T{}_Pe{}_M{}_Ca{}_wall{}'.format(MAX_T, Pe, M, Ca, psi_wall), cmap='RdBu')
    # plt.clim(0,1)


# plt.legend()


def bottom_top_wall(f_behind, g_behind, f, g):
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


def main():
    rect_corner_list = []

    # for rectangle block
    # while True:
    #     if count * 20 > 360:
    #         break
    #     if flag:
    #         rect_corner_list.append(((count * 20, 60), ((count + 1) * 20, 80)))
    #         rect_corner_list.append(((count * 20, 140), ((count + 1) * 20, 160)))
    #         rect_corner_list.append(((count * 20, 220), ((count + 1) * 20, 240)))
    #         rect_corner_list.append(((count * 20, 300), ((count + 1) * 20, 320)))
    #         flag = False
    #         count += 2
    #         continue
    #     elif not flag:
    #         rect_corner_list.append(((count * 20, 20), ((count + 1) * 20, 40)))
    #         rect_corner_list.append(((count * 20, 100), ((count + 1) * 20, 120)))
    #         rect_corner_list.append(((count * 20, 180), ((count + 1) * 20, 200)))
    #         rect_corner_list.append(((count * 20, 260), ((count + 1) * 20, 280)))
    #         rect_corner_list.append(((count * 20, 340), ((count + 1) * 20, 360)))
    #         flag = True
    #         count += 2
    # block_psi_all, corner_list = setblock(rect_corner_list)
    cr = Createblock(H, W)
    bb = Bounce_back(H, W)
    # circle_list = []
    # circle_list.append(((int(W/2), int(H/2)), 30))
    # r = 13
    # circle_list.append(((count * 2 * r, xx + r), r))
    # circle_list.append(((count * 2 * r, 2 * xx + 3 * r), r))
    # circle_list.append(((count * 2 * r, 3 * xx + 5 * r + 1),  r + 1))
    # circle_list.append(((count * 2 * r, 3 * xx + 5 * r), r))
    # count = 2
    # while True:
    #     if count * 2 * r > 370:
    #         break
    #     if flag:
    #         circle_list.append(((count * 2 * r, 3 * r), r + 5))
    #         circle_list.append(((count * 2 * r, xx + 5 * r), r + 5))
    #         circle_list.append(((count * 2 * r, 2 * xx + 7 * r), r + 5))
    #         circle_list.append(((count * 2 * r, 3 * xx + 9 * r), r + 5))
    #         print(3 * xx + 5 * r)
    #         print(2 * xx + 3 * r)
    #         flag = False
    #         count += 2
    #         continue
    #     elif not flag:
    #         circle_list.append(((count * 2 * r, xx + r), r + 5))
    #         circle_list.append(((count * 2 * r, 2 * xx + 3 * r), r + 5))
    #         circle_list.append(((count * 2 * r, 3 * xx + 5 * r), r + 5))
    #         flag = True
    #         count += 2
    # ellipse_list = []
    mabiki = MAX_T // 100
    # circle_list = []
    # r = 10
    # xx = 10
    # count = 3
    flag = True
    # while True:
    #    if count * (xx + r) > 380:
    #        break
    #    for i in range(block_num):
    #        circle_list.append(((count * (r + xx), (2 * i + 1) * (r + xx)), r))
    #    count += 2

    # while True:
    #    if count * (xx + r) > 380:
    #        break
    #    if flag:
    #        for i in range(block_num):
    #            circle_list.append(((count * (r + xx), (2 * i + 1) * (r + xx)), r))
    #        count += 2
    #        flag = False
    #    else:
    #        for i in range(block_num - 1):
    #            circle_list.append(((count * (r + xx), (2 * i + 2) * (r + xx)), r))
    #        count += 2
    #        flag = True
    count = 1
    while True:
        if (count + 1) * 20 > 380:
            break
        if flag:
            for i in range(4):
                rect_corner_list.append(
                    ((count * 20, 60 * (i + 1) + i * 20), ((count + 1) * 20, 60 * (i + 1) + (i + 1) * 20)))
            flag = False
            count += 2
            continue
        else:
            for i in range(5):
                rect_corner_list.append(
                    ((count * 20, 60 * i + (i + 1) * 20), ((count + 1) * 20, 60 * i + (i + 2) * 20)))
            flag = True
            count += 2
    block_psi_all, corner_list = cr.setblock(rect_corner_list)

    # block_psi_all, side_list, concave_list, convex_list = cr.setCirleblock(circle_list)
    block_mask = np.where(block_psi_all == 1, True, False)
    mask = np.logical_not(block_mask)
    cm = Compute(mask)
    cc = np.array([cm.psi])
    for i in range(MAX_T):
        for j in range(9):
            cm.F[j] = cm.getLarge_F(j)
            cm.feq[j] = cm.getfeq(j)
            cm.geq[j] = cm.getgeq(j)
            cm.f[j][mask] = cm.getF(j)
            cm.g[j][mask] = cm.getG(j)
        if i % mabiki == 0:
            cc = np.append(cc, np.array([cm.psi]), axis=0)
        print("timestep:{}".format(i))
        f_behind = copy.deepcopy(cm.f)
        g_behind = copy.deepcopy(cm.g)
        stream(cm.f, cm.g)
        # bb.halfway_bounceback_circle(side_list, concave_list, convex_list, f_behind, g_behind, cm.f, cm.g)
        bb.halfway_bounceback_rec(corner_list, f_behind, g_behind, cm.f, cm.g)
        bottom_top_wall(f_behind[:, 1:-1], g_behind[:, 1:-1], cm.f[:, 1:-1], cm.g[:, 1:-1])
        cm.zou_he_boundary_inlet()
        cm.zou_he_boundary_outlet()
        cm.rho = cm.getRho()
        cm.udpatePsi()
        cm.nabla_psix = cm.getNabla_psix()
        cm.nabla_psiy = cm.getNabla_psiy()
        cm.nabla_psi2 = cm.getNabla_psi2()
        cm.mu = cm.getMu()
        cm.ux = cm.getUx()
        cm.uy = cm.getUy()
        cm.p = cm.getP()
        cm.mix_tau = cm.getMix_tau()
    y = [i for i in range(H)]
    x = [i for i in range(W)]
    # fig = plt.figure()
    # plt.colorbar(plt.pcolor(x, y, cc[0], cmap='RdBu'))
    # ani = animation.FuncAnimation(fig, update, fargs=(x, y, cc), frames=int(len(cc)))
    # ani.save('../movies/bbc_b_num{}_Pe{}_M{}_Ca{:.4f}_wall{}_Re{:.2f}_sigma{}_tau{}_Eta{}.mp4'.format(block_num, Pe, M, Ca, psi_wall, Re, R_sigma, tau, Eta), fps=10)
    plt.figure()
    plt.pcolor(x, y, cm.psi, cmap='RdBu')
    plt.colorbar()
    # plt.legend()
    # plt.grid()
    plt.show()
    # plt.savefig('../images/bbc_b_num{}_Pe{}_M{}_Ca{:.4f}_wall{}_Re{:.2f}_sigma{}_tau{}_Eta{}.png'.format(block_num, Pe, M, Ca, psi_wall, Re, R_sigma, tau, Eta))


if __name__ == '__main__':
    np.seterr(all='raise')
    try:
        t1 = time.time()
        main()
        t2 = time.time()
        print((t2 - t1) / 60)
        # subprocess.call(["gdrive upload ../movies/bbc_b_num{}_Pe{}_M{}_Ca{:.4f}_wall{}_Re{:.2f}_sigma{}_tau{}_Eta{}.mp4".format(block_num, Pe, M, Ca, psi_wall, Re, R_sigma, tau, Eta)], shell=True)
        # subprocess.call(["gdrive upload ../images/bbc_b_num{}_Pe{}_M{}_Ca{:.4f}_wall{}_Re{:.2f}_sigma{}_tau{}_Eta{}.png".format(block_num, Pe, M, Ca, psi_wall, Re, R_sigma, tau, Eta)], shell=True)
    except Warning as e:
        print(e)
