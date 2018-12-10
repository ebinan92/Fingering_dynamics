import numpy as np
import sympy
import matplotlib.pyplot as plt
import copy
import cv2
from scipy.ndimage.morphology import binary_fill_holes
import matplotlib.animation as animation
from scipy import optimize
from scipy.ndimage.filters import convolve
import math
import time

H = 300  # lattice dimensions
W = 300
MAX_T = 3000
psi_wall = 0.0  # wettability on block and wall
Pe = 100  # Peclet number
Ca = 7.66 * 10.0 ** (-3)  # Capillary number
M = 20.0  # Eta non_newtonian / Eta newtonian
R_Nu = 10.0 ** (-6)  # physical kinematic viscosity of newtonian
tau = 1 / (3 - np.sqrt(3))  # relaxation time
rho0 = 1.0  # non-dimensional pressure
n_non = 1.0  # rho0 power-law parameter
Eta_n = 0.023  # Eta newtonian
R_sigma = 0.045  # physical interfacial tension
C_W = 4.0 * 10.0 ** (-5) / W  # conversion width
C_rho = 1 / rho0 * 10.0 ** 3  # conversion pressure
v1 = (tau - 0.5) / 3  # non-dimensional kinematic viscosity of newtonian
C_t = v1 / R_Nu * (C_W ** 2)  # conversion time step
DELTA_X = 1  # lattice spacing
DELTA_T = 1  # time step
x_array = np.arange(1.0, 1.7, 0.01) * DELTA_T * 100
sigma = R_sigma * (C_t ** 2) / (C_rho * (C_W ** 3))  # interfacial tension
u0 = Ca * sigma / (rho0 * v1)  # inlet velocity
c = DELTA_X / DELTA_T  # particle streaming speed
cs = c / np.sqrt(3)  # lattice speed of sound
xi = 2.0 * DELTA_X  # interface thickness
kappa = (3 / 4) * sigma * xi  # interfacial tension
a = - 2 * kappa / (xi ** 2)
gamma = u0 * W / (-a * Pe) / ((tau - 0.5) * DELTA_T)
x = sympy.symbols('x')
print("u0:{}".format(u0))


class Compute:
    def __init__(self, block_mask):
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
        self.psi[:, :20] = 1.0
        self.block_mask = block_mask
        self.left_wall = np.full((H, 1), 1.0).astype(float)
        self.right_wall = np.full((H, 1), -1.0).astype(float)
        self.gamma = gamma
        self.psi_wall_list = np.full((1, W + 2), psi_wall).astype(float)
        self.nabla_psix = self.getNabla_psix()
        self.nabla_psiy = self.getNabla_psiy()
        self.nabla_psi2 = self.getNabla_psi2()
        self.rho = np.ones((H, W), dtype=float) * rho0  # macroscopic density
        self.ux = np.zeros((H, W), dtype=float)
        self.uy = np.zeros((H, W), dtype=float)
        self.f = np.array([np.zeros((H, W), dtype=float) for i in range(9)])
        self.g = np.array([np.zeros((H, W), dtype=np.float128) for i in range(9)])
        self.feq = np.array([np.zeros((H, W), dtype=float) for i in range(9)])
        self.geq = np.array([np.zeros((H, W), dtype=np.float128) for i in range(9)])
        self.mu = self.getMu()
        self.F = np.array([np.zeros((H, W), dtype=float) for i in range(9)]).astype(float)
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
            if i >= 5:
                self.f[i] = self.getfeq(i)
                self.g[i] = self.getgeq(i)

    def getP(self):
        return (cs ** 2) * self.rho + self.psi * self.mu

    def updateP(self):
        self.p = (cs ** 2) * self.rho + self.psi * self.mu
        # print("p:{}".format(self.p.mean()))

    def updateU(self):
        temp1 = np.zeros((H, W), dtype=float)
        temp2 = np.zeros((H, W), dtype=float)
        for i in range(9):
            temp1 += self.f[i] * self.e[i][0]
            temp2 += self.f[i] * self.e[i][1]
        self.ux = (temp1 + self.mu * self.nabla_psix * DELTA_T / 2) / self.rho
        self.uy = (temp2 + self.mu * self.nabla_psiy * DELTA_T / 2) / self.rho
        # print("ux:{}, uy:{}".format(self.ux.mean(), self.uy.mean()))

    def getMu(self):
        mu = a * self.psi - a * (self.psi ** 3) - kappa * self.nabla_psi2
        return mu

    def updateMu(self):
        self.mu = a * self.psi - a * (self.psi ** 3) - kappa * self.nabla_psi2
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
                / (cs ** 4)) * self.nabla_psix
               + ((self.e[n][1] - self.uy) / (cs ** 2) + self.e[n][1] * (
                        self.e[n][0] * self.ux + self.e[n][1] * self.uy) / (
                          cs ** 4)) * self.nabla_psiy)
        return f

    def updateMix_tau(self):
        v2 = v1 * M
        mix_v = np.divide(2 * v1 * v2, (v1 * (1.0 - self.psi) + v2 * (1.0 + self.psi)))
        mix_tau = 3 * mix_v / (c ** 2) + 0.5 * DELTA_T
        self.mix_tau = mix_tau
        # print("mix_tau:{}".format(mix_tau.mean()))

    def getMix_tau(self):
        v2 = v1 * M
        mix_v = np.divide(2 * v1 * v2, (v1 * (1.0 - self.psi) + v2 * (1.0 + self.psi)))
        mix_tau = 3 * mix_v / (c ** 2) + 0.5 * DELTA_T
        return mix_tau

    def updatePsi(self):
        self.psi = np.sum(self.g, axis=0)
        # self.psi[int(H / 2 - 10):int(H / 2 + 10), int(W / 4 - 20):int(W / 4)] = psi_wall
        self.psi[self.block_mask] = psi_wall

    def getNabla_psix(self):
        f = np.zeros((H, W), dtype=float)
        psi_with_block = copy.deepcopy(self.psi)
        psi_with_block[self.block_mask] = psi_wall
        temp = np.hstack((self.left_wall, np.hstack((psi_with_block, self.right_wall))))
        temp = np.vstack((self.psi_wall_list, np.vstack((temp, self.psi_wall_list))))
        for i in range(9):
            if i == 0 or i == 2 or i == 4:
                continue
            elif i == 1:
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
        return f / (12 * DELTA_X)

    def getNabla_psiy(self):
        f = np.zeros((H, W), dtype=float)
        psi_with_block = copy.deepcopy(self.psi)
        psi_with_block[self.block_mask] = psi_wall
        temp = np.hstack((self.left_wall, np.hstack((psi_with_block, self.right_wall))))
        temp = np.vstack((self.psi_wall_list, np.vstack((temp, self.psi_wall_list))))
        for i in range(9):
            if i == 0 or i == 1 or i == 3:
                continue
            elif i == 2:
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
        return f / (12 * DELTA_X)

    def getNabla_psi2(self):
        f = np.zeros((H, W), dtype=float)
        psi_with_block = copy.deepcopy(self.psi)
        psi_with_block[self.block_mask] = psi_wall
        temp = np.hstack((self.left_wall, np.hstack((psi_with_block, self.right_wall))))
        temp = np.vstack((self.psi_wall_list, np.vstack((temp, self.psi_wall_list))))
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
        return f / (6 * (DELTA_X ** 2))

    # def getPsi_wall_list(self):
    #     psi_wall = sympy.symbols('psi')
    #     out = sympy.solve(1 / 2 * psi_wall ** 3 - 3 / 2 * psi_wall + np.cos(Theta))[0].as_real_imag()
    #     return np.full((1, W), out[0]).astype(float)

    def updateF(self):
        for i in range(9):
            self.f[i] = self.f[i] - DELTA_T / self.mix_tau * (self.f[i] - self.feq[i]) + self.F[i]

    def updateG(self):
        for i in range(9):
            self.g[i] = self.g[i] - 1 / tau * (self.g[i] - self.geq[i])

    """http://phelafel.technion.ac.il/~drorden/project/ZouHe.pdf"""

    def zou_he_boundary_inlet(self):
        ux = u0
        rho_inlet = 1 / (1 - ux) * (self.f[0][:, 0] + self.f[2][:, 0] + self.f[4][:, 0] + 2 * (
                self.f[3][:, 0] + self.f[6][:, 0] + self.f[7][:, 0]))
        psi_in = 1.0 - (
                self.g[0][:, 0] + self.g[2][:, 0] + self.g[3][:, 0] + self.g[4][:, 0] + self.g[6][:, 0] + self.g[7][
                                                                                                          :, 0])
        for i in range(9):
            if i == 1:
                self.f[i][1:-1, 0] = self.f[3][1:-1, 0] + 1.5 * ux * rho_inlet[1:-1]
                self.g[i][1:-1, 0] = self.w[i] * psi_in[1:-1] / (self.w[1] + self.w[5] + self.w[8])
            if i == 5:
                self.f[i][1:-1, 0] = self.f[7][1:-1, 0] - 0.5 * (
                        self.f[2][1:-1, 0] - self.f[4][1:-1, 0]) + 1.0 / 6.0 * ux * rho_inlet[1:-1]
                self.g[i][1:-1, 0] = self.w[i] * psi_in[1:-1] / (self.w[1] + self.w[5] + self.w[8])
            if i == 8:
                self.f[i][1:-1, 0] = self.f[6][1:-1, 0] + 0.5 * (
                        self.f[2][1:-1, 0] - self.f[4][1:-1, 0]) + 1.0 / 6.0 * ux * rho_inlet[1:-1]
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

    def zou_he_boundary_outlet(self):
        ux = u0
        rho_outlet = 1 / (1 - ux) * (self.f[0][:, -1] + self.f[2][:, -1] + self.f[4][:, -1] + 2 * (
                self.f[1][:, -1] + self.f[5][:, -1] + self.f[8][:, -1]))
        psi_out = -1.0 - (self.g[0][:, -1] + self.g[1][:, -1] + self.g[2][:, -1] + self.g[4][:, -1] + self.g[5][:, -1] +
                          self.g[8][:, -1])
        self.f[3][:, -1] = self.f[1][:, -1] + 1.5 * ux * rho_outlet
        self.g[3][:, -1] = self.w[3] * psi_out / (self.w[3] + self.w[6] + self.w[7])
        self.f[6][:, -1] = self.f[8][:, -1] - 0.5 * (
                self.f[2][:, -1] - self.f[4][:, -1]) + 1.0 / 6.0 * ux * rho_outlet
        self.g[6][:, -1] = self.w[6] * psi_out / (self.w[3] + self.w[6] + self.w[7])
        self.f[7][:, -1] = self.f[5][:, -1] + 0.5 * (
                self.f[2][:, -1] - self.f[4][:, -1]) + 1.0 / 6.0 * ux * rho_outlet
        self.g[7][:, -1] = self.w[7] * psi_out / (self.w[3] + self.w[6] + self.w[7])


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


class Createblock:
    # ブロックの枠組みを指定　頂点で決める
    def getRectangleBlock(self, bottom_left, top_right):
        block = np.zeros((H, W)).astype(np.uint8)
        cv2.rectangle(block, bottom_left, top_right, (1, 0, 0))
        return block

    #
    def getCircleBlock(self, center, radius):
        block = np.zeros((H, W)).astype(np.uint8)
        cv2.circle(block, center, radius, (1, 0, 0))
        block_psi = binary_fill_holes(block).astype(int)
        block_psi[center[1] - radius, center[0]] = 3
        # for i in range(radius, 0, -1):
        #     for j in range(-radius, radius, 1):
        #         if block_psi[center[1] - i, center[0] + j]

        #result is dilated for marking the corners, not important
        # dst = cv2.dilate(dst,None)
        # Threshold for an optimal value, it may vary depending on the image.
        #block_psi[dst>0.01*dst.max()]=[0,0,255]
        return block_psi

    def getCorner(self, block):
        points = cv2.findNonZero(block)
        x, y, w, h = cv2.boundingRect(points)
        top_right = (x + w - 1, y + h - 1)
        top_left = (x, y + h - 1)
        bottom_right = (x + w - 1, y)
        bottom_left = (x, y)
        corner = {"top_left": top_left, 'bottom_left': bottom_left, 'top_right': top_right,
                  'bottom_right': bottom_right}
        return corner


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

"""https://www.math.nyu.edu/~billbao/report930.pdf"""
# mid-grid halfway bounce back
def halfway_bounceback(corner_list, f_behind, g_behind, f, g):
    n_barrier = np.zeros((H, W), dtype=bool)
    s_barrier = np.zeros((H, W), dtype=bool)
    e_barrier = np.zeros((H, W), dtype=bool)
    w_barrier = np.zeros((H, W), dtype=bool)
    nw_corner = np.zeros((H, W), dtype=bool)
    ne_corner = np.zeros((H, W), dtype=bool)
    sw_corner = np.zeros((H, W), dtype=bool)
    se_corner = np.zeros((H, W), dtype=bool)
    for cor in corner_list:
        n_barrier[cor['top_left'][1]+1, cor['top_left'][0] + 1:cor['top_right'][0]] = True
        s_barrier[cor['bottom_left'][1]-1, cor['top_left'][0] + 1:cor['top_right'][0]] = True
        w_barrier[cor['bottom_left'][1] + 1:cor['top_left'][1], cor['top_right'][0]+1] = True
        e_barrier[cor['bottom_left'][1] + 1:cor['top_left'][1], cor['top_left'][0]-1] = True
        nw_corner[cor['top_right'][1]+1, cor['top_right'][0]+1] = True
        ne_corner[cor['top_left'][1]+1, cor['top_left'][0]-1] = True
        sw_corner[cor['bottom_right'][1]-1, cor['bottom_right'][0]+1] = True
        se_corner[cor['bottom_left'][1]-1, cor['bottom_left'][0]-1] = True
        # print(cor['top_left'], cor['top_right'], cor['bottom_left'], cor['bottom_right'])

    for i in range(9):
        if i == 1:
            f[i][nw_corner] = f_behind[3][nw_corner]
            g[i][nw_corner] = g_behind[3][nw_corner]
            f[i][sw_corner] = f_behind[3][sw_corner]
            g[i][sw_corner] = g_behind[3][sw_corner]
            f[i][w_barrier] = f_behind[3][w_barrier]
            g[i][w_barrier] = g_behind[3][w_barrier]
        elif i == 2:
            f[i][nw_corner] = f_behind[4][nw_corner]
            g[i][nw_corner] = g_behind[4][nw_corner]
            f[i][ne_corner] = f_behind[4][ne_corner]
            g[i][ne_corner] = g_behind[4][ne_corner]
            f[i][n_barrier] = f_behind[4][n_barrier]
            g[i][n_barrier] = g_behind[4][n_barrier]
        elif i == 3:
            f[i][ne_corner] = f_behind[1][ne_corner]
            g[i][ne_corner] = g_behind[1][ne_corner]
            f[i][se_corner] = f_behind[1][se_corner]
            g[i][se_corner] = g_behind[1][se_corner]
            f[i][e_barrier] = f_behind[1][e_barrier]
            g[i][e_barrier] = g_behind[1][e_barrier]
        elif i == 4:
            f[i][sw_corner] = f_behind[2][sw_corner]
            g[i][sw_corner] = g_behind[2][sw_corner]
            f[i][se_corner] = f_behind[2][se_corner]
            g[i][se_corner] = g_behind[2][se_corner]
            f[i][s_barrier] = f_behind[2][s_barrier]
            g[i][s_barrier] = g_behind[2][s_barrier]
        elif i == 5:
            f[i][nw_corner] = f_behind[7][nw_corner]
            g[i][nw_corner] = g_behind[7][nw_corner]
            f[i][n_barrier] = f_behind[7][n_barrier]
            g[i][n_barrier] = g_behind[7][n_barrier]
            f[i][w_barrier] = f_behind[7][w_barrier]
            g[i][w_barrier] = g_behind[7][w_barrier]
        elif i == 6:
            f[i][ne_corner] = f_behind[8][ne_corner]
            g[i][ne_corner] = g_behind[8][ne_corner]
            f[i][n_barrier] = f_behind[8][n_barrier]
            g[i][n_barrier] = g_behind[8][n_barrier]
            f[i][e_barrier] = f_behind[8][e_barrier]
            g[i][e_barrier] = g_behind[8][e_barrier]
        elif i == 7:
            f[i][se_corner] = f_behind[5][se_corner]
            g[i][se_corner] = g_behind[5][se_corner]
            f[i][s_barrier] = f_behind[5][s_barrier]
            g[i][s_barrier] = g_behind[5][s_barrier]
            f[i][e_barrier] = f_behind[5][e_barrier]
            g[i][e_barrier] = g_behind[5][e_barrier]
        elif i == 8:
            f[i][sw_corner] = f_behind[6][sw_corner]
            g[i][sw_corner] = g_behind[6][sw_corner]
            f[i][s_barrier] = f_behind[6][s_barrier]
            g[i][s_barrier] = g_behind[6][s_barrier]
            f[i][w_barrier] = f_behind[6][w_barrier]
            g[i][w_barrier] = g_behind[6][w_barrier]


def setblock(rect_corner_list):
    cr = Createblock()
    corner_list = []
    #0 or 1の配列　1がblockのあるところを示す
    block_psi_all = np.zeros((H, W), dtype=int)
    for rect in rect_corner_list:
        block = cr.getRectangleBlock(rect[0], rect[1])
        block_psi = binary_fill_holes(block).astype(int)
        corner = cr.getCorner(block)
        corner_list.append(corner)
        print(block_psi.shape)
        block_psi_all = block_psi_all + block_psi
    return block_psi_all, corner_list

def main():
    rect_corner_list = [((15, 60), (35, 80))]
    rect_corner_list.append(((15, 140), (35, 160)))
    rect_corner_list.append(((15, 220), (35, 240)))
    rect_corner_list.append(((55, 20), (75, 40)))
    rect_corner_list.append(((55, 82), (75, 102)))
    rect_corner_list.append(((55, 150), (75, 170)))
    rect_corner_list.append(((55, 218), (75, 238)))
    rect_corner_list.append(((85, 116), (105, 136)))
    rect_corner_list.append(((85, 186), (105, 206)))
    rect_corner_list.append(((85, 252), (105, 272)))
    rect_corner_list.append(((120, 14), (140, 34)))
    rect_corner_list.append(((120, 82), (140, 102)))
    rect_corner_list.append(((120, 150), (140, 170)))
    rect_corner_list.append(((120, 218), (140, 238)))
    rect_corner_list.append(((120, 286), (140, 306)))
    block_psi_all, corner_list = setblock(rect_corner_list)
    block_mask = np.where(block_psi_all == 1, True, False)
    cm = Compute(block_mask)
    for i in range(MAX_T):
        f_behind = copy.deepcopy(cm.f)
        g_behind = copy.deepcopy(cm.g)
        stream(cm.f, cm.g)
        halfway_bounceback(corner_list, f_behind, g_behind, cm.f, cm.g)
        bottom_top_wall(f_behind[:, 1:-1], g_behind[:, 1:-1], cm.f[:, 1:-1], cm.g[:, 1:-1])
        cm.zou_he_boundary_inlet()
        cm.zou_he_boundary_outlet()
        cm.updateRho()
        cm.updatePsi()
        cm.nabla_psix = cm.getNabla_psix()
        cm.nabla_psiy = cm.getNabla_psiy()
        cm.nabla_psi2 = cm.getNabla_psi2()
        cm.updateMu()
        cm.updateP()
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
        #main()
        cr = Createblock()
        block = cr.getCircleBlock((50, 60), 30)
        y = [i for i in range(H)]
        x = [i for i in range(W)]
        plt.figure()
        plt.pcolor(x, y, block, label="MAX_T:{}, Pe:{}, M:{}".format(MAX_T, Pe, M))
        plt.colorbar()
        plt.legend()
        plt.show()
        t2 = time.time()
        print((t2 - t1) / 60)
    except Warning as e:
        print(e)
