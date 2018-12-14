#!pip install -U -q PyDrive
import numpy as np
import sympy
import matplotlib.pyplot as plt
import copy
import cv2
import math
from multiprocessing import Pool
from scipy.ndimage.morphology import binary_fill_holes
import matplotlib.animation as animation
from scipy import optimize
from scipy.ndimage.filters import convolve
import math
import time
# from pydrive.auth import GoogleAuth
# from pydrive.drive import GoogleDrive
# from google.colab import auth
# from oauth2client.client import GoogleCredentials
# auth.authenticate_user()
# gauth = GoogleAuth()
# gauth.credentials = GoogleCredentials.get_application_default()
# drive = GoogleDrive(gauth)

H = 380  # lattice dimensions
W = 380
MAX_T = 82000
psi_wall = -1.0  # wettability on block and wall
Pe = 40  # Peclet number
C_W = 2.0 * (10 ** (-5)) / W  # conversion width
Ca = 7.33 * 10 ** (-3)  # Capillary number
M = 10.0  # Eta non_newtonian / Eta newtonian
R_Nu = 10 ** (-6)  # physical kinematic viscosity of newtonian
tau = 1 / (3.0 - math.sqrt(3))  # relaxation time
rho0 = 1.0  # non-dimensional pressure
n_non = 1.0  # rho0 power-law parameter
Eta_n = 0.023  # Eta newtonian
R_sigma = 0.045  # physical interfacial tension
C_rho = 1.0 * 10 ** 3  # conversion pressure
v1 = (tau - 0.5) / 3  # non-dimensional kinematic viscosity of newtonian
C_t = v1 / R_Nu * (C_W ** 2)  # conversion time step
#x_array = np.arange(1.0, 1.7, 0.01)* 100
sigma = R_sigma * (C_t ** 2) / (C_rho * (C_W ** 3))  # interfacial tension
u0 = Ca * sigma / (rho0 * v1)  # inlet velocity
xi = 2.0  # interface thickness
kappa = 0.75 * sigma * xi  # interfacial tension
a = - 2.0 * kappa / (xi ** 2)
gamma = u0 * W / (-a * Pe) / (tau - 0.5)
x = sympy.symbols('x')
print("u0:{}".format(u0))

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
            print(self.e[i], self.w[i])
        self.psi = np.full((H, W), -1.0)
        self.psi[:, :10] = 1.0
        self.block_mask = np.logical_not(mask)
        self.psi[self.block_mask] = psi_wall
        self.left_wall = np.full((H, 1), 1.0)
        self.right_wall = np.full((H, 1), -1.0)
        self.gamma = gamma
        self.psi_wall_list = np.full((1, W + 2), psi_wall)
        self.nabla_psix = np.zeros((H, W))[mask]
        self.nabla_psiy = np.zeros((H, W))[mask]
        self.nabla_psi2 = np.zeros((H, W))[mask]
        self.rho = np.ones((H, W))[mask] * rho0  # macroscopic density
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
        self.mu = self.getMu()
        # self.uy = self.getUy()
        self.p = self.getP()
        self.mix_tau = self.getMix_tau()
        for i in range(9):
            self.f[i][self.mask] = self.getfeq(i)
            self.g[i][self.mask] = self.getgeq(i)

    def getP(self):
        return 1/3 * self.rho + self.psi[self.mask] * self.mu

    def getUx(self):
        temp = np.zeros((H, W))[self.mask]
        for i in range(9):
            temp += self.f[i][self.mask] * self.e[i][0]
        ux = (temp + self.mu * self.nabla_psix / 2) / self.rho
        return ux
        # print("ux:{}, uy:{}".format(self.ux.mean(), self.uy.mean()))
    def getUy(self):
        temp = np.zeros((H, W))[self.mask]
        for i in range(9):
            temp += self.f[i][self.mask] * self.e[i][1]
        uy = (temp + self.mu * self.nabla_psiy / 2) / self.rho
        return uy

    def getMu(self):
        mu = a * self.psi[self.mask] * (1.0 - np.power(self.psi[self.mask], 2)) - kappa * self.nabla_psi2
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
               * self.nabla_psix + ((self.e[n][1] - self.uy) * 3 + self.e[n][1] * (
                        self.e[n][0] * self.ux + self.e[n][1] * self.uy) * 9) * self.nabla_psiy)
        return f

    def getMix_tau(self):
        v2 = v1 * M
        mix_v = np.divide(2 * v1 * v2, (v1 * (1.0 - self.psi[self.mask]) + v2 * (1.0 + self.psi[self.mask])))
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
        temp = np.vstack((self.psi_wall_list, np.vstack((temp, self.psi_wall_list))))
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
        return f[self.mask] / 12

    def getNabla_psiy(self):
        f = np.zeros((H, W))
        psi_with_block = copy.deepcopy(self.psi)
        psi_with_block[self.block_mask] = psi_wall
        temp = np.hstack((self.left_wall, np.hstack((psi_with_block, self.right_wall))))
        temp = np.vstack((self.psi_wall_list, np.vstack((temp, self.psi_wall_list))))
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
        return f[self.mask] / 12

    def getNabla_psi2(self):
        f = np.zeros((H, W))
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
        return f[self.mask] / 6

    def getF(self, i):
        f = self.f[i][self.mask] - 1.0 / self.mix_tau * (self.f[i][self.mask] - self.feq[i]) + self.F[i]
        return f

    def getG(self, i):
        g = self.g[i][self.mask] - 1.0 / tau * (self.g[i][self.mask] - self.geq[i])
        return g

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
        rho_outlet = 1 / (1 + ux) * (self.f[0][:, -1] + self.f[2][:, -1] + self.f[4][:, -1] + 2 * (
                self.f[1][:, -1] + self.f[5][:, -1] + self.f[8][:, -1]))
        psi_out = -1.0 - (self.g[0][:, -1] + self.g[1][:, -1] + self.g[2][:, -1] + self.g[4][:, -1] + self.g[5][:, -1] +
                          self.g[8][:, -1])
        self.f[3][:, -1] = self.f[1][:, -1] - 1.5 * ux * rho_outlet
        self.g[3][:, -1] = self.w[3] * psi_out / (self.w[3] + self.w[6] + self.w[7])
        self.f[6][:, -1] = self.f[8][:, -1] - 0.5 * (
                self.f[2][:, -1] - self.f[4][:, -1]) - 1.0 / 6.0 * ux * rho_outlet
        self.g[6][:, -1] = self.w[6] * psi_out / (self.w[3] + self.w[6] + self.w[7])
        self.f[7][:, -1] = self.f[5][:, -1] + 0.5 * (
                self.f[2][:, -1] - self.f[4][:, -1]) - 1.0 / 6.0 * ux * rho_outlet
        self.g[7][:, -1] = self.w[7] * psi_out / (self.w[3] + self.w[6] + self.w[7])

        self.f[2][0, -1] = self.f[4][0, -1]
        self.g[2][0, -1] = self.g[4][0, -1]
        self.f[4][-1, -1] = self.f[2][-1, -1]
        self.g[4][-1, -1] = self.g[2][-1, -1]



def create_circle(n, r):
    y, x = np.ogrid[-int(H / 2): int(H / 2), -r: n - r]
    mask = x ** 2 + y ** 2 <= r ** 2
    return mask


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


class Createblock:
    # ブロックの枠組みを指定　頂点で決める
    def getRectangleBlock(self, bottom_left, top_right):
        block = np.zeros((H, W)).astype(np.uint8)
        cv2.rectangle(block, bottom_left, top_right, (1, 0, 0))
        return block


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
        #print(cor['top_left'], cor['top_right'], cor['bottom_right'], cor['bottom_left'])
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
            f[i][w_barrier] = f_behind[3][w_barrier]
            g[i][w_barrier] = g_behind[3][w_barrier]
        elif i == 2:
            f[i][n_barrier] = f_behind[4][n_barrier]
            g[i][n_barrier] = g_behind[4][n_barrier]
        elif i == 3:
            f[i][e_barrier] = f_behind[1][e_barrier]
            g[i][e_barrier] = g_behind[1][e_barrier]
        elif i == 4:
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
        #print(block_psi.shape)
        block_psi_all = block_psi_all + block_psi
    print(block_psi_all.max(), "max")
    return block_psi_all, corner_list

def update(i, x, y, cc):
    print(i)
    plt.cla()
    plt.pcolor(x, y, cc[i], label='MAX_T{}_Pe{}_M{}_Ca{}_wall{}'.format(MAX_T, Pe, M, Ca, psi_wall), cmap='RdBu')
    # plt.clim(0,1)
    plt.legend()

def main():
    rect_corner_list = []
    count = 1
    flag = True
    mabiki = MAX_T // 150
    while True:
        if count * 20 > 360:
            break
        if flag:
            rect_corner_list.append(((count * 20, 60), ((count+1) * 20, 80)))
            rect_corner_list.append(((count * 20, 140), ((count+1) * 20, 160)))
            rect_corner_list.append(((count * 20, 220), ((count+1) * 20, 240)))
            rect_corner_list.append(((count * 20, 300), ((count+1) * 20, 320)))
            flag = False
            count += 2
            continue
        elif not flag:
            rect_corner_list.append(((count * 20, 20), ((count+1) * 20, 40)))
            rect_corner_list.append(((count * 20, 100), ((count+1) * 20, 120)))
            rect_corner_list.append(((count * 20, 180), ((count+1) * 20, 200)))
            rect_corner_list.append(((count * 20, 260), ((count+1) * 20, 280)))
            rect_corner_list.append(((count * 20, 340), ((count+1) * 20, 360)))
            flag = True
            count += 2
    block_psi_all, corner_list = setblock(rect_corner_list)
    block_mask = np.where(block_psi_all == 1, True, False)
    mask = np.logical_not(block_mask)
    cm = Compute(mask)
    cc = np.array([cm.psi])
    for i in range(MAX_T):
        f_behind = copy.deepcopy(cm.f)
        g_behind = copy.deepcopy(cm.g)
        stream(cm.f, cm.g)
        halfway_bounceback(corner_list, f_behind, g_behind, cm.f, cm.g)
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
        for j in range(9):
            cm.F[j] = cm.getLarge_F(j)
            cm.feq[j] = cm.getfeq(j)
            cm.geq[j] = cm.getgeq(j)
            cm.f[j][mask] = cm.getF(j)
            cm.g[j][mask] = cm.getG(j)
        if i % mabiki == 0:
            cc = np.append(cc, np.array([cm.psi]), axis=0)
            print("timestep:{}".format(i))
    y = [i for i in range(H)]
    x = [i for i in range(W)]
    fig = plt.figure()
    plt.colorbar(plt.pcolor(x, y, cc[0], cmap='RdBu'))
    ani = animation.FuncAnimation(fig, update, fargs=(x, y, cc), frames=int(len(cc)))
    ani.save('MAX_T{}_Pe{}_M{}_Ca{}_wall{}.mp4'.format(MAX_T, Pe, M, Ca, psi_wall), fps=10)
    plt.figure()
    plt.pcolor(x, y, cm.psi, label="MAX_T:{}, Pe:{}, M:{}, psi_wall:{}".format(MAX_T, Pe, M, psi_wall),cmap='RdBu')
    plt.colorbar()
    plt.legend()
    #plt.show()
    plt.savefig('MAX_T:{}, Pe:{}, M:{}, psi_wall:{}.png'.format(MAX_T, Pe, M, psi_wall))


if __name__ == '__main__':
    np.seterr(all='raise')
    try:
        t1 = time.time()
        main()
        t2 = time.time()
        print((t2 - t1) / 60)
        # upload_file_2 = drive.CreateFile()
        # upload_file_2.SetContentFile('MAX_T{}_Pe{}_M{}_Ca{}_wall{}.mp4'.format(MAX_T, Pe, M, Ca, psi_wall))
        # upload_file_2.Upload()
        # upload_file_1 = drive.CreateFile()
        # upload_file_1.SetContentFile('MAX_T:{}, Pe:{}, M:{}, psi_wall:{}.png'.format(MAX_T, Pe, M, psi_wall))
        # upload_file_1.Upload()
    except Warning as e:
        print(e)
