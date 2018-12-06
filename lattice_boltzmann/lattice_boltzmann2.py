import numpy as np
import sympy
import matplotlib.pyplot as plt
from scipy import optimize
from scipy.ndimage.filters import convolve
import math
import time

H = 300  # lattice dimensions
W = 300
DELTA_T = 5.0 * 10 ** (-3)  # time step
DELTA_X = 1.0 * 10 ** (-5)  # lattice spacing
MAX_T = 1
sigma = 0.045  # interfacial tension
Pe = 100.0  # peclet number
u0 = 0.0 # initial velocity
rho = 1000.0
n_non = 1.0  # power-law parameter
Eta_n = 0.023 # Eta newtonian
M = 20.0  # Eta non_newtonian / Eta newtonian
Theta = np.pi / 4  # contact angle
v1 = 1.0 * 10.0 ** (-6)  # kinetic viscosity of newtonian
c = DELTA_X / DELTA_T  # particle streaming speed
cs = c / np.sqrt(3.0)
xi = 2 * DELTA_X
kappa = 3 / 4 * sigma * xi
a = - 2 * kappa / (xi ** 2)
tau = 1 / (3 - np.sqrt(3))
#gamma = u0 * W / (-a * Pe) / ((tau - 0.5) * DELTA_T)
#print(gamma)
gamma = 0.00 * 10.0 ** (-10)
#gamma = 0.0
b = np.array([i for i in range(1,10)]).reshape(3,3)
print(np.roll(b, 1, axis=1))

class Compute:
    def __init__(self):
        self.e = np.array([np.array([0.0, 0.0]) for i in range(9)])  # lattice velocity
        self.w = np.array([0.0 for i in range(9)])
        self.rho = np.ones((H, W), dtype=float) * rho # macroscopic density
        self.ux = np.zeros((H, W), dtype=float)
        self.uy = np.zeros((H, W), dtype=float)
        self.p = np.full((H, W), self.rho * cs ** 2)
        self.f = np.array([np.zeros((H + 2, W), dtype=float) for i in range(9)])
        self.g = np.array([np.zeros((H + 2, W), dtype=float) for i in range(9)])
        self.feq = np.array([np.zeros((H + 2, W), dtype=float) for i in range(9)])
        self.geq = np.array([np.zeros((H + 2, W), dtype=float) for i in range(9)])
        self.psi = np.full((H + 2, W), -1.0)
        circle = create_circle(H + 2, 40)
        circle = circle[1:-1].T
        circle[0, :] = 0
        psi_wall = self.getPsi_wall()
        psi_wall = 0.5
        print("psi_wall:{}".format(psi_wall))
        self.psi[H + 1, :] = psi_wall
        self.psi[0, :] = psi_wall
        self.psi[circle] = 1.0
        #print(self.psi[0, :].mean())
        self.mu = self.getMu()
        self.F = np.array([np.zeros((H + 2, W), dtype=float) for i in range(9)]).astype(float)
        self.mix_tau = np.zeros((H+2, W))
        for i in range(9):
            if i == 0:
                self.e[i] = np.array([0.0, 0.0])
                self.w[i] = 4.0 / 9.0
                self.A0 = self.getA0()
                self.feq[i] = self.getfeq(i)
                self.f[i] = self.getfeq(i)
                self.B0 = self.getB0()
                self.geq[i] = self.getgeq(i)
                self.g[i] = self.getgeq(i)
            elif i < 5:
                self.e[i] = np.array([np.cos((i - 1) * np.pi / 2), np.sin((i - 1) * np.pi / 2)]) * c
                self.w[i] = 1.0 / 9.0
                self.A1_8 = self.getA1_8()
                self.f[i] = self.getfeq(i)
                self.feq[i] = self.getfeq(i)
                self.B1_8 = self.getB1_8()
                self.geq[i] = self.getgeq(i)
                self.g[i] = self.getgeq(i)
            if i >= 5:
                self.e[i] = np.array([np.cos((i - 5) * np.pi / 2 + np.pi / 4),
                                      np.sin(np.pi * ((i - 5.0)/ 2 + 1 / 4))]) * c * np.sqrt(2)
                self.w[i] = 1.0 / 36.0
                self.f[i] = self.getfeq(i)
                self.feq[i] = self.getfeq(i)
                self.geq[i] = self.getgeq(i)
                self.g[i] = self.getgeq(i)
            print(self.e[i])

    # def updateP(self):
    #     self.p = np.sum(self.f, axis=0) * (cs ** 2)
    #     print("p:{}".format(self.p.mean()))

    def updateP(self):
        self.p = self.rho / 3.0 - a/2.0*self.psi**2 - 3.0/4.0*a*self.psi**4-kappa*self.psi*self.getNabla_psi2()-kappa/2*(self.getNabla_psix()+self.getNabla_psiy())**2
        print("p:{}".format(self.p.mean()))

    def updateU(self):
        temp1 = np.zeros((H + 2, W), dtype=float)
        temp2 = np.zeros((H + 2, W), dtype=float)
        for i in range(9):
            temp1 += self.f[i] * self.e[i][0]
            temp2 += self.f[i] * self.e[i][1]
        self.ux[1:-1] = ((temp1 + self.mu * self.getNabla_psix() * DELTA_T / 2)/self.rho)[1:-1]
        self.uy[1:-1] = ((temp2 + self.mu * self.getNabla_psiy() * DELTA_T / 2)/self.rho)[1:-1]
        print("ux:{}, uy:{}".format(self.ux.mean(), self.uy.mean()))
    #     print("p:{}".format(self.p.mean()))

    def getMu(self):
        mu = a * self.psi - a * (self.psi ** 3) - kappa * self.getNabla_psi2()
        return mu

    def updateMu(self):
        self.mu = a * self.psi - a * (self.psi ** 3) - kappa * self.getNabla_psi2()
        print("mu:{}".format(self.mu.mean()))

    def updateRho(self):
        self.rho = np.sum(self.f, axis=0)  # macroscopic density
        self.rho[0, :] = rho
        self.rho[H+1, :] = rho
        print("rho:{}".format(self.rho.mean()))

    def getA0(self):
        a0 = (self.rho - 3.0 * (1.0 - self.w[0]) * self.p / c ** 2) / self.w[0]
        return a0

    def getA1_8(self):
        a1_8 = 3 * self.p / c ** 2
        return a1_8

    def getB0(self):
        b0 = (self.psi - 3.0 * (1.0 - self.w[0]) * gamma * self.mu / (c ** 2)) / self.w[0]
        return b0

    def getB1_8(self):
        b1_8 = 3 * gamma * self.mu / c ** 2
        #print("b1_8:{}".format(b1_8.mean()))
        return b1_8

    def getfeq(self, n):
        if n == 0:
            feq = self.w[n] * (self.getA0() + self.rho * (
                    3 * (self.e[n][0] * self.ux + self.e[n][1] * self.uy) / (c ** 2) + 9 * (
                    self.e[n][0] * self.ux + self.e[n][1] * self.uy) ** 2 / (
                            2 * (c ** 4)) - 3 * (self.ux ** 2 + self.uy ** 2) / (2 * c ** 2)))
        else:
            feq = self.w[n] * (self.getA1_8() + self.rho * (
                    3 * (self.e[n][0]*self.ux + self.e[n][1]*self.uy) / c ** 2 + 9 * (self.e[n][0]*self.ux + self.e[n][1]*self.uy) ** 2 / (
                    2 * c ** 4) - 3 * (self.ux**2 + self.uy**2) / (2 * c ** 2)))
        #print("feq{}:{}".format(n, feq.mean()))
        return feq

    def getgeq(self, n):
        if n == 0:
            geq = self.w[n] * (self.getB0() + self.psi * (
                    3 * (self.e[n][0] * self.ux + self.e[n][1] * self.uy) / (c ** 2) + 9 * (
                    self.e[n][0] * self.ux + self.e[n][1] * self.uy) ** 2 / (
                            2 * (c ** 4)) - 3 * (self.ux ** 2 + self.uy ** 2) / (2 * (c ** 2))))
        else:
            geq = self.w[n] * (self.getB1_8() + self.psi * (
                    3 * (self.e[n][0]*self.ux + self.e[n][1]*self.uy) / c ** 2 + 9 * (self.e[n][0]*self.ux + self.e[n][1]*self.uy) ** 2 / (
                    2 * c ** 4) - 3 * (self.ux**2 + self.uy**2) / (2 * c ** 2)))
        #print("geq{}:{}".format(n, geq.mean()))
        return geq

    def getLarge_F(self, n):
        f = DELTA_T * self.w[n] * (1 - DELTA_T / (2 * self.mix_tau)) * ((
                                                                                (self.e[n][0] - self.ux) / cs ** 2 + self.e[n][0] * (self.e[n][0]*self.ux + self.e[n][1]*self.uy) / (cs ** 4)) *
                                                                        self.mu * self.getNabla_psix() + ((self.e[n][1] - self.uy) / cs ** 2
                                                                                                          + self.e[n][1] * (self.e[n][0]*self.ux + self.e[n][1]*self.uy) / (cs ** 4)) * self.mu * self.getNabla_psiy())
        return f

    # def getLarge_F(self, n):
    #     f = DELTA_T * self.w[n] * (1 - DELTA_T / (2 * self.mix_tau)) * ((
    #         (self.e[n][0] - self.ux) / cs ** 2 + self.e[n][0] * (self.e[n][0]*self.ux + self.e[n][1]*self.uy) / (cs ** 4)) *
    #         self.mu * self.getNabla_psix() + ((self.e[n][1] - self.uy) / cs ** 2
    #          + self.e[n][1] * (self.e[n][0]*self.ux + self.e[n][1]*self.uy) / (cs ** 4)) * self.mu * self.getNabla_psiy())
    #     return f

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
        v2 = v1 * 5
        mix_v = np.divide(2 * v1 * v2, (v1 * (1.0 + self.psi) + v2 * (1.0 - self.psi)))
        #mix_v = 2 * v1 * v2 / (v2 * (1.0 - self.psi) + v1 * (1.0 - self.psi))
        mix_tau = 3 * mix_v / (c ** 2) + 0.5 * DELTA_T
        #mix_tau = np.ones((H+2, W))
        #mix_tau = np.full((H+2, W), 1.0)
        self.mix_tau[1:-1] = mix_tau[1:-1]
        print("mix_tau:{}".format(mix_tau.mean()))
        #print(t2 - t1)
        return mix_tau

    # def updateMix_tau(self):
    #     tau_psi = ((1.0 - self.psi)*0.16 + (1.0 + self.psi)*0.16*0.25)*0.50
    #     mix_tau = tau_psi / (self.rho * (cs ** 2) * DELTA_T) + 0.50
    # #     #mix_tau = np.ones((H+2, W))
    #
    # #     mix_tau = np.full((H+2, W), 1.0)
    #     self.mix_tau = mix_tau
    #     print("mix_tau:{}".format(mix_tau.mean()))
    # #     #print(t2 - t1)
    #     return mix_tau

    def updatePsi(self):
        self.psi[1:-1, :] = np.sum(self.g, axis=0)[1:-1, :]
        print("psi:{}".format(self.psi.mean()))

    # ∇φ
    def getNabla_psix(self):
        f = np.zeros((H+2, W), dtype=float)
        for i in range(9):
            f += np.roll(self.psi, -int(self.e[i][0]), axis=1) * self.e[i][0]
        return f / (6 * DELTA_X)

    def getNabla_psiy(self):
        f = np.zeros((H+2, W), dtype=float)
        for i in range(9):
            f += np.roll(self.psi, int(self.e[i][1]), axis=0) * self.e[i][1]
        return f / (6 * DELTA_X)

    # ∇^2φ
    def getNabla_psi2(self):
        fxx = np.zeros((H+2, W), dtype=float)
        fyy = np.zeros((H+2, W), dtype=float)
        for i in range(9):
            fxx += np.roll(self.psi, -int(self.e[i][0]), axis=1)
            fyy += np.roll(self.psi, int(self.e[i][1]), axis=0)
        return (fxx + fyy - 2 * 8 * self.psi) / (3 * DELTA_X)

    def getPsi_wall(self):
        psi_wall = sympy.symbols('psi')
        out = sympy.solve(1 / 2 * psi_wall ** 3 - 3 / 2 * psi_wall + np.cos(Theta))[0].as_real_imag()
        return out[0]

    def updateF(self):
        for i in range(9):
            self.f[i] = self.f[i] - DELTA_T / self.mix_tau * (self.f[i] - self.feq[i]) + self.F[i]

    def updateG(self):
        for i in range(9):
            self.g[i] = self.g[i] - 1 / tau * (self.g[i] - self.geq[i])
            #print("g{}:{}".format(i, self.g[i].mean()))


def create_circle(n, r):
    y, x = np.ogrid[-int((H + 2) / 2): n - int((H + 2) / 2), -r: n - r]
    mask = x ** 2 + y ** 2 <= r ** 2
    return mask


def stream(f, g):

    for i in range(9):
        if i == 0:
            continue
        elif i == 1:
            f[i] = np.roll(f[i], 1, axis=1)
            g[i] = np.roll(g[i], 1, axis=1)
        elif i == 2:
            f[i] = np.roll(f[i], -1, axis=0)
            g[i] = np.roll(g[i], -1, axis=0)
        elif i == 3:
            f[i] = np.roll(f[i], -1, axis=1)
            g[i] = np.roll(g[i], -1, axis=1)
        elif i == 4:
            f[i] = np.roll(f[i], 1, axis=0)
            g[i] = np.roll(g[i], 1, axis=0)
        elif i == 5:
            f[i] = np.roll(np.roll(f[i], 1, axis=1), -1, axis=0)
            g[i] = np.roll(np.roll(g[i], 1, axis=1), -1, axis=0)
        elif i == 6:
            f[i] = np.roll(np.roll(f[i], -1, axis=1), -1, axis=0)
            g[i] = np.roll(np.roll(g[i], -1, axis=1), -1, axis=0)
        elif i == 7:
            f[i] = np.roll(np.roll(f[i], -1, axis=1), 1, axis=0)
            g[i] = np.roll(np.roll(g[i], -1, axis=1), 1, axis=0)
        elif i == 8:
            f[i] = np.roll(np.roll(f[i], 1, axis=1), 1, axis=0)
            g[i] = np.roll(np.roll(g[i], 1, axis=1), 1, axis=0)

    # Zou-He and Open boundary
    # for i in range(9):
    #     if i == 0:
    #  f[i][H, :] = 0.0
    #         g[i][H, :] = 0.0       continue
    #     elif i == 1:
    #         f[i][1:-1, 0] = f[3][1:-1, 0] + 2/3*rho * v1
    #         g[i][1:-1, 0] = g[3][1:-1, 0] + 2/3*rho * v1
    #     elif i == 5:
    #         f[i][1:-1, 0] = f[7][1:-1, 0] - 1/2*(f[2][1:-1, 0] - f[4][1:-1, 0]) + 1/6*rho * u0 + 1/2*rho*v1
    #         g[i][1:-1, 0] = g[7][1:-1, 0] - 1/2*(g[2][1:-1, 0] - g[4][1:-1, 0]) + 1/6*rho * u0 + 1/2*rho*v1
    #     elif i == 8:
    #         f[i][1:-1, 0] = f[6][1:-1, 0] + 1/2*(f[2][1:-1, 0] - f[4][1:-1, 0]) + 1/6*rho * u0 - 1/2*rho*v1
    #         g[i][1:-1, 0] = g[6][1:-1, 0] + 1/2*(g[2][1:-1, 0] - g[4][1:-1, 0]) + 1/6*rho * u0 - 1/3*rho*v1

def halfway_bounceback(f_behind, g_behind, f, g):
    for i in range(9):
        # f[i][0, :] = 0.0
        # g[i][0, :] = 0.0
        # f[i][H+1, :] = 0.0
        # g[i][H+1, :] = 0.0
        if i == 0:
            continue
        elif i == 1:
            f[i][1, :] = 0.0
            g[i][1, :] = 0.0
            f[i][H, :] = 0.0
            g[i][H, :] = 0.0
            continue
        elif i == 2:
            f[i][1, :] = f_behind[4][1, :]
            g[i][1, :] = g_behind[4][1, :]
            f[i][H, :] = 0
            g[i][H, :] = 0
        elif i == 3:
            f[i][1, :] = 0.0
            g[i][1, :] = 0.0
            f[i][H, :] = 0.0
            g[i][H, :] = 0.0
            continue
        elif i == 4:
            f[i][1, :] = 0.0
            g[i][1, :] = 0.0
            f[i][H, :] = f_behind[2][H, :]
            g[i][H, :] = g_behind[2][H, :]
        elif i == 5:
            f[i][1, :] = f_behind[7][1, :]
            g[i][1, :] = g_behind[7][1, :]
            f[i][H, :] = 0.0
            g[i][H, :] = 0.0
        elif i == 6:
            f[i][1, :] = f_behind[8][1, :]
            g[i][1, :] = g_behind[8][1, :]
            f[i][H, :] = 0.0
            g[i][H, :] = 0.0
        elif i == 7:
            f[i][1, :] = 0.0
            g[i][1, :] = 0.0
            f[i][H, :] = f_behind[5][H, :]
            g[i][H, :] = g_behind[5][H, :]
        elif i == 8:
            f[i][1, :] = 0.0
            g[i][1, :] = 0.0
            f[i][H, :] = f_behind[6][H, :]
            g[i][H, :] = g_behind[6][H, :]

def reset_wall(f, g):
    for i in range(9):
        f[i][0, :] = 0.0
        g[i][0, :] = 0.0
        f[i][H+1, :] = 0.0
        g[i][H+1, :] = 0.0



def main():
    cm = Compute()
    for i in range(MAX_T):
        #reset_wall(cm.f, cm.g)
        f_behind = cm.f.copy()
        g_behind = cm.g.copy()
        stream(cm.f, cm.g)
        halfway_bounceback(f_behind, g_behind, cm.f, cm.g)
        #reset_wall(cm.f, cm.g)
        cm.updateRho()
        cm.updateP()
        cm.updatePsi()
        cm.updateMu()
        cm.updateU()
        cm.updateMix_tau()
        for j in range(9):
            cm.F[j] = cm.getLarge_F(j)
            cm.feq[j] = cm.getfeq(j)
            cm.geq[j] = cm.getgeq(j)
        cm.updateF()
        cm.updateG()

        print("timestep:{}".format(i))


    y = [i for i in range(H + 2)]
    x = [i for i in range(W)]
    plt.figure()
    plt.pcolor(x, y, cm.g[1])
    plt.colorbar()
    plt.show()


if __name__ == '__main__':
    main()
