import numpy as np
import copy

class Bounce_back:
    def __init__(self, H, W):
        self.H = H
        self.W = W

    """https://www.math.nyu.edu/~billbao/report930.pdf"""

    def left_boundary(self, f_behind, g_behind, f, g):
        f[1][:, 0] = copy.deepcopy(f_behind[4][:, 0])
        g[1][:, 0] = copy.deepcopy(g_behind[4][:, 0])
        f[5][:, 0] = copy.deepcopy(f_behind[7][:, 0])
        g[5][:, 0] = copy.deepcopy(g_behind[7][:, 0])
        f[8][:, 0] = copy.deepcopy(f_behind[6][:, 0])
        g[8][:, 0] = copy.deepcopy(g_behind[6][:, 0])

        # mid-grid halfway bounce back

    def halfway_bounceback_rec(self, corner_list, f_behind, g_behind, f, g):
        self.H = f[0].shape[0]
        self.W = f[0].shape[1]
        n_barrier = np.zeros((self.H, self.W), dtype=bool)
        s_barrier = np.zeros((self.H, self.W), dtype=bool)
        e_barrier = np.zeros((self.H, self.W), dtype=bool)
        w_barrier = np.zeros((self.H, self.W), dtype=bool)
        nw_corner = np.zeros((self.H, self.W), dtype=bool)
        ne_corner = np.zeros((self.H, self.W), dtype=bool)
        sw_corner = np.zeros((self.H, self.W), dtype=bool)
        se_corner = np.zeros((self.H, self.W), dtype=bool)
        for cor in corner_list:
            # print(cor['top_left'], cor['top_right'], cor['bottom_right'], cor['bottom_left'])
            n_barrier[cor['top_left'][1] + 1, cor['top_left'][0] + 1:cor['top_right'][0]] = True
            s_barrier[cor['bottom_left'][1] - 1, cor['top_left'][0] + 1:cor['top_right'][0]] = True
            w_barrier[cor['bottom_left'][1] + 1:cor['top_left'][1], cor['top_right'][0] + 1] = True
            e_barrier[cor['bottom_left'][1] + 1:cor['top_left'][1], cor['top_left'][0] - 1] = True
            nw_corner[cor['top_right'][1] + 1, cor['top_right'][0] + 1] = True
            ne_corner[cor['top_left'][1] + 1, cor['top_left'][0] - 1] = True
            sw_corner[cor['bottom_right'][1] - 1, cor['bottom_right'][0] + 1] = True
            se_corner[cor['bottom_left'][1] - 1, cor['bottom_left'][0] - 1] = True
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

    # mid-grid halfway bounce back
    def halfway_bounceback_circle(self, side_list, concave_list, convex_list, f_behind, g_behind, f, g):
        n_barrier = side_list[0]
        s_barrier = side_list[1]
        e_barrier = side_list[2]
        w_barrier = side_list[3]
        nw_vex = convex_list[0]
        ne_vex = convex_list[1]
        sw_vex = convex_list[2]
        se_vex = convex_list[3]
        nw_cave = concave_list[0]
        ne_cave = concave_list[1]
        sw_cave = concave_list[2]
        se_cave = concave_list[3]

        for i in range(9):
            if i == 1:
                f[i][w_barrier] = f_behind[3][w_barrier]
                g[i][w_barrier] = g_behind[3][w_barrier]
                f[i][nw_cave] = f_behind[3][nw_cave]
                g[i][nw_cave] = g_behind[3][nw_cave]
                f[i][sw_cave] = f_behind[3][sw_cave]
                g[i][sw_cave] = g_behind[3][sw_cave]
            elif i == 2:
                f[i][n_barrier] = f_behind[4][n_barrier]
                g[i][n_barrier] = g_behind[4][n_barrier]
                f[i][nw_cave] = f_behind[4][nw_cave]
                g[i][nw_cave] = g_behind[4][nw_cave]
                f[i][ne_cave] = f_behind[4][ne_cave]
                g[i][ne_cave] = g_behind[4][ne_cave]
            elif i == 3:
                f[i][e_barrier] = f_behind[1][e_barrier]
                g[i][e_barrier] = g_behind[1][e_barrier]
                f[i][ne_cave] = f_behind[1][ne_cave]
                g[i][ne_cave] = g_behind[1][ne_cave]
                f[i][se_cave] = f_behind[1][se_cave]
                g[i][se_cave] = g_behind[1][se_cave]
            elif i == 4:
                f[i][s_barrier] = f_behind[2][s_barrier]
                g[i][s_barrier] = g_behind[2][s_barrier]
                f[i][sw_cave] = f_behind[2][sw_cave]
                g[i][sw_cave] = g_behind[2][sw_cave]
                f[i][se_cave] = f_behind[2][se_cave]
                g[i][se_cave] = g_behind[2][se_cave]
            elif i == 5:
                f[i][nw_vex] = f_behind[7][nw_vex]
                g[i][nw_vex] = g_behind[7][nw_vex]
                f[i][nw_cave] = f_behind[7][nw_cave]
                g[i][nw_cave] = g_behind[7][nw_cave]
                f[i][n_barrier] = f_behind[7][n_barrier]
                g[i][n_barrier] = g_behind[7][n_barrier]
                f[i][w_barrier] = f_behind[7][w_barrier]
                g[i][w_barrier] = g_behind[7][w_barrier]
            elif i == 6:
                f[i][ne_vex] = f_behind[8][ne_vex]
                g[i][ne_vex] = g_behind[8][ne_vex]
                f[i][ne_cave] = f_behind[8][ne_cave]
                g[i][ne_cave] = g_behind[8][ne_cave]
                f[i][n_barrier] = f_behind[8][n_barrier]
                g[i][n_barrier] = g_behind[8][n_barrier]
                f[i][e_barrier] = f_behind[8][e_barrier]
                g[i][e_barrier] = g_behind[8][e_barrier]
            elif i == 7:
                f[i][se_vex] = f_behind[5][se_vex]
                g[i][se_vex] = g_behind[5][se_vex]
                f[i][se_cave] = f_behind[5][se_cave]
                g[i][se_cave] = g_behind[5][se_cave]
                f[i][s_barrier] = f_behind[5][s_barrier]
                g[i][s_barrier] = g_behind[5][s_barrier]
                f[i][e_barrier] = f_behind[5][e_barrier]
                g[i][e_barrier] = g_behind[5][e_barrier]
            elif i == 8:
                f[i][sw_vex] = f_behind[6][sw_vex]
                g[i][sw_vex] = g_behind[6][sw_vex]
                f[i][sw_cave] = f_behind[6][sw_cave]
                g[i][sw_cave] = g_behind[6][sw_cave]
                f[i][s_barrier] = f_behind[6][s_barrier]
                g[i][s_barrier] = g_behind[6][s_barrier]
                f[i][w_barrier] = f_behind[6][w_barrier]
                g[i][w_barrier] = g_behind[6][w_barrier]
