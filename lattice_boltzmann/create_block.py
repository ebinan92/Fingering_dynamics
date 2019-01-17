import numpy as np
import cv2
from scipy.ndimage.morphology import binary_fill_holes
import copy


class Createblock:
    def __init__(self, H, W):
        self.H = H
        self.W = W

    # ブロックの枠組みを指定　頂点で決める
    def getRectangleblock(self, bottom_left, top_right):
        block = np.zeros((self.H, self.W)).astype(np.uint8)
        cv2.rectangle(block, bottom_left, top_right, (1, 0, 0))
        return block

    # 中心座標と半径で決める
    def getCicleblock(self, center, radius):
        block = np.zeros((self.H, self.W)).astype(np.uint8)
        cv2.circle(block, center, radius, (1, 0, 0))
        # # 安定のため頂点を削除
        # block[center[1], center[0] + radius] = 0
        # block[center[1], center[0] - radius] = 0
        # block[center[1] - radius, center[0]] = 0
        # block[center[1] + radius, center[0]] = 0
        # block[center[1], center[0] + radius -1] = 1
        # block[center[1], center[0] - radius +1] = 1
        # block[center[1] - radius + 1, center[0]] = 1
        # block[center[1] + radius -1, center[0]] = 1
        return block

    def getEllipseblock(self, center, axes, angle):
        block = np.zeros((self.H, self.W)).astype(np.uint8)
        cv2.ellipse(block, (center, axes, angle), (1, 0, 0))
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

    # 8つのコーナーと4つの面で場合わけ
    def setCirleblock(self, circle_list):
        convex_list = []
        concave_list = []
        side_list = []
        side_top = []
        side_left = []
        side_right = []
        side_bottom = []
        convex_top_right = []
        convex_bottom_right = []
        convex_top_left = []
        convex_bottom_left = []
        concave_top_right = []
        concave_bottom_right = []
        concave_top_left = []
        concave_bottom_left = []
        block_psi_all = np.zeros((self.H, self.W), dtype=int)
        for circle in circle_list:
            # print("cirlcle")
            x = copy.deepcopy(circle[0][0])
            y = copy.deepcopy(circle[0][1])
            r = copy.deepcopy(circle[1])
            block = self.getCicleblock((x, y), r)
            block_psi = binary_fill_holes(block).astype(int)
            block_psi_all = block_psi + block_psi_all
            side_top_temp = []
            side_left_temp = []
            convex_top_right_temp = []
            concave_top_right_temp = []
            # y軸方向 円の第一象限のみコーナーを検出し, 他は対称性をいかして算出
            jj = r
            for i in range(1, r + 2):
                # print("I:{}".format(i))
                flag = False
                # x軸方向　右端から検出
                for j in range(jj, 0, -1):
                    # print("J:{}".format(j))

                    # 壁にめり込めば次に行く
                    if block[y + i, x + j]:
                        jj = j + 1
                        break
                    # 左, 下, 左下のマスを確認
                    flag_bottom = block[y + i - 1, x + j]
                    flag_left = block[y + i, x - 1 + j]
                    flag_left_bottom = block[y + i - 1, x - 1 + j]

                    # if (x + j) > 390 or (y + i) > 390:
                    # print(x + j, y + i)
                    if flag_left and flag_bottom:
                        concave_top_right_temp.append((x + j, y + i))
                        print("concave:{}".format((x + j, y + i)))
                    elif flag_left:
                        side_left_temp.append((x + j, y + i))
                        print("side_left:{}".format((x + j, y + i)))
                    elif not flag_left and flag_bottom:
                        side_top_temp.append((x + j, y + i))
                        # print(x + j, y + i)
                        print("side_top:{}".format((x + j, y + i)))
                    elif flag_left_bottom:
                        convex_top_right_temp.append((x + j, y + i))
                        print("convex:{}".format((x + j, y + i)))

            for t in side_top_temp:
                # print(t)
                side_top.append((2 * x - t[0], t[1]))
                # print("左:{}".format((2 * x - t[0], t[1])))
                side_bottom.append((t[0], 2 * y - t[1]))
                # print("下:{}".format((t[0], 2 * y - t[1])))
                side_bottom.append((2 * x - t[0], 2 * y - t[1]))
                # print("左下:{}".format((2 * x - t[0], 2 * y - t[1])))
            side_top.append((x, y + r + 1))
            side_top.extend(side_top_temp)
            side_bottom.append((x, y - r - 1))
            # print("sfasdfasdf:{}".format((x, y - r)))
            # print("side_top:{}, side_bottom:{}".format(len(side_top), len(side_bottom)))
            #
            # temp2 = []
            for l in side_left_temp:
                side_right.append((2 * x - l[0], l[1]))
                side_left.append((l[0], 2 * y - l[1]))
                side_right.append((2 * x - l[0], 2 * y - l[1]))
            side_left.extend(side_left_temp)
            side_left.append((x + r + 1, y))
            side_right.append((x - r - 1, y))

            # # print("side_left:{}, side_right:{}".format(len(side_left), len(side_right)))
            #
            for c in concave_top_right_temp:
                concave_top_left.append((2 * x - c[0], c[1]))
                concave_bottom_right.append((c[0], 2 * y - c[1]))
                concave_bottom_left.append((2 * x - c[0], 2 * y - c[1]))
            concave_top_right.extend(concave_top_right_temp)
            # print("concave_top_right:{}, concave_top_left:{}, concave_bottom_right:{}, concave_bottom_left:{}"
            # .format(len(concave_top_right), len(concave_top_left), len(concave_bottom_right),
            #         len(concave_bottom_left)))

            for v in convex_top_right_temp:
                convex_top_left.append((2 * x - v[0], v[1]))
                convex_bottom_right.append((v[0], 2 * y - v[1]))
                convex_bottom_left.append((2 * x - v[0], 2 * y - v[1]))
            convex_top_right.extend(convex_top_right_temp)
            # # print("convex_top_right:{}, convex_top_left:{}, convex_bottom_right:{}, convex_bottom_left:{}"
            # #       .format(len(convex_top_right), len(convex_top_left), len(convex_bottom_right),
            # #               len(convex_bottom_left)))

        # mask化
        top_mask = np.zeros((self.H, self.W), dtype=bool)
        for ori in side_top:
            top_mask[ori[1], ori[0]] = True

        bottom_mask = np.zeros((self.H, self.W), dtype=bool)
        for ori in side_bottom:
            bottom_mask[ori[1], ori[0]] = True

        right_mask = np.zeros((self.H, self.W), dtype=bool)
        for ori in side_right:
            right_mask[ori[1], ori[0]] = True

        left_mask = np.zeros((self.H, self.W), dtype=bool)
        for ori in side_left:
            left_mask[ori[1], ori[0]] = True

        cave_tr_mask = np.zeros((self.H, self.W), dtype=bool)
        for ori in concave_top_right:
            cave_tr_mask[ori[1], ori[0]] = True

        cave_tl_mask = np.zeros((self.H, self.W), dtype=bool)
        for ori in concave_top_left:
            cave_tl_mask[ori[1], ori[0]] = True

        cave_br_mask = np.zeros((self.H, self.W), dtype=bool)
        for ori in concave_bottom_right:
            cave_br_mask[ori[1], ori[0]] = True

        cave_bl_mask = np.zeros((self.H, self.W), dtype=bool)
        for ori in concave_bottom_left:
            cave_bl_mask[ori[1], ori[0]] = True

        vex_tr_mask = np.zeros((self.H, self.W), dtype=bool)
        for ori in convex_top_right:
            vex_tr_mask[ori[1], ori[0]] = True

        vex_tl_mask = np.zeros((self.H, self.W), dtype=bool)
        for ori in convex_top_left:
            vex_tl_mask[ori[1], ori[0]] = True

        vex_br_mask = np.zeros((self.H, self.W), dtype=bool)
        for ori in convex_bottom_right:
            vex_br_mask[ori[1], ori[0]] = True

        vex_bl_mask = np.zeros((self.H, self.W), dtype=bool)
        for ori in convex_bottom_left:
            vex_bl_mask[ori[1], ori[0]] = True

        # 12コのリストを3つにまとめる
        concave_list.append(cave_tr_mask)
        concave_list.append(cave_tl_mask)
        concave_list.append(cave_br_mask)
        concave_list.append(cave_bl_mask)
        side_list.append(top_mask)
        side_list.append(bottom_mask)
        side_list.append(right_mask)
        side_list.append(left_mask)
        convex_list.append(vex_tr_mask)
        convex_list.append(vex_tl_mask)
        convex_list.append(vex_br_mask)
        convex_list.append(vex_bl_mask)

        return block_psi_all, side_list, concave_list, convex_list

    def setEllipseblock(self, ellipse_list):
        #eclipse_list = dict{'c_x':, 'c_y': , 'r_x': ,'r_y':}
        convex_list = []
        concave_list = []
        side_list = []
        side_top = []
        side_left = []
        side_right = []
        side_bottom = []
        convex_top_right = []
        convex_bottom_right = []
        convex_top_left = []
        convex_bottom_left = []
        concave_top_right = []
        concave_bottom_right = []
        concave_top_left = []
        concave_bottom_left = []
        block_psi_all = np.zeros((self.H, self.W), dtype=int)
        for circle in ellipse_list:
            # print("cirlcle")
            x = circle['c_x']
            y = circle['c_y']
            r_x = int(circle['r_x'] / 2)
            r_y = int(circle['r_y'] / 2)
            block = self.getEllipseblock((x, y), (int(r_x * 2), int(r_y * 2)), circle['angle'])
            block_psi = binary_fill_holes(block).astype(int)
            block_psi_all = block_psi + block_psi_all
            side_top_temp = []
            side_left_temp = []
            convex_top_right_temp = []
            concave_top_right_temp = []
            # y軸方向 円の第一象限のみコーナーを検出し, 他は対称性をいかして算出
            jj = r_x + 1
            for i in range(1, r_y + 2):
                # print("I:{}".format(i))
                #  x軸方向　右端から検出
                for j in range(jj, 0, -1):
                    # print("J:{}".format(j))

                    # 壁にめり込めば次に行く
                    if block[y + i, x + j]:
                        jj = j + 1
                        break
                    # 左, 下, 左下のマスを確認
                    flag_bottom = block[y + i - 1, x + j]
                    flag_left = block[y + i, x - 1 + j]
                    flag_left_bottom = block[y + i - 1, x - 1 + j]

                    # if (x + j) > 390 or (y + i) > 390:
                    # print(x + j, y + i)
                    if flag_left and flag_bottom:
                        concave_top_right_temp.append((x + j, y + i))
                        print("concave:{}".format((x + j, y + i)))
                    elif flag_left:
                        side_left_temp.append((x + j, y + i))
                        print("side_left:{}".format((x + j, y + i)))
                    elif not flag_left and flag_bottom:
                        side_top_temp.append((x + j, y + i))
                        # print(x + j, y + i)
                        print("side_top:{}".format((x + j, y + i)))
                    elif flag_left_bottom:
                        convex_top_right_temp.append((x + j, y + i))
                        print("convex:{}".format((x + j, y + i)))

            for t in side_top_temp:
                # print(t)
                side_top.append((2 * x - t[0], t[1]))
                # print("左:{}".format((2 * x - t[0], t[1])))
                side_bottom.append((t[0], 2 * y - t[1]))
                # print("下:{}".format((t[0], 2 * y - t[1])))
                side_bottom.append((2 * x - t[0], 2 * y - t[1]))
                # print("左下:{}".format((2 * x - t[0], 2 * y - t[1])))
            side_top.append((x, y + r_y + 1))
            side_top.extend(side_top_temp)
            side_bottom.append((x, y - r_y - 1))
            # print("sfasdfasdf:{}".format((x, y - r)))
            # print("side_top:{}, side_bottom:{}".format(len(side_top), len(side_bottom)))
            #
            # temp2 = []
            for l in side_left_temp:
                side_right.append((2 * x - l[0], l[1]))
                side_left.append((l[0], 2 * y - l[1]))
                side_right.append((2 * x - l[0], 2 * y - l[1]))
            side_left.extend(side_left_temp)
            side_left.append((x + r_x + 1, y))
            side_right.append((x - r_x - 1, y))

            # # print("side_left:{}, side_right:{}".format(len(side_left), len(side_right)))
            #
            for c in concave_top_right_temp:
                concave_top_left.append((2 * x - c[0], c[1]))
                concave_bottom_right.append((c[0], 2 * y - c[1]))
                concave_bottom_left.append((2 * x - c[0], 2 * y - c[1]))
            concave_top_right.extend(concave_top_right_temp)
            # print("concave_top_right:{}, concave_top_left:{}, concave_bottom_right:{}, concave_bottom_left:{}"
            # .format(len(concave_top_right), len(concave_top_left), len(concave_bottom_right),
            #         len(concave_bottom_left)))

            for v in convex_top_right_temp:
                convex_top_left.append((2 * x - v[0], v[1]))
                convex_bottom_right.append((v[0], 2 * y - v[1]))
                convex_bottom_left.append((2 * x - v[0], 2 * y - v[1]))
            convex_top_right.extend(convex_top_right_temp)
            # # print("convex_top_right:{}, convex_top_left:{}, convex_bottom_right:{}, convex_bottom_left:{}"
            # #       .format(len(convex_top_right), len(convex_top_left), len(convex_bottom_right),
            # #               len(convex_bottom_left)))

        # mask化
        top_mask = np.zeros((self.H, self.W), dtype=bool)
        for ori in side_top:
            top_mask[ori[1], ori[0]] = True

        bottom_mask = np.zeros((self.H, self.W), dtype=bool)
        for ori in side_bottom:
            bottom_mask[ori[1], ori[0]] = True

        right_mask = np.zeros((self.H, self.W), dtype=bool)
        for ori in side_right:
            right_mask[ori[1], ori[0]] = True

        left_mask = np.zeros((self.H, self.W), dtype=bool)
        for ori in side_left:
            left_mask[ori[1], ori[0]] = True

        cave_tr_mask = np.zeros((self.H, self.W), dtype=bool)
        for ori in concave_top_right:
            cave_tr_mask[ori[1], ori[0]] = True

        cave_tl_mask = np.zeros((self.H, self.W), dtype=bool)
        for ori in concave_top_left:
            cave_tl_mask[ori[1], ori[0]] = True

        cave_br_mask = np.zeros((self.H, self.W), dtype=bool)
        for ori in concave_bottom_right:
            cave_br_mask[ori[1], ori[0]] = True

        cave_bl_mask = np.zeros((self.H, self.W), dtype=bool)
        for ori in concave_bottom_left:
            cave_bl_mask[ori[1], ori[0]] = True

        vex_tr_mask = np.zeros((self.H, self.W), dtype=bool)
        for ori in convex_top_right:
            vex_tr_mask[ori[1], ori[0]] = True

        vex_tl_mask = np.zeros((self.H, self.W), dtype=bool)
        for ori in convex_top_left:
            vex_tl_mask[ori[1], ori[0]] = True

        vex_br_mask = np.zeros((self.H, self.W), dtype=bool)
        for ori in convex_bottom_right:
            vex_br_mask[ori[1], ori[0]] = True

        vex_bl_mask = np.zeros((self.H, self.W), dtype=bool)
        for ori in convex_bottom_left:
            vex_bl_mask[ori[1], ori[0]] = True

        # 12コのリストを3つにまとめる
        concave_list.append(cave_tr_mask)
        concave_list.append(cave_tl_mask)
        concave_list.append(cave_br_mask)
        concave_list.append(cave_bl_mask)
        side_list.append(top_mask)
        side_list.append(bottom_mask)
        side_list.append(right_mask)
        side_list.append(left_mask)
        convex_list.append(vex_tr_mask)
        convex_list.append(vex_tl_mask)
        convex_list.append(vex_br_mask)
        convex_list.append(vex_bl_mask)

        return block_psi_all, side_list, concave_list, convex_list


    def setblock(self, rect_corner_list):
        corner_list = []
        # 0 or 1の配列　1がblockのあるところを示す
        block_psi_all = np.zeros((self.H, self.W), dtype=int)
        for rect in rect_corner_list:
            block = self.getRectangleblock(rect[0], rect[1])
            block_psi = binary_fill_holes(block).astype(int)
            corner = self.getCorner(block)
            corner_list.append(corner)
            # print(block_psi.shape)
            block_psi_all = block_psi_all + block_psi
        # print(block_psi_all.max(), "max")
        return block_psi_all, corner_list
