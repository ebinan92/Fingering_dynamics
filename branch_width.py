import numpy as np
import cv2
import pandas as pd
import copy
import time

# 動画は2値化されて8bitに変換されているものを想定しています。
filepath = './movies/201801005_ctab50__nasal100_4ml_Q100_fps30_plain_4times.avi'
filename = filepath.split('/')[-1].split('.')[0]

output = pd.DataFrame()
cap = cv2.VideoCapture(filepath)
DELTA_F = 20  # number of differentiDELTA_F2 = 5al frame　差分をとるフレームの数
DELTA_F2 = 5  # 2つの差分間のフレーム数 DELTA_F % DELTA_F2 = 0 となる必要があります。
noise = 15  # MedianBlur noise must be odd number 大きな値ほどノイズをとれる
th = 100  # threshold　たぶんなんでもいいです
branch_num = 25  # Number of branches to count　大きめにとっておけば問題なし
min_contour = 25  # 枝と認識する最小の周囲長

video_frame = cap.get(cv2.CAP_PROP_FRAME_COUNT)  # number of frames
video_fps = cap.get(cv2.CAP_PROP_FPS)  # fps
video_len_sec = video_frame / video_fps  # video time(s)
video_Width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
video_Height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

index = 0

fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
out = cv2.VideoWriter('{}.m4v'.format(filename), fourcc, 2, (int(video_Width), int(video_Height)))


#  2つのフレームの差分をとって重心と輪郭画像を出力
def getframediff(ahead_frame, behind_frame):
    global index
    diff = cv2.absdiff(ahead_frame, behind_frame)

    # 二値化
    diff[diff < th] = 0
    diff[diff >= th] = 255
    diff = cv2.medianBlur(diff, noise)

    # 輪郭取得
    image, contours, hierarchy = cv2.findContours(diff, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    # 輪郭が大きい順に並べて上位を取得
    contours = sorted(contours, key=len, reverse=True)[0:branch_num]
    center = []
    for cnt in contours:
        mu = cv2.moments(cnt)
        if len(cnt) < min_contour:
            break

        if mu["m00"] == 0 or mu["m00"] == 0:
            continue

        #  重心取得
        x, y = int(mu["m10"] / mu["m00"]), int(mu["m01"] / mu["m00"])
        d = {'x': x, 'y': y, 'index': index, 'flag': True}
        cv2.circle(image, tuple([x, y]), 1, 100, 2, 4)
        center.append(d)
        index += 1

    return image, center


# 枝の太さとその他
def getbranch_width(ahead_center, behind_center, behind_diff):
    branch_index = []
    branch_width = []
    img_draw_list = []
    y_vector = np.array([0, 1])

    for center in behind_center:

        # 重心が一番近いものを同じ枝とみなす
        next_center_enu = min(enumerate(ahead_center),
                              key=lambda x: (x[1]['x'] - center['x']) ** 2 + (x[1]['y'] - center['y']) ** 2)
        # print(next_center_enu[1])
        next_center = next_center_enu[1]
        next_center_index = next_center_enu[0]

        if ahead_center[next_center_index]['flag'] is False:
            continue

        # 枝の番号を継承
        ahead_center[next_center_index]['flag'] = False
        ahead_center[next_center_index]['index'] = copy.deepcopy(center['index'])
        next_vector = [next_center['x'] - center['x'], next_center['y'] - center['y']]

        # 回転させる角度
        si_ta = angle_between(y_vector, next_vector)

        # ここに足す90or回転行列に足すか

        # 画像サイズの取得(横, 縦)
        size = (int(video_Width), int(video_Height))

        # 回転変換行列の算出
        rotation_matrix = cv2.getRotationMatrix2D(tuple([center['x'], center['y']]), si_ta, 1.0)

        # アフィン変換 画像の回転
        img_rot = cv2.warpAffine(behind_diff, rotation_matrix, size, flags=cv2.INTER_CUBIC)

        # 画像の中心座標を重心に移動
        M = np.float32([[1, 0, - center['x'] + video_Width / 2], [0, 1, - center['y'] + video_Height / 2]])
        img_rot_center = cv2.warpAffine(img_rot, M, size)

        # 幅のカウント
        i = 0
        j = 0
        while True:
            flag1 = False
            flag2 = False
            if img_rot_center[int(video_Width / 2) + i][int(video_Height / 2)] != 0:
                i += 1
                flag1 = True

            if img_rot_center[int(video_Width / 2) - j][int(video_Height / 2)] != 0:
                j += 1
                flag2 = True

            if flag1 is False and flag2 is False:
                width = i + j + 1
                break

        draw_dict = {'center_x': center['x'], 'center_y': center['y'], 'min_x': center['x'] - j,
                     'max_x': center['x'] + i, 'index': center['index']}
        img_draw_list.append(draw_dict)
        branch_index.append(center['index'])
        branch_width.append(width)
        # 　新しい行を作成
        s = pd.Series(branch_width, index=branch_index, name=video_len_sec / video_frame * (count - DELTA_F2))

    for x in range(len(ahead_center)):
        ahead_center[x]['flag'] = True

    # draw
    for draw_dict in img_draw_list:
        #  枝の幅描画
        cv2.line(behind_diff, (draw_dict['min_x'], draw_dict['center_y']), (draw_dict['max_x'], draw_dict['center_y']),
                 (150, 150, 150), 3)

        #  中心座標描画
        cv2.circle(behind_diff, tuple([draw_dict['center_x'], draw_dict['center_y']]), 1, 100, 2, 4)

        #  枝の番号描画
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(behind_diff, str(draw_dict['index']), (draw_dict['center_x'], draw_dict['center_y']), font, 1,
                    (255, 255, 255), 1)
    return s, behind_diff, ahead_center


#  2つのベクトルの角度を求める
def angle_between(p1, p2):
    ang1 = np.arctan2(*p1[::-1])
    ang2 = np.arctan2(*p2[::-1])
    return np.rad2deg((ang1 - ang2) % (2 * np.pi))


frame_list = np.array([])
count = 0
f_count = 0
diff_count = 0
while cap.isOpened():
    # 動画の再生が早すぎるときはディレイを入れる
    # time.sleep(0.01)

    ret, frame = cap.read()
    if video_frame <= f_count * DELTA_F2:
        break

    if count % DELTA_F2 == 0:
        print('f_count:{}'.format(f_count))
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        if f_count == 0:
            frame_list = np.array([frame])
            f_count += 1
            count += 1
            continue

        else:
            if DELTA_F > DELTA_F2 * f_count:
                frame_list = np.append(frame_list, np.array([frame]), axis=0)
                f_count += 1
                count += 1
                continue

            else:
                frame_list = np.append(frame_list, np.array([frame]), axis=0)
                f_diff, center = getframediff(frame_list[f_count], frame_list[f_count - int(DELTA_F // DELTA_F2)])

                if diff_count == 0:
                    f_diff1 = f_diff
                    center1 = center
                    diff_count += 1
                    f_count += 1
                    count += 1
                    continue

                if diff_count % 2 == 0:
                    f_diff1 = f_diff
                    center1 = center
                    s, img, center1 = getbranch_width(center1, center2, f_diff2)
                    output = output.append(s)
                    out.write(cv2.cvtColor(img, cv2.COLOR_GRAY2BGR))
                    cv2.imshow("frame2", img)

                if diff_count % 2 == 1:
                    f_diff2 = f_diff
                    center2 = center
                    s, img, center2 = getbranch_width(center2, center1, f_diff1)
                    output = output.append(s)
                    out.write(cv2.cvtColor(img, cv2.COLOR_GRAY2BGR))
                    cv2.imshow("frame2", img)

                diff_count += 1
        f_count += 1
    count += 1
    if cv2.waitKey(1) & 0xFF == 27:
        break

output.to_csv(filename + '.csv')
out.release()
cap.release()
cv2.destroyAllWindows()
