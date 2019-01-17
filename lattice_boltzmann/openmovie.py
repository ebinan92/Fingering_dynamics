import matplotlib.pyplot as plt
import pickle
import matplotlib.animation as animation


def update(i, x, y, cc):
    print(i)
    plt.cla()
    plt.pcolor(x, y, cc[i], cmap='RdBu')
    # plt.clim(0,1)
    # plt.legend()


f = open('./f_list_bb_random_MAXT_50000_Ca_0.007_Pe_200.txt', 'rb')
cc = pickle.load(f)
H = 960
W = 980
y = [i for i in range(H)]
x = [i for i in range(W)]
fig = plt.figure()
plt.colorbar(plt.pcolor(x, y, cc[0], cmap='RdBu'))
ani = animation.FuncAnimation(fig, update, fargs=(x, y, cc), frames=int(len(cc)))
ani.save('../movies/bb_random_MAXT_50000_Ca_0.007_Pe_200.mp4', fps=10)