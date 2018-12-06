import numpy as np
import matplotlib.pyplot as plt

N = 100 #number of points
M = 50
L = 2 * np.pi #interval of data



'''one dimensional'''
def onedim():
    x = np.arange(0.0, L, L/float(N))
    y = np.sin(x) + 0.05 * np.random.random(size=x.shape)
    dy_analytical = np.cos(x)

    a = np.asarray(range(0, int(N / 2)))*2 * np.pi / L
    b = np.asarray(range(int(-N / 2 + 1),0))*2 * np.pi / L
    k = np.append(a,np.append(0,b))
    fd = np.real(np.fft.ifft(1.0j * k * np.fft.fft(y)))
    plt.plot(x, y, label='sin')
    plt.plot(x,dy_analytical,label='cos')
    plt.plot(x,fd,label='fft der cos')
    plt.legend(loc='lower left')
    plt.show()


'''two dimensional'''
def twodim():
    x = np.arange(0, L, L/float(N)) #this does not include the endpoint
    y = np.arange(0, L, L/float(M))
    X,Y = np.meshgrid(x,y)

    z = np.sin(X) + np.cos(Y)
    dz = np.cos(X) - np.sin(Y)
    ddz = -np.sin(X) - np.cos(Y)

    a = np.asarray(range(0, int(N / 2)))*2 * np.pi / L
    b = np.asarray(range(int(-N / 2 + 1),0))*2 * np.pi / L
    km = np.append(a,np.append(0,b))
    c = np.asarray(range(0, int(M / 2)))*2 * np.pi / L
    d = np.asarray(range(int(-M / 2 + 1),0))*2 * np.pi / L
    kn = np.append(c,np.append(0,d))
    # #km = np.asarray(range(N))* 2 * np.pi / L
    # #kn = np.asarray(range(N))* 2 * np.pi / L

    km_km = np.tile(km, (M, 1))
    kn_kn = np.tile(kn, (N, 1)).T
    print(km.shape,kn.shape)

    # fft_z = np.fft.fft2(z)
    # z_x = np.fft.ifft2(1.0j*fft_z*km_km).real
    # z_y = np.fft.ifft2(1.0j*fft_z*kn_kn).real
    # zz = z_x+z_y

    fft_z = np.fft.fft2(z)
    kmkn2 = km_km**2 + kn_kn**2
    # z_z_x = np.fft.ifft2(-fft_ddz*(km_km**2)).real
    # z_z_y = np.fft.ifft2(-fft_ddz*(kn_kn**2)).real
    ddzbyfft = np.real(np.fft.ifft2(-kmkn2*fft_z))

    plt.figure()
    plt.subplot(1,2,1)
    plt.pcolor(x, y, ddz,label="ddz = -sinx -cosy")
    plt.colorbar()
    plt.legend()

    plt.subplot(1,2,2)
    plt.pcolor(x,y,ddzbyfft,label = "ddz = ifft2(z) ")
    plt.colorbar()
    plt.legend()
    plt.show()

if __name__ == '__main__':
    onedim()
    #twodim()
