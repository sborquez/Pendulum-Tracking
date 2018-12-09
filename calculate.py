import numpy as np
from scipy.fftpack import fft, fftfreq
from scipy.constants import g
import cv2
import seaborn as sns; sns.set()
import matplotlib.pyplot as plt

def get_frequency(x, min, max, interval):
    sec = interval
    n = len(x)
    dt = sec/n

    t = np.arange(0, sec, dt)

    #centrate
    A = (max - min)*0.5
    xx =  x + A - max

    #P = t[81] - t [35]
    #F = 1/P
    fx = fft(xx)
    tf = fftfreq(n, dt)
    return np.abs(2*tf[np.argmax(fx)])


def calculate_center(points, range_x, range_y, fps, save=False):
    p = np.array(points)
    x =  p[:,0,0,0]
    y =  p[:,0,0,1]
    seconds = len(x)/fps

    fx = get_frequency(x, range_x[0], range_x[1], seconds)
    #fy = get_frequency(y, range_y[0], range_y[1], seconds)
    print(f"FREC: {fx}")


    l = g*(1/2*np.pi*fx)**2
    print(f"LEN: {l}[m]")

    #ly = g*(1/2*np.pi*fy)**2
    #l = 0.5*(lx + ly)

    ox = int(0.5*(range_x[1] + range_x[0]))
    oy = int(range_y[1] - l)

    if save:
        np.savetxt(f"x{len(x)}.txt", x)
        np.savetxt(f"y{len(y)}.txt", y)
    return l, ox, oy, fx, None

def plot(points, range_x, range_y, fps):

    p = np.array(points)
    x =  p[:,0,0,0]
    y =  p[:,0,0,1]
    seconds = len(x)/fps

    vx = np.diff(x)
    vy = np.diff(y)
    t = np.arange(0, seconds, 1/fps)

    plt.figure(figsize=(40, 10))

    # x[pixel] vs time[second]
    plt.subplot(221)
    plt.plot(t, x, "o-")
    plt.ylabel("x(t) [pix]")
    plt.xlabel("t [sec]")
    plt.title('x[pixel] vs time[second]')
    plt.grid(True)

    # y[pixel] vs time[second]
    plt.subplot(222)
    plt.plot(t, y, "ro-")
    plt.ylabel("y(t) [pix]")
    plt.xlabel("t [sec]")
    plt.title('y[pixel] vs time[second]')
    plt.grid(True)

    # vx[pixel/sec] vs time[second]
    plt.subplot(223)
    plt.plot(t[1:], vx, "o-")
    #plt.bar(freq_x, abs(spectrum_x))
    plt.ylabel("vx(t) [pix/sec]")
    plt.xlabel("t [sec]")
    plt.title('vx[pixel/sec] vs time[second]')
    plt.grid(True)

    # vy[pixel/sec] vs time[second]
    plt.subplot(224)
    plt.plot(t[1:], vy, "ro-")
    #plt.bar(freq_y, abs(spectrum_y))
    plt.ylabel("vy(t) [pix/sec]")
    plt.xlabel("t [sec]")
    plt.title('vy[pixel/sec] vs time[second]')
    plt.grid(True)

    plt.show()