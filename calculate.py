import numpy as np
from scipy.fftpack import fft, fftfreq
from scipy.constants import g
import cv2
import seaborn as sns; sns.set()
import matplotlib.pyplot as plt

def get_points(gray):
    circles = cv2.HoughCircles(gray,cv2.HOUGH_GRADIENT,1,20,
                            param1=50,param2=30,minRadius=0,maxRadius=55)
    points = circles[0,:,0:2].reshape(circles.shape[1], 1, 2).astype(np.float32)
    #print(points.shape)
    #for i in circles[0,:]:
        # draw the outer circle
    #    cv2.circle(gray,(i[0],i[1]),i[2],(0,255,0),2)
        # draw the center of the circle
    #    cv2.circle(gray,(i[0],i[1]),2,(0,0,255),3)

    #cv2.imshow('detected circles',gray)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
    return points

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


def calculate_frec(points, min_x, max_x, min_y, max_y, fps):
    p = np.array(points)
    fx = []
    fy = []
    seconds = p.shape[1]/fps
    for i in range(p.shape[1]):
        x =  p[:,i,0,0]
        y =  p[:,i,0,1]
        fxi = get_frequency(x, min_x[i], max_x[i], seconds)
        fyi = get_frequency(y, min_y[i], max_y[i], seconds)
        fx.append(fxi)
        fy.append(fyi)
        print(f"FREC: {fxi}, {fyi}")
    return fx, fy

def plot(points, range_x, range_y, fps):

    p = np.array(points)
    #print(p.shape)
    seconds = p.shape[0]/fps
    t = np.arange(0, seconds, 1/fps)

    plt.figure(figsize=(40, 10))

    # x[pixel] vs time[second]
    plt.subplot(221)
    for i in range(p.shape[1]):
        x =  p[:,i,0,0]
        plt.plot(t, x, "o-")
    plt.ylabel("x(t) [pix]")
    plt.xlabel("t [sec]")
    plt.title('x[pixel] vs time[second]')
    plt.grid(True)

    # y[pixel] vs time[second]
    plt.subplot(222)
    for i in range(p.shape[1]):
        y =  p[:,i,0,1]
        plt.plot(t, y, "o-")
    plt.ylabel("y(t) [pix]")
    plt.xlabel("t [sec]")
    plt.title('y[pixel] vs time[second]')
    plt.grid(True)



    vx = np.diff(x)
    vy = np.diff(y)
    # vx[pixel/sec] vs time[second]
    plt.subplot(223)
    for i in range(p.shape[1]):
        x =  p[:,i,0,0]
        vx = np.diff(x)
        plt.plot(t[1:], vx, "o-")
    #plt.bar(freq_x, abs(spectrum_x))
    plt.ylabel("vx(t) [pix/sec]")
    plt.xlabel("t [sec]")
    plt.title('vx[pixel/sec] vs time[second]')
    plt.grid(True)

    # vy[pixel/sec] vs time[second]
    plt.subplot(224)
    for i in range(p.shape[1]):
        y =  p[:,i,0,1]
        vy = np.diff(y)
        plt.plot(t[1:], vy, "o-")
    #plt.bar(freq_y, abs(spectrum_y))
    plt.ylabel("vy(t) [pix/sec]")
    plt.xlabel("t [sec]")
    plt.title('vy[pixel/sec] vs time[second]')
    plt.grid(True)

    plt.show()