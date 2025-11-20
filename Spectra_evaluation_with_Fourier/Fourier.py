
def pacfft (x1,t, range, x2=None, yshift = 0,maxt= 500, printing = False):
    import numpy as np
    import scipy.fftpack
    import matplotlib.pyplot as plt
    import scipy.signal as signal
    import matplotlib.ticker as ticker

    """x1 data, t is time, range is the range of the data to be used for the Fourier transformation, x2 is the fit"""
    Mradfactor = 0.1592 #factor to conveert from Hz to Mrad
    k = len(x1) # len of array used to cut the data symmetrically
    # cut and mirroring the data at the y axis

    if x2 is not None:
        x1[0:2] = x2[0:2]
    x1mir  = np.flip(x1)
    tmir =  -np.flip(t)
    x1mir= np.append(x1mir,x1[1:])
    x1mir = x1mir - yshift
    tmir = np.append(tmir,t[1:])
    
    
  
    x1 = np.array(x1mir)[k-range-1:k+range]
    t = np.array(tmir)[k-range-1:k+range]*10**-3 # 10^-3 for unit 
    
    # apply a window function to the data was in the thesis
    window = np.kaiser(len(x1), beta=20)
    # gauss window
    # use a different window function
    #window = np.hanning(len(x1))
    

    X1 = scipy.fftpack.fft(x1*window)
    # find the peaks in the Fourier transformation using signal 
# calculate the x-axis. Using the periodicity of the data
    N = len(X1)       
    T = t[1]-t[0]
    f = np.linspace(0.0, 1.0/(2.0*T), N//2)
# plot the Fourier transformation
    
    fig, ax = plt.subplots()
    ax.set_xlim(0, 1000)
    fy1 = 2.0/N * np.abs(X1[:N//2])
    ax.plot(f/Mradfactor, fy1, label = 'Data')
    ax.set_xlabel('Frequency [MHz]')
    ax.set_ylabel('Amplitude')
    ax.set_xlim(0,maxt)
    
    # get peaks and plot them 
    peaks = signal.find_peaks(fy1, height=np.mean(fy1)*1.5)
    if peaks[0].size == 0:
        print('No peaks found')
    if peaks[0].size > 9:
        print('More than 9 peaks found -> change max height')
        peaks = signal.find_peaks(fy1, height=max(fy1)*0.9)
    for i in peaks[0]:
        ax.plot(f[i]/Mradfactor, fy1[i], 'ro')
        # add a label to the peak
        ax.text(f[i]/Mradfactor, fy1[i], str(round(f[i]/Mradfactor,2)))
    # if a second data set is given for the fit, do the same for this fit
    fy2 = None
    if x2 is not None:
        x2mir = np.flip(x2)
        x2mir= np.append(x2mir,x2[1:])
        x2mir = x2mir - yshift

        x2 = np.array(x2mir)[k-range-1:k+range]
        X2 = scipy.fftpack.fft(x2*window)
        fy2 = 2.0/N * np.abs(X2[:N//2])
        #ax.plot(f/Mradfactor, fy2, label = 'Fit', color = 'red', ls = '--')
    ax.legend()
    ax.yaxis.set_major_formatter(ticker.ScalarFormatter())
    ax.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
    # plot t over x 
    fig, ay = plt.subplots()
    if printing == True:
        ay.plot(t, x1)
        if x2 is not None:
            ay.plot(t, x2, color = 'red')

        ay.set_xlabel('Time [s]')
        ay.set_ylabel('Amplitude')
        # clear the figure 
        #plt.show()
    # clear all plots 
    plt.close('all')
    return f/Mradfactor, fy1, fy2

# from this script create a reverse script that does the inverse Fourier transformation
def irfft(y1, n):
    import numpy as np
    import scipy.fftpack
    # do an inverse Fourier transformation on the data using irfft
    realfourier = scipy.fftpack.irfft(y1, n=n) * n/2
    return realfourier
