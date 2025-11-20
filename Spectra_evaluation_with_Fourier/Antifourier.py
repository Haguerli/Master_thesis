import matplotlib.pyplot as plt
import pandas as pd
import os as os
import matplotlib.ticker as ticker
import scipy.signal as signal
import numpy as np
from os import path
import sys
sys.path.append(path.abspath('Fourier.py'))
from Fourier import pacfft
from scipy.fft import irfft
from matplotlib.widgets import Button, Slider



#current_folder = "/Users/hannes/Nextcloud/2ndCdNightmare/For_presentations/First_time/"
#current_folder = "/Users/hannes/Nextcloud/2ndCdNightmare/For_presentations/SD_45_comparison/"
#current_folder = "/Users/hannes/Nextcloud/2ndCdNightmare/For_presentations/SD_0_comparison/"
#current_folder = "/Users/hannes/Nextcloud/2ndCdNightmare/For_presentations/PPLN_deteccomparison/"
#current_folder = "/Users/hannes/Nextcloud/2ndCdNightmare/For_presentations/Temperature/"
#current_folder = "/Users/hannes/Nextcloud/2ndCdNightmare/For_presentations/PPLN_difference/"
#current_folder = "/Users/hannes/Nextcloud/2ndCdNightmare/For_presentations/PPLN_SD_spectra_or_background/"
current_folder = "/Users/hannes/Nextcloud/2ndCdNightmare/For_presentations/PPLN_5/"
colnames=['TIME', 'X', 'Fit', 'Z'] 
# Search for files with filenames containing ".dat" and "result" in all subfolders
# list all folder in current folder
file_list = os.listdir(current_folder)
# sort list 
file_list.sort()
print(file_list)
# remove from file list all files that do not contain .dat
file_list = [x for x in file_list if ".dat" in x]
k = pd.read_csv(current_folder + file_list[0], sep="  ", skiprows=4, names=colnames, engine='python')
# convert every , to . 
k = k.apply(lambda x: x.str.replace(',','.'))
# convert the strings to floats
k = k.apply(pd.to_numeric, errors='coerce')
# open the dat file of the first file in the list and save the content in a pandas dataframe
a = {}

for i in file_list:
    a[i] = pd.read_csv(current_folder + i, sep="  ", skiprows=4, engine='python', names=colnames)
    # convert every , to .
    a[i] = a[i].apply(lambda x: x.str.replace(',','.'))
    # convert the strings to floats
    a[i] = a[i].apply(pd.to_numeric, errors='coerce')
def gauss(x, a, x0, sigma):
    return a*np.exp(-(x-x0)**2/(2*sigma**2))
def realfourier(y2, xfreq, minor, major):
    y2 = np.array(y2)
# change all values from 0 minor to 0 and all from major to infinte to 0 
    y2[xfreq < minor] = 0
    y2[xfreq > major] = 0
    return irfft(y2, n=len(xfreq)) * len(xfreq)/2

Mradfactor = 0.1592
num = 0
lena = 400
x = np.array(a[file_list[num]]['X'])
t = np.array(a[file_list[num]]['TIME'])
x2 = np.array(a[file_list[num]]['Fit'])
xfreq, y1, y2 = pacfft(x,t, lena, x2 = x2, yshift=+0.0, maxt=600, printing = False) # xshift2 = 0.1, ignore = 0)
save1 = y1
save2 = y2


fig, ax = plt.subplots(2, 1 , figsize=(14, 14))
maxx = 600
minor_int = 300
major_int = 450 
ax[0].plot(xfreq[:maxx], y1[:maxx], label = 'Data')
ax[0].plot(xfreq[:maxx], y2[:maxx], label = 'Fit')
ax[0].set_xlabel('Frequency [MRad/s]')
ax[0].set_ylabel('Amplitude')
ax[0].set_xlim(0, 600)
ax[0].set_ylim(0, max(y1[:maxx])*1.1)
# increase distance between plots 
plt.subplots_adjust(hspace=0.4)


# make vertical lines at minor and major and color the area between them


fig.subplots_adjust( bottom=0.25)

real2 = ax[1].plot(t[:300]*2, realfourier(y2, xfreq, minor_int, major_int)[:300], label = 'Fourier back of Fit')
axfreq = fig.add_axes([0.25, 0.1, 0.65, 0.03])
minor_slider = Slider(
    ax=axfreq,
    label='Minor Frequency [MRad/s]',
    valmin=0,
    valmax=700,
    valinit=minor_int,
)
# add a second slider beneath the first one
axfreq2 = fig.add_axes([0.25, 0.05, 0.65, 0.03])
major_slider = Slider(
    ax=axfreq2,
    label='Major Frequency [MRad/s]',
    valmin=0,
    valmax=700,
    valinit=major_int,
)
# add a button in the upper right corner
axcolor = 'lightgoldenrodyellow'
resetax = fig.add_axes([0.8, 0.025, 0.1, 0.04])
reset_button = Button(resetax, 'Reset', color=axcolor, hovercolor='0.975')



rp = 0
k = ax[0].axvline(minor_int, lw=2)
k2 = ax[0].axvline(major_int, lw=2, color = 'red')
# fill the space between k and k2
def update1(val):
    global rp
    rp = (rp + 1)% 2
    update(0)
def update(val):
    # modulo 2 to avoid that the function is called twice
    color = ['blue', 'orange']
    
    label = [ 'Fourier back of Data', 'Fourier back of Fit']
    if rp == 0:
        y = save1
        reset_button.label.set_text('Data')
        
    else:
        y = save2
        reset_button.label.set_text('Fit')
    time = minor_slider.val
    k.set_xdata(time)
    k2.set_xdata(major_slider.val)
    real2[0].set_ydata(realfourier(y, xfreq, minor_slider.val, major_slider.val)[:300])
    # change label of the plot
    real2[0].set_label(label[rp])
    # change color of the plot
    real2[0].set_color(color[rp])   
    ax[1].legend()
    
    fig.canvas.draw_idle()
    
minor_slider.on_changed(update)
major_slider.on_changed(update)
reset_button.on_clicked(update1)


ax[1].plot(t, x, label = 'Data', alpha = 0.5, color = 'black')
ax[1].set_ylim(min(x[5:200]), max(x[5:300]))
ax[1].set_xlabel('Time [ns]')
ax[1].set_ylabel('R(t)')
ax[1].set_xlim(0, 200)
ax[1].legend()
ax[0].legend()
ax[0].set_title('Fourier Transform')
ax[1].set_title('Inverse Fourier Transform/ Real space')
plt.show()


