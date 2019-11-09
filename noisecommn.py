import librosa as li
import numpy as np
import scipy
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import noisereduce as nsr
y, sample_rate = li.load(r"E:\\python\\file.wav")
noisy_part = y
y = nsr.reduce_noise(audio_clip=y, noise_clip=noisy_part, verbose=True)
print("Sampling rate: "+str(sample_rate))
plt.subplot(2,1,1)
plt.title("Time domain samples")
plt.stem(y[1950:2000])
ydash=np.fft.fft(y)
plt.subplot(2,1,2)
plt.title("Frequency domain samples")
plt.stem(ydash[1950:2000])
plt.show()
zero_crossings = []
energy = []
entropy_of_energy = []
frames=[]
length_series=len(y)
for i in range(0,length_series,int(sample_rate/5.0)):
     frame_self = y[i:i+int(sample_rate/5.0):1]
     frames.append(frame_self)
     z = li.zero_crossings(frame_self)
     arr = np.nonzero(z)
     zero_crossings.append(len(arr[0]))
     e = li.feature.rmse(frame_self)
     energy.append(np.mean(e))
     ent = 0.0
     m = np.mean(e)
     for j in range(0,len(e[0])):
          q = np.absolute(e[0][j] - m)
          ent = ent + (q * np.log10(q))
     entropy_of_energy.append(ent)
f_list_1 = []
f_list_1.append(zero_crossings)
f_list_1.append(energy)
f_list_1.append(entropy_of_energy)
f_np_1 = np.array(f_list_1)
f_np_1 = np.transpose(f_np_1)[:-1]
kmeans = KMeans(n_clusters=2, random_state=0).fit(f_np_1)
result=kmeans.predict(f_np_1)
plt.subplot(3,1,1)
plt.title("Audio Analog Signal")
plt.plot(y[1950:2000])
plt.subplot(3,1,2)
plt.title("Spectogram")
plt.specgram(y,Fs=sample_rate)
plt.subplot(3,1,3)
plt.title("Audio Digital Signal")
plt.plot(result, marker='d', color='blue', drawstyle='steps')
plt.show()
