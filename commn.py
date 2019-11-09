import librosa as li
import librosa.display
import numpy as np
import pyaudio
import wave
import scipy
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 32000
CHUNK = 960
RECORD_SECONDS = 3
frames=[]
WAVE_OUTPUT_FILENAME = "file.wav"
audio = pyaudio.PyAudio()
stream = audio.open(format=FORMAT, channels=CHANNELS,
                rate=RATE, input=True,
                frames_per_buffer=CHUNK)
print("recording...")
for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
     data = stream.read(CHUNK)
     frames.append(data)
print("finished recording")
waveFile = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
waveFile.setnchannels(CHANNELS)
waveFile.setsampwidth(audio.get_sample_size(FORMAT))
waveFile.setframerate(RATE)
waveFile.writeframes(b''.join(frames))
waveFile.close()
y, sample_rate = li.load(r"E:\\python\\file.wav")
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
D = li.amplitude_to_db(np.abs(li.stft(y)), ref=np.max)
plt.subplot(3,1,1)
plt.title("Audio Analog Signal")
plt.plot(y[1950:2000])
plt.subplot(3,1,2)
plt.title("Spectogram")
librosa.display.specshow(D, y_axis='linear')
plt.colorbar(format='%+2.0f dB')
plt.subplot(3,1,3)
plt.title("Audio Digital Signal")
plt.plot(result, marker='d', color='blue', drawstyle='steps')
plt.show()
stream.stop_stream()
stream.close()
audio.terminate()
