import librosa
import librosa.display
import matplotlib.pyplot as plt

#Grafico del punto 1 (audio lento)


# Cargar el archivo de audio

archivo1 = "lapachos_lento.wav" 
x, srr = librosa.load(archivo1,sr=100000) #Sample Rate fijao a ws = 100kHz


fig_señal = plt.figure(figsize=(14, 5))
librosa.display.waveshow(x, sr=srr) #grafico del sample
plt.title('Forma de onda del audio(lapachos lento)')
plt.xlabel('Tiempo (s)')
plt.ylabel('Amplitud')

# 
plt.xlim(0.4,3.0)
plt.axvline(x=0.493,ymin=0,ymax=1, color='r') #arranca l
plt.axvline(x=0.827,ymin=0,ymax=1, color='r') #termina l arranca a
plt.axvline(x=1.267,ymin=0,ymax=1, color='r') #termina a
plt.axvline(x=1.464,ymin=0,ymax=1, color='b') #arranca p
plt.axvline(x=1.480,ymin=0,ymax=1, color='b') #termina p arranca a
plt.axvline(x=1.906,ymin=0,ymax=1, color='b') #termina a 
plt.axvline(x=2.053,ymin=0,ymax=1, color='r') #arranca ch
plt.axvline(x=2.152,ymin=0,ymax=1, color='r') #termina ch arranca o
plt.axvline(x=2.521,ymin=0,ymax=1, color='r') #termina o arranca s
plt.axvline(x=2.801,ymin=0,ymax=1, color='r') #termina s

plt.show()
