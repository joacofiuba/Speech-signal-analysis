import librosa
import librosa.display
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
#Grafico del punto 1 (audio lento)


# Cargar el archivo de audio
archivo_lento = "lapachos_lento.wav"  
x_lento, sr = librosa.load(archivo_lento, sr=100000)

vocales = [
    ("[a1] varios períodos", 1.00, 1.20),
    ("[a1] un período",      1.079, 1.082),
    ("[a2] varios períodos", 1.65, 1.70),
    ("[a2] un período",      1.689, 1.693),
    ("[o] varios períodos",  2.30, 2.35),
    ("[o] un período",       2.324, 2.328),
]

for label, ti, tf in vocales:
    segmento = x_lento[int(ti*sr):int(tf*sr)]

    # calculo de FFT
    X = np.fft.fft(segmento) # magnitudes complejas
    freqs = np.fft.fftfreq(len(segmento), d=1/sr) # frecuencias asociadas a cada complejo

    # Solo mitad positiva
    X_mag = np.abs(X[:len(X)//2])  #tomo el modulo para poder graficar, solo tomo las frecuencias positivas
    freqs_pos = freqs[:len(freqs)//2] #me quedo con lo que esté por debajo de fs/2 (50kHz) porque por encima tengo aliasing (Nyquist)

    plt.figure()
    plt.plot(freqs_pos, X_mag)
    plt.title(label)
    plt.xlabel('Frecuencia (Hz)')
    plt.ylabel('Módulo')
    plt.xlim(0, 4000)
    plt.show()
