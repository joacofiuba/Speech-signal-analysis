import librosa
import librosa.display
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
from scipy.signal import find_peaks


archivo = "lapachos_rapido.wav"  
x, sr = librosa.load(archivo, sr=100000)

vocales = [
    ("[a1] varios períodos", 0.400, 0.450),
    ("[a1] un período",      0.4095, 0.4135),
    ("[a2] varios períodos", 0.750, 0.800),
    ("[a2] un período",      0.776, 0.780),
    ("[o] varios períodos",  1.100, 1.150),
    ("[o] un período",       1.197, 1.202),
]

for label, ti, tf in vocales:
    segmento = x[int(ti*sr):int(tf*sr)]
 
    # calculo de FFT
    X = np.fft.fft(segmento) # magnitudes complejas
    freqs = np.fft.fftfreq(len(segmento), d=1/sr) # frecuencias asociadas a cada complejo
 
    # Solo mitad positiva
    X_mag = np.abs(X[:len(X)//2])  #tomo el modulo para poder graficar, solo tomo las frecuencias positivas
    freqs_pos = freqs[:len(freqs)//2] #me quedo con lo que esté por debajo de fs/2 (50kHz) porque por encima tengo aliasing (Nyquist)
 
    # Identificación de formantes
    peaks, _ = find_peaks(X_mag, prominence=20)
    print(f"\n{label} - Primeros formantes:")
    for p in peaks[:3]:
        print(f"  {freqs_pos[p]:.1f} Hz")
 
    plt.figure()
    plt.plot(freqs_pos, X_mag)
    plt.title(label)
    plt.xlabel('Frecuencia (Hz)')
    plt.ylabel('Módulo')
    plt.xlim(0, 2500)
    plt.show()
 
