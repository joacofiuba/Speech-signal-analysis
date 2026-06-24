import librosa
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import find_peaks, freqz

archivo = "lapachos_rapido.wav"

sr = 16000
x, sr = librosa.load(archivo, sr=sr)

vocales = [
    ("[a1] varios períodos", 0.400, 0.450),
    ("[a1] un período",      0.4095, 0.4135),
    ("[a2] varios períodos", 0.750, 0.800),
    ("[a2] un período",      0.776, 0.780),
    ("[o] varios períodos",  1.100, 1.150),
    ("[o] un período",       1.197, 1.202),
]

# Orden LPC estándar para voz a 16 kHz es 18.
orden_lpc = int(sr / 1000) + 2 

for label, ti, tf in vocales:
    segmento = x[int(ti*sr):int(tf*sr)]
    N = len(segmento)
 
    # Optimización 2: Enventanado para reducir fugas espectrales (spectral leakage)
    segmento = segmento * np.hanning(N)
 
    #Computa solo frecuencias positivas, 
    # reduciendo a la mitad la memoria y el tiempo de cálculo.
    X = np.fft.rfft(segmento)
    freqs_pos = np.fft.rfftfreq(N, d=1/sr)
    X_mag = np.abs(X)
 
    # Cálculo LPC
    segmento_pre = librosa.effects.preemphasis(segmento)
    a = librosa.lpc(segmento_pre, order=orden_lpc)
    
    # freqz con worN garantizando el mismo tamaño que el vector de rfft
    _, h = freqz(1, a, worN=len(freqs_pos))
    lpc_mag = np.abs(h)
    
    # Normalización del LPC para el gráfico
    lpc_mag = lpc_mag * (np.max(X_mag) / np.max(lpc_mag))
 

    # Gráfico
    plt.figure()
    plt.plot(freqs_pos, X_mag, label='Espectro FFT', alpha=0.5)
    plt.plot(freqs_pos, lpc_mag, color='red', linewidth=2, label='Envolvente LPC')
    plt.title(label)
    plt.xlabel('Frecuencia (Hz)')
    plt.ylabel('Módulo')
    plt.xlim(0, 2500) # Límite visual preservado
    plt.legend()
    plt.show()
