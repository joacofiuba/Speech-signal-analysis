import librosa
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import freqz

# Cargar el archivo de audio
archivo = "lapachos_lento.wav"  
x, sr = librosa.load(archivo, sr=100000)

vocales = [
    ("[a1] varios períodos", 1.00, 1.20),
    ("[a1] un período",      1.079, 1.082),
    ("[a2] varios períodos", 1.65, 1.70),
    ("[a2] un período",      1.689, 1.693),
    ("[o] varios períodos",  2.30, 2.35),
    ("[o] un período",       2.324, 2.328),
]

for label, ti, tf in vocales:
    segmento = x[int(ti*sr):int(tf*sr)]
 
    # Cálculo de FFT
    X = np.fft.fft(segmento)
    freqs_pos = np.fft.fftfreq(len(segmento), d=1/sr)[:len(segmento)//2]
    X_mag = np.abs(X[:len(segmento)//2])
 
    # --- Cálculo LPC Mejorado ---
    
    # 1. Submuestreo a 8 kHz
    sr_lpc = 8000
    segmento_lpc = librosa.resample(y=segmento, orig_sr=sr, target_sr=sr_lpc)
    
    # 2. Pre-énfasis: Aplana el espectro para que el LPC capture mejor F2 y F3
    coef_preenfasis = 0.97
    segmento_pre = librosa.effects.preemphasis(segmento_lpc, coef=coef_preenfasis)
    
    # 3. Enventanado
    segmento_pre = segmento_pre * np.hamming(len(segmento_pre))
    
    # 4. Cálculo LPC (Orden 12 es el estándar óptimo para 8kHz)
    orden_lpc = 12
    a_lpc = librosa.lpc(segmento_pre, order=orden_lpc)
    
    # 5. Respuesta en frecuencia del filtro LPC
    idx_max = np.searchsorted(freqs_pos, sr_lpc / 2)
    freqs_eval = freqs_pos[:idx_max]
    _, h_lpc = freqz(1, a_lpc, worN=freqs_eval, fs=sr_lpc)
    
    # 6. Des-énfasis visual: Restaura la pendiente espectral original
    w_eval = 2 * np.pi * freqs_eval / sr_lpc
    h_pre = 1 - coef_preenfasis * np.exp(-1j * w_eval)
    
    magnitud_h_pre = np.abs(h_pre)
    
    # Evitar el artefacto de división por casi cero en DC (0 Hz)
    # Congelamos el valor del divisor por debajo de los 100 Hz
    idx_100hz = np.searchsorted(freqs_eval, 100)
    if idx_100hz > 0:
        magnitud_h_pre[:idx_100hz] = magnitud_h_pre[idx_100hz]
        
    envolvente = np.abs(h_lpc) / magnitud_h_pre
    # 7. Normalización restringida al rango de ploteo
    idx_limite = np.searchsorted(freqs_eval, 2500)
    if idx_limite > 0:
        max_espectro = np.max(X_mag[:idx_limite])
        max_envolvente = np.max(envolvente[:idx_limite])
        if max_envolvente > 0:
            envolvente = envolvente * (max_espectro / max_envolvente)

    # Ploteo
    plt.figure()
    plt.plot(freqs_pos, X_mag, label='Espectro')
    plt.plot(freqs_eval, envolvente, color='red', linewidth=2, label='Envolvente LPC')
    plt.title(label)
    plt.xlabel('Frecuencia (Hz)')
    plt.ylabel('Módulo')
    plt.xlim(0, 2500)
    plt.legend()
    plt.show()
