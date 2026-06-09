# ============================================
# Análisis de vocales con STFT (banda angosta vs banda ancha)
# ============================================

# Importamos librerías necesarias
import numpy as np
import matplotlib.pyplot as plt
import librosa
from scipy.signal import get_window, ShortTimeFFT


archivo = "lapachos_lento.wav"   # Nombre del archivo de audio
fs = 44100                       # Frecuencia de muestreo deseada
x, sr = librosa.load(archivo, sr=fs)  # x = señal de audio, sr = frecuencia real usada

# --------------------------------------------
# 2. Definir intervalos de las vocales
# --------------------------------------------
# Cada tupla (t0, t1) indica inicio y fin en segundos
vowel_intervals = [
    (0.827, 1.267),  # vocal "a"
    (1.480, 1.906),  # vocal "a"
    (2.152, 2.521)   # vocal "o"
]

# --------------------------------------------
# 3. Función para calcular espectrograma STFT
# --------------------------------------------
def stft_spectrogram(x_seg, sr, w_ancho):
    """
    Calcula el espectrograma STFT de un segmento de audio.
    Parámetros:
      - x_seg: segmento de señal
      - sr: frecuencia de muestreo
      - w_ancho: ancho de la ventana (en muestras)
    """
    # Ventana Hann para suavizar bordes
    w = get_window('hann', w_ancho)

    # Configuración de la STFT
    SFT = ShortTimeFFT(
        win=w,
        hop=max(1, w_ancho//4),  # paso entre ventanas (solapamiento)
        fs=sr,
        fft_mode='onesided',
        mfft=16384,              # tamaño de la FFT (alta resolución en frecuencia)
        dual_win=None,
        scale_to=None,
        phase_shift=0
    )

    # Calcular STFT
    Sx = SFT.stft(x_seg)

    # Convertir a decibelios
    Sx_dB = 20 * np.log10(np.abs(Sx) + 1e-10)

    # extent = límites de tiempo y frecuencia para graficar
    return Sx_dB, SFT.extent(len(x_seg))

# --------------------------------------------
# 4. Loop sobre cada vocal
# --------------------------------------------
for i, (t0, t1) in enumerate(vowel_intervals, start=1):
    # Convertir tiempos a índices de muestra
    s = int(t0 * sr)
    e = int(t1 * sr)
    x_seg = x[s:e]

    # Banda angosta: ventana larga → mejor resolución en frecuencia
    S_narrow, ext_n = stft_spectrogram(x_seg, sr, w_ancho=2500)

    # Banda ancha: ventana corta → mejor resolución en tiempo
    S_wide, ext_w = stft_spectrogram(x_seg, sr, w_ancho=200)

    # --------------------------------------------
    # 5. Graficar resultados
    # --------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Espectrograma banda angosta
    im0 = axes[0].imshow(S_narrow, origin='lower', aspect='auto',
                         extent=ext_n, vmax=30, vmin=-20, cmap='viridis')
    axes[0].set_ylim(0, 7000)  # rango de frecuencias relevante para voz
    axes[0].set_title(f'Vocal {i} Banda angosta ({t0:.3f}-{t1:.3f}s)')
    axes[0].set_xlabel('Tiempo [s]')
    axes[0].set_ylabel('Frecuencia [Hz]')
    fig.colorbar(im0, ax=axes[0], label='Magnitud [dB]')

    # Espectrograma banda ancha
    im1 = axes[1].imshow(S_wide, origin='lower', aspect='auto',
                         extent=ext_w, vmax=30, vmin=-20, cmap='viridis')
    axes[1].set_ylim(0, 7000)
    axes[1].set_title(f'Vocal {i} Banda ancha ({t0:.3f}-{t1:.3f}s)')
    axes[1].set_xlabel('Tiempo [s]')
    fig.colorbar(im1, ax=axes[1], label='Magnitud [dB]')

    plt.tight_layout()
    plt.show()

# ============================================
# Explicación conceptual:
# - Banda angosta (ventana larga): se ven claramente los armónicos y la f0.
# - Banda ancha (ventana corta): se ven mejor los formantes (resonancias de la vocal).
# ============================================
