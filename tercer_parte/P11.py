import numpy as np
import matplotlib.pyplot as plt
import librosa
from scipy.signal import ShortTimeFFT, get_window
import soundfile as sf

archivo = "lapachos_rapido.wav"
fs = 44100 

x, _ = librosa.load(archivo, sr=fs)

# --- 1. TFCT PARA PROCESAMIENTO (Banda Angosta) ---
w_proc = 2048
win_proc = get_window('hann', w_proc)
SFT_proc = ShortTimeFFT(win=win_proc, hop=w_proc//4, fs=fs, fft_mode='onesided', mfft=4096, phase_shift=0)

# Análisis
Sx = SFT_proc.stft(x)

# Modificación: Interpolación lineal de columnas para duplicar la longitud
n_bins, n_frames = Sx.shape
Sx_mod = np.zeros((n_bins, n_frames * 2 - 1), dtype=complex)
Sx_mod[:, ::2] = Sx
Sx_mod[:, 1::2] = (Sx[:, :-1] + Sx[:, 1:]) / 2.0

# Síntesis
x_rec = SFT_proc.istft(Sx_mod)
sf.write("P11_lapachos_TFCT_mitad.wav", x_rec, fs)



# TFCT PARA ESPECTROGRAMA

archivo = "P11_lapachos_TFCT_mitad.wav"

w_ancho = 2500 #aprox 56ms de ventana
w = get_window(('hann'), w_ancho)

x, sr = librosa.load(archivo, sr=fs) 
#hop = salto de el sliding de la ventana, w_ancho//4 (división entera) implica un solapamiento entre cada iteración de un 25% de la ventana 
SFT = ShortTimeFFT(win=w, hop=w_ancho//8, fs=fs,fft_mode='onesided', mfft=16384, dual_win=None, scale_to=None, phase_shift=0)
Sx = SFT.stft(x)

# Calcular la magnitud del espectrograma en decibeles (añadiendo un offset para evitar log(0))
Sx_dB = 20 * np.log10(np.abs(Sx) + 1e-10)

# Configurar y generar la gráfica
fig, ax = plt.subplots(figsize=(10, 6))


# SFT.extent(len(y)) retorna (t_min, t_max, f_min, f_max)
im = ax.imshow(Sx_dB, origin='lower', aspect='auto', extent=SFT.extent(len(x)), vmax = 30, vmin = -30, cmap='viridis') # gráfico de la matriz
ax.set_ylim(0, 7000)
ax.set_title('Espectrograma - Banda Angosta - Lapachos rápida Interpolada')
ax.set_xlabel('Tiempo [s]')
ax.set_ylabel('Frecuencia [Hz]')
fig.colorbar(im, ax=ax, label='Magnitud [dB]')

plt.tight_layout()
plt.show()


# Espectrograma lapachos rapido

archivo = "lapachos_rapido.wav"

w_ancho = 2500 #aprox 56ms de ventana
w = get_window(('hann'), w_ancho)

x, sr = librosa.load(archivo, sr=fs) 
#hop = salto de el sliding de la ventana, w_ancho//4 (división entera) implica un solapamiento entre cada iteración de un 25% de la ventana 
SFT = ShortTimeFFT(win=w, hop=w_ancho//8, fs=fs,fft_mode='onesided', mfft=16384, dual_win=None, scale_to=None, phase_shift=0)
Sx = SFT.stft(x)

# Calcular la magnitud del espectrograma en decibeles (añadiendo un offset para evitar log(0))
Sx_dB = 20 * np.log10(np.abs(Sx) + 1e-10)

# Configurar y generar la gráfica
fig, ax = plt.subplots(figsize=(10, 6))


# SFT.extent(len(y)) retorna (t_min, t_max, f_min, f_max)
im = ax.imshow(Sx_dB, origin='lower', aspect='auto', extent=SFT.extent(len(x)), vmax = 30, vmin = -30, cmap='viridis') # gráfico de la matriz
ax.set_ylim(0, 7000)
ax.set_title('Espectrograma - Banda Angosta - Lapachos rápida Interpolada')
ax.set_xlabel('Tiempo [s]')
ax.set_ylabel('Frecuencia [Hz]')
fig.colorbar(im, ax=ax, label='Magnitud [dB]')

plt.tight_layout()
plt.show()
