import numpy as np
import matplotlib.pyplot as plt
import librosa
from scipy.signal import ShortTimeFFT, get_window
import soundfile as sf

archivo = "lapachos_lento.wav"
fs = 44100 

x, _ = librosa.load(archivo, sr=fs)

# --- 1. TFCT PARA PROCESAMIENTO (Banda Angosta) ---
w_proc = 2048
win_proc = get_window('hann', w_proc)
SFT_proc = ShortTimeFFT(win=win_proc, hop=w_proc//4, fs=fs, fft_mode='onesided', mfft=w_proc, phase_shift=0)

# Análisis, modificación y síntesis
Sx = SFT_proc.stft(x)
Sx_mod = Sx[:, ::2] 
x_rec = SFT_proc.istft(Sx_mod)

sf.write("lapachos_doble_vel_tfct.wav", x_rec, fs)


# --- 2. TFCT PARA ESPECTROGRAMA (Banda Ancha - Parámetros originales) ---
w_spec = 400
win_spec = get_window('hann', w_spec)
SFT_spec = ShortTimeFFT(win=win_spec, hop=w_spec//8, fs=fs, fft_mode='onesided', mfft=4096, phase_shift=0)

# Cálculo exclusivo para la gráfica
Sx_rec = SFT_spec.stft(x_rec)
Sx_dB = 20 * np.log10(np.abs(Sx_rec) + 1e-10)


# --- 3. CONFIGURACIÓN DE GRÁFICA ---
fig, ax = plt.subplots(figsize=(10, 6))

im = ax.imshow(Sx_dB, origin='lower', aspect='auto', extent=SFT_spec.extent(len(x_rec)), vmax=30, vmin=-10, cmap='viridis')
ax.set_ylim(0, 2500)
ax.set_title('Espectrograma - Banda Ancha (Señal Procesada)')
ax.set_xlabel('Tiempo [s]')
ax.set_ylabel('Frecuencia [Hz]')
fig.colorbar(im, ax=ax, label='Magnitud [dB]')

plt.tight_layout()
plt.show()
