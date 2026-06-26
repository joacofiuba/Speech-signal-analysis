import numpy as np
from scipy.io import wavfile
from scipy.signal import firwin, lfilter
import matplotlib.pyplot as plt 
import librosa
from scipy.signal import ShortTimeFFT, get_window


# Frecuencia de muestreo definida manualmente (estándar de calidad CD, adecuada para voz)
fs = 44100 

# Cargar la señal (se ignora la frecuencia de muestreo del archivo original)
_, x = wavfile.read("lapachos_lento.wav")

# Parámetros de decimación
M = 2  
nyquist = fs / 2
cutoff = nyquist / M  

# Diseño del filtro FIR antialiasing con ventana de Hamming
numtaps = 101  
h = firwin(numtaps, cutoff/nyquist, window="hamming")

# Aplicación del filtro antialiasing
x_filt = lfilter(h, 1.0, x)

# Decimación (submuestreo eliminando puntos)
x_dec = x_filt[::M]

# Guardar la señal resultante en el mismo directorio.
# Al guardarla con la fs original, la velocidad de reproducción se duplica.
wavfile.write("P8_lapachos_lento_decimado.wav", fs, x_dec.astype(np.int16))



# ESPECTROGRAMA: 


archivo = "P8_lapachos_lento_decimado.wav"
fs = 44100 # (estandar) 


# ancho de la ventana -> determina si la STFT es de banda angosta o ancha.
# banda angosta: permite tener mayor resolución espectral
# banda ancha: permite tener mayor resolución temporal.


# 1/sr = duración de cada n de la secuencia x[n]. 
# w_ancho = cantidad de n => n * 1/sr = ancho temporal de la ventana
w_ancho = 300
w = get_window('hann', w_ancho)


x, sr = librosa.load(archivo, sr=fs) 

#hop = salto de el sliding de la ventana, w_ancho//4 (división entera) implica un solapamiento entre cada iteración de un 25% de la ventana 
#mmft = zero-padding. agregado para poder hacer zoom sin pixelearse.
SFT = ShortTimeFFT(win=w, hop=w_ancho//4, fs=fs,fft_mode='onesided', mfft=4096, dual_win=None, scale_to=None, phase_shift=0)
Sx = SFT.stft(x)

# Calcular la magnitud del espectrograma en decibeles (añadiendo un offset para evitar log(0))
Sx_dB = 20 * np.log10(np.abs(Sx) + 1e-10)

# Configurar y generar la gráfica
fig, ax = plt.subplots(figsize=(10, 6))


# SFT.extent(len(y)) retorna (t_min, t_max, f_min, f_max)
im = ax.imshow(Sx_dB, origin='lower', aspect='auto', extent=SFT.extent(len(x)), vmax = 30, vmin = -10, cmap='viridis') # gráfico de la matriz
ax.set_ylim(0, 4000)
ax.set_title('Espectrograma - Banda Ancha')
ax.set_xlabel('Tiempo [s]')
ax.set_ylabel('Frecuencia [Hz]')
fig.colorbar(im, ax=ax, label='Magnitud [dB]')

plt.tight_layout()
plt.show()
