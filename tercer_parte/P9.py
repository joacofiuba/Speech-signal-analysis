import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.signal import firwin, lfilter

# 1. Carga de señales leyendo dinámicamente la frecuencia de muestreo
fs_rapida, x_rapida = wavfile.read("lapachos_rapido.wav")
fs_lenta, x_lenta = wavfile.read("lapachos_lento.wav")

# 2. Parámetros de interpolación
L = 2  
fs_nueva = fs_rapida * L
nyquist_nuevo = fs_nueva / 2

# La frecuencia de corte es el límite de Nyquist de la señal original
# para eliminar las imágenes espectrales generadas por la inserción de ceros.
cutoff = fs_rapida / 2  

# 3. Diseño del filtro FIR anti-imagen con ventana de Hamming
numtaps = 101  
# Se multiplica por L para compensar la reducción de amplitud al insertar ceros
h = firwin(numtaps, cutoff/nyquist_nuevo, window="hamming") * L

# 4. Upsampling: Inserción de L-1 ceros entre cada muestra original
x_up = np.zeros(len(x_rapida) * L)
x_up[::L] = x_rapida

# 5. Aplicación del filtro anti-imagen (Interpolación)
x_interp = lfilter(h, 1.0, x_up)

# Guardar la señal resultante.
# Al guardarla con la fs_rapida original, la reproducción ocurre a la mitad de la velocidad.
wavfile.write("lapachos_rapido_interpolado.wav", fs_rapida, x_interp.astype(np.int16))


# --- CÁLCULO Y GRAFICACIÓN DE FFT COMPARATIVA ---
def mostrar_fft_ax(signal_audio, fs, ax, titulo, color):
    N = len(signal_audio)
    freqs = np.fft.rfftfreq(N, d=1/fs)
    fft_vals = np.abs(np.fft.rfft(signal_audio)) / N
    
    ax.plot(freqs, fft_vals, color=color)
    ax.set_ylabel('Magnitud')
    ax.set_xlim(0, 2000)
    ax.set_title(titulo)
    ax.grid(True, alpha=0.3)

# Se generan 2 subgráficos: la referencia (lenta) y el resultado interpolado
fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

mostrar_fft_ax(x_lenta, fs_lenta, axes[0], 'FFT - Lapachos Lenta (Referencia)', 'black')
mostrar_fft_ax(x_interp, fs_rapida, axes[1], 'FFT - Lapachos Rápida Interpolada Clásica', 'red')

axes[1].set_xlabel('Frecuencia [Hz]')

plt.tight_layout()
plt.show()
