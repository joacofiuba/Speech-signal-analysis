import numpy as np
from scipy.io import wavfile
from scipy.signal import firwin, lfilter

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
wavfile.write("lapachos_lento_decimado.wav", fs, x_dec.astype(np.int16))
