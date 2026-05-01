import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

# Archivo de audio lento
archivo_lento = r"C:\Users\Usuario\OneDrive\Escritorio\Señales y sistemas\TP1\Lapachos_lento.wav"
x_lento, sr = librosa.load(archivo_lento, sr=100000)

# --- Segmentación manual aproximada ---
# Ajusta estos tiempos según tus marcas del punto 1
t_a_inicio, t_a_fin = 0.827, 1.267   # segmento de la vocal [a]
t_s_inicio, t_s_fin = 2.521, 2.801   # segmento de la fricativa [s]

# Convertir a índices de muestras
idx_a_inicio, idx_a_fin = int(t_a_inicio*sr), int(t_a_fin*sr)
idx_s_inicio, idx_s_fin = int(t_s_inicio*sr), int(t_s_fin*sr)

segmento_a = x_lento[idx_a_inicio:idx_a_fin]
segmento_s = x_lento[idx_s_inicio:idx_s_fin]

# --- Graficar segmentos ---
plt.figure(figsize=(12,4))
librosa.display.waveshow(segmento_a, sr=sr)
plt.title("Segmento vocal [a] (cuasi-periódico)")
plt.xlabel("Tiempo (s)")
plt.ylabel("Amplitud")
plt.show()

plt.figure(figsize=(12,4))
librosa.display.waveshow(segmento_s, sr=sr)
plt.title("Segmento fricativo [s] (no periódico)")
plt.xlabel("Tiempo (s)")
plt.ylabel("Amplitud")
plt.show()

# --- Estimación de periodo y frecuencia fundamental ---
# Usamos autocorrelación para la vocal [a]
autocorr = np.correlate(segmento_a, segmento_a, mode='full')
autocorr = autocorr[len(autocorr)//2:]  # mitad positiva

# Buscar el primer máximo significativo (ignorar el pico en lag=0)
lag = np.argmax(autocorr[1:]) + 1
periodo_muestras = lag
periodo_segundos = periodo_muestras / sr
frecuencia_fundamental = 1 / periodo_segundos

print(f"Periodo estimado [a]: {periodo_segundos:.6f} s")
print(f"Frecuencia fundamental [a]: {frecuencia_fundamental:.2f} Hz")

