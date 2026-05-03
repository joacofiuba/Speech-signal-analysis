import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

# Archivo de audio lento
archivo_lento = r"lapachos_lento.wav"
archivo_rapido = r"lapachos_rapido.wav"
x_lento, sr = librosa.load(archivo_lento, sr=100000)

# --- Segmentación manual aproximada ---
t_a_inicio, t_a_fin = 0.827, 1.267  # segmento de la vocal [a]
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

f0 = librosa.yin(segmento_a, fmin=50, fmax=400, sr=sr, frame_length=4002)
f0_media = f0[f0 > 0].mean()  # ignorar frames sin pitch detectado
periodo_segundos = 1 / f0_media
print(f"Frecuencia fundamental [a]: {f0_media:.2f} Hz")
print(f"Periodo estimado [a]: {periodo_segundos:.6f} s")
