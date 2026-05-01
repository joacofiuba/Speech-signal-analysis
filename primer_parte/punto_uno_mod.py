import librosa
import librosa.display
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

#Grafico del punto 1 (audio lento)


# Cargar el archivo de audio

archivo1 = "lapachos_lento.wav"  # Cambia esto por el camino a tu archivo
x, srr = librosa.load(archivo1,sr=100000) #sr es tasa de muestreo


fig_señal, ax = plt.subplots(figsize=(14, 5))
librosa.display.waveshow(x, sr=srr, ax=ax)
ax.set_title('Forma de onda del audio(lapachos lento)')
ax.set_xlabel('Tiempo (s)')
ax.set_ylabel('Amplitud')
ax.set_xlim(0.4,3.0)

# Fondos por tipo fonético
ax.axvspan(0.493, 0.827, color='blue',  alpha=0.15) # l  - transitorio
ax.axvspan(0.827, 1.267, color='green', alpha=0.15) # a  - vocal
ax.axvspan(1.464, 1.480, color='blue',  alpha=0.15) # p  - explosivo
ax.axvspan(1.480, 1.906, color='green', alpha=0.15) # a  - vocal
ax.axvspan(2.053, 2.152, color='red',   alpha=0.15) # ch - fricativa
ax.axvspan(2.152, 2.521, color='green', alpha=0.15) # o  - vocal
ax.axvspan(2.521, 2.801, color='red',   alpha=0.15) # s  - fricativa

# Líneas verticales
ax.axvline(x=0.493, color='blue',  alpha=0.7) #arranca l
ax.axvline(x=0.827, color='blue',  alpha=0.7) #termina l
ax.axvline(x=0.827, color='green', alpha=0.7) #arranca a
ax.axvline(x=1.267, color='green', alpha=0.7) #termina a
ax.axvline(x=1.464, color='blue',  alpha=0.7) #arranca p
ax.axvline(x=1.480, color='blue',  alpha=0.7) #termina p
ax.axvline(x=1.480, color='green', alpha=0.7) #arranca a
ax.axvline(x=1.906, color='green', alpha=0.7) #termina a
ax.axvline(x=2.053, color='red',   alpha=0.7) #arranca ch
ax.axvline(x=2.152, color='red',   alpha=0.7) #termina ch
ax.axvline(x=2.152, color='green', alpha=0.7) #arranca o
ax.axvline(x=2.521, color='green', alpha=0.7) #termina o
ax.axvline(x=2.521, color='red',   alpha=0.7) #arranca s
ax.axvline(x=2.801, color='red',   alpha=0.7) #termina s

# Leyenda
leyenda = [
    Patch(facecolor='green', alpha=0.5, label='Vocal (cuasi-periódico)'),
    Patch(facecolor='red',   alpha=0.5, label='Fricativa/Africada (no periódico)'),
    Patch(facecolor='blue',  alpha=0.5, label='Transitorio (l, p)'),
]
ax.legend(handles=leyenda, loc='upper right')

plt.show()
