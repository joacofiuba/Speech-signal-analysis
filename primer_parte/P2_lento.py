import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

archivo_lento = r"lapachos_lento.wav"
x_lento, sr = librosa.load(archivo_lento, sr=100000)

# Vector de fonemas: (etiqueta, t_inicio, t_fin)
fonemas = [
    ("l",   0.650, 0.700), # unicamente segmento periodico
    ("a1",  0.827, 1.267), # todo el fonema
    ("a1",  1.100, 1.150),  # unicamente segmento peródico
    ("a2",  1.650, 1.700), # unicamente segmento periódico
    ("o",   2.300, 2.350), # unicamente segmento periódico 
    ("s",   2.521, 2.801), # todo el fonemaa 
]

def graficar_fonemas(señal, sr, fonemas):
    for etiqueta, t_ini, t_fin in fonemas:
        idx_ini = int(t_ini * sr)
        idx_fin = int(t_fin * sr)
        segmento = señal[idx_ini:idx_fin]
        t = np.linspace(t_ini, t_fin, len(segmento))

        plt.figure(figsize=(12, 4))
        plt.plot(t, segmento)
        plt.title(f"Segmento [{etiqueta}]")
        plt.xlabel("Tiempo (s)")
        plt.ylabel("Amplitud")
        plt.tight_layout()
        plt.show()

graficar_fonemas(x_lento, sr, fonemas)

