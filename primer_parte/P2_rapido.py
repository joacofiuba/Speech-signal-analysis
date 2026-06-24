import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

archivo_lento = "lapachos_rapido.wav" 
x_lento, sr = librosa.load(archivo_lento, sr=100000)

# Vector de fonemas: (etiqueta, t_inicio, t_fin)
fonemas = [
    ("l",   0.200, 0.351), #unicamente segmento periodico
    ("a1",  0.351, 0.517), # todo el intervalo
    ("a2",  0.696, 0.885), # unicamente segmento periódico
    ("o",   1.122, 1.341), #unicamente segmento periódico 
    ("s",   1,341, 1.530), # todo el intervalo 
]

def graficar_fonemas(señal, sr, fonemas):
    for etiqueta, t_ini, t_fin in fonemas:
        idx_ini = int(t_ini * sr)
        idx_fin = int(t_fin * sr)
        segmento = señal[idx_ini:idx_fin]
        t = np.linspace(t_ini, t_fin, len(segmento))

        plt.figure(figsize=(12, 4))
        plt.plot(t, segmento)
        plt.title(f"Segmento_rapido [{etiqueta}]")
        plt.xlabel("Tiempo (s)")
        plt.ylabel("Amplitud")
        plt.tight_layout()
        plt.show()

graficar_fonemas(x_lento, sr, fonemas)

