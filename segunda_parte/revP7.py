import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf
import librosa
import librosa.display

# Configuración 
INPUT_FILE   = "lapachos_lento.wav"
OUTPUT_FILE  = "lapachos_modificado.wav"
PITCH_FACTOR = 0.7

FRAME_LENGTH = 2048
HOP_LENGTH   = 512

fs = 44100

# Carga y normalización de audio 
audio, _ = sf.read(INPUT_FILE)
if audio.ndim > 1:
    audio = np.mean(audio, axis=1)
audio = audio.astype(np.float64)
audio = audio / np.max(np.abs(audio))

# Detección automática de segmentos periódicos (voiced)
# Se corre YIN sobre toda la señal. Cada frame donde YIN devuelve un F0
# válido (no NaN) se considera voiced. Frames consecutivos voiced se
# agrupan en segmentos; se descartan segmentos muy cortos (< MIN_DURACION_SEG).

MIN_DURACION_SEG = 0.03 # segundos mínimos para considerar un segmento válido

f0_global = librosa.yin(
    audio,
    fmin=200, fmax=300,
    sr=fs,
    frame_length=FRAME_LENGTH,
    hop_length=HOP_LENGTH
)

# ── Umbral de energía por frame ───────────────────────────────────────────────
# Se calcula la energía RMS de cada frame con el mismo hop que el YIN global.
# Un frame es voiced solo si tiene F0 válido (YIN) Y energía suficiente.

UMBRAL_ENERGIA = 0.02  # ajustar si se incluye ruido o se cortan fonemas débiles

rms_global = librosa.feature.rms(
    y=audio,
    frame_length=FRAME_LENGTH,
    hop_length=HOP_LENGTH
)[0]

# Alinear longitudes (YIN y RMS pueden diferir en 1 frame)
n_frames     = min(len(f0_global), len(rms_global))
f0_global    = f0_global[:n_frames]
rms_global   = rms_global[:n_frames]

frames_voiced = ~np.isnan(f0_global) & (rms_global > UMBRAL_ENERGIA)



# Agrupar frames voiced consecutivos en segmentos (inicio, fin) en segundos
PERIODIC_SEGMENTS = []
en_segmento       = False

for idx_frame, es_voiced in enumerate(frames_voiced):

    tiempo_frame = idx_frame * HOP_LENGTH / fs

    if es_voiced and not en_segmento:
        # Inicio de un nuevo segmento voiced
        tiempo_inicio = tiempo_frame
        en_segmento   = True

    elif not es_voiced and en_segmento:
        # Fin del segmento voiced
        tiempo_fin  = tiempo_frame
        duracion    = tiempo_fin - tiempo_inicio
        en_segmento = False

        if duracion >= MIN_DURACION_SEG:
            PERIODIC_SEGMENTS.append((tiempo_inicio, tiempo_fin))

# Cerrar el último segmento si la señal termina en voiced
if en_segmento:
    tiempo_fin = len(audio) / fs
    duracion   = tiempo_fin - tiempo_inicio
    if duracion >= MIN_DURACION_SEG:
        PERIODIC_SEGMENTS.append((tiempo_inicio, tiempo_fin))


# Buffers de salida 
# output acumula las piezas solapadas; output_weights acumula las ventanas
# para luego normalizar el solapamiento.
output          = np.copy(audio)
output_weights  = np.ones_like(audio)
all_pitch_marks = []


# TD-PSOLA por segmento 
for start_time, end_time in PERIODIC_SEGMENTS:

    start_sample  = int(start_time * fs)
    end_sample    = int(end_time   * fs)
    segment_audio = audio[start_sample:end_sample]
    N             = len(segment_audio)

    # Paso 1: Estimación de F0 con YIN 
    # YIN devuelve un valor de F0 por cada hop dentro del segmento.
    f0 = librosa.yin(
        segment_audio,
        fmin=200, fmax=300,
        sr=fs,
        frame_length=FRAME_LENGTH,
        hop_length=HOP_LENGTH
    )

    valid_f0 = f0[~np.isnan(f0)]
    if len(valid_f0) == 0:
        continue

    f0_mediana = np.median(valid_f0)
    T0_medio   = int(fs / f0_mediana)  # período promedio en muestras


    # ── Paso 2: Detección de Pitch Marks ──────────────────────────────────────
    # Una pitch mark es el pico máximo de amplitud dentro de cada ciclo vocal.
    # Se ubica la primera mark en el primer ciclo (primeras T0_medio muestras).
    # Luego, para cada ciclo siguiente se busca el pico en una ventana centrada
    # en la posición esperada (mark_anterior + T0_local), con tolerancia ±T0/3.

    primera_mark = np.argmax(segment_audio[:T0_medio])
    pitch_marks  = [primera_mark]
    marca_actual = primera_mark

    while True:
        # F0 local en la posición actual (índice de frame YIN)
        frame_idx = min(int(marca_actual / HOP_LENGTH), len(f0) - 1)
        f0_local  = f0[frame_idx] if not np.isnan(f0[frame_idx]) else f0_mediana
        T0_local  = int(fs / f0_local)

        # Posición esperada del siguiente pico
        posicion_esperada = marca_actual + T0_local

        # Si el siguiente ciclo cae fuera del segmento, terminar
        if posicion_esperada >= N - T0_local:
            break

        # Ventana de búsqueda: ±1/3 del período alrededor de la posición esperada
        busqueda_inicio = posicion_esperada - T0_local // 3
        busqueda_fin    = min(posicion_esperada + T0_local // 3, N)

        if busqueda_inicio >= N or busqueda_inicio >= busqueda_fin:
            break

        # El pico dentro de la ventana es la nueva pitch mark
        pico_relativo = np.argmax(segment_audio[busqueda_inicio:busqueda_fin])
        nueva_mark    = busqueda_inicio + pico_relativo

        # Seguridad: si no avanzó, forzar posición esperada
        if nueva_mark <= marca_actual:
            nueva_mark = posicion_esperada

        pitch_marks.append(nueva_mark)
        marca_actual = nueva_mark

    pitch_marks = np.array(pitch_marks)

    if len(pitch_marks) < 3:
        continue

    # Guardar pitch marks en coordenadas globales (para el gráfico)
    all_pitch_marks.extend(pitch_marks + start_sample)


    # Paso 3: Calcular nuevas posiciones de síntesis
    # Para bajar el pitch se estiran los intervalos entre marks:
    # nuevo_intervalo = T0_local / PITCH_FACTOR  (PITCH_FACTOR < 1 → intervalo más largo)

    new_marks = [pitch_marks[0]]

    for i in range(1, len(pitch_marks)):
        frame_idx  = min(int(pitch_marks[i] / HOP_LENGTH), len(f0) - 1)
        f0_local   = f0[frame_idx] if not np.isnan(f0[frame_idx]) else f0_mediana
        T0_local   = int(fs / f0_local)

        intervalo_estirado = int(T0_local / PITCH_FACTOR)
        nueva_posicion     = new_marks[-1] + intervalo_estirado
        new_marks.append(nueva_posicion)

    new_marks = np.array(new_marks)

    # Limpiar la zona del segmento en el buffer de salida
    output        [start_sample:end_sample] = 0.0
    output_weights[start_sample:end_sample] = 0.0


    # ── Paso 4: Overlap-Add (OLA) ─────────────────────────────────────────────
    # Para cada nueva mark de síntesis:
    #   1. Encontrar la mark original más cercana → extraer el ciclo vocal original
    #   2. Escalar la ventana al nuevo período (PITCH_FACTOR < 1 → ventana más ancha)
    #   3. Aplicar ventana de Hanning al ciclo (suaviza bordes para evitar clicks)
    #   4. Sumar la pieza ventaneada en la posición de síntesis (overlap-add)

    for j in range(1, len(new_marks) - 1):

        # Pitch mark original más cercana a esta posición de síntesis
        distancias   = np.abs(pitch_marks - new_marks[j])
        idx_original = np.argmin(distancias)

        # Necesitamos vecinos para calcular el ancho de ventana → descartar extremos
        if idx_original == 0 or idx_original >= len(pitch_marks) - 1:
            continue

        mark_original = pitch_marks[idx_original]

        # Ancho de ventana izquierdo y derecho, escalados al nuevo período
        ancho_izq_original = mark_original - pitch_marks[idx_original - 1]
        ancho_der_original = pitch_marks[idx_original + 1] - mark_original

        ancho_izq_escalado = int(ancho_izq_original / PITCH_FACTOR)
        ancho_der_escalado = int(ancho_der_original / PITCH_FACTOR)

        # Extraer la pieza del segmento original centrada en mark_original
        inicio_extraccion = mark_original - ancho_izq_escalado
        fin_extraccion    = mark_original + ancho_der_escalado

        if inicio_extraccion < 0 or fin_extraccion >= N:
            continue

        pieza         = segment_audio[inicio_extraccion:fin_extraccion]
        ventana       = np.hanning(len(pieza))
        pieza_ventana = pieza * ventana

        # Posición de destino en el buffer de salida
        inicio_destino = (start_sample + new_marks[j]) - ancho_izq_escalado
        fin_destino    = inicio_destino + len(pieza_ventana)

        if inicio_destino < 0 or fin_destino >= len(output):
            continue

        # Acumular en el buffer (overlap-add)
        output        [inicio_destino:fin_destino] += pieza_ventana
        output_weights[inicio_destino:fin_destino] += ventana


# Normalización final 
# Dividir por los pesos acumulados para corregir el solapamiento,
# luego normalizar al pico máximo.
output_weights[output_weights < 1e-4] = 1.0
output = output / output_weights
output = output / np.max(np.abs(output))

sf.write(OUTPUT_FILE, output, fs)




# Gráficos 
t = np.arange(len(audio)) / fs
 
# PÁGINA 1: Señal original vs modificada en el tiempo
plt.figure(figsize=(14, 5))
plt.plot(t, audio,  label="Original",   alpha=0.7)
plt.plot(t, output, label="Modificada", alpha=0.7)
plt.title("Señal Original vs Modificada")
plt.xlabel("Tiempo [s]"); plt.ylabel("Amplitud")
plt.legend(); plt.grid()
 
# PÁGINA 2: Transformada de Fourier (FFT)
plt.figure(figsize=(14, 5))
freqs = np.fft.rfftfreq(len(audio), 1 / fs)
plt.plot(freqs, np.abs(np.fft.rfft(audio)),  label="FFT Original",   alpha=0.7)
plt.plot(freqs, np.abs(np.fft.rfft(output)), label="FFT Modificada", alpha=0.7)
plt.xlim(0, 4000)
plt.title("Transformada de Fourier")
plt.xlabel("Frecuencia [Hz]"); plt.ylabel("Magnitud")
plt.legend(); plt.grid()
 
# PÁGINA 3: Pitch marks detectadas y segmentos voiced
plt.figure(figsize=(14, 5))
plt.plot(t, audio, alpha=0.7, label="Audio Original")
all_pitch_marks = np.array(all_pitch_marks)
marcas_validas  = all_pitch_marks[all_pitch_marks < len(audio)]
plt.scatter(marcas_validas / fs, audio[marcas_validas], color='red', s=10, label='Pitch Marks')
for start_time, end_time in PERIODIC_SEGMENTS:
    plt.axvspan(start_time, end_time, alpha=0.2, color='gray')
plt.title("Pitch Marks Detectadas (segmentos voiced automáticos)")
plt.xlabel("Tiempo [s]"); plt.ylabel("Amplitud")
plt.legend(); plt.grid()
 

# espectrogramas
from scipy.signal import ShortTimeFFT, get_window
 
SPEC_W_ANCHO = 2500
SPEC_HOP     = SPEC_W_ANCHO // 4
 
ventana_spec = get_window('hann', SPEC_W_ANCHO)
SFT = ShortTimeFFT(
    win=ventana_spec,
    hop=SPEC_HOP,
    fs=fs,
    fft_mode='onesided',
    mfft=4096,
    phase_shift=0
)
 
# PÁGINA 4: Espectrograma Original
Sx_orig    = SFT.stft(audio)
Sx_orig_dB = 20 * np.log10(np.abs(Sx_orig) + 1e-10)
 
plt.figure(figsize=(10, 6))
plt.imshow(
    Sx_orig_dB,
    origin='lower', aspect='auto',
    extent=SFT.extent(len(audio)),
    vmin=-10, vmax=30,
    cmap='viridis'
)
plt.ylim(0, 4000)
plt.title("Espectrograma Original (Voz Femenina)")
plt.xlabel("Tiempo [s]"); plt.ylabel("Frecuencia [Hz]")
plt.colorbar(label='Magnitud [dB]')
 
# PÁGINA 5: Espectrograma Modificado
Sx_mod    = SFT.stft(output)
Sx_mod_dB = 20 * np.log10(np.abs(Sx_mod) + 1e-10)
 
plt.figure(figsize=(10, 6))
plt.imshow(
    Sx_mod_dB,
    origin='lower', aspect='auto',
    extent=SFT.extent(len(output)),
    vmin=-10, vmax=30,
    cmap='viridis'
)
plt.ylim(0, 4000)
plt.title("Espectrograma Modificada (Voz Masculina)")
plt.xlabel("Tiempo [s]"); plt.ylabel("Frecuencia [Hz]")
plt.colorbar(label='Magnitud [dB]')
 
plt.show()
