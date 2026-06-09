import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf
import librosa
from scipy.signal import find_peaks

# =========================================================
# CONFIGURACION
# =========================================================

INPUT_FILE = "lapachos_lento.wav" 
OUTPUT_FILE = "lapachos_modificado.wav" 

# ---------------------------------------------------------
# FACTOR DE CAMBIO DE PITCH
# Mujer -> Hombre = 0.7
# Hombre -> Mujer = 1.4
# ---------------------------------------------------------

PITCH_FACTOR = 0.7

# =========================================================
# TRAMOS PERIODICOS MANUALES
# =========================================================
#
# FORMATO:
# (inicio_en_segundos, fin_en_segundos)
#
# EJEMPLO:
# hay dos "a" y una "o"
#
# =========================================================

PERIODIC_SEGMENTS = [
    (0.493, 0.827),# L

    (0.827, 1.267),   # primera "a"

    (1.480, 1.906),   # segunda "a"

    (2.152, 2.521),   # "o"

]

# =========================================================
# PARAMETROS DSP
# =========================================================

FRAME_LENGTH = 2048
HOP_LENGTH = 80

# =========================================================
# CARGAR AUDIO
# =========================================================

audio, fs = sf.read(INPUT_FILE)

# Stereo -> Mono
if len(audio.shape) > 1:
    audio = np.mean(audio, axis=1)

audio = audio.astype(np.float64)

# Normalizar
audio = audio / np.max(np.abs(audio))

# =========================================================
# SALIDA FINAL
# =========================================================

output = np.copy(audio)

# =========================================================
# LISTAS PARA GRAFICOS
# =========================================================

all_pitch_marks = []

# =========================================================
# PROCESAR CADA TRAMO PERIODICO
# =========================================================

for segment_idx, (start_time, end_time) in enumerate(PERIODIC_SEGMENTS):

    print(f"\nProcesando tramo {segment_idx+1}")

    # -----------------------------------------------------
    # CONVERTIR TIEMPO -> MUESTRAS
    # -----------------------------------------------------

    start_sample = int(start_time * fs)
    end_sample = int(end_time * fs)

    segment_audio = audio[start_sample:end_sample]

    # =====================================================
    # DETECCION DE PITCH CON YIN
    # =====================================================

    f0 = librosa.yin(
        segment_audio,
        fmin=80,
        fmax=300,
        sr=fs,
        frame_length=FRAME_LENGTH,
        hop_length=HOP_LENGTH
    )

    # =====================================================
    # VOICED / UNVOICED
    # =====================================================

    voiced = ~np.isnan(f0)

    f0_clean = np.copy(f0)

    for i in range(len(f0_clean)):

        if np.isnan(f0_clean[i]):
            f0_clean[i] = 0

    # =====================================================
    # DETECCION DE PITCH MARKS
    # =====================================================

    pitch_marks = []

    for i in range(len(f0_clean)):

        if not voiced[i]:
            continue

        pitch = f0_clean[i]

        if pitch <= 0:
            continue

        # periodo en muestras

        T0 = int(fs / pitch)

        center = int(i * HOP_LENGTH)

        # -------------------------------------------------
        # BUSQUEDA LOCAL ALREDEDOR DEL PITCH ESPERADO
        # -------------------------------------------------

        search_start = max(0, center - T0 // 2)
        search_end = min(len(segment_audio),
                         center + T0 // 2)

        local_segment = segment_audio[
            search_start:search_end
        ]

        if len(local_segment) == 0:
            continue

        peaks, _ = find_peaks(local_segment)

        if len(peaks) == 0:
            continue

        # pico mas fuerte

        best_peak = peaks[
            np.argmax(
                np.abs(local_segment[peaks])
            )
        ]

        pitch_mark = search_start + best_peak

        pitch_marks.append(pitch_mark)

    pitch_marks = np.array(
        sorted(list(set(pitch_marks)))
    )

    # =====================================================
    # GUARDAR MARKS PARA GRAFICOS
    # =====================================================

    global_marks = pitch_marks + start_sample

    all_pitch_marks.extend(global_marks)

    # =====================================================
    # CREAR NUEVAS POSICIONES
    # =====================================================

    new_marks = [pitch_marks[0]]

    for i in range(1, len(pitch_marks)):

        original_distance = (
            pitch_marks[i] -
            pitch_marks[i - 1]
        )

        new_distance = int(
            original_distance / PITCH_FACTOR
        )

        new_marks.append(
            new_marks[-1] + new_distance
        )

    new_marks = np.array(new_marks)

    # =====================================================
    # INICIALIZAR CON AUDIO ORIGINAL
    # =====================================================
    #
    # Las partes unvoiced permanecen intactas
    #
    # =====================================================

    processed_segment = np.copy(segment_audio)

    # =====================================================
    # TD-PSOLA
    # =====================================================

    for i in range(
        1,
        min(len(pitch_marks)-1,
            len(new_marks)-1)
    ):

        # -------------------------------------------------
        # SOLO PROCESAR FRAMES VOICED
        # -------------------------------------------------

        frame_idx = int(pitch_marks[i] / HOP_LENGTH)

        if frame_idx >= len(voiced):
            continue

        if not voiced[frame_idx]:
            continue

        old_mark = pitch_marks[i]
        new_mark = new_marks[i]

        # -------------------------------------------------
        # PERIODOS LOCALES
        # -------------------------------------------------

        T0_left = (
            pitch_marks[i] -
            pitch_marks[i - 1]
        )

        T0_right = (
            pitch_marks[i + 1] -
            pitch_marks[i]
        )

        left = int(T0_left)
        right = int(T0_right)

        # -------------------------------------------------
        # EXTRAER SEGMENTO
        # -------------------------------------------------

        start_old = old_mark - left
        end_old = old_mark + right

        if start_old < 0 or end_old >= len(segment_audio):
            continue

        piece = segment_audio[start_old:end_old]

        # -------------------------------------------------
        # VENTANA HANN
        # -------------------------------------------------

        window = np.hanning(len(piece))

        piece_windowed = piece * window

        # -------------------------------------------------
        # OVERLAP ADD
        # -------------------------------------------------

        start_new = new_mark - left
        end_new = start_new + len(piece_windowed)

        if start_new < 0 or end_new >= len(processed_segment):
            continue

        # -------------------------------------------------
        # REEMPLAZAR SOLO REGION VOICED
        # -------------------------------------------------

        processed_segment[
            start_new:end_new
        ] = piece_windowed

    # =====================================================
    # NORMALIZAR
    # =====================================================

    if np.max(np.abs(processed_segment)) > 0:

        processed_segment = (
            processed_segment /
            np.max(np.abs(processed_segment))
        )

    # =====================================================
    # REEMPLAZAR EN AUDIO FINAL
    # =====================================================

    output[start_sample:end_sample] = processed_segment[
        :len(segment_audio)
    ]

# =========================================================
# NORMALIZACION FINAL
# =========================================================

output = output / np.max(np.abs(output))

# =========================================================
# GUARDAR AUDIO
# =========================================================

sf.write(OUTPUT_FILE, output, fs)

print("\nAudio guardado como:", OUTPUT_FILE)

# =========================================================
# GRAFICO 1
# ORIGINAL VS MODIFICADA
# =========================================================

t = np.arange(len(audio)) / fs

plt.figure(figsize=(14,6))

plt.plot(
    t,
    audio,
    label="Original",
    alpha=0.7
)

plt.plot(
    t,
    output,
    label="Modificada",
    alpha=0.7
)

plt.title("Señal Original vs Modificada")
plt.xlabel("Tiempo [s]")
plt.ylabel("Amplitud")
plt.legend()
plt.grid()

# =========================================================
# GRAFICO 2
# FFT
# =========================================================

freqs = np.fft.rfftfreq(len(audio), 1/fs)

fft_original = np.abs(np.fft.rfft(audio))
fft_output = np.abs(np.fft.rfft(output))

plt.figure(figsize=(14,6))

plt.plot(
    freqs,
    fft_original,
    label="FFT Original"
)

plt.plot(
    freqs,
    fft_output,
    label="FFT Modificada"
)

plt.xlim(0, 4000)

plt.title("Transformada de Fourier")
plt.xlabel("Frecuencia [Hz]")
plt.ylabel("Magnitud")
plt.legend()
plt.grid()

# =========================================================
# GRAFICO 3
# PITCH MARKS
# =========================================================

plt.figure(figsize=(14,6))

plt.plot(t, audio)

all_pitch_marks = np.array(all_pitch_marks)

valid_marks = all_pitch_marks[
    all_pitch_marks < len(audio)
]

plt.scatter(
    valid_marks / fs,
    audio[valid_marks],
    color='red',
    s=10,
    label='Pitch Marks'
)

plt.title("Pitch Marks Detectadas")
plt.xlabel("Tiempo [s]")
plt.ylabel("Amplitud")
plt.legend()
plt.grid()

# =========================================================
# MARCAR TRAMOS PERIODICOS
# =========================================================

for start_time, end_time in PERIODIC_SEGMENTS:

    plt.axvspan(
        start_time,
        end_time,
        alpha=0.2
    )

plt.show()
