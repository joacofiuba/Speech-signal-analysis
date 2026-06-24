import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf
import librosa
import librosa.display

# config:
INPUT_FILE = "lapachos_lento.wav" 
OUTPUT_FILE = "lapachos_modificado.wav" 
PITCH_FACTOR = 0.7  # Factor requerido (Mujer -> Hombre)
PERIODIC_SEGMENTS = [(0.493, 0.827), (0.827, 1.267), (1.480, 1.906), (2.152, 2.521)]
FRAME_LENGTH, HOP_LENGTH = 2048, 64  

audio, fs = sf.read(INPUT_FILE)
if len(audio.shape) > 1: audio = np.mean(audio, axis=1)
audio = audio.astype(np.float64) / np.max(np.abs(audio))

# Inicialización global para Overlap-Add
output = np.copy(audio)
output_weights = np.ones_like(audio)
all_pitch_marks = []


#TD-PSOLA:
for start_time, end_time in PERIODIC_SEGMENTS:
    start_sample, end_sample = int(start_time * fs), int(end_time * fs)
    segment_audio = audio[start_sample:end_sample]

    # 1. Estimación de Pitch con Yin
    f0 = librosa.yin(segment_audio, fmin=70, fmax=350, sr=fs, frame_length=FRAME_LENGTH, hop_length=HOP_LENGTH)
    valid_f0 = f0[~np.isnan(f0)]
    if len(valid_f0) == 0: continue
    mean_f0 = np.median(valid_f0)
    T0_mean = int(fs / mean_f0)

    # 2. Análisis: Detección de Pitch Marks ciclo por ciclo
    pitch_marks = [np.argmax(segment_audio[:T0_mean])]
    current_mark = pitch_marks[0]
    
    while current_mark < len(segment_audio) - T0_mean:
        frame_idx = min(int(current_mark / HOP_LENGTH), len(f0)-1)
        current_f0 = f0[frame_idx] if not np.isnan(f0[frame_idx]) else mean_f0
        T0_local = int(fs / current_f0)
        
        # Buscar el pico exacto del siguiente ciclo vocal
        search_start = (current_mark + T0_local) - T0_local // 3
        search_end = min((current_mark + T0_local) + T0_local // 3, len(segment_audio))
        if search_start >= len(segment_audio) or search_start == search_end: break
        
        next_peak = search_start + np.argmax(segment_audio[search_start:search_end])
        if next_peak <= current_mark: next_peak = current_mark + T0_local
        pitch_marks.append(next_peak)
        current_mark = next_peak

    pitch_marks = np.array(pitch_marks)
    if len(pitch_marks) < 3: continue
    all_pitch_marks.extend(pitch_marks + start_sample)

    # 3. Síntesis: Calcular nuevas posiciones estiradas (Bajar el tono)
    new_marks = [pitch_marks[0]]
    for i in range(1, len(pitch_marks)):
        frame_idx = min(int(pitch_marks[i] / HOP_LENGTH), len(f0)-1)
        current_f0 = f0[frame_idx] if not np.isnan(f0[frame_idx]) else mean_f0
        new_marks.append(new_marks[-1] + int((fs / current_f0) / PITCH_FACTOR))
    new_marks = np.array(new_marks)

    # Limpiar la zona del tramo para la resíntesis
    output[start_sample:end_sample], output_weights[start_sample:end_sample] = 0.0, 0.0

    # 4. Overlap-Add con ventana escalada (Engrosamiento de timbre)
    for j in range(1, len(new_marks) - 1):
        idx_original = np.argmin(np.abs(pitch_marks - new_marks[j]))
        if idx_original == 0 or idx_original >= len(pitch_marks) - 1: continue
        old_mark = pitch_marks[idx_original]
        
        # Escalar límites de la ventana para adaptar al nuevo período
        T0_left_sc = int((old_mark - pitch_marks[idx_original - 1]) / PITCH_FACTOR)
        T0_right_sc = int((pitch_marks[idx_original + 1] - old_mark) / PITCH_FACTOR)
        
        start_old, end_old = old_mark - T0_left_sc, old_mark + T0_right_sc
        if start_old < 0 or end_old >= len(segment_audio): continue
        
        piece_windowed = segment_audio[start_old:end_old] * np.hanning(end_old - start_old)
        start_new = (start_sample + new_marks[j]) - T0_left_sc
        end_new = start_new + len(piece_windowed)
        
        if start_new >= 0 and end_new < len(output):
            output[start_new:end_new] += piece_windowed
            output_weights[start_new:end_new] += np.hanning(end_old - start_old)

# Normalización final por solapamiento y volumen peak
output_weights[output_weights < 1e-4] = 1.0
output = (output / output_weights) / np.max(np.abs(output / output_weights))
sf.write(OUTPUT_FILE, output, fs)



# graficos
t = np.arange(len(audio)) / fs

# PAGINA 1: Señal original vs modificada en el tiempo
plt.figure(figsize=(14, 5))
plt.plot(t, audio, label="Original", alpha=0.7)
plt.plot(t, output, label="Modificada", alpha=0.7)
plt.title("Señal Original vs Modificada"); plt.xlabel("Tiempo [s]"); plt.ylabel("Amplitud"); plt.legend(); plt.grid()

# PAGINA 2: Transformada de Fourier (FFT)
plt.figure(figsize=(14, 5))
freqs = np.fft.rfftfreq(len(audio), 1/fs)
plt.plot(freqs, np.abs(np.fft.rfft(audio)), label="FFT Original", alpha=0.7)
plt.plot(freqs, np.abs(np.fft.rfft(output)), label="FFT Modificada", alpha=0.7)
plt.xlim(0, 4000); plt.title("Transformada de Fourier"); plt.xlabel("Frecuencia [Hz]"); plt.ylabel("Magnitud"); plt.legend(); plt.grid()

# PAGINA 3: Pitch marks detectadas
plt.figure(figsize=(14, 5))
plt.plot(t, audio, alpha=0.7, label="Audio Original")
all_pitch_marks = np.array(all_pitch_marks)
plt.scatter(all_pitch_marks / fs, audio[all_pitch_marks[all_pitch_marks < len(audio)]], color='red', s=10, label='Pitch Marks')
for start_time, end_time in PERIODIC_SEGMENTS: plt.axvspan(start_time, end_time, alpha=0.2, color='gray')
plt.title("Pitch Marks Detectadas"); plt.xlabel("Tiempo [s]"); plt.ylabel("Amplitud"); plt.legend(); plt.grid()

# PAGINA 4: Espectrograma Original
plt.figure(figsize=(14, 5))
D_orig = librosa.amplitude_to_db(np.abs(librosa.stft(audio, n_fft=FRAME_LENGTH, hop_length=HOP_LENGTH)), ref=np.max)
librosa.display.specshow(D_orig, sr=fs, hop_length=HOP_LENGTH, x_axis='time', y_axis='hz',cmap='viridis')
plt.colorbar(format='%+2.0f dB'); plt.ylim(0, 4000); plt.title("Espectrograma Original (Voz Femenina)")

# PAGINA 5: Espectrograma Modificado
plt.figure(figsize=(14, 5))
D_mod = librosa.amplitude_to_db(np.abs(librosa.stft(output, n_fft=FRAME_LENGTH, hop_length=HOP_LENGTH)), ref=np.max)
librosa.display.specshow(D_mod, sr=fs, hop_length=HOP_LENGTH, x_axis='time', y_axis='hz',cmap='viridis')
plt.colorbar(format='%+2.0f dB'); plt.ylim(0, 4000); plt.title("Espectrograma Modificada (Voz Masculina)")

plt.show()
