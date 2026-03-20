### INFORME DE LABORATORIO #3.
# ANALISIS ESPECTRAL DE LA VOZ

### DESCRIPCIÓN 
En este laboratorio se realizó el análisis espectral de señales de voz utilizando técnicas de procesamiento digital de señales. En la Parte A, se adquirieron seis grabaciones de voz correspondientes a tres hablantes masculinos y tres femeninos, quienes pronunciaron la frase “Sara cocina sopa sin sal cinco veces” durante aproximadamente cinco segundos. Las señales fueron almacenadas en formato .wav y posteriormente importadas en Python para su análisis en el dominio del tiempo y en el dominio de la frecuencia mediante la Transformada de Fourier (FFT). A partir de este procesamiento se calcularon características espectrales de cada señal, tales como frecuencia fundamental, frecuencia media, brillo (centroide espectral) e intensidad o energía.

En la Parte B, se realizó el análisis de estabilidad de la voz mediante el cálculo de jitter y shimmer, parámetros que permitieron evaluar las variaciones en la frecuencia fundamental y en la amplitud de la señal entre ciclos consecutivos. Estas medidas se utilizaron para analizar la estabilidad de la señal vocal y su comportamiento a lo largo del tiempo.

Finalmente, en la Parte C, se compararon los resultados obtenidos entre las diferentes grabaciones con el fin de identificar posibles diferencias entre las voces masculinas y femeninas y analizar el comportamiento de los parámetros espectrales y temporales calculados. Este análisis permitió observar cómo las características de la señal de voz pueden variar entre distintos hablantes y cómo pueden ser estudiadas mediante herramientas de procesamiento digital de señales.

### OBJETIVOS
Emplear técnicas de análisis espectral para la diferenciación o clasificación de señales de voz según el género. 
Capturar y procesar señales de voz masculinas y femeninas.
Aplicar la Transformada de Fourier como herramienta de análisis espectral de la voz.
Extraer parámetros característicos de la señal de voz: frecuencia fundamental, frecuencia media, brillo, intensidad, jitter y shimmer.
Comparar las diferencias principales entre señales de voz de hombres y mujeres a partir de su análisis en frecuencia.
Desarrollar conclusiones sobre el comportamiento espectral de la voz humana en función del género. 

###  PARTE A.
En la Parte A del laboratorio se realizó la adquisición y el análisis inicial de señales de voz. Para ello, se grabaron seis muestras de audio correspondientes a tres hablantes masculinos y tres femeninos, quienes pronunciaron la frase “Sara cocina sopa sin sal cinco veces” durante aproximadamente cinco segundos. Las grabaciones se almacenaron en formato .wav, procurando mantener condiciones de muestreo similares entre todas las señales.

Posteriormente, los archivos de audio fueron importados en Python para su procesamiento. En primer lugar, cada señal fue representada en el dominio del tiempo mediante gráficas de la forma de onda. Luego, se aplicó la Transformada de Fourier (FFT) con el fin de analizar el contenido frecuencial de las señales y obtener su espectro de magnitud.

A partir de este análisis se calcularon diferentes características espectrales de cada señal de voz, incluyendo frecuencia fundamental, frecuencia media, brillo (centroide espectral) e intensidad o energía. Estos parámetros permitieron describir el comportamiento de cada grabación y sirvieron como base para el análisis y comparación de las voces en las siguientes partes del laboratorio.

# DIAGRAMA
<img width="2000" height="5000" alt="image" src="https://github.com/estmanuelamancera/Lab3-2026/blob/main/INICIO%20(1).png" />

# CÓDIGO
## Conectar Google Drive
```
# ---------------------------------------------------------
# 1. CONECTAR GOOGLE DRIVE
# ---------------------------------------------------------
from google.colab import drive
drive.mount('/content/drive')
```
Permite que Google Colab acceda a Google Drive.
## Importar librerías
```
# ---------------------------------------------------------
# 2. IMPORTAR LIBRERIAS
# ---------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.fft import fft, fftfreq
import pandas as pd
```
Las librerias permiten trabajar con vectores hacer cálculos matemáticos, manejar señales digitales, graficar señales y leer archivos de audio.
La funcion scipy.fft sirve para calcular la Transformada de Fourier,convierte señal en tiempo en señal en frecuencia, etso permite saber que frecuencia tiene la voz.
## Importar librerías
```
# ---------------------------------------------------------
# 3. RUTA (TUS AUDIOS ESTAN EN MI UNIDAD)
# ---------------------------------------------------------
ruta = "/content/drive/MyDrive/"
```
Aquí se define la ubicación donde se encuentran almacenados los archivos de audio dentro de Google Drive. Esta ruta se utiliza posteriormente para acceder a cada uno de los archivos .wav que contienen las grabaciones de voz realizadas durante el laboratorio.
## Lista archivos de audio 
```
# ---------------------------------------------------------
# 4. ARCHIVOS DE AUDIO
# ---------------------------------------------------------
archivos = [
"hombre1.wav",
"hombre2.wav",
"hombre3.wav",
"mujer1.wav",
"mujer2.wav",
"mujer3.wav"
]
```
En esta sección se crea una lista con los nombres de los archivos de audio que serán analizados. Cada archivo corresponde a una grabación de voz realizada durante la práctica. Esta lista permite automatizar el análisis de las señales mediante un ciclo que procesa cada archivo de forma secuencial.
## Función de análisis de la señal
```
# ---------------------------------------------------------
# 5. FUNCION ANALISIS DE VOZ
# ---------------------------------------------------------
def analizar_voz(signal, fs):

    N = len(signal)

    # FFT
    yf = fft(signal)
    xf = fftfreq(N, 1/fs)

    magnitud = np.abs(yf)

    # solo frecuencias positivas
    xf = xf[:N//2]
    magnitud = magnitud[:N//2]

    # frecuencia fundamental
    f0 = xf[np.argmax(magnitud)]

    # frecuencia media
    f_media = np.sum(xf * magnitud) / np.sum(magnitud)

    # brillo
    brillo = f_media

    # energia
    energia = np.sum(signal**2) / N

    return f0, f_media, brillo, energia, xf, magnitud

```
En este bloque se define una función encargada de realizar el análisis espectral de cada señal de voz. Primero se obtiene el número total de muestras de la señal. Luego se aplica la Transformada Rápida de Fourier (FFT) para convertir la señal del dominio del tiempo al dominio de la frecuencia. A partir del espectro obtenido se calcula la magnitud de cada componente frecuencial, considerando únicamente las frecuencias positivas.

Posteriormente se determina la frecuencia fundamental, identificando la frecuencia asociada al mayor valor de magnitud del espectro. También se calcula la frecuencia media, que corresponde al promedio ponderado de las frecuencias según su magnitud. Este valor se utiliza como una aproximación del brillo o centroide espectral. Finalmente, se calcula la energía de la señal, definida como el promedio del cuadrado de la amplitud de la señal. La función devuelve estos parámetros junto con los vectores de frecuencia y magnitud necesarios para generar las gráficas.

## Procesamiento de los archivos de audio
```
# ---------------------------------------------------------
# 6. ANALIZAR LOS AUDIOS
# ---------------------------------------------------------
resultados = []

for archivo in archivos:

    fs, signal = wavfile.read(ruta + archivo)

    # convertir a mono si es estereo
    if len(signal.shape) > 1:
        signal = signal[:,0]

    # normalizar
    signal = signal / np.max(np.abs(signal))

    tiempo = np.arange(len(signal)) / fs

    f0, f_media, brillo, energia, xf, magnitud = analizar_voz(signal, fs)

    resultados.append([
        archivo,
        f0,
        f_media,
        brillo,
        energia
    ])

    # colores
    if "mujer" in archivo:
        color = "fuchsia"
    else:
        color = "blue"
```
En esta sección se realiza el procesamiento de cada una de las grabaciones de voz. Primero se crea una lista para almacenar los resultados obtenidos. Luego, mediante un ciclo, el programa recorre cada archivo de audio y lo carga utilizando la función de lectura de archivos .wav. Si la grabación es estéreo, se selecciona un solo canal para convertirla en una señal mono. Posteriormente, la señal se normaliza para mantener una escala de amplitud uniforme entre todas las grabaciones. Finalmente, se calcula el vector de tiempo y se aplica la función de análisis espectral para obtener las características de la señal, las cuales se almacenan en la lista de resultados.
## Gráfica de la señal en el dominio del tiempo
```
# -----------------------------------------------------
    # GRAFICA DOMINIO DEL TIEMPO
    # -----------------------------------------------------
    plt.figure(figsize=(10,4))
    plt.plot(tiempo, signal, color=color)
    plt.title("Señal de voz en el tiempo - " + archivo)
    plt.xlabel("Tiempo (segundos)")
    plt.ylabel("Amplitud")
    plt.grid()
    plt.show()
```
Este bloque genera la representación de cada señal de voz en el dominio del tiempo. La gráfica muestra cómo varía la amplitud de la señal a lo largo del tiempo, lo cual permite observar la forma de onda de la voz, su duración aproximada y posibles pausas o irregularidades en la grabación.
## Análisis de Calidad de la Señal y Relación Señal-Ruido (SNR)
```python
frame_length = int(0.02 * fs)
hop_length = int(0.01 * fs)

frames = []
energia = []

for i in range(0, len(x) - frame_length, hop_length):
    frame = x[i:i+frame_length]
    frames.append(frame)
    energia.append(np.sum(frame**2))

frames = np.array(frames)
energia = np.array(energia)

# --------------------------------------------
# Separación señal / ruido
# --------------------------------------------
umbral = np.percentile(energia, 40)

ruido_frames = energia < umbral
senal_frames = energia >= umbral

ruido = frames[ruido_frames].flatten()
senal = frames[senal_frames].flatten()

# --------------------------------------------
# SNR
# --------------------------------------------
snr = calcular_snr(senal, ruido)
print(f"SNR: {snr:.2f} dB")
```
Para el cálculo del SNR, se implementó un algoritmo basado en segmentación temporal de la señal, utilizando ventanas de 20 ms con un solapamiento del 50%. Para cada ventana, se estimó la energía como criterio de discriminación entre actividad vocal y ruido de fondo. Se definió un umbral adaptativo a partir del percentil 40 de la distribución de energía, lo que permitió clasificar automáticamente las tramas en segmentos de señal (alta energía) y ruido (baja energía). Finalmente, se calculó la relación señal-ruido (SNR) mediante la razón logarítmica entre las potencias promedio de la señal y del ruido, expresada en decibeles (dB).

| Sujeto | Género | SNR Original (Sin filtrar) | Clasificación de Calidad |
| :--- | :---: | :---: | :--- |
| Audio 1 (Tomas) | Masculino | **19.49 dB** | Óptima |
| Audio 2 | Masculino | **18.42 dB** | Buena |
| Audio 3 | Masculino | **18.40 dB** | Buena |
| Audio 4 | Femenino | **17.66 dB** | Buena |
| Audio 5 | Femenino | **14.83 dB** | Aceptable |
| Audio 6 | Femenino | **20.27 dB** | Excelente |
## Gráfica del espectro de frecuencias
```
# -----------------------------------------------------
    # GRAFICA FFT
    # -----------------------------------------------------
    plt.figure(figsize=(10,4))
    plt.plot(xf, magnitud, color=color)
    plt.title("Espectro de frecuencias (FFT) - " + archivo)
    plt.xlabel("Frecuencia (Hz)")
    plt.ylabel("Magnitud")
    plt.grid()
    plt.show()
```
En esta sección se genera la representación de la señal en el dominio de la frecuencia. La gráfica corresponde al espectro de magnitud obtenido mediante la Transformada de Fourier. Este espectro permite visualizar cómo se distribuye la energía de la señal en diferentes frecuencias y facilita la identificación de componentes dominantes como la frecuencia fundamental.

## Organización final de los resultados
```
# ---------------------------------------------------------
# 7. TABLA FINAL
# ---------------------------------------------------------
tabla = pd.DataFrame(
    resultados,
    columns=[
        "Archivo",
        "Frecuencia Fundamental (Hz)",
        "Frecuencia Media (Hz)",
        "Brillo",
        "Intensidad (Energia)"
    ]
)

```
Finalmente, los valores calculados para cada grabación se organizan en una tabla utilizando la librería Pandas. Esta tabla contiene el nombre del archivo analizado y los parámetros espectrales calculados: frecuencia fundamental, frecuencia media, brillo e intensidad. La presentación de los resultados en forma tabular facilita la comparación entre las distintas señales de voz y sirve como base para el análisis posterior entre voces masculinas y femeninas.

# GRAFICAS
<img width="800" height="700" alt="image" src="https://github.com/estmanuelamancera/Lab3-2026/blob/main/hombrea1.png" />
<img width="800" height="700" alt="image" src="https://github.com/estmanuelamancera/Lab3-2026/blob/main/hombrea2.png" />
<img width="1131" height="843" alt="image" src="https://github.com/user-attachments/assets/def7c341-1fa8-4c92-96e5-d3823221fa9b" />
<img width="800" height="700" alt="image" src="https://github.com/estmanuelamancera/Lab3-2026/blob/main/hombreb1.png" />
<img width="800" height="700" alt="image" src="https://github.com/estmanuelamancera/Lab3-2026/blob/main/hombreb2.png" />
<img width="1169" height="823" alt="image" src="https://github.com/user-attachments/assets/82a5f521-22da-4591-9c6c-6fc659a0252d" />
<img width="800" height="700" alt="image" src="https://github.com/estmanuelamancera/Lab3-2026/blob/main/hombrec1.png" />
<img width="800" height="700" alt="image" src="https://github.com/estmanuelamancera/Lab3-2026/blob/main/hombrec2.png" />
<img width="1142" height="826" alt="image" src="https://github.com/user-attachments/assets/0a13c6e9-2119-48e2-a1e8-bb83cbb0519a" />
<img width="800" height="700" alt="image" src="https://github.com/estmanuelamancera/Lab3-2026/blob/main/mujera1.png" />
<img width="800" height="700" alt="image" src="https://github.com/estmanuelamancera/Lab3-2026/blob/main/mujera2.png" />
<img width="1298" height="827" alt="image" src="https://github.com/user-attachments/assets/65d007f1-a2f1-4f76-894e-31f4cb7e9fcd" />
<img width="800" height="700" alt="image" src="https://github.com/estmanuelamancera/Lab3-2026/blob/main/mujerb1.png" />
<img width="800" height="700" alt="image" src="https://github.com/estmanuelamancera/Lab3-2026/blob/main/mujerb2.png" />
<img width="1114" height="807" alt="image" src="https://github.com/user-attachments/assets/1b51eb93-8360-453e-bebd-60f18c3e35d6" />
<img width="800" height="700" alt="image" src="https://github.com/estmanuelamancera/Lab3-2026/blob/main/mujerc1.png" />
<img width="800" height="700" alt="image" src="https://github.com/estmanuelamancera/Lab3-2026/blob/main/mujerc2.png" />
<img width="1078" height="807" alt="image" src="https://github.com/user-attachments/assets/f2161cfb-5222-4a0a-8b00-b6a4ba47d0c0" />


## TABLA RESULTADOS PARTE A
<img width="800" height="700" alt="image" src="https://github.com/estmanuelamancera/Lab3-2026/blob/main/tabla1.png" />

# PARTE B

En esta sección se realiza un análisis detallado de la estabilidad de la voz a partir de las grabaciones obtenidas en la Parte A. Se selecciona una muestra de voz masculina y una femenina, a las cuales se aplica un filtro pasa–banda dentro del rango típico de frecuencias de cada género (80–400 Hz para hombres y 150–500 Hz para mujeres) con el fin de eliminar componentes de ruido no deseados.
Posteriormente, se evalúan las variaciones temporales y de amplitud de las señales de voz mediante el cálculo del Jitter (fluctuación en la frecuencia fundamental entre ciclos consecutivos) y el Shimmer (variación en la amplitud pico a pico).
### Diagrama flujo 
![Infografía de periódico moderno ordenado colorido (1)](https://github.com/user-attachments/assets/fcf2b43d-49e2-4ff2-a347-d251d2542a89)


### Diseño del filtro pasa banda para hombres

![PB HOMBRE](https://github.com/estmanuelamancera/Lab3-2026/blob/main/WhatsApp%20Image%202026-03-17%20at%201.12.00%20PM.jpeg?raw=true)

![PB HOMBRE](https://github.com/estmanuelamancera/Lab3-2026/blob/main/WhatsApp%20Image%202026-03-17%20at%201.12.18%20PM.jpeg?raw=true)
### Diseño del filtro pasa banda para mujeres

![PB MUJER](https://github.com/estmanuelamancera/Lab3-2026/blob/main/WhatsApp%20Image%202026-03-17%20at%201.12.34%20PM.jpeg?raw=true)

### Código filtro pasabanda

```python
import numpy as np
import librosa
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, find_peaks

# --- 1. CARGA DEL ARCHIVO Y SEGMENTACIÓN ---
archivo = 'Tomas.m4a.wav'  # <--- CAMBIA ESTO por el nombre de tu archivo
fs = 44100
t_inicio, t_fin = 2.3, 2.8  # Intervalo de 0.5 segundos

# Cargamos el audio completo primero para definir 'y_full'
try:
    y_full, _ = librosa.load(archivo, sr=fs, mono=True)
    print(" Archivo cargado correctamente.")
except Exception as e:
    print(f" Error al cargar el archivo: {e}")
    y_full = None

if y_full is not None:
    start_sample = int(t_inicio * fs)
    end_sample = int(t_fin * fs)
    y_segmento = y_full[start_sample:end_sample]

    # --- 2. FILTRO BUTTERWORTH ORDEN 3 ---
    nyq = 0.5 * fs
    low, high = 80 / nyq, 400 / nyq
    b, a = butter(3, [low, high], btype='bandpass')

    y_filtrada = filtfilt(b, a, y_segmento)
    y_norm = y_filtrada / np.max(np.abs(y_filtrada))

    # --- 3. DETECCIÓN DE PICOS ROBUSTA ---
    distancia_min = int(fs / 180)
    altura_min = 0.5
    prominencia_min = 0.2

    picos, _ = find_peaks(y_norm,
                          distance=distancia_min,
                          height=altura_min,
                          prominence=prominencia_min)

    # --- 4. CÁLCULOS ---
    tiempos_picos = (picos + start_sample) / fs
    if len(picos) > 2:
        Ti = np.diff(tiempos_picos)
        f0_real = 1 / np.mean(Ti)
        jitter_rel = (np.mean(np.abs(np.diff(Ti))) / np.mean(Ti)) * 100

        # --- 5. GRÁFICA ---
        plt.figure(figsize=(15, 6))
        t_eje = np.linspace(t_inicio, t_fin, len(y_norm))

        plt.plot(t_eje, y_norm, color='teal', label='Señal Filtrada')
        plt.plot(t_eje[picos], y_norm[picos], "ro", markersize=8, label='Picos detectados')

        plt.title(f'Análisis 0.5s | F0: {f0_real:.2f} Hz | Jitter: {jitter_rel:.4f}%')
        plt.xlabel('Tiempo (segundos)')
        plt.ylabel('Amplitud Normalizada')

        # MOSTRAR TODO EL RANGO (2.3 a 2.8)
        plt.xlim(t_inicio, t_fin+0.1)

        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()

        print(f"Resultados: F0 = {f0_real:.2f} Hz, Jitter = {jitter_rel:.4f}%")

        # --- CÁLCULO DE PARÁMETROS SOLICITADOS ---

# 1. Calcular los periodos Ti (en segundos)
# tiempos_picos ya contiene la ubicación en tiempo de cada punto rojo
Ti = np.diff(tiempos_picos) 

# 2. Calcular el Jitter Absoluto (Promedio de las diferencias entre periodos)
# np.diff(Ti) nos da (T2-T1), (T3-T2), etc.
jitter_abs = np.mean(np.abs(np.diff(Ti)))

# 3. Calcular el Jitter Relativo (%)
periodo_promedio = np.mean(Ti)
jitter_rel = (jitter_abs / periodo_promedio) * 100

# --- MOSTRAR RESULTADOS PARA EL INFORME ---
print(f"{' CÁLCULOS DE JITTER ':=^40}")
print(f"Primeros 5 periodos (Ti): {Ti[:5]} segundos")
print(f"Número total de periodos: {len(Ti)}")
print("-" * 40)
print(f"JITTER ABSOLUTO: {jitter_abs:.6f} segundos")
print(f"JITTER RELATIVO: {jitter_rel:.4f} %")
print(f"{'':=^40}")

```

### Filtro señal masculina

![Hombre](https://github.com/estmanuelamancera/Lab3-2026/blob/main/filtro%20en%20hombre%20.png?raw=true)

### Filtro señal femenina

![Mujer](https://github.com/estmanuelamancera/Lab3-2026/blob/main/filtro%20mujer.png?raw=true)

### Medición del jitter

### Código de captura Jitter

```python

import numpy as np
import librosa
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, find_peaks

# --- 1. CARGA DEL ARCHIVO Y SEGMENTACIÓN ---
archivo = 'Tomas.m4a.wav'  
fs = 44100
t_inicio, t_fin = 2.3, 2.8  # Intervalo de 0.5 segundos

# Cargamos el audio completo primero para definir 'y_full'
try:
    y_full, _ = librosa.load(archivo, sr=fs, mono=True)
    print(" Archivo cargado correctamente.")
except Exception as e:
    print(f" Error al cargar el archivo: {e}")
    y_full = None

if y_full is not None:
    start_sample = int(t_inicio * fs)
    end_sample = int(t_fin * fs)
    y_segmento = y_full[start_sample:end_sample]

    # --- 2. FILTRO BUTTERWORTH ORDEN 3 ---
    nyq = 0.5 * fs
    low, high = 80 / nyq, 400 / nyq
    b, a = butter(3, [low, high], btype='bandpass')

    y_filtrada = filtfilt(b, a, y_segmento)
    y_norm = y_filtrada / np.max(np.abs(y_filtrada))

    # --- 3. DETECCIÓN DE PICOS ROBUSTA ---
    distancia_min = int(fs / 180)
    altura_min = 0.5
    prominencia_min = 0.2

    picos, _ = find_peaks(y_norm,
                          distance=distancia_min,
                          height=altura_min,
                          prominence=prominencia_min)

    # --- 4. CÁLCULOS ---
    tiempos_picos = (picos + start_sample) / fs
    if len(picos) > 2:
        Ti = np.diff(tiempos_picos)
        f0_real = 1 / np.mean(Ti)
        jitter_rel = (np.mean(np.abs(np.diff(Ti))) / np.mean(Ti)) * 100

        # --- 5. GRÁFICA ---
        plt.figure(figsize=(15, 6))
        t_eje = np.linspace(t_inicio, t_fin, len(y_norm))

        plt.plot(t_eje, y_norm, color='teal', label='Señal Filtrada')
        plt.plot(t_eje[picos], y_norm[picos], "ro", markersize=8, label='Picos detectados')

        plt.title(f'Análisis 0.5s | F0: {f0_real:.2f} Hz | Jitter: {jitter_rel:.4f}%')
        plt.xlabel('Tiempo (segundos)')
        plt.ylabel('Amplitud Normalizada')

        # MOSTRAR TODO EL RANGO (2.3 a 2.8)
        plt.xlim(t_inicio, t_fin+0.1)

        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()

        print(f"Resultados: F0 = {f0_real:.2f} Hz, Jitter = {jitter_rel:.4f}%")

        # --- CÁLCULO DE PARÁMETROS SOLICITADOS ---

# 1. Calcular los periodos Ti (en segundos)
# tiempos_picos ya contiene la ubicación en tiempo de cada punto rojo
Ti = np.diff(tiempos_picos)

# 2. Calcular el Jitter Absoluto (Promedio de las diferencias entre periodos)
# np.diff(Ti) nos da (T2-T1), (T3-T2), etc.
jitter_abs = np.mean(np.abs(np.diff(Ti)))

# 3. Calcular el Jitter Relativo (%)
periodo_promedio = np.mean(Ti)
jitter_rel = (jitter_abs / periodo_promedio) * 100

# --- MOSTRAR RESULTADOS PARA EL INFORME ---
print(f"{' CÁLCULOS DE JITTER ':=^40}")
print(f"Primeros 5 periodos (Ti): {Ti[:5]} segundos")
print(f"Número total de periodos: {len(Ti)}")
print("-" * 40)
print(f"JITTER ABSOLUTO: {jitter_abs:.6f} segundos")
print(f"JITTER RELATIVO: {jitter_rel:.4f} %")
print(f"{'':=^40}")

```
### Resultados obtenidos

![Grafica jitter picos ](https://github.com/estmanuelamancera/Lab3-2026/blob/main/CAPTURA%20JITTER%20Y%20PICOS.png?raw=true)

========== CÁLCULOS DE JITTER ==========

Primeros 5 periodos (Ti): [0.00834467 0.00834467 0.00825397 0.00829932 0.00823129] segundos
Número total de periodos: 8

==================================

JITTER ABSOLUTO: 0.000045 segundos
JITTER RELATIVO: 0.5472 %
FRECUENCIA FUNDAMENTAL: 120.66 Hz

### Medición del shimmer 
### Código captura shimmer

```python
# --- CÁLCULO DE SHIMMER ---

# 1. Obtener las amplitudes Ai de cada pico detectado
# y_norm[picos] nos da la altura de cada punto rojo
Ai = y_norm[picos]

# 2. Shimmer Absoluto (en dB)
# Comparamos la razón entre amplitudes consecutivas en escala logarítmica
shimmer_db = np.mean(np.abs(20 * np.log10(Ai[1:] / Ai[:-1])))

# 3. Shimmer Relativo (%)
# Diferencia promedio de amplitud entre picos consecutivos
diff_amp = np.mean(np.abs(np.diff(Ai)))
shimmer_rel = (diff_amp / np.mean(Ai)) * 100

# --- RESULTADOS ---
print(f"{' CÁLCULOS DE SHIMMER ':=^40}")
print(f"Amplitudes detectadas (Ai): {Ai[:5]}") # Primeras 5 alturas
print("-" * 40)
print(f"SHIMMER ABSOLUTO: {shimmer_db:.4f} dB")
print(f"SHIMMER RELATIVO: {shimmer_rel:.4f} %")
print(f"{'':=^40}")

# --- 5. GRÁFICA DE VALIDACIÓN DE SHIMMER ---
plt.figure(figsize=(12, 5))

# Graficamos la señal de fondo en gris para que resalten los picos
t_eje = np.linspace(t_inicio, t_fin, len(y_norm))
plt.plot(t_eje, y_norm, color='lightgray', alpha=0.5, label='Señal de Voz')

# Graficamos los picos detectados
plt.plot(t_eje[picos], Ai, 'ro', markersize=8, label='Amplitudes (Ai)')

# Dibujamos la línea que une los picos (Envolvente de amplitud)
plt.plot(t_eje[picos], Ai, 'r--', alpha=0.6, label='Variación de Amplitud (Shimmer)')

# Configuración de la gráfica
plt.title(f'Análisis de Shimmer | Shimmer Rel: {shimmer_rel:.4f}% | Shimmer Abs: {shimmer_db:.4f} dB', fontsize=14)
plt.xlabel('Tiempo (segundos)')
plt.ylabel('Amplitud Normalizada')
plt.legend(loc='upper right')
plt.grid(True, linestyle=':', alpha=0.6)

# Mantenemos el zoom para ver el detalle de los primeros ciclos
plt.xlim(t_inicio, t_inicio + 0.1)
plt.ylim(0, 1.1) # Enfocamos en la parte positiva para ver las amplitudes

plt.tight_layout()
plt.show()

import numpy as np
import matplotlib.pyplot as plt

# 1. Extraer las amplitudes de los picos que ya detectamos
Ai = y_norm[picos]

# 2. Calcular la diferencia absoluta en dB entre picos consecutivos
# Fórmula: |20 * log10(A_{i+1} / A_i)|
shimmer_ciclo_a_ciclo_db = np.abs(20 * np.log10(Ai[1:] / Ai[:-1]))

# 3. El Shimmer Absoluto final es el promedio de esas diferencias
shimmer_abs_final = np.mean(shimmer_ciclo_a_ciclo_db)

# --- GRAFICAR EL SHIMMER ABSOLUTO ---
plt.figure(figsize=(10, 5))

# Graficamos las diferencias de cada par de picos
plt.stem(range(len(shimmer_ciclo_a_ciclo_db)), shimmer_ciclo_a_ciclo_db,
         linefmt='g-', markerfmt='go', basefmt='k-')

# Dibujamos una línea roja que represente el promedio (el valor final)
plt.axhline(y=shimmer_abs_final, color='r', linestyle='--',
            label=f'Shimmer Abs Promedio: {shimmer_abs_final:.4f} dB')

plt.title('Variación de Amplitud Ciclo a Ciclo (Shimmer Absoluto)')
plt.xlabel('Número de transición entre picos')
plt.ylabel('Diferencia de Amplitud (dB)')
plt.legend()
plt.grid(alpha=0.3)
plt.show()

print(f"Tu Shimmer Absoluto es: {shimmer_abs_final:.4f} dB")

```
### Resultados obtenidos 

![Shimmer](https://github.com/estmanuelamancera/Lab3-2026/blob/main/grafica%20shimmer.png?raw=true)

![Variacion amplitud](https://github.com/estmanuelamancera/Lab3-2026/blob/main/variacion%20amplitud%20shimmer.png?raw=true)


SHIMMER ABSOLUTO: 0.4989 dB
SHIMMER RELATIVO: 5.7470 %

#### Resultados Hombre y Mujer
### 📊 Resultados del Análisis Acústico (Jitter y Shimmer)

| Sujeto | Jitter Absoluto (s) | Jitter Relativo (%) | Shimmer Absoluto (dB) | Shimmer Relativo (%) |
| :--- | :---: | :---: | :---: | :---: |
| **Hombre ** | 0.000045 | 0.5472% | 0.4989 | 5.7470% |
| **Mujer ** | 0.064180 | 157.6787% | 1.3133 | 16.0415% |

## PARTE C 

# 1. Registro de Adquisición
   
| Parámetro                    | Valor/Descripción                          | Observaciones                                      |
|-----------------------------|--------------------------------------------|---------------------------------------------------|
| Frecuencia de muestreo (Fs) | 44100 Hz                                   | Definida en el código con librosa.load()          |
| Resolución                  | 16 bits (formato WAV)                      | Estándar en audio digital                         |
| Canales                     | Mono (1 canal)                             | Señal convertida a mono (mono=True)               |
| Duración analizada          | 0.5 segundos (2.3 s – 2.8 s)               | Segmento seleccionado para análisis               |
| Duración total              | ~3 segundos                                | Depende de la grabación original                  |
| Condiciones ambientales     | Controladas                                | Sin ruido significativo en la grabación           |
| Calidad general             | Buena                                      | Señal sin saturación, adecuada para análisis      |

Las grabaciones se realizaron en condiciones controladas, obteniendo una buena calidad de señal sin presencia significativa de ruido ambiental ni saturación. Se evidenció variabilidad en la duración y en la amplitud de las muestras, lo cual puede atribuirse a diferencias en la intensidad vocal y en la ejecución de cada hablante. Con el fin de mantener cierta homogeneidad en la adquisición, la distancia al micrófono se mantuvo aproximadamente constante en 10 cm desde la fuente de emisión. Adicionalmente, se seleccionaron segmentos específicos de la señal y se aplicó un filtrado pasa-banda para mejorar la calidad del análisis y la extracción de características.
# ¿Qué diferencias se observan en la frecuencia fundamental?
Se observó que la frecuencia fundamental en la voz masculina es menor en comparación con la voz femenina, lo cual coincide con el comportamiento esperado debido a diferencias fisiológicas en las cuerdas vocales. Sin embargo, en el caso de la señal femenina, la estimación de los parámetros presenta inconsistencias asociadas al proceso de detección de picos, lo que afecta la precisión de los cálculos derivados.
# 2. Análisis Comparativo: Voces Masculinas vs Femeninas

| Sujeto | Jitter Absoluto (s) | Jitter Relativo (%) | Shimmer Absoluto (dB) | Shimmer Relativo (%) |
| :--- | :---: | :---: | :---: | :---: |
| **Hombre ** | 0.000045 | 0.5472% | 0.4989 | 5.7470% |
| **Mujer ** | 0.064180 | 157.6787% | 1.3133 | 16.0415% |

# ¿Qué otras diferencias notan en términos de brillo, media o intensidad?
En términos de brillo, frecuencia media, mediana e intensidad, se observan diferencias entre las señales de voz analizadas. La voz femenina presenta un mayor brillo, lo cual está asociado a una mayor concentración de energía en altas frecuencias, en concordancia con su mayor frecuencia fundamental. Por el contrario, la voz masculina muestra un espectro dominado por componentes de baja frecuencia, lo que se traduce en un menor brillo.

En relación con la frecuencia media y la mediana espectral, se espera que ambas sean mayores en la voz femenina, debido a la mayor presencia de componentes de alta frecuencia. Esto indica que la energía de la señal femenina está distribuida hacia frecuencias más elevadas, mientras que en la voz masculina se concentra en rangos más bajos, reflejando diferencias fisiológicas en la producción vocal.

En cuanto a la intensidad, esta no presenta una diferencia claramente definida entre ambos casos, ya que depende en gran medida de la forma de emisión durante la grabación. No obstante, el análisis del shimmer evidencia variaciones en la amplitud de la señal; en particular, la señal femenina presenta valores elevados, lo que sugiere una mayor inestabilidad en la amplitud. Sin embargo, estos resultados pueden estar influenciados por el proceso de detección de picos, lo cual afecta la estimación precisa de este parámetro.

# Conclusiones sobre el comportamiento de la voz en hombres y mujeres a partir de los análisis realizados.

A partir de los resultados obtenidos, se evidencia que la señal masculina presenta valores de jitter (0.5472%) y shimmer (5.7470%) dentro de rangos cercanos a los esperados para una voz estable, lo que indica una vibración relativamente uniforme tanto en periodo como en amplitud. En contraste, la señal femenina muestra valores significativamente elevados (jitter de 157.6787% y shimmer de 16.0415%), los cuales no son consistentes con el comportamiento fisiológico normal de la voz humana.

Esta discrepancia sugiere que el cálculo de los parámetros en la señal femenina se vio afectado por el proceso de detección de picos, probablemente debido a una selección inadecuada de parámetros frente a una frecuencia fundamental más alta. Esto resalta la sensibilidad de los métodos de análisis temporal ante cambios en las características de la señal.

En términos prácticos, los resultados confirman que la estimación de parámetros como jitter y shimmer depende críticamente del preprocesamiento, particularmente del filtrado y la detección de ciclos. Por lo tanto, es necesario adaptar los parámetros del algoritmo según el tipo de voz analizada para garantizar mediciones confiables.
# Importancia clínica del jitter y shimmer en el análisis de la voz.
Los parámetros jitter y shimmer son fundamentales en el análisis clínico de la voz, ya que permiten evaluar la estabilidad de la vibración de las cuerdas vocales desde un enfoque cuantitativo. El jitter mide las variaciones ciclo a ciclo en la frecuencia, mientras que el shimmer cuantifica las variaciones en la amplitud de la señal vocal. Estos parámetros son ampliamente utilizados en la práctica clínica para la detección y caracterización de trastornos vocales.

Diversos estudios han demostrado que valores elevados de jitter y shimmer están asociados con alteraciones en la calidad vocal, como disfonías funcionales y patologías laríngeas. En particular, el jitter ha sido identificado como un indicador sensible para diferenciar tipos de trastornos vocales [1]. Asimismo, el análisis conjunto de jitter y shimmer ha mostrado utilidad para identificar pacientes con disfonía y evaluar la efectividad de tratamientos terapéuticos [2].

Además, estos parámetros presentan relevancia en el análisis de enfermedades neurológicas. En patologías como la enfermedad de Parkinson, se han observado variaciones significativas en jitter y shimmer, lo que permite utilizarlos como biomarcadores en el monitoreo de la progresión de la enfermedad [3]. Sin embargo, es importante considerar que estos parámetros pueden verse afectados por condiciones de medición como la intensidad vocal, por lo que se requiere un adecuado control experimental para garantizar resultados confiables [4].

En conjunto, el jitter y el shimmer constituyen herramientas esenciales en el análisis acústico de la voz, proporcionando información objetiva sobre la estabilidad fonatoria y contribuyendo tanto al diagnóstico como al seguimiento de alteraciones vocales.

### CONCLUSIONES
En esta práctica se analizaron características espectrales y temporales de señales de voz masculina y femenina, permitiendo identificar diferencias relevantes asociadas tanto a factores fisiológicos como al procesamiento digital aplicado. Se evidenció que la frecuencia fundamental y los parámetros espectrales como la media y la mediana permiten diferenciar de manera clara entre ambos tipos de voz. Asimismo, el análisis de jitter y shimmer permitió evaluar la estabilidad de la señal vocal, destacando la importancia de estos parámetros en el estudio de la calidad de la voz. Sin embargo, también se comprobó que estos indicadores son altamente sensibles al preprocesamiento, lo que resalta la necesidad de ajustar adecuadamente los parámetros de filtrado y detección para obtener resultados confiables.

Finalmente, esta práctica demuestra la utilidad del procesamiento digital de señales en aplicaciones reales como el análisis biomédico de la voz, el reconocimiento automático de hablantes y el diagnóstico de patologías vocales. Estas herramientas permiten extraer información relevante a partir de señales acústicas, constituyendo una base fundamental para desarrollos en ingeniería biomédica y procesamiento de audio.

## REFERENCIAS  
[1] D. M. Bless, G. M. Hirano and T. Feder, “Videostroboscopic evaluation of the larynx,” Ear and Hearing, vol. 6, no. 4, pp. 194–200, 1985.

[2] M. A. Maryn, P. Roy, B. De Bodt, P. Van Cauwenberge and G. Corthals, “Acoustic measurement of overall voice quality: A meta-analysis,” Journal of the Acoustical Society of America, vol. 126, no. 5, pp. 2619–2634, 2009.

[3] J. R. Orozco-Arroyave et al., “Characterization methods for the detection of multiple voice disorders: Neurological, functional, and laryngeal diseases,” IEEE Journal of Biomedical and Health Informatics, vol. 19, no. 6, pp. 1820–1828, 2015.

[4] J. Wolfe and R. J. Garnier, “Effects of vocal intensity on measurements of jitter and shimmer,” Journal of Voice, vol. 28, no. 4, pp. 412–420, 2014.

