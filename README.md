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
