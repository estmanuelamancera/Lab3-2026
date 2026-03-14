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
