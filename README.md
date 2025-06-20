# **INFORME DEL PROYECTO DE CLASIFICACIÓN DE VINOS EN BASE A CARACTERÍTICAS QUÍMICAS.**

## **Introducción:**

El objetivo de este proyecto es clasificar diferentes muestras de vino en distintos grupos basándose en la similitud de sus 13 componentes químicos (Alcohol, Ácido málico, Ceniza, Alcalinidad de la ceniza, Magnesio, Fenoles totales, Flavonoides, Fenoles no flavonoides, Proantocianinas, Intensidad del color, Tono, OD280/OD315 de vinos diluidos 1  y Prolina).

## **Descripción de los datos:**
Los datos utilizados en este proyecto provienen de Kaggle y están disponibles públicamente para su uso.

[Dataset en Kaggle](https://www.kaggle.com/datasets/harrywang/wine-dataset-for-clustering).

Este conjunto de datos contiene información sobre características químicas de los vinos.

### **Descripción de las columnas:**

- **Alcohol**: Porcentaje de alcohol etílico presente en el vino (expresado como volumen por volumen). El alcohol contribuye al cuerpo, la textura y la sensación en boca del vino. 

- **Malic acid (Ácido málico)**: Uno de los principales ácidos orgánicos encontrados en las uvas. Su concentración disminuye durante la maduración de la uva y la fermentación maloláctica.

- **Ash (Ceniza)**: Residuo inorgánico que queda después de quemar el vino. Representa la concentración total de minerales presentes.

- **Alcalinity of ash (Alcalinidad de la ceniza)**: Medida de la capacidad de la ceniza para neutralizar ácidos. Se relaciona con la presencia de carbonatos y otros compuestos alcalinos.

- **Magnesium (Magnesio)**: Un mineral esencial presente en las uvas y, por lo tanto, en el vino. Contribuye a diversas reacciones bioquímicas durante la fermentación y puede influir en el sabor.

- **Total phenols (Fenoles totales)**: Grupo de compuestos químicos presentes en la piel, las semillas y los tallos de la uva. Incluyen antocianos (pigmentos rojos), taninos y otros compuestos fenólicos. Los fenoles son cruciales para el color, la estructura (taninos que aportan astringencia y cuerpo) y el potencial de envejecimiento del vino. También contribuyen al sabor y aroma (notas especiadas, ahumadas, etc.).

- **Flavanoids (Flavonoides)**: Un subgrupo de los fenoles, importantes antioxidantes que contribuyen al color (especialmente en vinos blancos y rosados) y a la estructura del vino (taninos, aunque en menor medida que otros fenoles). Influyen en el color, la astringencia y el potencial antioxidante del vino.

- **Nonflavanoid phenols (Fenoles no flavonoides)**: Otro subgrupo de los fenoles, que incluye ácidos fenólicos y otros compuestos. Contribuyen al sabor, aroma y estabilidad del vino. Algunos pueden tener efectos antioxidantes.

- **Proanthocyanins (Proantocianidinas)**: Polímeros de flavonoides, un tipo específico de tanino que se encuentra en la piel y las semillas de la uva. Son responsables de la astringencia (sensación de sequedad en boca) y contribuyen al cuerpo y la estructura del vino tinto. También juegan un papel en la polimerización de los pigmentos y la estabilidad del color durante el envejecimiento.

- **Color intensity (Intensidad del color)**: Medida de la profundidad del color del vino, generalmente determinada espectrofotométricamente. En vinos tintos, una mayor intensidad de color a menudo se asocia con una mayor concentración de pigmentos y, potencialmente, con vinos más ricos y concentrados. En vinos blancos y rosados, la intensidad del color puede indicar la variedad de uva, la edad o el método de producción.

- **Hue (Tono)**: Describe el matiz o la tonalidad del color del vino (por ejemplo, rojo rubí, púrpura, amarillo dorado). El tono puede dar información sobre la edad del vino (los tintos jóvenes tienden a tener tonos más púrpuras, mientras que los más viejos pueden tener tonos teja o ladrillo; los blancos jóvenes pueden ser amarillo pálido con reflejos verdosos, mientras que los más evolucionados tienden a ser dorados). También puede ser característico de ciertas variedades de uva.

- **OD280/OD315 of diluted wines**: Medida de la absorbancia óptica del vino diluido a dos longitudes de onda específicas (280 nm y 315 nm) utilizando un espectrofotómetro. Esta medida se utiliza a menudo como un indicador del contenido fenólico total y la concentración de ciertos compuestos aromáticos. Un valor más alto generalmente indica una mayor concentración de compuestos fenólicos, que contribuyen al color, la estructura y el potencial antioxidante del vino.

- **Proline**: Un aminoácido presente en altas concentraciones en el vino. Su concentración puede estar relacionada con la variedad de uva, las condiciones de cultivo y el nivel de madurez de la uva. Algunos estudios sugieren una posible correlación indirecta con la calidad en ciertos casos.

## **Tecnologías Utilizadas**

- **Python**
- **Pandas**
- **Numpy**
- **Matplotlib**
- **Seaborn**
- **Scikit-learn**
- **Streamlit**

## **Análisis Exploratorio de Datos (EDA) y Preprocesamiento**

Se realizó un análisis para identificar correlaciones entre las variables y detectar patrones que pudieran influir en la predicción de los precios de las viviendas.

## **Estrategia de Modelado y Evaluación**

Para abordar este problema de aprendizaje no supervisado, se implementó una estrategia comparativa para evaluar el rendimiento de seis algoritmos de clustering fundamentales.

### **Modelos entrenados:**

- **K-Means**: Algoritmo de particionamiento que divide los datos en k grupos, minimizando la variabilidad dentro de cada cluster. Es eficiente y adecuado para clusters de forma esférica y tamaño similar.

- **DBSCAN (Density-Based Spatial Clustering of Applications with Noise)**: Algoritmo basado en densidad que identifica clusters de forma arbitraria y detecta automáticamente los puntos atípicos (ruido).

- **Mean-Shift**: Algoritmo también basado en densidad que busca modos o picos en la distribución de los datos. No requiere especificar el número de clusters, pero no es escalable en grandes volúmenes de datos.

- **Agglomerative Clustering (Clustering Jerárquico Aglomerativo)**: Construye una jerarquía de clusters fusionando los más cercanos en cada paso. Permite explorar estructuras jerárquicas y utilizar diferentes métricas de distancia.

- **HDBSCAN (Hierarchical Density-Based Spatial Clustering of Applications with Noise)**: Variante jerárquica de DBSCAN que permite detectar clusters de densidades variables y formas no esféricas. Es robusto ante ruido, no requiere especificar el número de clusters y genera una jerarquía para extraer la mejor partición basada en estabilidad.

- **Spectral Clustering**: Algoritmo que emplea la descomposición espectral de un grafo de similitudes para identificar estructuras complejas en los datos. Resulta eficaz cuando los clusters no tienen forma esférica y requieren capturar relaciones no lineales entre los puntos.

Para cada uno de estos modelos exploró el impacto de distintas estrategias de preprocesamiento sobre la calidad del agrupamiento.

### **Fases de experimentación:**
1. **Entrenamiento sin preprocesamiento**.
  Se aplicaron todos los algoritmos excepto Mean-Shift, ya que éste requiere de datos normalizados, directamente sobre los datos originales, sin tratar outliers, sesgos ni normalizar. Esto estableció una línea base para evaluar la sensibilidad de cada modelo frente a la escala y distribución original.

2. **Entrenamiento con normalización.**
  Se escalaron los datos con StandardScaler antes de aplicar los algoritmos. Esta etapa evaluó si un escalado uniforme mejora los resultados de los modelos sensibles a las distancias.

3. **Entrenamiento con tratamiento de outliers, sesgo y normalización.**
  Se aplicó un pipeline de preprocesamiento completo (detección y tratamiento de valores extremos, corrección del sesgo, normalización). Esto permitió probar si una distribución más simétrica y a escala mejora significativamente la calidad del clustering.

### **Métricas de Evaluación**:

Se utilizaron las siguientes métricas para evaluar el rendimiento de los algoritmos:

- **Silhouette Coefficient**: Evalúa la cohesión interna de los clusters frente a su separación. Valores próximos a 1 indican agrupamientos bien definidos.

- **Davies-Bouldin Index**: Mide la similitud entre cada cluster y el más cercano. Cuanto menor, mejor separados y más compactos son los clusters.

- **Índice de Calinski-Harabasz**: Evalúa la proporción entre la dispersión inter-cluster e intra-cluster. Valores altos indican una buena estructura de agrupamiento.

**Ajuste de Hiperparámetros y Evaluación Final**
Tras identificar los modelos y estrategias de preprocesamiento más prometedores, se ajustarán los hiperparámetros mediante búsqueda aleatoria (por ejemplo, k en K-Means o eps y min_samples en DBSCAN).

## **Resultados y Conclusiones**

El modelo **K-Means con 3 clústeres** fue el modelo seleccionado. Esta decisión se tomó priorizando la realidad del dominio y la utilidad práctica del modelo.

### **Métricas finales:**

- **Silhouette Score** = 0.573
- **Davies-Bouldin Score** = 0.531
- **Calinski-Harabasz Score** = 542.65

### **Caracterización de los clusters:**
Se han obtenido tres grupos principales con las siguientes características:

- **Clúster 0: Vinos Rosados Ligeros**: Vinos con baja intensidad de color, bajos fenoles y flavonoides, pero con una acidez suavizada por la alcalinidad. Esto encaja con rosados ligeros o incluso algunos blancos muy pálidos.

- **Clúster 1: Vinos Tintos Premium**: Vinos con perfil más robusto, ricos en fenoles, flavonoides y con buen potencial de envejecimiento. El alcohol es promedio, pero la concentración de compuestos de calidad es muy alta.

-**Clúster 2: Vinos Blancos Ácidos**: La combinación de alta acidez málica, alto alcohol y la intensidad de color más baja de los tres clústeres sugiere fuertemente un vino blanco con una acidez marcada, o quizás un rosado muy pálido y ácido.

## **Archivos del Proyecto:**

- **00 - wine-clustering.csv**
- **cargar_datos.py**: Código para cargar los datos.
- **model_training.py**: Código para el entrenamiento del modelo.
- **modelo_kmeans.pkl**: modelo entrenado.

## Ejecución Local

1. Clona el repositorio:
   git clone https://github.com/monicabernabe/Clasificacion_Vinos/tree/main
   
2. Instala las dependencias:
  pip install -r requirements.txt

3. Ejecuta la app:
  streamlit run app.py

## Autora
Mónica Bernabé
Ingeniera Técnica Industrial | Aseguramiento de calidad | Consultora en Validación de Sistemas | Ciencia de Datos