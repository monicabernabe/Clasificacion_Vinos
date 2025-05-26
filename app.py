import streamlit as st
import pandas as pd
import numpy as np
import pickle 

# 1. Cargar el Modelo
try:
    with open('03_model/modelo_kmeans.pkl', 'rb') as f:
        kmeans_model = pickle.load(f)
except FileNotFoundError:
    st.error("Error: El archivo 'modelo_kmeans.pkl' no se encontró.")
    st.stop() # Detiene la ejecución de la app si no encuentra los archivos

# 2. Títulos y Descripción de la App
st.set_page_config(
    page_title="Clasificador de Vinos",
    page_icon="🍷",
    layout="centered"
)

st.title("🍷 Clasificador de Vinos por Características Químicas")
st.markdown("""
¡Descubre a qué tipo de clúster pertenece tu vino basándote en su composición química!
Introduce los valores de las 13 características y deja que el modelo haga la predicción.
""")

st.markdown("---")

# 3. Definición de las Características y sus Rangos
# (Estos rangos son solo de ejemplo basados en los datos del dataset, puedes ajustarlos)
features_info = {
    "Alcohol": {"label": "Alcohol (%)", "min": 11.0, "max": 15.0},
    "Malic_Acid": {"label": "Ácido Málico (g/L)", "min": 0.7, "max": 6.0},
    "Ash": {"label": "Ceniza (g/L)", "min": 1.3, "max": 3.5},
    "Ash_Alcanity": {"label": "Alcalinidad de la Ceniza (meq/L)", "min": 10.0, "max": 30.0},
    "Magnesium": {"label": "Magnesio (mg/L)", "min": 70.0, "max": 165.0},
    "Total_Phenols": {"label": "Fenoles Totales", "min": 1.0, "max": 4.0},
    "Flavanoids": {"label": "Flavonoides", "min": 0.3, "max": 5.0},
    "Nonflavanoid_Phenols": {"label": "Fenoles No Flavonoides", "min": 0.1, "max": 0.7},
    "Proanthocyanins": {"label": "Proantocianidinas", "min": 0.4, "max": 3.6},
    "Color_Intensity": {"label": "Intensidad del Color", "min": 1.0, "max": 13.0},
    "Hue": {"label": "Tono", "min": 0.4, "max": 1.7},
    "OD280": {"label": "OD280/OD315 de Vinos Diluidos", "min": 1.2, "max": 4.0},
    "Proline": {"label": "Prolina (mg/L)", "min": 200.0, "max": 1700.0}
    }

# 4. Definición de los Nombres y Descripciones de los Clústeres
cluster_names = {
        0: {"name": "Vinos Rosados Ligeros", "description": "Vinos con baja intensidad de color, menores niveles de fenoles y flavonoides, y una acidez más suave. Generalmente ligeros y frescos."},
        1: {"name": "Vinos Tintos Premium", "description": "Vinos con alta concentración de fenoles, flavonoides y proantocianidinas, indicativos de gran estructura, cuerpo, complejidad y potencial de envejecimiento. Son vinos de alta calidad."},
        2: {"name": "Vinos Blancos Ácidos", "description": "Vinos con alta concentración de alcohol y ácido málico, y la intensidad de color más baja. Suelen tener una acidez marcada, característicos de muchos vinos blancos."}
    }

# 5. Inputs del Usuario
st.header("🔬 Ingresa las Características Químicas del Vino")

input_values = {}
# Crear dos columnas para los inputs para una mejor disposición
col1, col2 = st.columns(2)

features_list = list(features_info.keys())
half_features = len(features_list) // 2

with col1:
    for i in range(half_features):
        feature = features_list[i]
        info = features_info[feature]
        input_values[feature] = st.number_input(
            label=info["label"],
            min_value=info["min"],
            max_value=info["max"],
            value=float(np.mean([info["min"], info["max"]])), # Valor inicial en el promedio
            step=0.01,
            format="%.3f"
        )

with col2:
    for i in range(half_features, len(features_list)):
        feature = features_list[i]
        info = features_info[feature]
        input_values[feature] = st.number_input(
            label=info["label"],
            min_value=info["min"],
            max_value=info["max"],
            value=float(np.mean([info["min"], info["max"]])), # Valor inicial en el promedio
            step=0.01,
            format="%.3f"
        )

# 6. Botón de Predicción
if st.button("Clasificar Vino", help="Haz clic para ver a qué clúster pertenece tu vino."):
    # Convertir los inputs a un DataFrame (K-Means espera un 2D array)
    input_df = pd.DataFrame([input_values])

    # Realizar la predicción
    predicted_cluster = kmeans_model.predict(input_df)[0]

        # Mostrar el resultado
    st.markdown("---")
    st.subheader("🎉 ¡Tu vino pertenece al Clúster Predicho!")
    st.markdown(f"### **{cluster_names[predicted_cluster]['name']}**")
    st.info(cluster_names[predicted_cluster]['description'])

st.markdown("---")

# 7. Sección "Acerca de"
st.sidebar.title("Acerca de este Clasificador")
st.sidebar.markdown("""
Esta aplicación utiliza un modelo de **Machine Learning no supervisado (K-Means)** para agrupar vinos según sus características químicas. El modelo ha sido entrenado para identificar 3 tipos distintos de perfiles de vino, que se correlacionan con diferentes cultivares.
""")

st.sidebar.subheader("Los 3 Clústeres de Vino:")
for i in sorted(cluster_names.keys()):
    st.sidebar.markdown(f"**Clúster {i}: {cluster_names[i]['name']}**")
    st.sidebar.markdown(f"_{cluster_names[i]['description']}_")
    st.sidebar.markdown("---")

st.sidebar.subheader("Detalle de las Características Químicas:")

# Diccionario con las descripciones COMPLETAS de las características
full_features_description = {
    "Alcohol": {
        "label": "Alcohol (%)",
        "desc": "Porcentaje de alcohol etílico presente en el vino. Contribuye al cuerpo, la textura y la sensación en boca. Un nivel adecuado es crucial para el equilibrio. Altos niveles pueden dar una sensación de calidez y plenitud, mientras que bajos niveles pueden resultar en un vino ligero y delgado."
    },
    "Malic_Acid": {
        "label": "Ácido Málico (g/L)",
        "desc": "Uno de los principales ácidos orgánicos de la uva. Aporta acidez y frescura al vino. Altos niveles pueden dar un sabor verde o áspero, mientras que bajos niveles resultan en un vino más suave. Su concentración disminuye con la maduración de la uva y la fermentación maloláctica."
    },
    "Ash": {
        "label": "Ceniza (g/L)",
        "desc": "Residuo inorgánico que queda después de quemar el vino, representando la concentración total de minerales. No es un indicador directo de calidad, pero su composición puede influir en el sabor y la estabilidad, aportando cierta complejidad mineral."
    },
    "Ash_Alcanity": {
        "label": "Alcalinidad de la Ceniza (meq/L)",
        "desc": "Mide la capacidad de la ceniza para neutralizar ácidos, relacionándose con la presencia de carbonatos. Influye indirectamente en la acidez percibida: una alta alcalinidad puede contrarrestar la acidez, dando una sensación más suave."
    },
    "Magnesium": {
        "label": "Magnesio (mg/L)",
        "desc": "Un mineral esencial que contribuye a diversas reacciones bioquímicas durante la fermentación. Puede influir en el sabor, afectando la percepción de amargor y mineralidad, y es importante para la salud de la levadura y la estabilidad del vino."
    },
    "Total_Phenols": {
        "label": "Fenoles Totales",
        "desc": "Grupo de compuestos químicos (incluye antocianos y taninos) presentes en la piel, semillas y tallos de la uva. Son cruciales para el color, la estructura (astringencia y cuerpo) y el potencial de envejecimiento del vino. Contribuyen también a su sabor y aroma (notas especiadas, ahumadas)."
    },
    "Flavanoids": {
        "label": "Flavonoides",
        "desc": "Un subgrupo de los fenoles e importantes antioxidantes. Contribuyen al color (especialmente en vinos blancos y rosados) y a la estructura del vino, afectando la astringencia y la estabilidad del color."
    },
    "Nonflavanoid_Phenols": {
        "label": "Fenoles No Flavonoides",
        "desc": "Otro subgrupo de los fenoles, que incluye ácidos fenólicos. Contribuyen al sabor, aroma y estabilidad del vino, aportando notas amargas y terrosas. Su influencia en el color es menor, pero pueden intervenir en la oxidación."
    },
    "Proanthocyanins": {
        "label": "Proantocianidinas",
        "desc": "Polímeros de flavonoides y un tipo específico de tanino. Son responsables de la astringencia (sensación de sequedad en boca) y contribuyen significativamente al cuerpo y la estructura del vino tinto. También juegan un papel en la polimerización de pigmentos y la estabilidad del color durante el envejecimiento."
    },
    "Color_Intensity": {
        "label": "Intensidad del Color",
        "desc": "Medida de la profundidad del color del vino, generalmente determinada espectrofotométricamente. En vinos tintos, una mayor intensidad se asocia a menudo con una mayor concentración de pigmentos y vinos más ricos y concentrados."
    },
    "Hue": {
        "label": "Tono",
        "desc": "Describe el matiz o la tonalidad del color del vino (ej. rojo rubí, púrpura, amarillo dorado). Proporciona información sobre la edad del vino (los tintos jóvenes tienden a tonos más púrpuras) y puede ser característico de ciertas variedades de uva."
    },
    "OD280": {
        "label": "OD280/OD315 de Vinos Diluidos",
        "desc": "Medida de la absorbancia óptica del vino diluido a 280 nm y 315 nm. Es un indicador del contenido fenólico total y la concentración de ciertos compuestos aromáticos. Un valor más alto generalmente indica una mayor concentración de compuestos fenólicos, que contribuyen al color y la estructura."
    },
    "Proline": {
        "label": "Prolina (mg/L)",
        "desc": "Un aminoácido presente en altas concentraciones en el vino. Aunque no contribuye directamente al sabor o aroma, su concentración puede estar relacionada con la variedad de uva, las condiciones de cultivo y el nivel de madurez de la uva."
    }
}

for feature_key in features_info.keys(): # Iteramos sobre las mismas claves que usamos para los inputs
    info = full_features_description[feature_key] # Obtenemos la descripción completa
    st.sidebar.markdown(f"**{info['label']}**")
    st.sidebar.markdown(f"_{info['desc']}_") # Usamos 'desc' para la descripción completa
    st.sidebar.markdown(f"- _Rango típico:_ {features_info[feature_key]['min']} - {features_info[feature_key]['max']}") # Usamos los rangos de features_info
    st.sidebar.markdown("")

st.sidebar.markdown("---")

st.sidebar.info("Desarrollado para el proyecto de clasificación de vinos.")