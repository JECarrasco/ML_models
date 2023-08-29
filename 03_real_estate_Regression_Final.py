import pandas as pd
import seaborn as sns
df = pd.read_csv(r"C:\Users\darfa\OneDrive\Escritorio\practica ds\df\real_estate.csv")

df.head()


import matplotlib.pyplot as plt
%matplotlib inline

### PREVISUALIZACION Y ANALISIS DE DATOS ###

# obtener la columna de etiquetas (La columna de etiquetas seran los precios de las casas)
label = df[df.columns[-1]]

# crear una figura para 2 subplots (2 rows, 1 column)
fig, ax = plt.subplots(2, 1, figsize = (9,12))

# trazar el histograma
ax[0].hist(label, bins=100)
ax[0].set_ylabel('Frequency')

# Agrega las lineas para el promedio, mediana y moda
ax[0].axvline(label.mean(), color='magenta', linestyle='dashed', linewidth=2)
ax[0].axvline(label.median(), color='cyan', linestyle='dashed', linewidth=2)

# trazar el boxplot   
ax[1].boxplot(label, vert=False)
ax[1].set_xlabel('Label')

# Agregar un titulo a la figura
fig.suptitle('Label Distribution')

# muestra la figura
fig.show()

# Podemos observar en el histograma que existen valores atipicos (outliers), tanto del extremo derecho como el izquierdo

### REMOVER LOS OUTLIERS ###

# solo dejamos el 70% de la df
df = df[df['price_per_unit']<70]
# obtener la columna de etiquetas
label = df[df.columns[-1]]

# Crear una figura para 2 subplots (2 rows, 1 column)
fig, ax = plt.subplots(2, 1, figsize = (9,12))

# Trazamos el histograma
ax[0].hist(label, bins=100)
ax[0].set_ylabel('Frequency')

# Agrega las lineas para el promedio, mediana y moda
ax[0].axvline(label.mean(), color='magenta', linestyle='dashed', linewidth=2)
ax[0].axvline(label.median(), color='cyan', linestyle='dashed', linewidth=2)

# Trazamos el boxplot   
ax[1].boxplot(label, vert=False)
ax[1].set_xlabel('Label')

# Agregamos un titulo a la figura
fig.suptitle('Label Distribution')

# Mostramos la figura
fig.show()

# Podemos observar que despues de dropear el 30% de la df extrema derecha, el histograma muestra una distribucion mas favorable de los datos


### OBSERVACION DE CORRELACIONES ###

# Trazamos un grafico de dispercion por cada columna entre la etiqueta y las demas columnas
for col in df[df.columns[0:-1]]:
    fig = plt.figure(figsize=(9, 6))
    ax = fig.gca()
    feature = df[col]
    correlation = feature.corr(label)
    plt.scatter(x=feature, y=label)
    plt.xlabel(col)
    plt.ylabel('Correlations')
    ax.set_title('Label vs ' + col + '- correlation: ' + str(correlation))
plt.show()

# Generamos un mapa de correlaciones para identificar si existe correlacion entre ambas columnas
corr = df.corr()
sns.heatmap(corr, cmap = 'Wistia', annot= True);
# Podemos ver que la mayor correlacion presente en los precios de casas son local_convenience_stores, latitud y longitud

# Caracteristicas categoricas

## Tambien podemos observar que tanto la columna transaction_date como local_convenience_stores son valores discretos, es por eso que seran tratados como atributos categoricos

# Trazamos un boxplot para la etiqueta con cada atributo categorico
for col in df[['transaction_date', 'local_convenience_stores']]:
    fig = plt.figure(figsize=(9, 6))
    ax = fig.gca()
    df.boxplot(column = 'price_per_unit', by = col, ax = ax)
    ax.set_title('Label by ' + col)
    ax.set_ylabel("Label Distribution by Categorical Variable")
plt.show()

### Separacion de las caracteristicas y la etiqueta, para luego ser dividida en df para entrenamiento y validacion ###
## Como la columna transaction_date no tiene valor en la prediccion sera omitida

from sklearn.model_selection import train_test_split

#Separamos los datos en x variable dependiente (todas las columnas excepto price_per_unit) y y variable independiente (solo la columna price_per_unit)
# Separamos las caracteristicas (Columna 1[house_age] al final con excepcion de una) y la etiqueta (ultima columna)
# Separate features (columns 1 [house_age] to the last but one) and labels (the last column)
X, y = df[df.columns[1:-1]].values, df[df.columns[-1]].values

# Split df 70%-30% into training set and test set
# Dividimos la df en 70% y 30% como set de entrenamiento y set de testeo
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=0)

# Imprimimos el resultado de la division
print ('Training Set: %d, rows\nTest Set: %d rows' % (X_train.shape[0], X_test.shape[0]))

### PREPROCESAMIENTO DE LA df Y ENTRENAMIENTO DEL MODELO ###
## Normalizacion de las caracteristicas para luego usar un randomforestregressor para entrenar el modelo

# Cargamos la libreria para el modelo
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.ensemble import RandomForestRegressor
import numpy as np

# Define preprocessing for numeric columns (scale them)
# Definir el preprocesamiento para columnas numéricas (escalarlas)
numeric_features = [0,1,3,4]
numeric_transformer = Pipeline(steps=[
    ('scaler', StandardScaler())])

# Combine preprocessing steps
# Combinar pasos de preprosesamiento

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
    ])

# Create preprocessing and training pipeline
# Crear el preprocesamiento y entrenar el pipeline
pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                           ('regressor', RandomForestRegressor())])


# fit the pipeline to train a linear regression model on the training set
# Ajustar el pipeline para entrenar el modelo de regresion lineal en un set de entrenamiento
model = pipeline.fit(X_train, (y_train))
print (model)

### EVALUACION DEL MODELO ###

# Cargamos la libreria para evaluar el modelo
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
%matplotlib inline

# Obtenemos las predicciones
predictions = model.predict(X_test)

# Mostramos las metricas
mse = mean_squared_error(y_test, predictions)
print("MSE:", mse)
rmse = np.sqrt(mse)
print("RMSE:", rmse)
r2 = r2_score(y_test, predictions)
print("R2:", r2)

# Trazamos las predicciones vs las reales
plt.scatter(y_test, predictions)
plt.xlabel('Actual Labels')
plt.ylabel('Predicted Labels')
plt.title('Predictions vs Actuals')
z = np.polyfit(y_test, predictions, 1)
p = np.poly1d(z)
plt.plot(y_test,p(y_test), color='magenta')
plt.show()

### USO DEL MODELO ENTRENADO ###
## Guardaremos el modelo de regresion para despues utilizarlo en futuras predicciones de price_per_unit

# Cargamos la libreria para guardar el modelo
import joblib

# Save the model as a pickle file

# Guardamos el modelo como un archivo Pickle
filename = './real_estate_model.pkl'
joblib.dump(model, filename)

# Cargamos el modelo dentro del archivo
loaded_model = joblib.load(filename)

# Generamos un array con los valores necesarios para generar la prediccion de precio (x) y omitiendo la transaction date
X_new = np.array([[16.2,289.3248,5,24.98203,121.54348],
                  [13.6,4082.015,0,24.94155,121.5038]])

# Usamos el modelo para predecir el price_per_unit
results = loaded_model.predict(X_new)
print('Predictions:')
for prediction in results:
    print(round(prediction,2))
    
# Podemos ver que se generaron 2 valores estos son los valores predecidos del precio por unidad.
# A medida que el modelo se vuelva mas preciso, las predicciones seran mas exactas.
# Podemos intentar practicar otros algoritmos de prediccion para ajustar mas la precision del modelo