# Importamos la libreria a utilizar
import pandas as pd 
import numpy as np
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt
import joblib

# Cargamos el df
df = pd.read_csv(r"C:\Users\darfa\OneDrive\Escritorio\practica ds\data\Linear multiple\Student_Performance.csv")

# Hours studied - Horas dedicadas a estudiar
# Previous scores - Calificacion anterior
# Extracurricular activities - Actividades extracurriculares
# Sleep Hours - Horas de sueño
# Sample Question Papers Practiced - Pruebas de practica realizadas
# Performance index - Calificacion final

# Empezaremos por renombrar las columnas para mayor facilidad en el manejo de las mismas
df = df.rename(columns={'Hours Studied': 'HoursStud','Previous Scores': 'PScore','Extracurricular Activities': 'Extracurricular','Sleep Hours': 'Sleep','Sample Question Papers Practiced':'Samples','Performance Index':'Score'})

# Obtenemos el label
label = df[df.columns[-1]]

## Graficar la distribucion de los datos
%matplotlib inline

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
ax[1].set_xlabel('label')

# Agregar un titulo a la figura
fig.suptitle('label Distribution')

# muestra la figura
fig.show()

# Los datos tienen una distribucion perfecta
# Generamos un mapa de correlaciones
corr = df.corr()
sns.heatmap(corr, cmap = 'Wistia', annot= True);

# Creacion de variables dummy
# Creamos variables dummy para la columna Extracurricular
df_dummies = pd.get_dummies(df['Extracurricular'], prefix='Extracurricular')

# Concatenar las variables dummy al DataFrame original
df = pd.concat([df, df_dummies], axis=1)

# Eliminar la columna original categórica si lo deseas
df.drop('Extracurricular', axis=1, inplace=True)

# Graficamos la distribucion de las demas columnas
df.hist(figsize=(10, 8), bins=20)  
# Agregar titulo
plt.suptitle("Distribucion por columna", y=1.02, fontsize=16)
plt.tight_layout()  # Para evitar que se superpongan los títulos
#Mostramos la grafica
plt.show()

# =============================================================================
# Generamos el modelo de regresion lineal
# =============================================================================

#Movemos la columna Score al final del df
cm = 'Score'

# Asegúrate de que la columna exista en el DataFrame
if cm in df.columns:
    # Obtiene la lista de todas las columnas en el DataFrame
    columnas = df.columns.tolist()

    # Mueve la columna 'charges' al final de la lista
    columnas.remove(cm)
    columnas.append(cm)

    # Reorganiza las columnas del DataFrame con el orden deseado
    df = df[columnas]
    
#Separamos las columnas en variables independientes y variables dependientes
X, y = df[df.columns[0:-1]].values, df[df.columns[-1]].values

# Dividimos los datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=0)

# Imprimimos el valor de la division
print ('Training Set: %d, rows\nTest Set: %d rows' % (X_train.shape[0], X_test.shape[0]))

# Entrenamos el modelo
regression = LinearRegression()

#Con este codigo se entrena el modelo ingresando datos de entrenamiento
R_lineal = regression.fit(X_train, y_train)

#Generamos las predicciones
lineal_pred = R_lineal.predict(X_test)

# Calculamos las metricas del modelo
mse = mean_squared_error(y_test, lineal_pred)
print("MSE:", mse)
rmse = np.sqrt(mse)
print("RMSE:", rmse)
r2 = r2_score(y_test, lineal_pred)
print("R2:", r2)
mae = mean_absolute_error(y_test, lineal_pred)
print("MAE:", mae)

# Trazamos las predicciones vs las reales
plt.scatter(y_test, lineal_pred)
plt.xlabel('Actual Labels')
plt.ylabel('Predicted Labels')
plt.title('lineal_pred vs Actuals')
z = np.polyfit(y_test, lineal_pred, 1)
p = np.poly1d(z)
plt.plot(y_test,p(y_test), color='magenta')
plt.show()

# La R2 del modelo es muy alto lo que podria indicar un sobre ajuste, por lo que tenemos que realizar nuevas predicciones y comparar
# Es posible que exista colinealidad entre la variable dependiente PScore y la variable Independiente Score
# Por lo que un futuro paso sea dropear dicha columna

# =============================================================================
# Buscamos colinealidad
# =============================================================================

# Dropeamos la columna PScore
df = df[["PScore","Score"]].copy()

# Volvemos a entrenar el modelo de regresion lineal solo con la columna de PScore

#Separamos las columnas en variables independientes y variables dependientes
X, y = df[df.columns[0:-1]].values, df[df.columns[-1]].values

# Dividimos los datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=0)

# Imprimimos el valor de la division
print ('Training Set: %d, rows\nTest Set: %d rows' % (X_train.shape[0], X_test.shape[0]))

# Entrenamos el modelo
regression = LinearRegression()

#Con este codigo se entrena el modelo ingresando datos de entrenamiento
R_lineal = regression.fit(X_train, y_train)

#Generamos las predicciones
lineal_pred = R_lineal.predict(X_test)

# Calculamos las metricas del modelo
mse = mean_squared_error(y_test, lineal_pred)
print("MSE:", mse)
rmse = np.sqrt(mse)
print("RMSE:", rmse)
r2 = r2_score(y_test, lineal_pred)
print("R2:", r2)
mae = mean_absolute_error(y_test, lineal_pred)
print("MAE:", mae)

# Trazamos las predicciones vs las reales
plt.scatter(y_test, lineal_pred)
plt.xlabel('Actual Labels')
plt.ylabel('Predicted Labels')
plt.title('lineal_pred vs Actuals')
z = np.polyfit(y_test, lineal_pred, 1)
p = np.poly1d(z)
plt.plot(y_test,p(y_test), color='magenta')
plt.show()

# Podemos observar que solo la columna PScore como variable dependiente genera una R2 de .83 lo que la vuelve bastante significativa
