import pandas as pd 
import numpy as np
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt
import joblib

# Cargamos el df
df = pd.read_csv(r"C:\Users\darfa\OneDrive\Escritorio\practica ds\data\score.csv")
## En este df tenemos 2 columnas la primera son las horas que el alumno dedico al estudio antes del examen, la segunda es la calificacion final
## El objetivo es crear un modelo de regresion que pueda predecir la calificacion del estudiante en base a las horas dedicadas de estudio

# Generamos un mapa de correlaciones
corr = df.corr()
sns.heatmap(corr, cmap = 'Wistia', annot= True);
## podemos observar que existe una fuerte correlacion entre la calificacion y las horas de estudio


### DISTRIBUCION DE DATOS ###
    # Generamos un boxplot para conocer la distribucion de los datos
def show_distribution(var_data):
    import matplotlib.pyplot as plt

    
# Obtenemos las estadisticas
    min_val = var_data.min()
    max_val = var_data.max()
    mean_val = var_data.mean()
    med_val = var_data.median()
    mod_val = var_data.mode()[0]

    print(var_data.name,'\nMinimum:{:.2f}\nMean:{:.2f}\nMedian:{:.2f}\nMode:{:.2f}\nMaximum:{:.2f}\n'.format(min_val,
                                                                                            mean_val,
                                                                                            med_val,
                                                                                            mod_val,
                                                                                            max_val))

    # Creamos una figura para dos subplots (2 filas y 1 columna)
    fig, ax = plt.subplots(2, 1, figsize = (10,4))

    # Trazamos el histograma   
    ax[0].hist(var_data)
    ax[0].set_ylabel('Frequency')

    # Agregamos lineas para el promedio, mediana y moda, ademas del valor min y max
    ax[0].axvline(x=min_val, color = 'gray', linestyle='dashed', linewidth = 2)
    ax[0].axvline(x=mean_val, color = 'cyan', linestyle='dashed', linewidth = 2)
    ax[0].axvline(x=med_val, color = 'red', linestyle='dashed', linewidth = 2)
    ax[0].axvline(x=mod_val, color = 'yellow', linestyle='dashed', linewidth = 2)
    ax[0].axvline(x=max_val, color = 'gray', linestyle='dashed', linewidth = 2)

    # Trazamos el boxplot
    ax[1].boxplot(var_data, vert=False)
    ax[1].set_xlabel('Value')

    # Le agregamos un titulo a la figura
    fig.suptitle(var_data.name)

    # Mostramos la figura
    fig.show()
    
#Insertamos las columnas para el boxplot
delayFields = ['Hours','Scores']
for col in delayFields:
    show_distribution(df[col])
    
    
### SEPARACION DE LOS DATOS ###

# Seleccionamos la etiqueta que sera la variable a predecir
label = df[df.columns[-1]]

#Separamos los datos en Variable independiente X y dependiente Y
X, y = df[df.columns[0:-1]].values, df[df.columns[-1]].values

#Separamos los datos en entrenamiento y testeo en un 70% y 30%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=0)

# Imprimimos el resultado de la division
print ('Training Set: %d, rows\nTest Set: %d rows' % (X_train.shape[0], X_test.shape[0]))

#### REGRESION LINEAL ####

# Entrenamos el modelo y le ingresamos los datos para entrenamiento
regression = LinearRegression()
R_lineal = regression.fit(X_train, y_train)

# Realizamos las predicciones del modelo de regresion lineal
lineal_pred = R_lineal.predict(X_test)

# Mostramos las metricas
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

### REGRESION DE VECINOS MAS CERCANOS KNN ###

k = 3  # Número de vecinos a considerar
# Creamos y entrenamos el algoritmo
knn_reg = KNeighborsRegressor(n_neighbors=k)
knn_reg.fit(X.reshape(-1, 1), y)

# Realizamos las predicciones
knn_pred = knn_reg.predict(X_test.reshape(-1, 1))

#Calculamos las metricas del modelo
mse = mean_squared_error(y_test, knn_pred)
print("MSE:", mse)
rmse = np.sqrt(mse)
print("RMSE:", rmse)
r2 = r2_score(y_test, knn_pred)
print("R2:", r2)
mae = mean_absolute_error(y_test, knn_pred)
print("MAE:", mae)

# Comparamos graficamente las predicciones con los datos reales
plt.scatter(y_test, knn_pred)
plt.xlabel('Actual Labels')
plt.ylabel('Predicted Labels')
plt.title('knn_pred vs Actuals')
z = np.polyfit(y_test, knn_pred, 1)
p = np.poly1d(z)
plt.plot(y_test,p(y_test), color='magenta')
plt.show()

## Si comparamos el coeficiente de R2 en ambos modelos podemos ver que el modelo con mayor puntaje es el de regresion lineal
## Por lo que procederemos a guardar dicho modelo

# Guardamos el modelo en el siguiente archivo .pkl
filename = './Hours_Score_Linear_regression.pkl'
joblib.dump(R_lineal, filename)

# Cargamos el modelo
Linear_loaded = joblib.load(filename)
# Agregamos nuevos datos para predecir las calificaciones
X_new = np.array([[2.7],
                  [4.8],
                  [3.8],
                  [6.9],
                  [7.8],
                  [1.0],
                  [2.0],
                  [10.0]])

N_pred = Linear_loaded.predict(X_new)
print('y_pred:')
for prediction in N_pred:
    print(round(prediction,2))