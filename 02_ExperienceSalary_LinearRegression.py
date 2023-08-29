# Cargamos las librerias
import pandas as pd
import numpy as np
import seaborn as sns

# Cargamos el df
df = pd.read_csv(r"C:\Users\darfa\OneDrive\Escritorio\practica ds\data\Salary_dataset.csv")

# LIMPIEZA Y ANALISIS #

#Sumatoria de los valores nulos
df.isnull().sum()
# Estadistica descriptiva
df.describe()

# Generamos un mapa de correlaciones para identificar si existe correlacion entre ambas columnas
corr = df.corr()
sns.heatmap(corr, cmap = 'Wistia', annot= True);
# Al encontrar una correlacion importante entre los datos de salario y años trabajados no es necesario modificar la data


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
delayFields = ['YearsExperience','Salary']
for col in delayFields:
    show_distribution(df[col])
    
    
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Seleccionamos la etiqueta que sera la variable a predecir
label = df[df.columns[-1]]

#Separamos los datos en Variable independiente X y dependiente Y
X, y = df[df.columns[0:-1]].values, df[df.columns[-1]].values

#Separamos los datos en entrenamiento y testeo en un 70% y 30%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=0)

# Imprimimos el resultado de la division
print ('Training Set: %d, rows\nTest Set: %d rows' % (X_train.shape[0], X_test.shape[0]))

# Entrenamos el modelo y le ingresamos los datos para entrenamiento
regression = LinearRegression()
R_lineal = regression.fit(X_train, y_train)

y_pred = R_lineal.predict(X_test)

# Cargamos la libreria para evaluar el modelo
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
# Mostramos las metricas
mse = mean_squared_error(y_test, y_pred)
print("MSE:", mse)
rmse = np.sqrt(mse)
print("RMSE:", rmse)
r2 = r2_score(y_test, y_pred)
print("R2:", r2)
mae = mean_absolute_error(y_test, y_pred)
print("MAE:", mae)


#Importamos la libreria para trazar las predicciones vs las reales
import matplotlib.pyplot as plt
# Trazamos las predicciones vs las reales
plt.scatter(y_test, y_pred)
plt.xlabel('Actual Labels')
plt.ylabel('Predicted Labels')
plt.title('y_pred vs Actuals')
z = np.polyfit(y_test, y_pred, 1)
p = np.poly1d(z)
plt.plot(y_test,p(y_test), color='magenta')
plt.show()
# Al ser un pequeño conjunto de datos podemos observar a detalle que los datos no coinciden a la perfeccion, sin embargo se acercan a los valores esperados

# Procederemos a guardar el modelo para hacer futuras predicciones
# Cargamos la libreria para guardar el modelo
import joblib

# Guardamos el modelo en el siguiente archivo .pkl
filename = './Experience_Salary_Linear_regression.pkl'
joblib.dump(R_lineal, filename)

# Cargamos el modelo
Linear_loaded = joblib.load(filename)

# Generamos un array con valores X para predecir el Salario del trabajador
X_new = np.array([[8.3],
                  [8.8],
                  [9.1],
                  [9.6],
                  [9.7],
                  [10.4],
                  [10.6],
                   [5],
                   [17]])

# Hacemos las nuevas predicciones llamando al modelo
results = Linear_loaded.predict(X_new)
print('y_pred:')
for prediction in results:
    print(round(prediction,2))
    
    
# En conclusion podemos ver como el modelo genera predicciones muy cercanas a los valores reales, aun cuando las metricas de evaluacion parecen no ser favorables para el modelo es necesario ajustarlo un poco mas para recibir predicciones mas certeraz
# Sera necesario practicar con otro algoritmo de regresion como puede ser el random forest regressor.


### REGRESION DE VECINOS MAS CERCANOS KNN ###

# Importamos el algoritmo a utilizar
from sklearn.neighbors import KNeighborsRegressor

k = 3  # Número de vecinos a considerar
# Creamos y entrenamos el algoritmo
model = KNeighborsRegressor(n_neighbors=k)
model.fit(X.reshape(-1, 1), y)

# Realizamos las predicciones
y_pred = model.predict(X_test.reshape(-1, 1))

# Realizamos Las predicciones de nuevos datos
Y_pred = model.predict(X_new.reshape(-1, 1))

#Calculamos las metricas del modelo
mse = mean_squared_error(y_test, y_pred)
print("MSE:", mse)
rmse = np.sqrt(mse)
print("RMSE:", rmse)
r2 = r2_score(y_test, y_pred)
print("R2:", r2)
mae = mean_absolute_error(y_test, y_pred)
print("MAE:", mae)

# Comparamos graficamente las predicciones con los datos reales
plt.scatter(y_test, y_pred)
plt.xlabel('Actual Labels')
plt.ylabel('Predicted Labels')
plt.title('y_pred vs Actuals')
z = np.polyfit(y_test, y_pred, 1)
p = np.poly1d(z)
plt.plot(y_test,p(y_test), color='magenta')
plt.show()

# Al utilizar un menor numero de K podemos observar que nuestro modelo se ajusta mejor
# El primer ejemplo fue con 10 vecinos, mientras el modelo mejor ajustado se da con 3 vecinos
# El siguiente modelo sera con Maquinas de vectores de soporte

### MAQUINAS DE VECTORES DE SOPORTE (SVM) ### 

# Importamos el algoritmo a utilizar
from sklearn.svm import SVR

# Entrenamiento del Modelo
model = SVR(kernel='linear')  # Puedes elegir diferentes kernels como 'linear', 'poly', 'rbf', etc.
model.fit(X.reshape(-1, 1), y)
    
# Realizamos las predicciones
y_pred = model.predict(X_test.reshape(-1, 1))

Y_pred = model.predict(X_new.reshape(-1, 1))

#Calculamos las metricas del modelo
mse = mean_squared_error(y_test, y_pred)
print("MSE:", mse)
rmse = np.sqrt(mse)
print("RMSE:", rmse)
r2 = r2_score(y_test, y_pred)
print("R2:", r2)
mae = mean_absolute_error(y_test, y_pred)
print("MAE:", mae)

# Comparamos graficamente las predicciones con los datos reales
plt.scatter(y_test, y_pred)
plt.xlabel('Actual Labels')
plt.ylabel('Predicted Labels')
plt.title('y_pred vs Actuals')
z = np.polyfit(y_test, y_pred, 1)
p = np.poly1d(z)
plt.plot(y_test,p(y_test), color='magenta')
plt.show()

# Podemos observar dado el score de r2 que el modelo no es significativo, por lo tanto quizas este algoritmo pudiera no ser el mejor