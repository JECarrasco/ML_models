# Importamos la libreria a utilizar
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
df = pd.read_csv(r"C:\Users\darfa\OneDrive\Escritorio\practica ds\data\Linear multiple\insurance.csv")

# Age - edad
# Sex - Sexo
# BMI - Indice de masa corporal
# Children - numero de hijos asegurados
# Smoker - Fumador (booleano si o no)
# Region - 4 regiones
# Charges - costo de individual de la poliza medica

# El objetivo principal de este df es generar un modelo que sea capaz de predecir los costos de las polizas de seguros.
# En este caso nuestra variable independiente "y" sera Charges, mientras que las demas variables seran los predictores

# Obtener la columna de label (Variable independiente) en este caso es la ultima columna
label = df[df.columns[-1]]

# Graficar la distribucion de los datos
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

### ELIMINACION DE OUTLIERS ###
def eliminar_outliers_iqr(data, column_name):
    Q1 = data[column_name].quantile(0.25)
    Q3 = data[column_name].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return data[(data[column_name] >= lower_bound) & (data[column_name] <= upper_bound)]


dfo = pd.DataFrame(label)

# Eliminar outliers usando el método del rango intercuartílico
column_name = 'charges'
label = eliminar_outliers_iqr(dfo, column_name)

# Convertimos label en lista
label = label[label.columns[-1]]


# Eliminamos los outliers utilizando la condicion de sustituir los valores mayores a > 50000 por valores nulos
df.loc[df['charges'] > 50000, 'charges'] = None

# Por ultimo eliminamos los valores nulos
df.dropna(subset=['charges'], inplace=True)

# Creamos de nuevo la columna label
label = df[df.columns[-1]]


#NOTA: Utilizar el metodo que mejor creas conveniente de los dos anteriores

# =============================================================================
# GRAFICAMOS NUEVAMENTE
# =============================================================================
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
# =============================================================================
# GENERAMOS UN BOXPLOT CON LAS CORRELACIONES
# =============================================================================
corr = df.corr()
sns.heatmap(corr, cmap = 'Wistia', annot= True);

#Podemos observar que no existe mucha correlacion entre las variables numericas, por lo que tendremos que generar las variables dummy

# =============================================================================
# CREACION DE VARIABLES DUMMY
# =============================================================================
# Creamos variables dummy para la columna Sex
df_dummies = pd.get_dummies(df['sex'], prefix='sex')

# Concatenar las variables dummy al DataFrame original
df = pd.concat([df, df_dummies], axis=1)

# Eliminar la columna original categórica si lo deseas
df.drop('sex', axis=1, inplace=True)

# Creamos variables dummy para la columna smoker
df_dummies = pd.get_dummies(df['smoker'], prefix='smoke')

# Concatenar las variables dummy al DataFrame original
df = pd.concat([df, df_dummies], axis=1)

# Eliminar la columna original categórica si lo deseas
df.drop('smoker', axis=1, inplace=True)

# Por ultimo creamos variables dummy para la columna region
df_dummies = pd.get_dummies(df['region'])

# Concatenar las variables dummy al DataFrame original
df = pd.concat([df, df_dummies], axis=1)

# Eliminar la columna original categórica si lo deseas
df.drop('region', axis=1, inplace=True)

# NOTA: volvemos a generar un mapa de correlaciones

# =============================================================================
# GENERAMOS EL MODELO DE REGRESION
# =============================================================================

#Movemos la columna charges al final del df|
cm = 'charges'

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

R_lineal = regression.fit(X_train, y_train)

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

### Como podemos observar, el valor de la r2 es igual a .77 por lo que el modelo acerto un 77% de las predicciones
# Es necesario ajustar mas los valores e incluso eliminar variables que tal vez no sean utiles al momento de generar las predicciones

# =============================================================================
# Guardamos el modelo y predecimos nuevos valores
# =============================================================================

#Guardamos el modelo
filename = './Insurance_Linear_regression.pkl'
joblib.dump(R_lineal, filename)

# Cargamos el modelo
Linear_loaded = joblib.load(filename)

# Generamos un array con nuevos valores
X_new = np.array([[54, 30.8, 3, 1, 0, 1, 0, 0, 0, 0, 1],
                  [55, 38.28, 0 ,0, 1, 1, 0, 0, 0, 1, 0],
                  [38, 19.3, 0, 0, 1, 0, 1, 0, 0, 0, 1]])

# Predecimos los nuevos valores
L_pred = Linear_loaded.predict(X_new)

print('y_pred:')
for prediction in N_pred:
    print(round(prediction,2))
    
# =============================================================================
# Random forest regressor
# =============================================================================

# Utilizando los datos preprocesados con anterioridad, podemos generar un nuevo modelo que nos permita comparar cual esta mejor ajustado
from sklearn.ensemble import RandomForestRegressor

#Entrenamos el modelo y generamos las predicciones
RFRModel = RandomForestRegressor()
RegresionRFR = RFRModel.fit(X_train, y_train)
RFR_predict = RegresionRFR.predict(X_test)


# Calculamos las metricas del modelo
mse = mean_squared_error(y_test, RFR_predict)
print("MSE:", mse)
rmse = np.sqrt(mse)
print("RMSE:", rmse)
r2 = r2_score(y_test, RFR_predict)
print("R2:", r2)
mae = mean_absolute_error(y_test, RFR_predict)
print("MAE:", mae)

# Trazamos las predicciones vs las reales
plt.scatter(y_test, RFR_predict)
plt.xlabel('Actual Labels')
plt.ylabel('Predicted Labels')
plt.title('lineal_pred vs Actuals')
z = np.polyfit(y_test, RFR_predict, 1)
p = np.poly1d(z)
plt.plot(y_test,p(y_test), color='magenta')
plt.show()

# Podemos observar que la R2 del modelo RFR tiene mayor precision que la R2 del modelo de regresion lineal .88 vs .77
## Por lo tanto se utilizara el modelo con la mejor presicion para hacer futuras predicciones.

# =============================================================================
# Guardamos el nuevo modelo y realizamos predicciones
# =============================================================================
#Guardamos el modelo
filename = './Insurance_RandomForest_regression.pkl'
joblib.dump(RegresionRFR, filename)

# Cargamos el modelo
RFR = joblib.load(filename)

#Realizamos las nuevas predicciones
RFR_pred = RFR.predict(X_new)

