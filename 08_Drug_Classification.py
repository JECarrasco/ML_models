# Cargamos todas las librerias necesarias
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn. metrics import accuracy_score, precision_score, recall_score, confusion_matrix, classification_report
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score
import joblib
from sklearn.preprocessing import LabelEncoder
%matplotlib inline

# Cargamos el df
df = pd.read_csv(r"C:\Users\darfa\OneDrive\Escritorio\practica ds\data\Clasification\drug200.csv")
df.head()

# El objetivo del siguiente algoritmo es crear un modelo de clasificacion capaz de clasificar el tipo de droga consumida
# Utilizando las variables del df se busca predecir la columna de Drug
# Age - edad 
# Sex - Sexo 
# BP - Nivel de presion de sangre
# Cholesterol - Nivel de colesterol
# Na_to_K - Racion de sodio a potacio en la sangre

# Mostramos los valores unicos dentro de cada columna para comprobar si son variables categoricas
for columna in df.columns:
    valores_unicos = df[columna].unique()
    print(f"Valores únicos en '{columna}': {valores_unicos}")

# Convertimos varibles categoricas a numericas 
label_encoder = LabelEncoder()
# Empezamos por la variable objetivo
df['Drug_encoded'] = label_encoder.fit_transform(df['Drug'])
# Con este metodo convertimos cada valor unico en un numero p.j: DrugY = 0, DrugA = 1, etc.

# Convertimos la columna Sex en varible binaria 0 = mujer 1 = hombre
df['Sex_b'] = label_encoder.fit_transform(df['Sex'])
df.drop('Sex', axis=1, inplace=True)


# Generamos los valores dummies en la columna BP
# Creamos variables dummy para la columna Extracurricular
df_dummies = pd.get_dummies(df['BP'], prefix='BP')

# Concatenar las variables dummy al DataFrame original
df = pd.concat([df, df_dummies], axis=1)

# Eliminar la columna original categórica si lo deseas
df.drop('BP', axis=1, inplace=True)

# Generamos los valores dummies en la columna Cholesterol
# Creamos variables dummy para la columna Extracurricular
df_dummies = pd.get_dummies(df['Cholesterol'], prefix='Cholesterol')

# Concatenar las variables dummy al DataFrame original
df = pd.concat([df, df_dummies], axis=1)

# Eliminar la columna original categórica si lo deseas
df.drop('Cholesterol', axis=1, inplace=True)

# Generamos un nuevo dataframe en donde dropeamos la variable categorica de drugs, quedandonos asi con la misma columna convertida a numerica
df_encod = df
df_encod.drop('Drug', axis = 1, inplace = True)

# Cambiamos el nombre de la columna Drug_encoded a Drug
df_encod = df_encod.rename(columns={'Drug_encoded': 'Drug'})

#Movemos la columna Drug al final del df
cm = 'Drug'

# Asegúrate de que la columna exista en el DataFrame
if cm in df_encod.columns:
    # Obtiene la lista de todas las columnas en el DataFrame
    columnas = df_encod.columns.tolist()

    # Mueve la columna 'Drug' al final de la lista
    columnas.remove(cm)
    columnas.append(cm)

    # Reorganiza las columnas del DataFrame con el orden deseado
    df_encod = df_encod[columnas]
    

# Separamos los datos en variable independiente y variable dependiente    
X, y = df_encod[df_encod.columns[0:-1]].values, df_encod[df_encod.columns[-1]].values

# Dividimos los datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=0)

# Imprimimos el valor de la division
print ('Training Set: %d, rows\nTest Set: %d rows' % (X_train.shape[0], X_test.shape[0]))

#Entrenamos el modelo
LR = LogisticRegression(solver='lbfgs', max_iter=1000)  # Selecciona el solucionador y otros parámetros
LR.fit(X_train, y_train)  # Entrena el modelo

# Generamos las prediciones
y_pred = LR.predict(X_test)  # Predice las etiquetas en el conjunto de prueba

# Evaluamos el modelo

# El evaluador Accuracy nos indica el porcentaje de aciertos en las predicciones, es decir cuantos valores predichos coinciden con los valores reales.
# Dicho evaluador tiene que tener un valor mayor a .5 para evitar predicciones al azar
print("Overall Accuracy:",accuracy_score(y_test, y_pred))
# La precisión es una medida que calcula la proporción de predicciones positivas correctas (verdaderos positivos) 
# en relación con el total de predicciones positivas (verdaderos positivos más falsos positivos).
print("Overall Precision:",precision_score(y_test, y_pred, average='macro'))
# Recall es similar a Precision
print("Overall Recall:",recall_score(y_test, y_pred, average='macro'))
#La matrix de confusion es una forma grafica de mostrar el total de falsos positivos, f negativos, verdaderos positivos y V negativos
confusion = confusion_matrix(y_test, y_pred)
# El reporte de clasificacion nos indica las metricas para calificar el modelo
report = classification_report(y_test, y_pred)


# Nuevas predicciones

# Guardamos nuestro modelo
filename = './Drug_Classifier.pkl'
joblib.dump(LR, filename)

# Cargamos nuestro modelo
model = joblib.load(filename)

# Obtenemos las predicciones de 4 pacientes
x_new = np.array([[58,38.247,0,0,1,0,1,0],
                  [56,25.395,0,1,0,0,1,0],
                  [20,35.639,1,1,0,0,0,1],
                  [15,16.725,0,1,0,0,0,1]])

# Generamos las predicciones
predictions = model.predict(x_new)
