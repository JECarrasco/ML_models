import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn. metrics import accuracy_score, precision_score, recall_score, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score
import joblib
%matplotlib inline

# Cargamos el archivo a trabajar
data = pd.read_csv(r"C:\Users\darfa\OneDrive\Escritorio\practica ds\data\Clasification\wine.csv")

# El objetivo del siguiente modelo es crear un algoritmo que prediga la variable de vino dada las caracteristicas dentro del df.

# Separamos los datos en caracteristicas y la etiqueta (valor que buscamos predecir)

features = ['Alcohol','Malic_acid','Ash','Alcalinity','Magnesium','Phenols','Flavanoids','Nonflavanoids','Proanthocyanins','Color_intensity','Hue','OD280_315_of_diluted_wines','Proline']
label = 'WineVariety'

# generamos las variables dependientes e independientes
X, y = data[features].values, data[label].values

for n in range(0,4):
    print("Wine", str(n+1), "\n  Features:",list(X[n]), "\n  Label:", y[n])
    

# Comparamos la distribucion de las caracteristicas de cada variable de vino
from matplotlib import pyplot as plt
%matplotlib inline

for col in features:
    data.boxplot(column=col, by=label, figsize=(6,6))
    plt.title(col)
plt.show()


# Separamos la data en train y test 70%-30%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=0)

# Mostramos la division
print ('Training cases: %d\nTest cases: %d' % (X_train.shape[0], X_test.shape[0]))

# Normalizamos las caracteristicas y entrenamos el modelo

# Definimos preprocesamiento para columnas numéricas (escalarlas):
    ##El código utiliza la clase StandardScaler del módulo preprocessing de Scikit-learn para estandarizar 
    #las características numéricas en el conjunto de datos. Eso significa que escala los valores de cada 
    #columna numérica para que tengan media cero y desviación estándar uno. Esto es útil para garantizar que
    #las características tengan un rango comparable y ayudar al algoritmo de regresión logística a 
    #converger más rápido y mejorar su rendimiento.
feature_columns = [0,1,2,3,4,5,6]
feature_transformer = Pipeline(steps=[
    ('scaler', StandardScaler())
    ])

# Creamos el preprocesado y la pipeline de entrenamiento:
preprocessor = ColumnTransformer(
    transformers=[
        ('preprocess', feature_transformer, feature_columns)])

# Creamos la pipeline de entrenamiento
pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                           ('regressor', LogisticRegression(solver='lbfgs', multi_class='auto'))])

# Ajustamos la pipeline para entrenar un modelo de regresion logistica en el set de entrenamiento
model = pipeline.fit(X_train, y_train)
print (model)    

# Evaluamos el modelo
# Obtenemos las predicciones de la data de test
predictions = model.predict(X_test)

# Obtenemos las metricas de evaluacion
print("Overall Accuracy:",accuracy_score(y_test, predictions))
print("Overall Precision:",precision_score(y_test, predictions, average='macro'))
print("Overall Recall:",recall_score(y_test, predictions, average='macro'))

# Graficamos una matriz de confusion
cm = confusion_matrix(y_test, predictions)
classes = ['Variety A','Variety B','Variety C']
plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
plt.colorbar()
tick_marks = np.arange(len(classes))
plt.xticks(tick_marks, classes, rotation=45)
plt.yticks(tick_marks, classes)
plt.title('Confusion Matrix')
plt.xlabel("Predicted Variety")
plt.ylabel("Actual Variety")
plt.show()


# Obtenemos los scores de probabilidad por clase
probabilities = model.predict_proba(X_test)

auc = roc_auc_score(y_test,probabilities, multi_class='ovr')
print('Average AUC:', auc)

# Obtenemos las metricas ROC por cada clase
fpr = {}
tpr = {}
thresh ={}
for i in range(len(classes)):    
    fpr[i], tpr[i], thresh[i] = roc_curve(y_test, probabilities[:,i], pos_label=i)
    
# Graficamos el cuadro de las metricas ROC
plt.plot(fpr[0], tpr[0], linestyle='--',color='orange', label=classes[0] + ' vs Rest')
plt.plot(fpr[1], tpr[1], linestyle='--',color='green', label=classes[1] + ' vs Rest')
plt.plot(fpr[2], tpr[2], linestyle='--',color='blue', label=classes[2] + ' vs Rest')
plt.title('Multiclass ROC curve')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive rate')
plt.legend(loc='best')
plt.show()

# Generamos predicciones con datos nuevos

# Guardamos nuestro modelo
filename = './wine_classifer.pkl'
joblib.dump(model, filename)

# Cargamos nuestro modelo
model = joblib.load(filename)

# Obtenemos las predicciones de dos muestras de vino
x_new = np.array([[13.72,1.43,2.5,16.7,108,3.4,3.67,0.19,2.04,6.8,0.89,2.87,1285],
                  [12.37,0.94,1.36,10.6,88,1.98,0.57,0.28,0.42,1.95,1.05,1.82,520]])

# Generamos las predicciones
predictions = model.predict(x_new)

# Obtenemos el resultado de las predicciones
for prediction in predictions:
    print(prediction, '(' + classes[prediction] +')')