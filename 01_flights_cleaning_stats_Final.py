# Cargamos el df y visualizamos
import pandas as pd
df = pd.read_csv(r"C:\Users\darfa\OneDrive\Escritorio\practica ds\data\flights.csv")
df.head()

# Hacemos una sumatoria de los valores nulos dentro de cada columna
df.isnull().sum()

# Comparamos los valores nulos en la columna depdel15 con depdelay
# Entendemos que si un vuelo se retrasa mas de 15 minutos se considera atrasado y se le asgina un 1 a la columna depdel15
df[df.isnull().any(axis=1)][['DepDelay','DepDel15']]

#Observamos la estadistica y concluimos que esos valores son 0 en todos los casos
df[df.isnull().any(axis=1)].DepDelay.describe()

#Remplazamos los valores NA por 0 una vez que revisamos la estadistica
df.DepDel15 = df.DepDel15.fillna(0)
df.isnull().sum()

### LIMPIEZA DE VALORES ATIPICOS ###

# Con esta funcion mostramos las estadisticas y distribucion para cada columna
def show_distribution(var_data):
    from matplotlib import pyplot as plt

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

# Llamamos la funcion para trazar los graficos en cada columna
delayFields = ['DepDelay','ArrDelay']
for col in delayFields:
    show_distribution(df[col])
    
    
    
#### OUTLIERS ####    
# Podemos observar mediante la grafica que hay valores atipicos (outliers) presentes en el extremo derecho e izquierdo
# Para solucionar dichos outliers recortaremos los datos de cada extremo 
# Quedando solo valores que se encuentren entre el percentil 1 y 90

# Recortamos los outliers para la columna ArrDelay manteniendo solamente los valores entre 1% y 90% del percentil 
ArrDelay_01pcntile = df.ArrDelay.quantile(0.01)
ArrDelay_90pcntile = df.ArrDelay.quantile(0.90)
df = df[df.ArrDelay < ArrDelay_90pcntile]
df = df[df.ArrDelay > ArrDelay_01pcntile]

# Recortamos los outliers para la columna DepDelay manteniendo solamente los valores entre 1% y 90% del percentil 
DepDelay_01pcntile = df.DepDelay.quantile(0.01)
DepDelay_90pcntile = df.DepDelay.quantile(0.90)
df = df[df.DepDelay < DepDelay_90pcntile]
df = df[df.DepDelay > DepDelay_01pcntile]

# Revisamos nuevamente los graficos de distribucion
for col in delayFields:
    show_distribution(df[col])

# Los outliers han disminuido considerablemente

### EXPLORACION DE LA DATA ###

# Comencemos con una vista general de las estadísticas de resumen para las columnas numéricas.
r_stats = df.describe()
r_stats.head()

# Promedio de los retrasos en salidas y llegadas de los vuelos
df[delayFields].mean()

# Comparamos el rendimiento de las aerolineas en terminos de rendimiento de retraso de llegada
for col in delayFields:
    df.boxplot(column=col, by='Carrier', figsize=(8,8))


# Analizamos si algunos dias de la semana tienden a ser mas propensos a atrasos que otros
for col in delayFields:
    df.boxplot(column=col, by='DayOfWeek', figsize=(8,8))

    
# Comparamos que aeropuerto tiene el mayor promedio de retrasos
departure_airport_group = df.groupby(df.OriginAirportName)

mean_departure_delays = pd.DataFrame(departure_airport_group['DepDelay'].mean()).sort_values('DepDelay', ascending=False)
mean_departure_delays.plot(kind = "bar", figsize=(12,12))
mean_departure_delays


# Analizamos si los retrasos en salidas tienden a provocar retrasos en llegadas mayores en comparacion con las salidas puntuales
df.boxplot(column='ArrDelay', by='DepDel15', figsize=(12,12))


# ¿Que ruta (aeropuerto origen a aeropuerto destino) tiene el mayor numero de retrasos?
# Creamos una columna de rutas
routes  = pd.Series(df['OriginAirportName'] + ' > ' + df['DestAirportName'])
df = pd.concat([df, routes.rename("Route")], axis=1)

# Hacemos un Group by de la columna
route_group = df.groupby(df.Route)
pd.DataFrame(route_group['ArrDel15'].sum()).sort_values('ArrDel15', ascending=False)

# Analizamos que ruta tiene el promedio de retrasos mas alto (minutos)
pd.DataFrame(route_group['ArrDelay'].mean()).sort_values('ArrDelay', ascending=False)
