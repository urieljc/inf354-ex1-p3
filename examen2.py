# -*- coding: utf-8 -*-
"""
Created on Sun Apr  4 17:47:15 2021

@author: BazanJuanCarlos
"""

#tipos de pre-procesamientos

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
 
#carga del dataset
dataset=pd.read_csv('master.csv')

#sabemos si existen datos vacios
a=dataset.isnull().any(axis=1).sum()
print(a[a>0])
print("----------------------------------------")

copia=dataset
copia.info()
print("----------------------------------------")
print(copia.describe(include=np.object))

#separamos los variables a dependientes e independientes
x=copia.loc[:,['country','year','age','suicides_no','population','HDI for year']].values
y=copia.loc[:,['sex']].values
#z=copia.loc[:,'age'].values
#imputacin de datos faltantes
#se ulitiza la media o moda para reemplazar
# en los campos vacios del dataset
print("----------------------------------------")

from sklearn.impute import SimpleImputer
imputer=SimpleImputer(missing_values=np.nan,strategy='mean',verbose=0) 
imputer=imputer.fit(x[:,5:])
x[:,5:]=imputer.transform(x[:,5:])
print("Termino la imputacion de datos")
print("----------------------------------------")

#variable categoricas
#varialbes tipo texto en seleccion

from sklearn.preprocessing import LabelEncoder

labelEncoder_x=LabelEncoder()

x[:,0]=labelEncoder_x.fit_transform(x[:,0])

x[:,2]=labelEncoder_x.fit_transform(x[:,2])

labelEncoder_y=LabelEncoder()
y=labelEncoder_y.fit_transform(y)


#dummy Encoding
#asignar peso o mayo validas a las unidades transformadas
#asignar que valores tiene mayor prioridad

#from sklearn.preprocessing import OneHotEncoder
#onehotencoder = OneHotEncoder(categorical_features=[0])
#x = onehotencoder.fit_transform(x).toarray()

#escalamiento de datos
#tranformar las escalas de los numeros
# se puede usar la estandarizacion
# Xstan=(x-mena(x)/desviacionStandar(x))

from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
x=scaler.fit_transform(x)

