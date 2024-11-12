import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
from Tarea import promedio,mediana,rango_intercuartil,mad,desviacion_estandar,moda

archivo=open("grupo_local.csv","r")
archivo.readline()
tipos_galaxias=[]
distancias=[]
for lin in archivo:
    tipos_galaxias.append(lin.split(",")[5])
    distancias.append(float(lin.split(",")[3]))
archivo.close()

cc=Counter(tipos_galaxias)
etiquetas=list(cc.keys())
valores=list(cc.values())

fig=plt.figure(figsize=(6,3.5),dpi=100)
ax1=fig.add_subplot(111)

ax1.bar(etiquetas,valores,color="olivedrab",width=0.8)

ax1.set_xlabel("Tipo de galaxia",fontsize=13)
ax1.set_ylabel("N",fontsize=13)
ax1.set_title("Tipos de galaxias en el Grupo local",fontsize=13)
plt.show()

print("Distancia promedio: ", promedio(distancia))
print("Distancia mediana: ", mediana(distancia))
print("Distancia IQR: ", rango_intercuartil(distancia))
print("Distancia MAD: ", mad(distancia))
print("Distancia desviacion_estandar: ", desviacion_estandar(distancia))
print("Moda de tipo de galaxias: ", moda(tipos_galaxias))
