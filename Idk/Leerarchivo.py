
from Tareas import moda,promedio,mediana,rango,varianza,desviacion_estandar,mad,rango_intercuartilico,percentil
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt

archivo=open("bsc_sel.dat","r")
archivo.readline()
HR = []
HD=[]
Vmag=[]
BV=[]
SpType=[]
for lin in archivo:
    HR.append(int(lin.split()[0]))
    HD.append(int(lin.split()[1]))
    if lin.split()[2]=='""':
        Vmag.append(np.nan)
    else:
        Vmag.append(float(lin.split()[2]))
    if lin.split()[3]=='""':
        BV.append(np.nan)
    else:
        BV.append(float(lin.split()[3]))
    if lin.split()[4]=='""':
        SpType.append(np.nan)
    else:
        SpType.append(str(lin.split()[4]))

archivo.close()
print(f"Rango: {rango(Vmag)}")
print(f"Promedio de Vmag: {promedio(Vmag)}")
print(f"Mediana de Vmag: {mediana(Vmag)}")

print(f"Percentil 1% {percentil(Vmag,1)}")

print(f"Varianza: {varianza(Vmag)}")
print(f"Desviacion estandar {desviacion_estandar(Vmag)}")
print(f"Rnago intercuartilico: {rango_intercuartilico(Vmag)}")
print(f"MAD: {mad(Vmag)}")



entrada= open("velocidades_Ocen.dat","r")
entrada.readline()
nombre = []
ra=[]
dec=[]
vhelio=[]
for lin in entrada:
    nombre.append(lin.split()[0])
    ra.append(float(lin.split()[1]))
    dec.append(float(lin.split()[2]))
    if lin.split()[3]=='""':
        vhelio.append(np.nan)
    else:
        vhelio.append(float(lin.split()[3]))
entrada.close()

resultado= promedio(vhelio)
print("promedio = ", resultado)



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
