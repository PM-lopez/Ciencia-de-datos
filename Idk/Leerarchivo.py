
from Tareas import moda,promedio,mediana,rango,varianza,desviacion_estandar,mad,rango_intercuartilico,percentil
import numpy as np
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
