
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







archivo=open("omegaCen.dat","r")
archivo.readline()
vhelio=[]
for lin in archivo:
    vhelio.append(float(lin.split()[8]))
archivo.close()

bines=[]
ancho=(310+150)/70
for i in range(70):
    borde=-150+i*ancho
    bines.append(borde)

fig=plt.figure(figsize=(4,3),dpi=100)
ax1=fig.add_subplot(111)
ax1.hist(vhelio,color="gray",alpha=0.7,bins=bines)
ax1.set_xlabel(r"V$_{helio}$ (km/s)", fontsize=13)
ax1.set_ylabel("N",fontsize=13)
ax1.set_xlim(-150,300)
ax1.set_title("Metodo normal")
plt.show()

binesS=int(np.log2(len(vhelio))+1)
fig=plt.figure(figsize=(4,3),dpi=100)
ax2=fig.add_subplot(111)
ax2.hist(vhelio,color="red",alpha=0.7,bins=binesS)
ax2.set_xlabel(r"V$_{helio}$ (km/s)", fontsize=13)
ax2.set_ylabel("N",fontsize=13)
ax2.set_title("Metodo Stutges")
plt.show()

anchoSc=3.49*desviacion_estandar(vhelio)*(len(vhelio))**(-1/3)
binesSc=int((300+150)/anchoSc)
fig=plt.figure(figsize=(4,3),dpi=100)
ax3=fig.add_subplot(111)
ax3.hist(vhelio,color="yellow",alpha=0.7,bins=binesSc)
ax3.set_xlabel(r"V$_{helio}$ (km/s)", fontsize=13)
ax3.set_ylabel("N",fontsize=13)
ax3.set_title("Regla de Scott")
plt.show()

anchoFD=2*rango_intercuartilico(vhelio)*(len(vhelio))**(-1/3)
binesFD=int((300+150)/anchoFD)
fig=plt.figure(figsize=(4,3),dpi=100)
ax4=fig.add_subplot(111)
ax4.hist(vhelio,color="blue",alpha=0.7,bins=binesFD)
ax4.set_xlabel(r"V$_{helio}$ (km/s)", fontsize=13)
ax4.set_ylabel("N",fontsize=13)
ax4.set_xlim(-250,300)
ax4.set_title("Regla de Freeman & Diaconis")
plt.show()
 
