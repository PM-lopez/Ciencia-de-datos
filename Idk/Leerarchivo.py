
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





archivo=open("estrellas_vecindadSolar_all.dat","r")
archivo.readline()
BP_tmp=[]
RP_tmp=[]
met_tmp=[]
for lin in archivo:
    met_tmp.append(float(lin.split()[3]))
    BP_tmp.append(float(lin.split()[6]))
    RP_tmp.append(float(lin.split()[7]))
archivo.close()
BP=[]
RP=[]
met=[]
for v0,v1,v2 in zip(met_tmp,BP_tmp,RP_tmp):
    if (v1>0) & (v2>0):
        met.append(v0)
        BP.append(v1)
        RP.append(v2)

print("covarianza= ",covarianza(BP,RP))
print("correlacion= ",correlacion(BP,RP))

fig=plt.figure(1,figsize=(4,4),dpi=120)
ax1=fig.add_subplot(111)
ax1.set_xlabel("BP")
ax1.set_ylabel("RP")
cm=plt.cm.get_cmap("jet")
sc=ax1.scatter(BP,RP,marker=".",ec="none",c=met,cmap=cm,s=1,vmin=0.5,vmax=0.3)
cb=plt.colorbar(sc)
cb.set_label("Metalicidad")
plt.show()










def derivada(f, x, h):
    """
    Retorna el gradiente como el limite del
    cuociente diferencial
    """
    return ( f(x + h) - f(x) ) / h


def ajuste_lineal_exacto(x,y):
    '''
    Determina los parámetros de minimo cuadrado para
    una ajuste lineal de la forma y = A + Bx usando
    las ecuaciones normales
    '''
    x_sq = [xv**2 for xv in x]
    x_y = [xv*yv for xv, yv in zip(x,y)]
    delta = len(x) * sum(x_sq) - sum(x)**2
    pendiente = (len(x) * sum(x_y) - sum(x) * sum(y)) / delta
    intercepto = (sum(x_sq) * sum(y) - sum(x) * sum(x_y)) / delta
    return pendiente, intercepto

def mse(x, y, theta):
    m,b = theta
    residuos = [(y_i - (m * x_i + b))**2 for x_i, y_i in zip(x,y)]
    mse = sum(residuos) / len(residuos)
    return mse


def limite_de_cuociente(x, y, f, v, i, h):
    """
    Retorna el limite de la diferencia de cuocientes
    para el i-esimo parametro de una función en el punto
    dado por v.
    x, y : datos
    f : función cuyo gradiente se quiere estimar
    v : punto en el que se quiere estimar el gradiente
    i : dimensión en la que se calcula el gradiente
    h : tamaño del diferencial
    """
    # Agregamos h solo al i-esimo elemento de v
    w = [v_j + (h if j==i else 0) for j,v_j in enumerate(v)]
    return (f(x, y, w) - f(x, y, v)) / h

def estimate_gradient(x, y, f, v, h=0.0001):
    return [limite_de_cuociente(x, y, f, v, i, h) for i in range(len(v))]

def paso_en_gradiente(v, gradient, step_size):
    """
    Se mueve un paso pequeño 'step_size' en la
    dirección del gradiente desde el punto v
    """
    assert len(v) == len(gradient)
    step = [step_size * g_i for g_i in gradient]
    return [a + b for a,b in zip(v, step)]

def gradiente_mse(x, y, theta):
    pendiente, intercepto = theta
    y_pred = [pendiente * xv + intercepto for xv in x]
    # Derivada parcial respecto a la pendiente
    g1 = 2 / len(x) * sum([ (y_p - y_d) * x_d for x_d, y_d, y_p in zip(x, y, y_pred) ])
    # Derivada parcial respecto al intercepto
    g2 = 2 / len(x) * sum([ (y_p - y_d) for x_d, y_d, y_p in zip(x, y, y_pred) ])
    return [g1, g2]

import numpy as np
import random
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
# Generamos datos con tendencia lineal
x = 2 * np.random.rand(100)
# pendiente = 3
# intercepto = 4
y = 4 + 3 * x + np.random.rand(100)

# comenzar con valores aleatorios para la pendiente
# y el intercepto
theta = [random.uniform(-1,1), random.uniform(-1,1)]
learning_rate = 0.001
n_iter_max = 100_000
gtol = 1e-6 # tolerancia en la norma del vector gradiente
# Gradient descent
iterar = True
n_iter = 0
thetas = []
while iterar:
    # Calcular el gradiente
    grad = gradiente_mse(x, y, theta)
    # Realizar un paso en la dirección contraria
    # al gradiente
    theta = paso_en_gradiente(theta, grad, -learning_rate)
    thetas.append(theta)
    # Check 1: ver si se alcanzó el número máximo de iteraciones
    if n_iter > n_iter_max:
        iterar = False
    # Check 2: revisar si la norma del gradiente ya alcanzó el
    # mínimo tamaño permitido por el criterio de tolerancia
    norm_grad = sum([g**2 for g in grad])**(1/2)
    if norm_grad < gtol:
        iterar = False
    # Contabilizar la iteracion
    n_iter += 1
print("Solucion: ", theta)
print("Num iteraciones: ", n_iter)

# Calcular el coeficiente de determinación R-squared
# Suma total de cuadrados: variacion total de y_i's respecto a su promedio
y_prom = np.mean(y)
suma_total_cuadrados = sum([v**2 for v in y - y_prom])
# Suma de errores cuadráticos
suma_sqerrors = sum([ (theta[0] * xv + theta[1] - yv)**2 for xv,yv in zip(x,y) ])
# calculo de r-squared
r_squared = 1 - suma_sqerrors / suma_total_cuadrados
print("R-squared: ", r_squared)

fig=plt.figure(1,figsize=(5,3.5),dpi=100)
fig.subplots_adjust(left=0.15,bottom=0.12,right=0.95,top=0.97,hspace=0.24,wspace=0.20)
ax1=fig.add_subplot(111)
ax1.set_xlabel("X")
ax1.set_ylabel("Y")
ax1.set_xlim(-0.2,2.2)
ax1.set_ylim(3.0,12.5)
ax1.scatter(x,y,marker=".",fc="black",ec="none",s=40,label="Puntos")
xv=np.linspace(0,2,100)
ax1.plot(xv,xv*theta[0]+theta[1],color="red",label="Ajuste lineal")
ax1.xaxis.set_minor_locator(AutoMinorLocator(4))
ax1.yaxis.set_minor_locator(AutoMinorLocator(5))
ax1.legend(loc=2,scatterpoints=1,handletextpad=0.001,fontsize=11)
plt.show()

n_epocas = 50
t0, t1 = 5, 50 # hiperparámetros de la learning schedule
def learning_schedule(t):
    return t0 / (t + t1)
theta = np.random.rand(2) # parámetros iniciales aleatorios
# El número de puntoss
m = len(x)
thetas = []
for epoca in range(n_epocas):
    for i in range(m):
        random_index = np.random.randint(m)
        xi = x[random_index:random_index+1]
        yi = y[random_index:random_index+1]
        grad = gradiente_mse(x, y, theta)
        learning_rate = learning_schedule(epoca * m + i)
        theta = paso_en_gradiente(theta, grad, -learning_rate)
        thetas.append(theta)
print(theta)
