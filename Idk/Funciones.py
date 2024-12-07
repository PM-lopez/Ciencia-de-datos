import math 
def promedio(lista):
    """
    calcula el promedio de una lista.
    
    parametros:
    -------------
    lista: lista de variables aleatorias
    
    retorna:
    ------------
    promedio : float
    """
    
    vals= []
    for v in lista:
        if math.isfinite(v):
            vals.append(v)
        
    promedio=sum(vals)/len(vals)
    return promedio





def mediana(vals_in):
    """
    Calcula la mediana de una lista de numeros
    Detecta y elimina valores NaN
    
    Paràmetros
    ----------
    vals: lista
        lista con los numeros
        
    Retorna
    -------
    mediana:float
        la mediana de los numeros (excluyendo NaNs)
    """
    
    
    #eliminamos los valores que sean NaNs
    vals=[]
    for v in vals_in:
        if math.isfinite(v):
            vals.append(v)
    
    #ordenar la lista
    vals.sort()
    if len(vals)%2!=0:
        k=len(vals)//2
        mediana=vals[k]
    else:
        k=len(vals)//2
        mediana=(vals[k-1]+vals[k])/2
    return mediana




def moda(vals):
    """
    calcula la moda de una lista conteniendo una
    variable categoriva nominal
    Parametros
    -----------
    vals: list
    lista de categotias
    Retorna
    -------
    moda: str
    la moda de la muestra
    """
    #encontrar el conjunto de elementos unicos
    categorias=[]
    for v in vals:
        if v not in categorias:
            categorias.append(v)
    #obtener el numero de cuentas en la muestra
    #para cada una de las categorias
    cuentas=[]
    for c in categorias:
        n=0
        for val in vals:
            if val==c:
                n=n+1
        cuentas.append(n)

    #guess and check
    i_max=0
    vals_max=cuentas[0]
    for i in range(1,len(cuentas)):
        if cuentas[i]> vals_max:  
            i_max=i
            vals_max=cuentas[i]
    # determinar todas las categorias que tengan el numero
    # maximo de cuentas	
    modas=[]
    for i in range(len(cuentas)):
        if cuentas[i]==vals_max:
            modas.append(categorias[i])
  
    #retorno la moda
    #moda= categorias[i_max]
    return modas



def rango(vals_in):
    """
    Calcula el rango de una lista de numeros
    Detecta y elimina valores NaN
    
    Paràmetros
    ----------
    vals: lista
        lista con los numeros
        
    Retorna
    -------
    rango:float
        rango de los numeros (excluyendo NaNs)
    """
    
    
    #eliminamos los valores que sean NaNs
    vals=[]
    for v in vals_in:
        if math.isfinite(v):
            vals.append(v)

    return max(vals)-min(vals)


def varianza(vals_in):
    """
    Calcula la varianza de una lista de numeros
    Detecta y elimina valores NaN
    
    Paràmetros
    ----------
    vals: lista
        lista con los numeros
        
    Retorna
    -------
    varianza:float
        varianza de los numeros (excluyendo NaNs)
    """
    
    
    #eliminamos los valores que sean NaNs
    vals=[]
    for v in vals_in:
        if math.isfinite(v):
            vals.append(v)
            
    #estimar el promedio
    prom=promedio(vals)
    
    #Estimamos las desviaciones cuadraticas medias
    dcm=[]
    for i in vals:
        dcm.append((v-prom)**2)
    varianza=sum(dcm)/len(vals)
    
    return varianza


def desviacion_estandar(vals_in):
    """
    Calcula la desviacion estandar de una lista de numeros
    Detecta y elimina valores NaN
    
    Paràmetros
    ----------
    vals: lista
        lista con los numeros
        
    Retorna
    -------
    desviacion estandar:float
        desviacion estandar de los numeros (excluyendo NaNs)
    """
    
    
    #eliminamos los valores que sean NaNs
    vals=[]
    for v in vals_in:
        if math.isfinite(v):
            vals.append(v)
        #estimar el promedio
    prom=promedio(vals)
    
    #Estimamos las desviaciones cuadraticas medias
    dcm=[]
    for i in vals:
        dcm.append((i-prom)**2)
    varianza=sum(dcm)/len(vals)
    
    return varianza**(1/2)


def percentil(vals_in,q,interpolacion="lineal"):
    """
    Calcula el percentil de una lista de numeros
    Detecta y elimina valores NaN
    
    Paràmetros
    ----------
    vals: lista
        lista con los numeros
    q: int
        percentil a ser calculado(0-100)
    Retorna
    -------
    percentil:float
        percentil de los numeros (excluyendo NaNs)
    """
    
    
    #eliminamos los valores que sean NaNs
    vals=[]
    for v in vals_in:
        if math.isfinite(v):
            vals.append(v)

    # Ordenar la lista dada como input in-place
    vals.sort()

    if interpolacion=="lineal":
        ieff=(len(vals)-1)*(q/100)
        i=int(ieff)
        j=min(i+1,len(vals)-1)
        fraction=ieff-i
        #interpolacion lineal
        percentile=vals[i]+(vals[j]-vals[i])*fraction

        return percentile
    
def rango_intercuartilico(vals_in):
    """
    Calcula el rango intercuartilico de una lista de numeros
    Detecta y elimina valores NaN
    
    Paràmetros
    ----------
    vals: lista
        lista con los numeros
        
    Retorna
    -------
    rango intercuartilico:float
        rango intercuartilico de los numeros (excluyendo NaNs)
    """
    
    
    #eliminamos los valores que sean NaNs
    vals=[]
    for v in vals_in:
        if math.isfinite(v):
            vals.append(v)
    iqr=percentil(vals,75)-percentil(vals,25)
    return iqr
def mad(vals_in):
    """
    Calcula el MAD de una lista de numeros
    Detecta y elimina valores NaN
    
    Paràmetros
    ----------
    vals: lista
        lista con los numeros
        
    Retorna
    -------
    MAD:float
        MAD de los numeros (excluyendo NaNs)
    """
    
    
    #eliminamos los valores que sean NaNs
    vals=[]
    for v in vals_in:
        if math.isfinite(v):
            vals.append(v)
    #calculamos la mediana de los datos orginales
    mediana1=mediana(vals)
    #creamos una lista vacia para ingresar cada dato restado con la mediana1 y su valor absoluto
    desviaciones_med=[]
    for i in vals:
        desviaciones_med.append(abs(i-mediana1))
    #por ultimo se calcula la mediana de la lista el cual sera el MAD
    mad=mediana(desviaciones_med)
    return mad

def covarianza(vals_x,vals_y):
    """
    Calcula la covarianza de una lista de numeros
    Detecta y elimina valores NaN
    
    Paràmetros
    ----------
    vals: lista
        lista con los numeros
        
    Retorna
    -------
    covarianza:float
        covarianza de los numeros (excluyendo NaNs)
    """
    
    
      #eliminamos los valores que sean NaNs
    x=[]
    y=[]
    
    for i in range(len(vals_x)):
        if math.isfinite(vals_x[i]) and math.isfinite(vals_y[i]): 
            x.append(vals_x[i])
            y.append(vals_y[i])
  

    p_x=promedio(x)
    p_y=promedio(y)
    
    tt=[]
    
    for xv,yv in zip(x,y):
        tt.append( (xv-p_x)*(yv-p_y))
        
    covarianza=sum(tt)/len(tt)
    return covarianza
    
def correlacion(x, y):
    """
    Calcula la correlación de Pearson entre dos listas de valores. Ignora valores NaN.
    Parámetros:
    -----------
    x, y: list
        Listas de valores numéricos. Deben tener la misma longitud.
    Retorna:
    --------
    correlacion: float
        Coeficiente de correlación de Pearson.
    """
    x_vals, y_vals = [], []
    for xi, yi in zip(x, y):
        if math.isfinite(xi) and math.isfinite(yi):
            x_vals.append(xi)
            y_vals.append(yi)
    
    cov = covarianza(x_vals, y_vals)
    std_x = desviacion_estandar(x_vals)
    std_y = desviacion_estandar(y_vals)
    
    return cov / (std_x * std_y)
