import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import random
import math
df=pd.read_csv('activos.csv')

def rendimiento(): #Rendimiento del portafolio de inversión
    activos=df.to_numpy(copy=True)
    a, r, rp, ri = [], 0, [], []
    for i in range(11):
        b=[]
        for j in range(1, 11): b.append((activos[i][j]-activos[i+1][j])/activos[i+1][j])
        a.append(b)
    return a #Tabla de rendimientos

rendimientos= rendimiento()
covarianza = np.cov(rendimientos, rowvar=False)

def rendimientoPActivo(): #Rendimientos por activos
    rp = []
    global rendimientos
    for j in range(10):
        r=0
        for i in range(11): r+=rendimientos[i][j]
        rp.append(r/11)
    return rp #Rendimiento promedio 

def rendP(w):
    rn, rp =rendimientoPActivo(), 0
    for n in range(10): rp += (rn[n]*w[n])
    return rp
    
def riesgoP(w):
    global covarianza
    re = 0
    for n in range(10):
        for m in range(10): re += (w[n]*w[m]*covarianza[n][m])
    return re

def fobj(w):
    rendimiento, riesgo = rendP(w), riesgoP(w)
    return rendimiento/riesgo

def restriccion(w):
    suma = sum(w)
    for i in range(10): w[i]=(w[i]/suma)
    return w

def cromosoma():
    w=[random.random() for i in range(10)]
    return restriccion(w)

def crearPoblacion():
    return [cromosoma() for i in range(100)]

poblacion = crearPoblacion()
fitness = []

def calcularFitness():
    global poblacion, fitness
    fitness = []
    for cromosoma in poblacion:
        fitness.append(fobj(cromosoma))

def ordenar():
    global poblacion, fitness
    ordenados = sorted(zip(fitness, poblacion), key = lambda x: x[0], reverse = True)
    poblacion = [cromosoma[1] for cromosoma in ordenados]
    fitness = [j[0] for j in ordenados]

def current(c1):
    c2, c3 = random.randint(0, 99), random.randint(0, 99)
    if c2 == c1: c2 = random.randint(0, 99)
    elif c2 == c3 or c1 == c3: c3 = random.randint(0, 99)
    return c2, c3

def vectorMutacion():
    global poblacion
    vectorM=[]
    for i in range(len(poblacion)):
        c2, c3 =current(i)
        f = random.random()
        vm = [abs(poblacion[i][j] + (f * (poblacion[c2][j]-poblacion[c3][j]))) for j in range(10)]
        vectorM.append(restriccion(vm))
    return vectorM

def cruza(vm):
    global poblacion
    hijos = []
    for i in range(len(poblacion)):
        cr, nuevoCrom = random.random(), []
        for j in range(10):
            if (random.random() == cr) or (random.randint(0, 9) == j):
                nuevoCrom.append(vm[int(i)][int(j)])
            else:
                nuevoCrom.append(poblacion[int(i)][int(j)])
        hijos.append(restriccion(nuevoCrom))
    return hijos

def reemplazo(hijos):
    global poblacion
    for i in range(len(poblacion)):
        if fobj(hijos[i])>fobj(poblacion[i]):
            poblacion[i] = hijos[i]

historialEvolucion = []
convergencia, fitnessAnt, iteraciones = 0, 0, 0
while convergencia<=70:
    vMutacion = vectorMutacion()
    hijos = cruza(vMutacion)
    reemplazo(hijos)
    calcularFitness()
    ordenar()
    historialEvolucion.append(fitness[0])
    if fitness[0] == fitnessAnt:
        convergencia +=1
    else:
        fitnessAnt = fitness[0]
        convergencia=0
    iteraciones +=1

plt.plot([x for x in range(len(historialEvolucion))], historialEvolucion)
plt.show()
empresas = ['HYUNDAI', 'HONDA', 'GM', 'VOLSKWAGEN', 'FORD', 'TESLA', 'MAZDA', 'BMW', 'TOYOTA', 'VOLVO']
plt.pie(poblacion[0], labels=empresas)
plt.title('PORTAFOLIO DE INVERSIÓN')
plt.show()
rend=rendP(poblacion[0])
ries=riesgoP(poblacion[0])
plt.bar(['RENDIMIENTO', 'RIESGO'], [rend, ries], color=['green','blue'], alpha=0.3)
plt.title('GRÁFICA DE RIESGO Y RENDIMIENTO FINAL')
plt.show()
print(rend)
print(ries)