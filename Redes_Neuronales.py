#########################################################################################
          #        Trabajo Práctico N°1 de Redes Neuronales        #
          #         Programa de Rodríguez Natalia Agustina         #
#########################################################################################

import numpy as np
import pandas as pd
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from matplotlib import style
import random
from semanas_epi import orden

style.use('ggplot')

""" valores """
tau=0.01
E=-65E-3
R=10E6
I=2E-9


########### Inciso A y B #################

# con I_e=0, f(V(t))=(E-V(t))/tau
# La función fun gráfica la recta f(V(t)) para un rango x de valores para V 

def fun(b):
    return (E-b)/tau

x=np.arange(-1,1,0.1)
tramo1=np.arange(-1,E,0.1)
tramo2=np.arange(E+0.1,1,0.1)
a=np.zeros(len(tramo1))

""" GRAFICA PARA LA ESTABILIDAD DEL PUNTO FIJO"""
"""
plt.plot(x,fun(x),color='m',label='f(t)')
plt.plot(tramo1,a,'r>')
plt.plot(tramo2,a,'r<')
plt.plot(E,fun(E),marker='o',color='black')
plt.annotate('PF Estable',xy=(E,fun(E)),xytext=(E,-50),arrowprops=dict(facecolor='m', shrink=0.05),)
plt.title('Retrato de Fase')
plt.xlabel('V')
plt.ylabel('dV/dt')
plt.legend()
plt.grid(True)
plt.show()
"""
""" CAMPO DE DIRECCIONES """
"""
nx, ny=.001,.001
x=np.arange(0,0.020,nx)
y=np.arange(-0.05,-0.04,ny)

X,Y=np.meshgrid(x,y)

dy= -Y/tau + (E+R*I)/tau
dx=np.ones(dy.shape)

plt.quiver(X,Y,dx,dy,color='purple')
plt.axhline(y=(E+R*I),xmin=0,xmax=1,linestyle='--', label='V*')
plt.ylabel('V(t)[volts]')
plt.xlabel('t[seg]')
plt.title('Campo de dirección')
plt.legend()
plt.show()
"""

################# INCISO D Y E #########################

paso=0.05E-3
t=np.arange(0,0.2,paso)

def sol_analitica(e,t):
    V=E+R*e
    y= V + (E-V) * np.exp(- (t/tau))
    return y


def ED(v,t):
    dvdt= (E+ R*I- v)/tau
    return dvdt


def rk4(f,y0,t):
    n=np.size(t)
    y=np.zeros((n,np.size(y0)))
    y[0]=y0
    for i in range(n-1):
        h=paso
        k1=f(y[i],t[i])
        k2=f(y[i] + k1*h/2., t[i] + h/2.)
        k3=f(y[i] + k2*h/2., t[i] + h/2.)
        k4=f(y[i] + k3*h,t[i] + h)
        y[i+1]=y[i] + (h/6.)* (k1+2*k2+2*k3+k4)
    return y

sol_rk4=rk4(ED,E,t)

"""
plt.plot(t,sol_analitica(I,t), color='b',linewidth=3,label='analítica')
plt.plot(t,sol_rk4,color='y',linewidth=1.5,label='numérica')
plt.axhline(y=(E+R*I),xmin=0,xmax=1,linestyle='--',label='V*')
plt.ylabel('V[volts]')
plt.xlabel('t[seg]')
plt.legend()
plt.show()
"""
################## INCISO F ####################

Vu=-50E-3

def rk4_umbral(f,y0,t):
    n=np.size(t)
    y=np.zeros((n,np.size(y0)))
    y[0]=y0
    for i in range(n-1):
        h=paso
        if(y[i]>=Vu):y[i]=y0
        k1=f(y[i],t[i])
        k2=f(y[i] + k1*h/2., t[i] + h/2.)
        k3=f(y[i] + k2*h/2., t[i] + h/2.)
        k4=f(y[i] + k3*h,t[i] + h)
        y[i+1]=y[i] + (h/6.)* (k1+2*k2+2*k3+k4)
    return y

sol_umbral=rk4_umbral(ED, E, t)

"""
plt.plot(t,sol_umbral,color='m',label='RK4')
plt.axhline(y=(Vu),xmin=0,xmax=1,linestyle='--',label='V umbral')
plt.ylabel('V[volts]')
plt.xlabel('t[seg]')
plt.legend()
plt.show()
"""

################## INCISO G E Y #####################

""" I=I_0*cos(t/30ms) """

def corriente(t):
    I_0=2.5E-9
    z=I_0*np.cos(t/30E-3)
    return z

def ED_I(v,t):
    dvdt= (E+ R*corriente(t) - v)/tau
    return dvdt

def rk4_I(f,y0,t):
    n=np.size(t)
    y=np.zeros((n,np.size(y0)))
    y[0]=y0
    for i in range(n-1):
        h=paso
        if(y[i]>=Vu):y[i]=y0
        k1=f(y[i],t[i])
        k2=f(y[i] + k1*h/2., t[i] + h/2.)
        k3=f(y[i] + k2*h/2., t[i] + h/2.)
        k4=f(y[i] + k3*h,t[i] + h)
        y[i+1]=y[i] + (h/6.)* (k1+2*k2+2*k3+k4)
    return y

sol_I=rk4_I(ED_I,E,t)

"""
plt.plot(t,sol_I,color='m',label='V(t)')
plt.axhline(y=(Vu),xmin=0,xmax=1,linestyle='--',label='V umbral')
plt.ylabel('V[volts]')
plt.xlabel('t[seg]')
plt.legend()
plt.show()
"""

""" I = a muchos términos """

def corriente1(t):
    I_0=0.35E-9
    z=I_0*(np.cos(t/3E-3) + np.sin(t/5E-3) + np.cos(t/7E-3) + np.sin(t/11E-3) + np.cos(t/13E-3))**2
    return z

def ED_I1(v,t):
    dvdt= (E+ R*corriente1(t) - v)/tau
    return dvdt

"""
sol_I1=rk4_I(ED_I1,E,t)
plt.plot(t,sol_I1,color='m',label='V(t)')
plt.axhline(y=(Vu),xmin=0,xmax=1,linestyle='--',label='V umbral')
plt.ylabel('V[volts]')
plt.xlabel('t[seg]')
plt.legend()
plt.show()
"""

############### INCISO H ###############

Ic=(Vu-E)/R
print(Ic)

def frecuencia(x):
    periodo= tau * np.log(R*x/(R*x - Vu + E))
    return (1./periodo)

corrientes=np.linspace(0.,2.5E-9,80)
valor=np.zeros(np.size(corrientes))

for i in range(0,np.size(corrientes)):
    if((R*corrientes[i]/(R*corrientes[i]-Vu+E))<=0.):
        valor[i]=0.
    else:
        valor[i]=frecuencia(corrientes[i])

#for i in range(0,len(corrientes)):
#    print(corrientes[i],valor[i])

"""
plt.scatter(corrientes,valor,color='m')
plt.ylabel('frecuencia[1/seg]')
plt.xlabel('I[nA]')
plt.show()
"""
######################### FIN :D #########################