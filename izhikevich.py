#########################################################################################
          #        Trabajo Práctico N°2 de Redes Neuronales        #
          #         Programa de Rodríguez Natalia Agustina         #
#########################################################################################


import numpy as np
import matplotlib.pyplot as plt

############## PARAMETROS ##################################
a=0.02
b=0.2
c=-65
d=8

v0=-70    #VALOR INICIAL DE POTENCIAL
inicio=0  #VALOR INICIAL DE LA CORRIENTE
aux=10    #ALTURA DEL PASO DE CORRIENTE
umbral=30   #POTENCIAL UMBRAL
t_i=0       #TIEMPO DE INICIO 
t_f=200     #TIEMPO FINAL
Dt=0.1     #PASO TEMPORAL
t_top=50   #TIEMPO PARA LA CORRIENTE AUXILIAR

#############################################################
parametros=np.array([a,b,c,d],dtype=float)
t=np.arange(t_i,t_f,Dt)

I=np.zeros(np.size(t))

############### PARA RS ###################
I4=np.zeros(np.size(t))
t1=10
t2=100
t3=102
Ia=0.2
Ib=10
inicial=0


########### PARA GRAFICAR LA CORRIENTE DE RS #######
for i in range(0,np.size(I4)):
    if (t[i]<t1): I4[i]=inicial
    if (t[i]>=t1) and (t[i]<t2): I4[i]=Ia
    if (t[i]>=t2) and (t[i]<t3): I4[i]=Ib
    if (t[i]>=t3): I4[i]=Ia
###########################################


########## PARA GRAFICAR LA CORRIENTE #######
for i in range(0,np.size(I)):
    if (t[i]<=t_top): I[i]=inicio
    if (t[i]>t_top):I[i]=aux
##############################################



""" EL MODELO """
def modelo(z,t,I,p1,p2):
    corriente=inicio
    if (t>t_top): corriente=aux

    ########### Para RS #################
    """
    if (t<t1): corriente=inicial
    if (t>=t1) and (t<t2): corriente=Ia
    if (t>=t2) and (t<t3): corriente=Ib
    if (t>=t3): corriente=Ia
    """
    #####################################

    v=z[0]
    u=z[1]

    dvdt= 0.04*v*v + 5*v + 140 - u + corriente
    dudt= p1*((p2*v) - u)

    return np.array([dvdt,dudt])


""" CONDICION INICIAL """
z0=np.zeros(2)
z0[0]=v0                   #V_0
z0[1]=parametros[1]*z0[0]  #U_0



################# RK4 ###########################
def rk4(f,y0,t,args=()):
    n=len(t)
    y=np.zeros((n,len(y0)))
    y[0]=y0
    for i in range(n-1):
        h=Dt
        if(y[i][0] >= umbral):
            y[i][0]=c
            y[i][1]=y[i][1]+parametros[3]
        k1=f(y[i],t[i],*args)
        k2=f(y[i] + k1*h/2., t[i] + h/2.,*args)
        k3=f(y[i] + k2*h/2., t[i] + h/2.,*args)
        k4=f(y[i] + k3*h,t[i] + h,*args)
        y[i+1]=y[i] + (h/6.)* (k1+2*k2+2*k3+k4)
    return y
#################################################

solucion=rk4(modelo,z0,t,args=(I,parametros[0],parametros[1]))
V=solucion[:,0]
U=solucion[:,1]


#plt.xlim(0,200)
plt.plot(t,V,color='m',label='v(t)')
plt.plot(t,U,color='pink',label='u(t)')
plt.plot(t,I,color='green',label='I(t)')
plt.xlabel('t')
plt.ylabel('V')
plt.title('Low-Thershold Spiking (LTS)')
plt.legend()
plt.show()



