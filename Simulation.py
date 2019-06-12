#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 27 16:15:28 2019

@author: bertrambrovang
"""


from collections import OrderedDict
from typing import Dict, Callable
from inspect import signature
import numpy.random as rd
import matplotlib.pyplot as plt
import json
from functools import partial
import numpy as np
import math as ma
import time

from time import sleep, monotonic

import qcodes as qc
#from doNd import do2d
from qcodes import Station
from qcodes.dataset.experiment_container import new_experiment
from qcodes.dataset.database import initialise_database
from qcodes.tests.instrument_mocks import DummyInstrument
from qcodes.dataset.param_spec import ParamSpec
from qcodes.dataset.measurements import Measurement
from qcodes.dataset.plotting import plot_by_id
from qcodes.dataset.data_set import load_by_id
from qcodes.dataset.data_export import get_shaped_data_by_runid
from qcodes.instrument.base import Instrument
#from radialsearch import radialSearch2D,radialSearch3Dsphere
from scipy.spatial import Delaunay,ConvexHull
import plotly.plotly as py
import plotly.figure_factory as FF
import plotly.graph_objs as go
from plotly.offline import plot, init_notebook_mode
import matplotlib.tri as mtri
import matplotlib as mpl
from mpl_toolkits import mplot3d

from typing import Callable, Sequence, Union, Tuple, List, Optional
import os
import datetime
import math
import numpy as np
import matplotlib
from qcodes.instrument.base import _BaseParameter
from qcodes import config
import itertools as itera
#from export_functions import export_by_id
#from totalplottest import multiRadialPlot
#dot to gate capacitances        
#Cg=np.array([[1.73,0.635,0.24,0.66,0,0,0,0],\
#             [0.435,1.91,0.27,0.2,0,0,0,0],\
#             [0.38,0.20,2.63,0.2,0,0,0,0],\
#             [0.168,0.2,0.2,1.4,0,0,0,0]])*1e-18

Cg=np.array([[1.73,0.235,0.24,0.66,0,0,0,0],\
             [0.235,1.91,0.27,0.2,0,0,0,0],\
             [0.38,0.20,2.63,0.2,0,0,0,0],\
             [0.168,0.2,0.2,1.4,0,0,0,0]])*1e-18

##fake 
#Cg=np.array([[5,0,0,0,0.1,0.1,0,0],\
#             [0,5,0,0,0.1,0.1,0,0],\
#             [0,0,5,0,0,0,0,0],\
#             [0,0,0,0.0001,0,0,0,0]])*1e-18


#dot to dot capacitances
c0102=1.05e-18
c0103=0.75e-18
c0104=1.05e-18
c0203=1.05e-18
c0204=0.75e-18
c0304=1.05e-18

side=1.05e-18
diag=0.75e-18

#run this to initialize a new setup, 
#test(numberofelectronsprdot,dots,cgmatrix,cdotcapacitances,,,,,,,sweepgate1,sweepgate2,,sweepgate4=None)
# dots range from 2-4 and sweepgates from 0-7 where gate3 and 4 can be left as None
#np.array([[8*10**(-23)],[5*10**(-22)],[8*10**(-22)]])
# t has to be a list [t] with all tunnel couplings t=[(1-2),(2-3),(3-4),(1-4),(1-3),(2-4)]
#test(5,[2,3],Cg,[0,4e-3,0,0,0,0],[0,0,0,0],side,diag,0,1,2,3) 
"""
test(2,[1,2],Cg,[0,0,0,0,0,0],[0,0,0,0],side,diag,0,1,2,3) 
"""
#run this for 2d radialsearch
"""
radialSearch2D(simtbpar.v1,0.001,simtbpar.v2,0.001,simtb01.CD,pointsTheta=250,maxR=0.080,pointsR=120)
"""
#run this for 3d radialsearch
"""
radialSearch3Dsphere(simtbpar.v1, 0.000001, simtbpar.v2 , 0.000001, simtbpar.v3, 0.000001, simtb01.CD,pointsSphere = 750, minR=0.0005,maxR=0.3,pointsR=300)
"""

#run this for do2d
"""
do2d(simtbpar.v1,0,0.1,100,0.001,simtbpar.v2,0,0.1,100,0.001,simtb01.CD)
"""

#plotting multiple radialSearch3Dsphere data sets in one plot
"""
multiRadialPlot(73,74,75) #dataids
"""
def test(n,dots,cgo,t,r,side,diag,sweep1,sweep2,sweep3=None,sweep4=None):
    """
    the test function initializes a mock setup, defines the dummy instruments
    """
    global station,simtbpar,simtb01
    
    
    Electrons=np.arange(n+1)
    emptyMatrix=[[0],[0],[0],[0]]
    for i in dots:
        emptyMatrix[i-1]=np.array(Electrons)

    stateList=np.array(list(itera.product(emptyMatrix[0],emptyMatrix[1],emptyMatrix[2],emptyMatrix[3])))

    tM=tMatrix(n,dots,stateList,t,r)
    
    cg=np.zeros(cgo.shape)
    
    dots=np.array(dots)-1

    
    if len(dots)==1:
        C=np.zeros([4,4])
        cg[dots[0],dots[0]]=cgo[dots[0],dots[0]] #gives cgnew a dot to own gate cap
        for i in range(4): #gives C diagonal elements
                if i==dots[0]: 
                    C[i,i]=sum(cg[i])
                elif i!=dots[0]:
                    C[i,i]=1e-30
                    
    
    elif len(dots)==2:
        C=np.zeros([4,4])
        for i in dots: #gives cgnew capacitances used with these dots
            for j in dots: 
                cg[i,j]=cgo[i,j]
                
                if i!=j: #else:    
                    if j-i!=2 and j-i!=-2:
                        C[i,j]=-side
                        C[i,i]=C[i,i]+side
                    else:
                        C[i,j]=-diag
                        C[i,i]=C[i,i]+diag
                elif i==j: #if i==j:
                    if i in dots:
                        C[i,i]=C[i,i]+sum(cg[i])
    
        for i in range(4):
             if i not in dots:
                        C[i,i]=1e-30  #set dot to gate for dots not active
    

    
    elif len(dots)==3:
        C=np.zeros([4,4])
        for i in dots:            
            for j in dots: 
                cg[i,j]=cgo[i,j]
                
                if i!=j: #else:    
                    if j-i!=2 and j-i!=-2:
                        C[i,j]=-side
                        C[i,i]=C[i,i]+side
                    else:
                        C[i,j]=-diag
                        C[i,i]=C[i,i]+diag
                elif i==j: #if i==j:
                    if i in dots:
                        C[i,i]=C[i,i]+sum(cg[i])
    
                        
                        
        for i in range(4):
             if i not in dots:
                        C[i,i]=1e-30  #set dot to gate for dots not active
            
    elif len(dots)==4:
        cg=cgo
        C_11=sum(cg[0])+2*side+diag
        C_22=sum(cg[1])+2*side+diag
        C_33=sum(cg[2])+2*side+diag
        C_44=sum(cg[3])+2*side+diag
        #capacitance matrix needed in calculation 
        
        C=np.array([[C_11,-side,-diag,-side],\
                    [-side,C_22,-side,-diag],\
                    [-diag,-side,C_33,-side],\
                    [-side,-diag,-side,C_44]])
    
    
    Cinv=np.linalg.inv(C)
    
    
    #initializes dummy instruments if not already started in kernel
    try:
       simtbpar = DummyInstrument('simtbpar', gates=['v1', 'v2','v3','v4'])
    except KeyError:
        print('simtbpar already loaded')


    try:
       simtb01 = DummyInstrument('simtb01', gates=['Emin','Emint','CD','CDt','Q1Exp','Q2Exp','Q3Exp','Q4Exp','E0','E1','E2','E3','E4','E5','E6','E7','E8','E9','Ediff','Estate1','Estate2','Estate3','Estate1t','Estate2t','Estate3t'])
    except KeyError:
        print('simtb01 already loaded')

        
    #station=qc.Station(simtbpar,simtb01)  IS STATION IMPORTANT?       
    
    
    name=Cinv1(n,dots,Cinv,tM,stateList,sweep1,sweep2,sweep3,sweep4,cg)
    next(name)
    
    nameEminCD=CinvEminCD(n,dots,Cinv,tM,stateList,sweep1,sweep2,sweep3,sweep4,cg)
    next(nameEminCD)
    
    nameE=CinvE(n,dots,Cinv,tM,stateList,sweep1,sweep2,sweep3,sweep4,cg)
    next(nameE)
    
    nameExQ=CinvExQ(n,dots,Cinv,tM,stateList,sweep1,sweep2,sweep3,sweep4,cg)
    next(nameExQ)
    
    
    #customgetter 1 sends back Emin and customgetter 2 sends back the Charge state with min. energy
    def EminGet(dac):
        val1 = nameEminCD.send((simtbpar.v1.get(),simtbpar.v2.get(),simtbpar.v3.get(),simtbpar.v4.get()))[0]
        next(nameEminCD)
        return val1
    
    def EmintGet(dac):
        val1 = nameEminCD.send((simtbpar.v1.get(),simtbpar.v2.get(),simtbpar.v3.get(),simtbpar.v4.get()))[1]
        next(nameEminCD)
        return val1
     
    def CDGet(dac):
        val2 = nameEminCD.send((simtbpar.v1.get(),simtbpar.v2.get(),simtbpar.v3.get(),simtbpar.v4.get()))[2]
        next(nameEminCD)
        return val2
    
    def CDtGet(dac):
        val2 = nameEminCD.send((simtbpar.v1.get(),simtbpar.v2.get(),simtbpar.v3.get(),simtbpar.v4.get()))[3]
        next(nameEminCD)
        return val2
    
    def Q1ExpGet(dac):
        val2 = nameExQ.send((simtbpar.v1.get(),simtbpar.v2.get(),simtbpar.v3.get(),simtbpar.v4.get()))[0]
        next(nameExQ)
        return val2
    
    def Q2ExpGet(dac):
        val2 = nameExQ.send((simtbpar.v1.get(),simtbpar.v2.get(),simtbpar.v3.get(),simtbpar.v4.get()))[1]
        next(nameExQ)
        return val2
    
    def Q3ExpGet(dac):
        val2 = nameExQ.send((simtbpar.v1.get(),simtbpar.v2.get(),simtbpar.v3.get(),simtbpar.v4.get()))[2]
        next(nameExQ)
        return val2
    
    def Q4ExpGet(dac):
        val2 = nameExQ.send((simtbpar.v1.get(),simtbpar.v2.get(),simtbpar.v3.get(),simtbpar.v4.get()))[3]
        next(nameExQ)
        return val2

    def E0Get(dac):
        val1 = nameE.send((simtbpar.v1.get(),simtbpar.v2.get(),simtbpar.v3.get(),simtbpar.v4.get()))[0]
        next(nameE)
        return val1

    def E1Get(dac):
        val1 = nameE.send((simtbpar.v1.get(),simtbpar.v2.get(),simtbpar.v3.get(),simtbpar.v4.get()))[1]
        next(nameE)
        return val1

    def E2Get(dac):
        val1 = nameE.send((simtbpar.v1.get(),simtbpar.v2.get(),simtbpar.v3.get(),simtbpar.v4.get()))[2]
        next(nameE)
        return val1
    
    def E3Get(dac):
        val1 = nameE.send((simtbpar.v1.get(),simtbpar.v2.get(),simtbpar.v3.get(),simtbpar.v4.get()))[3]
        next(nameE)
        return val1
     
    def E4Get(dac):
        val1 = nameE.send((simtbpar.v1.get(),simtbpar.v2.get(),simtbpar.v3.get(),simtbpar.v4.get()))[4]
        next(nameE)
        return val1
    
    def E5Get(dac):
        val1 = nameE.send((simtbpar.v1.get(),simtbpar.v2.get(),simtbpar.v3.get(),simtbpar.v4.get()))[5]
        next(nameE)
        return val1
    
    def E6Get(dac):
        val1 = nameE.send((simtbpar.v1.get(),simtbpar.v2.get(),simtbpar.v3.get(),simtbpar.v4.get()))[6]
        next(nameE)
        return val1
    
    def E7Get(dac):
        val1 = nameE.send((simtbpar.v1.get(),simtbpar.v2.get(),simtbpar.v3.get(),simtbpar.v4.get()))[7]
        next(nameE)
        return val1
    
    def E8Get(dac):
        val1 = nameE.send((simtbpar.v1.get(),simtbpar.v2.get(),simtbpar.v3.get(),simtbpar.v4.get()))[8]
        next(nameE)
        return val1

    def E9Get(dac):
        val1 = nameE.send((simtbpar.v1.get(),simtbpar.v2.get(),simtbpar.v3.get(),simtbpar.v4.get()))[9]
        next(nameE)
        return val1   
    
    def Estate1Get(dac):
        val1 = name.send((simtbpar.v1.get(),simtbpar.v2.get(),simtbpar.v3.get(),simtbpar.v4.get()))[0]
        next(name)
        return val1
    
    def Estate2Get(dac):
        val1 = name.send((simtbpar.v1.get(),simtbpar.v2.get(),simtbpar.v3.get(),simtbpar.v4.get()))[1]
        next(name)
        return val1
    
    def Estate3Get(dac):
        val1 = name.send((simtbpar.v1.get(),simtbpar.v2.get(),simtbpar.v3.get(),simtbpar.v4.get()))[2]
        next(name)
        return val1
    
    def Estate1tGet(dac):
        val1 = name.send((simtbpar.v1.get(),simtbpar.v2.get(),simtbpar.v3.get(),simtbpar.v4.get()))[3]
        next(name)
        return val1
    
    def Estate2tGet(dac):
        val1 = name.send((simtbpar.v1.get(),simtbpar.v2.get(),simtbpar.v3.get(),simtbpar.v4.get()))[4]
        next(name)
        return val1
    
    def Estate3tGet(dac):
        val1 = name.send((simtbpar.v1.get(),simtbpar.v2.get(),simtbpar.v3.get(),simtbpar.v4.get()))[5]
        next(name)
        return val1
        
    simtb01.Emin.get=partial(EminGet,simtbpar)
    simtb01.Emint.get=partial(EmintGet,simtbpar) 
    simtb01.CD.get=partial(CDGet,simtbpar)
    simtb01.CDt.get=partial(CDtGet,simtbpar)
    simtb01.Q1Exp.get=partial(Q1ExpGet,simtbpar)
    simtb01.Q2Exp.get=partial(Q2ExpGet,simtbpar)
    simtb01.Q3Exp.get=partial(Q3ExpGet,simtbpar)
    simtb01.Q4Exp.get=partial(Q4ExpGet,simtbpar)
    simtb01.E0.get=partial(E0Get,simtbpar)
    simtb01.E1.get=partial(E1Get,simtbpar)
    simtb01.E2.get=partial(E2Get,simtbpar)
    simtb01.E3.get=partial(E3Get,simtbpar)
    simtb01.E4.get=partial(E4Get,simtbpar)
    simtb01.E5.get=partial(E5Get,simtbpar)
    simtb01.E6.get=partial(E6Get,simtbpar)
    simtb01.E7.get=partial(E7Get,simtbpar)
    simtb01.E8.get=partial(E8Get,simtbpar)
    simtb01.E9.get=partial(E9Get,simtbpar)
    simtb01.Estate1.get=partial(Estate1Get,simtbpar)
    simtb01.Estate2.get=partial(Estate2Get,simtbpar)
    simtb01.Estate3.get=partial(Estate3Get,simtbpar)
    simtb01.Estate1t.get=partial(Estate1tGet,simtbpar)
    simtb01.Estate2t.get=partial(Estate2tGet,simtbpar)
    simtb01.Estate3t.get=partial(Estate3tGet,simtbpar)
    

def Cinv1(n,dots,Cinv,tM,stateList,sweep1,sweep2,sweep3,sweep4,cg): #call with all the cg....
    print('Cinv1')
    #gate voltages
    V=np.array([0,0,0,0,0,0,0,0],dtype=object) #USE THIS when you want to set specific gates to specific voltages
    while True:
        (Vs1,Vs2,Vs3,Vs4) = yield
        V[sweep1]=Vs1 #gets send gatevoltages from do1d,do2d or another method of measuring
        V[sweep2]=Vs2 
        V[sweep3]=Vs3
        V[sweep4]=Vs4
        
        Q=cg*V #Q is send to Energy
        
        k=0
        
        if (tM==0).all():
           Em=np.empty([(n+1)**len(dots)],dtype=object)
           for state in stateList:
               Em[k]=Energy(state,Cinv,Q)
               k+=1
               
           yield Em  
                
        else:
            Em=np.empty([(n+1)**len(dots)],dtype=object)
            for state in stateList:
                Em[k]=Energy(state,Cinv,Q)
                k+=1
            placeHolder=tM+np.diag(Em)
            eigVal,eigVec=np.linalg.eig(placeHolder.astype(np.float64))
            Emt=eigVal[:][np.where(abs(eigVec[:,:])==np.amax(abs(eigVec[:,:]),axis=0))[1]]
                
            yield (Em[16],Em[12],Em[8],Emt[16],Emt[12],Emt[8])
        
def CinvEminCD(n,dots,Cinv,tM,stateList,sweep1,sweep2,sweep3,sweep4,cg): #call with all the cg....
    print('CinvEminCD')
    #gate voltages
    V=np.array([0,0,0,0,0,0,0,0],dtype=object) #USE THIS when you want to set specific gates to specific voltages
    while True:
        (Vs1,Vs2,Vs3,Vs4) = yield
        V[sweep1]=Vs1 #gets send gatevoltages from do1d,do2d or another method of measuring
        V[sweep2]=Vs2 
        V[sweep3]=Vs3
        V[sweep4]=Vs4
    
        Q=cg*V #Q is send to Energy
        
        k=0
        
        if (tM==0).all():
            Em=np.empty([(n+1)**len(dots)],dtype=object)
            for state in stateList:
                Em[k]=Energy(state,Cinv,Q)
                k+=1
            Emt=Em
        
        else:
            
            Em=np.empty([(n+1)**len(dots)],dtype=object)
            for state in stateList:
                Em[k]=Energy(state,Cinv,Q)
                k+=1
            placeHolder=tM+np.diag(Em)
            eigVal,eigVec=np.linalg.eig(placeHolder.astype(np.float64))           
            Emt=eigVal[:][np.where(abs(eigVec[:,:])==np.amax(abs(eigVec[:,:]),axis=0))[1]]
            
        yield (np.min(Em),np.min(Emt),np.argmin(Em).astype(float),np.argmin(Emt).astype(float))

def CinvE(n,dots,Cinv,tM,stateList,sweep1,sweep2,sweep3,sweep4,cg): #call with all the cg....
    print('CinvE')
    #gate voltages
    V=np.array([0,0,0,0,0,0,0,0],dtype=object) #USE THIS when you want to set specific gates to specific voltages
    while True:
    
        (Vs1,Vs2,Vs3,Vs4) = yield
        V[sweep1]=Vs1 #gets send gatevoltages from do1d,do2d or another method of measuring
        V[sweep2]=Vs2 
        V[sweep3]=Vs3
        V[sweep4]=Vs4
    
        Q=cg*V #Q is send to Energy
        
        k=0
        
        if (tM==0).all():
            Em=np.empty([(n+1)**len(dots)],dtype=object)
            for state in stateList:
                Em[k]=Energy(state,Cinv,Q)
                k+=1
        
        else:
            Em=np.empty([(n+1)**len(dots)],dtype=object)
            for state in stateList:
                Em[k]=Energy(state,Cinv,Q)
                k+=1
            placeHolder=tM+np.diag(Em)
            eigVal,eigVec=np.linalg.eig(placeHolder.astype(np.float64))           
            Em=eigVal[:][np.where(abs(eigVec[:,:])==np.amax(abs(eigVec[:,:]),axis=0))[1]]
                
                
                
        sortedE=np.sort(Em)
        yield (Em[0],Em[1],Em[2],Em[3],Em[4],Em[5],Em[6],Em[7],Em[8],Em[0])
        #yield (sortedE[0],sortedE[1],sortedE[2],sortedE[3],sortedE[4],sortedE[5],sortedE[6],sortedE[7],sortedE[8],sortedE[9])
        
def CinvExQ(n,dots,Cinv,tM,stateList,sweep1,sweep2,sweep3,sweep4,cg): #call with all the cg....
    #gate voltages
    V=np.array([0,0,0,0,0,0,0,0],dtype=object) #USE THIS when you want to set specific gates to specific voltages
    while True:
    
        (Vs1,Vs2,Vs3,Vs4) = yield
        V[sweep1]=Vs1 #gets send gatevoltages from do1d,do2d or another method of measuring
        V[sweep2]=Vs2 
        V[sweep3]=Vs3
        V[sweep4]=Vs4
    
        Q=cg*V #Q is send to Energy
        
        k=0
        
        if (tM==0).all():
            Em=np.empty([(n+1)**len(dots)],dtype=object)
            for state in stateList:
                Em[k]=Energy(state,Cinv,Q)
                k+=1
           
        else:
           Em=np.empty([(n+1)**len(dots)],dtype=object)
           
           for state in stateList:
               Em[k]=Energy(state,Cinv,Q)
               k+=1
            
           placeHolder=tM+np.diag(Em)
           eigVal,eigVec=np.linalg.eig(placeHolder.astype(np.float64))           
           Em=eigVal[:][np.where(abs(eigVec[:,:])==np.amax(abs(eigVec[:,:]),axis=0))[1]]
                
                
           q1exp=sum((eigVec[:,np.argmin(eigVal)]**2)*stateList[:,0])
           q2exp=sum((eigVec[:,np.argmin(eigVal)]**2)*stateList[:,1])
           q3exp=sum((eigVec[:,np.argmin(eigVal)]**2)*stateList[:,2])
           q4exp=sum((eigVec[:,np.argmin(eigVal)]**2)*stateList[:,3])
                
        yield (q1exp,q2exp,q3exp,q4exp,(q1exp+q2exp+q3exp+q4exp))
        

def Energy(state,Cinv,Q):
    e=1.6021766208*10**(-19)
    Qvec=np.array([-e*state[0]+sum(Q[0]),-e*state[1]+sum(Q[1]),-e*state[2]+sum(Q[2]),-e*state[3]+sum(Q[3])])
    
    M=np.empty([len(Qvec),len(Qvec)],dtype=object)
    H=np.empty([len(Qvec)],dtype=object)
    K=np.empty([len(Qvec)],dtype=object)
    
    M=Cinv*Qvec
    H=np.sum(M,1)

    K=Qvec*H
        
    U=0.5*np.sum(K)  
    U=U*6.24e18
    return U


        
 # FUNCTION ENDS HERE, BELOW IS FOR TAKING MEASUREMENTS
#for Emin
"""
initialise_database()
new_experiment(name='MOCKTR01test',
                          sample_name="no sample")

meas = Measurement()
meas.register_parameter(simtbpar.v1)  # register the first independent parameter
meas.register_parameter(simtbpar.v2)  # register the second independent parameter
meas.register_parameter(simtb01.Emin, setpoints=(simtbpar.v1,simtbpar.v2))  # now register the dependent oone


meas.write_period = 2


with meas.run() as datasaver:

    for set_v1 in np.linspace(0, 0.06, 100):
        for set_v2 in np.linspace(0,0.1,100):
            simtbpar.v1.set(set_v1)
            simtbpar.v2.set(set_v2)
            get_v = simtb01.Emin.get()
            datasaver.add_result((simtbpar.v1, set_v1),
                                 (simtbpar.v2, set_v2),
                                 (simtb01.Emin, get_v))

    dataid = datasaver.run_id  # convenient to have for plotting
    
plot_by_id(dataid)
"""



#            ax.set_xlabel('V1')
#            ax.set_ylabel('V2')
#            ax.set_zlabel('V3')
#            
#            ax.set_title('dataid=%i' %dataid) #add start state to title, somehow
#            x=ds.get_parameter_data('simtbpar_v1')['simtbpar_v1']['simtbpar_v1']
#            y=ds.get_parameter_data('simtbpar_v2')['simtbpar_v2']['simtbpar_v2']    
    
    
def inone(v1,v2,v3,v4):
    initialise_database()
    new_experiment(name='MOCKTR01test',
                              sample_name="no sample")
    
    meas = Measurement()
    meas.register_parameter(simtbpar.v1)  # register the first independent parameter
    meas.register_parameter(simtbpar.v2)  # register the second independent parameter
    meas.register_parameter(simtbpar.v3)  # register the first independent parameter
    meas.register_parameter(simtbpar.v4)  # register the second independent parameter
    
    meas.register_parameter(simtb01.Q1Exp, setpoints=(simtbpar.v1,simtbpar.v2))  # now register the dependent oone
    meas.register_parameter(simtb01.Q2Exp, setpoints=(simtbpar.v1,simtbpar.v2))  # now register the dependent oone
    meas.register_parameter(simtb01.Q3Exp, setpoints=(simtbpar.v1,simtbpar.v2))  # now register the dependent oone
    meas.register_parameter(simtb01.Q4Exp, setpoints=(simtbpar.v1,simtbpar.v2))  # now register the dependent oone
    
    simtbpar.v1.set(v1)
    simtbpar.v2.set(v2)
    simtbpar.v3.set(v3)
    simtbpar.v4.set(v4)
    
    
# for CD
def CDsweep2d(points,axis):
    initialise_database()
    new_experiment(name='MOCKTR01test',
                              sample_name="no sample")
    
    meas = Measurement()
    meas.register_parameter(simtbpar.v1)  # register the first independent parameter
    meas.register_parameter(simtbpar.v2)  # register the second independent parameter
    meas.register_parameter(simtb01.Q1Exp, setpoints=(simtbpar.v1,simtbpar.v2))  # now register the dependent oone
    
    meas.write_period = 2

    k=0
    with meas.run() as datasaver:
    
        for set_v1 in np.linspace(axis[0],axis[1], points):
            for set_v2 in np.linspace(axis[2],axis[3],points):
                simtbpar.v1.set(set_v1)
                simtbpar.v2.set(set_v2)
                test=simtb01.Q1Exp.get()
                datasaver.add_result((simtbpar.v1, set_v1),
                                     (simtbpar.v2, set_v2),
                                     (simtb01.Q1Exp, test))
                k=k+1
    
    dataid = datasaver.run_id  # convenient to have for plotting
        
    plot_by_id(dataid)
    print("dataid= %i" %dataid)

#for CD and Emin
"""
initialise_database()
new_experiment(name='MOCKTR01test',
                          sample_name="no sample")

meas = Measurement()
meas.register_parameter(simtbpar.v1)  # register the first independent parameter
meas.register_parameter(simtbpar.v2)  # register the second independent parameter
meas.register_parameter(simtb01.CD, setpoints=(simtbpar.v1,simtbpar.v2))  # now register the dependent oone
meas.register_parameter(simtb01.Emin, setpoints=(simtbpar.v1,simtbpar.v2))  # now register the dependent oone


meas.write_period = 2


with meas.run() as datasaver:

    for set_v1 in np.linspace(0, 0.1, 100):
        for set_v2 in np.linspace(0,0.1,100):
            simtbpar.v1.set(set_v1)
            simtbpar.v2.set(set_v2)
            get_v = simtb01.CD.get()
            get_v2 = simtb01.Emin.get()
            datasaver.add_result((simtbpar.v1, set_v1),
                                 (simtbpar.v2, set_v2),
                                 (simtb01.CD, get_v),
                                 (simtb01.Emin, get_v2))

    dataid = datasaver.run_id  # convenient to have for plotting
    
plot_by_id(dataid) #wont work
"""
        

#use this for creating 3 dimensional sweeps.
def sweep3d(points):
    initialise_database()
    new_experiment(name='MOCKTR01test',
                              sample_name="no sample")
    
    meas = Measurement()
    meas.register_parameter(simtbpar.v1)  # register the first independent parameter
    meas.register_parameter(simtbpar.v2)  # register the second independent parameter
    meas.register_parameter(simtbpar.v3)  # register the second independent parameter
    meas.register_parameter(simtb01.CD, setpoints=(simtbpar.v1,simtbpar.v2,simtbpar.v3))  # now register the dependent oone
    
    
    meas.write_period = 2
    
    
    with meas.run() as datasaver:
    
        for set_v1 in np.linspace(0, 0.15, points):
            for set_v2 in np.linspace(0,0.15,points):
                for set_v3 in np.linspace(0,0.15,points):
                    simtbpar.v1.set(set_v1)
                    simtbpar.v2.set(set_v2)
                    simtbpar.v3.set(set_v3)
                    get_v = simtb01.CD.get()
                    
                    datasaver.add_result((simtbpar.v1, set_v1),
                                         (simtbpar.v2, set_v2),
                                         (simtbpar.v3, set_v3),
                                         (simtb01.CD, get_v))
    
        dataid = datasaver.run_id  # convenient to have for plotting

    ds = qc.dataset.data_export.load_by_id(dataid) #returns only points on sphere
    x=ds.get_parameter_data('simtbpar_v1')['simtbpar_v1']['simtbpar_v1']
    y=ds.get_parameter_data('simtbpar_v2')['simtbpar_v2']['simtbpar_v2']
    z=ds.get_parameter_data('simtbpar_v3')['simtbpar_v3']['simtbpar_v3']
    k=ds.get_parameter_data('simtb01_CD')['simtb01_CD']['simtb01_CD']
    
    fig1=plt.figure()
    ax=plt.axes(projection='3d')
    hej=ax.scatter3D(x, y, z,c=k,cmap='inferno');
    cbar=fig1.colorbar(hej)
    
    ax.set_xlabel('V1')
    ax.set_ylabel('V2')
    ax.set_zlabel('V3')
    
    ax.set_title('dataid=%i' %dataid) #add start state to title, somehow

def spaghetti(points):
    initialise_database()
    new_experiment(name='MOCKTR01test',
                              sample_name="no sample")
    
    meas = Measurement()
    meas.register_parameter(simtbpar.v1)
    meas.register_parameter(simtbpar.v2)# register the first independent parameter
    meas.register_parameter(simtb01.E0, setpoints=(simtbpar.v1,simtbpar.v2))  # now register the dependent oone
    meas.register_parameter(simtb01.E1, setpoints=(simtbpar.v1,simtbpar.v2))  # now register the dependent oone
    meas.register_parameter(simtb01.E2, setpoints=(simtbpar.v1,simtbpar.v2))  # now register the dependent oone
    meas.register_parameter(simtb01.E3, setpoints=(simtbpar.v1,simtbpar.v2))  # now register the dependent oone
    meas.register_parameter(simtb01.E4, setpoints=(simtbpar.v1,simtbpar.v2))  # now register the dependent oone
    meas.register_parameter(simtb01.E5, setpoints=(simtbpar.v1,simtbpar.v2))  # now register the dependent oone
    meas.register_parameter(simtb01.E6, setpoints=(simtbpar.v1,simtbpar.v2))  # now register the dependent oone
    meas.register_parameter(simtb01.E7, setpoints=(simtbpar.v1,simtbpar.v2))  # now register the dependent oone
    meas.register_parameter(simtb01.E8, setpoints=(simtbpar.v1,simtbpar.v2))  # now register the dependent oone
    
    meas.write_period = 2
    
    with meas.run() as datasaver:
    
        set_v2=0.1
        simtbpar.v2.set(set_v2)
    
        for set_v1 in np.linspace(0, 0.15, points):
            simtbpar.v1.set(set_v1)
            get_E0 = simtb01.E0.get()
            get_E1 = simtb01.E1.get()
            get_E2 = simtb01.E2.get()
            get_E3 = simtb01.E3.get()
            get_E4 = simtb01.E4.get()
            get_E5 = simtb01.E5.get()
            get_E6 = simtb01.E6.get()
            get_E7 = simtb01.E7.get()
            get_E8 = simtb01.E8.get()
            
            datasaver.add_result((simtbpar.v1, set_v1),
                                 (simtbpar.v2, set_v2),
                                 (simtb01.E0, get_E0),
                                 (simtb01.E1, get_E1),
                                 (simtb01.E2, get_E2),
                                 (simtb01.E3, get_E3),
                                 (simtb01.E4, get_E4),
                                 (simtb01.E5, get_E5),
                                 (simtb01.E6, get_E6),
                                 (simtb01.E7, get_E7),
                                 (simtb01.E8, get_E8))
    
        dataid = datasaver.run_id  # convenient to have for plotting

    ds = qc.dataset.data_export.load_by_id(dataid) #returns only points on sphere
    x=ds.get_parameter_data('simtbpar_v1')['simtbpar_v1']['simtbpar_v1']
    E0Data=ds.get_parameter_data('simtb01_E0')['simtb01_E0']['simtb01_E0']
    E1Data=ds.get_parameter_data('simtb01_E1')['simtb01_E1']['simtb01_E1']
    E2Data=ds.get_parameter_data('simtb01_E2')['simtb01_E2']['simtb01_E2']
    E3Data=ds.get_parameter_data('simtb01_E3')['simtb01_E3']['simtb01_E3']
    E4Data=ds.get_parameter_data('simtb01_E4')['simtb01_E4']['simtb01_E4']
    E5Data=ds.get_parameter_data('simtb01_E5')['simtb01_E5']['simtb01_E5']
    E6Data=ds.get_parameter_data('simtb01_E6')['simtb01_E6']['simtb01_E6']
    E7Data=ds.get_parameter_data('simtb01_E7')['simtb01_E7']['simtb01_E7']
    E8Data=ds.get_parameter_data('simtb01_E8')['simtb01_E8']['simtb01_E8']

    
    fig=plt.figure()
    ax = plt.subplot(111)

    plt.plot(x,E0Data, label='(0,0)')
    plt.plot(x,E1Data, label='(0,1)')
    plt.plot(x,E2Data, label='(0,2)')
    plt.plot(x,E3Data, label='(1,0)')
    plt.plot(x,E4Data, label='(1,1)')
    plt.plot(x,E5Data, label='(1,2)')
    plt.plot(x,E6Data, label='(2,0)')
    plt.plot(x,E7Data, label='(2,1)')
    plt.plot(x,E8Data, label='(2,2)')
    ax.legend(bbox_to_anchor=(1.0, 1.03), fontsize=14)
    plt.xlabel('V1 [V]', fontsize=14)
    plt.ylabel('Energy [eV]', fontsize=14)
    plt.title('Energy of states', fontsize=14)
    plt.grid()
    plt.savefig('spaghotti',bbox_inches='tight')
    fig.show()
#    
#    
#    
#    ax=plt.axes(projection='3d')
#    hej=ax.scatter3D(x, y, z,c=k,cmap='inferno');
#    cbar=fig1.colorbar(hej)
#    
#    ax.set_xlabel('V1')
#    ax.set_ylabel('V2')
#    ax.set_zlabel('V3')
#    
#    ax.set_title('dataid=%i' %dataid) #add start state to title, somehow


def CDsweep4d(points):
    initialise_database()
    new_experiment(name='MOCKTR01test',
                              sample_name="no sample")
    
    meas = Measurement()
    meas.register_parameter(simtbpar.v1)  # register the first independent parameter
    meas.register_parameter(simtbpar.v2)
    meas.register_parameter(simtbpar.v3)  # register the first independent parameter
    meas.register_parameter(simtbpar.v4) # register the second independent parameter
    meas.register_parameter(simtb01.CD, setpoints=(simtbpar.v1,simtbpar.v2,simtbpar.v3,simtbpar.v4))  # now register the dependent oone
    
    
    meas.write_period = 2
    
    
    with meas.run() as datasaver:
    
        for set_v1 in np.linspace(0, 0.15, points):
            for set_v2 in np.linspace(0,0.15,points):
                 for set_v3 in np.linspace(0,0.15,points):
                      for set_v4 in np.linspace(0,0.15,points):
                        simtbpar.v1.set(set_v1)
                        simtbpar.v2.set(set_v2)
                        simtbpar.v3.set(set_v3)
                        simtbpar.v4.set(set_v4)
                        
                        get_v = simtb01.CD.get()
                        datasaver.add_result((simtbpar.v1, set_v1),
                                             (simtbpar.v2, set_v2),
                                             (simtbpar.v3, set_v3),
                                             (simtbpar.v4, set_v4),
                                             (simtb01.CD, get_v)) #(simtbpar.v2, 0.15-0.7*set_v1),(simtbpar.v4, 0.3*(0.15-0.7*set_v1)-0.195*(set_v2)-0.1*(set_v1)),
    
#        dataid = datasaver.run_id  # convenient to have for plotting
#        
#    ds = qc.dataset.data_export.load_by_id(dataid) #returns only points on sphere
#    k=ds.get_parameter_data('simtb01_CD')['simtb01_CD']['simtb01_CD']
#    plt.imshow(np.flip(k.reshape(points,points),0))    
    
    
    #plot_by_id(dataid)

    #print("dataid= %i" %dataid)

#simtbpar.v1.set(set_v1)
#                simtbpar.v2.set(0.15-0.7*set_v1)
#                
#                simtbpar.v3.set(set_v2)
#                simtbpar.v4.set(0.3*(0.15-0.7*set_v1)-0.195*(set_v2)-0.1*(set_v1))







# THIS PLOTS THE HULL

def radplot3d(dataid):
    """
    data id of the radialSearch3Dsphere run
    """
    
    
    ds = qc.dataset.data_export.load_by_id(dataid) #returns only points on sphere
    x=ds.get_parameter_data('simtbpar_v1')['simtbpar_v1']['simtbpar_v1']
    y=ds.get_parameter_data('simtbpar_v2')['simtbpar_v2']['simtbpar_v2']
    z=ds.get_parameter_data('simtbpar_v3')['simtbpar_v3']['simtbpar_v3']
    k=ds.get_parameter_data('simtb01_CD')['simtb01_CD']['simtb01_CD']
    
        
    
    
    oldrange=max(k)-min(k)
    newk=(((k-min(k))*1)/oldrange)
    inferno=mpl.cm.get_cmap('terrain',12)
    

    points=np.empty([len(x),3])
    for i in range(0,len(k)):
        points[i]=[x[i],y[i],z[i]]
    hull=ConvexHull(points) #using convexhull makes it so that it requires a lot more points on the sphere to
    # get more simplices.
    
    
    
    
    
    
    ax = plt.axes(projection='3d')

    for s in hull.simplices:
        s = np.append(s, s[0])  # Here we cycle back to the first coordinate
        ax.plot_trisurf(points[s, 0], points[s, 1], points[s, 2],color=inferno(newk[s][0]))#find a way to plot color on each surface given by k
    
#    ax.set_xlim([0,0.1])
#    ax.set_ylim([0,0.1])
#    ax.set_zlim([0,0.1])
    
    
    ax.set_xlabel('17')
    ax.set_ylabel('12')
    ax.set_zlabel('43')
    
    ax.set_title('dataid=%i' %dataid) #add start state to title, somehow
    plt.show()
    #return newk

def findradstart(dataid,state):
    """
    dataid from a CD run, state is the statenumber which you want to search.    
    """
    ds = qc.dataset.data_export.load_by_id(dataid)
#    x=ds.get_parameter_data('simtbpar_v1')['simtbpar_v1']['simtbpar_v1']
    y=ds.get_parameter_data('simtbpar_v2')['simtbpar_v2']['simtbpar_v2']
    z=ds.get_parameter_data('simtbpar_v3')['simtbpar_v3']['simtbpar_v3']
    k=ds.get_parameter_data('simtb01_CD')['simtb01_CD']['simtb01_CD']
    x=z
    
    indexes= np.where(np.array(k)==state)[0]

    xb=np.array(x)[indexes]
    yb=np.array(y)[indexes]
#    zb=np.array(z)[indexes]
    
#    def median(x):
#        l=len(x)
#        le=ma.floor(l/2)
#        return x[le]
#    
#    xrad=median(xb)
#    yrad=median(yb)
#    
    def middle(x):
        high=max(x)
        low=min(x)
        return (high+low)/2
    
    xmid=middle(xb)
    #print(xmid)
    ymid=middle(yb)
    #print(ymid)
#    zmid=middle(zb)
    #zrad=median(zb)
#    simtbpar.v1.set(xrad)
#    simtbpar.v2.set(yrad)
#    simtbpar.v3.set(zrad)
    #print('x=', xrad, 'y=',yrad ,'z=',zrad)
    p=[xmid,ymid]#,zrad
    return p

def getCharges(n,dots,state):
    k=0
    
    if dots == 2:
        Index=np.empty([(n+1)**2],dtype=object)
        for i in range(0,n+1):
            for j in range(0,n+1):
                Index[k]=np.array([i,j])
                k=k+1
    elif dots == 3:
        Index=np.empty([(n+1)**3],dtype=object)
        for i in range(0,n+1):
            for j in range(0,n+1):
                for h in range(0,n+1):
                    Index[k]=np.array([i,j,h])
                    k=k+1
    elif dots == 4:
        Index=np.empty([(n+1)**4],dtype=object)
        for i in range(0,n+1):
            for j in range(0,n+1):
                for h in range(0,n+1):
                    for g in range(0,n+1):
                        Index[k]=np.array([i,j,h,g])
                        k=k+1
    
    return Index[state]

def getState(n,dots,n1,n2,n3=None,n4=None):
    
    k=0
    h=0
    if dots == 2:
        Index=np.empty([(n+1)**2],dtype=object)
        for i in range(0,n+1):
            for j in range(0,n+1):
                Index[k]=np.array([i,j])
                k=k+1

        for x in Index:
           l=np.array_equal(np.array([n1,n2]),x)
           
           if l == True:
               break
           h=h+1
               
    elif dots == 3:
        Index=np.empty([(n+1)**3],dtype=object)
        for i in range(0,n+1):
            for j in range(0,n+1):
                for p in range(0,n+1):
                    Index[k]=np.array([i,j,p])
                    k=k+1
        for x in Index:
           l=np.array_equal(np.array([n1,n2,n3]),x)
           if l == True:
               break
           h=h+1
           
    elif dots == 4:
        Index=np.empty([(n+1)**4],dtype=object)
        for i in range(0,n+1):
            for j in range(0,n+1):
                for p in range(0,n+1):
                    for g in range(0,n+1):
                        Index[k]=np.array([i,j,p,g])
                        k=k+1
        for x in Index:
           l=np.array_equal(np.array([n1,n2,n3]),x)
           if l == True:
               break
           h=h+1
    return h

def sweep2din3d(points):
    initialise_database()
    new_experiment(name='MOCKTR01test',
                              sample_name="no sample")
    
    meas = Measurement()
    meas.register_parameter(simtbpar.v1)  # register the first independent parameter
    meas.register_parameter(simtbpar.v2)  # register the second independent parameter
    meas.register_parameter(simtbpar.v3)  # register the second independent parameter
    meas.register_parameter(simtb01.CD, setpoints=(simtbpar.v1,simtbpar.v2,simtbpar.v3))  # now register the dependent oone
    
    
    meas.write_period = 2
    
    #Choose starting location
    #simtbpar.v1.set(0.025)
    #simtbpar.v2.set(0.075)
    #simtbpar.v3.set(0.125)
    
    #lower corner is at: 0.0259, 0.0272, 0.0186
    #and vector to higher corner is 0.0496, 0.05, 0.0436
    start=np.array([0.0259,
                    0.0272,
                    0.0186])
    vec=np.array([0.0496,
                  0.05,
                  0.0436])
    perp1=np.cross(vec,np.array([1,0,0]))
    perp2=np.cross(vec,perp1)
    len1=np.linalg.norm(perp1)
    len2=np.linalg.norm(perp2)
    perp1=perp1/len1
    perp2=perp2/len2
    k=1
    x0,x1=-0.1,0.1
    y0,y1=-0.1,0.1
    for dist in np.linspace(0.2,0.50,3):
        #print(dist)
        #print(perp1)
        with meas.run() as datasaver:
        
            for set_v1 in np.linspace(y0, y1, points):
                for set_v2 in np.linspace(x0,x1,points):
                    #simtbpar.v1.set(start[0]+dist*vec[0]+0*set_v1-0.00440096*set_v2)
                    #simtbpar.v2.set(start[1]+dist*vec[1]+0.04636*set_v1+0.00248*set_v2)
                    #simtbpar.v3.set(start[2]+dist*vec[2]+-0.05*set_v1+0.00216256*set_v2)
                    
                    simtbpar.v1.set(start[0]+dist*vec[0]+perp1[0]*set_v1+perp2[0]*set_v2)
                    simtbpar.v2.set(start[1]+dist*vec[1]+perp1[1]*set_v1+perp2[1]*set_v2)
                    simtbpar.v3.set(start[2]+dist*vec[2]+perp1[2]*set_v1+perp2[2]*set_v2)
                    
                    
                    
                    get_v = simtb01.CD.get()
                    
                    datasaver.add_result((simtbpar.v1, set_v1),
                                         (simtbpar.v2, set_v2),
                                         (simtbpar.v3, set_v2),
                                         (simtb01.CD, get_v))
        
            dataid = datasaver.run_id  # convenient to have for plotting
    
        ds = qc.dataset.data_export.load_by_id(dataid) #returns only points on sphere
        CD=ds.get_parameter_data('simtb01_CD')['simtb01_CD']['simtb01_CD']
        #E=ds.get_parameter_data('simtb01_Emin')['simtb01_Emin']['simtb01_Emin']
        #CD[np.where(CD!=43)]=0
        
        plt.figure(k)
        #plt.axes()
        plt.imshow(CD.reshape(points,points),extent= [x0,x1,y1,y0])
        plt.title(r'cut dist : %s' %dist)
        plt.colorbar()
#        savepath=r'C:\Users\torbj\Google Drev\UNI\Bachelor\gif/'
#        fname=(savepath+r'figure_%i' %dataid+r'.png')
#        
#        plt.savefig(fname=fname)
#        p=findradstart(dataid,43)
#        plt.scatter(p[0],p[1])
        k=k+1  #cycles figures


def Qsweep3d(points):
    initialise_database()
    new_experiment(name='MOCKTR01test',
                              sample_name="no sample")
    
    meas = Measurement()
    meas.register_parameter(simtbpar.v1)  # register the first independent parameter
    meas.register_parameter(simtbpar.v2)  # register the second independent parameter
    meas.register_parameter(simtbpar.v3)  # register the second independent parameter
    meas.register_parameter(simtb01.Q2Exp, setpoints=(simtbpar.v1,simtbpar.v2,simtbpar.v3))  # now register the dependent oone
    meas.register_parameter(simtb01.Q1Exp, setpoints=(simtbpar.v1,simtbpar.v2,simtbpar.v3))  # now register the dependent oone
    meas.register_parameter(simtb01.Q3Exp, setpoints=(simtbpar.v1,simtbpar.v2,simtbpar.v3))  # now register the dependent oone
    
    
    meas.write_period = 2
    
    
    with meas.run() as datasaver:
    
        for set_v1 in np.linspace(0, 0.06, points):
            for set_v2 in np.linspace(0,0.06,points):
                for set_v3 in np.linspace(0,0.06,points):
                    simtbpar.v1.set(set_v1)
                    simtbpar.v2.set(set_v2)
                    simtbpar.v3.set(set_v3)
                    get_v2 = simtb01.Q2Exp.get()
                    get_v1 = simtb01.Q1Exp.get()
                    get_v3 = simtb01.Q3Exp.get()
                    datasaver.add_result((simtbpar.v1, set_v1),
                                         (simtbpar.v2, set_v2),
                                         (simtbpar.v3, set_v3),
                                         (simtb01.Q1Exp, get_v1),
                                         (simtb01.Q2Exp, get_v2),
                                         (simtb01.Q3Exp, get_v3))
        
        dataid = datasaver.run_id  # convenient to have for plotting
        
    ds = qc.dataset.data_export.load_by_id(dataid) #returns only points on sphere
    x=ds.get_parameter_data('simtbpar_v1')['simtbpar_v1']['simtbpar_v1']
    y=ds.get_parameter_data('simtbpar_v2')['simtbpar_v2']['simtbpar_v2']
    z=ds.get_parameter_data('simtbpar_v3')['simtbpar_v3']['simtbpar_v3']
    Q1=ds.get_parameter_data('simtb01_Q1Exp')['simtb01_Q1Exp']['simtb01_Q1Exp']
    Q2=ds.get_parameter_data('simtb01_Q2Exp')['simtb01_Q2Exp']['simtb01_Q2Exp']
    Q3=ds.get_parameter_data('simtb01_Q3Exp')['simtb01_Q3Exp']['simtb01_Q3Exp']
    
    
    fig1=plt.figure()
    ax=plt.axes(projection='3d')
    hej=ax.scatter3D(x, y, z,c=Q1,cmap='inferno');
    cbar=fig1.colorbar(hej)
    
    ax.set_xlabel('V1')
    ax.set_ylabel('V2')
    ax.set_zlabel('V3')
    
        #E=ds.get_parameter_data('simtb01_Emin')['simtb01_Emin']['simtb01_Emin']
    #plot_by_id(dataid)
    #print("dataid= %i" %dataid)

def tMatrix(n,dots,stateList,t,r):
    
    import numpy as np
    n=n+1
    
    t=np.array(t)
    r=np.array(r)
    
    w, h = n**len(dots), n**len(dots);
    diagonalMatrix = np.array([[0 for x in range(w)] for y in range(h)],dtype=object)

    #Insert the states on the diagonal [0,0],[0,1]...
    k=0
    for i in range(0,n**len(dots)):
        diagonalMatrix[k][k]=stateList[k]
        k=k+1
                
    for i in range(0,n**len(dots)):
        for j in range(0,n**len(dots)):
            dtm=np.array(diagonalMatrix[i][i])-np.array(diagonalMatrix[j][j])
            if i==j:
                pass
            elif all((np.sort(dtm.T)==np.array([-1,  0,  0,  1]))):
                if (dtm[2])==0 and (dtm[3])==0:   #1-2
                    diagonalMatrix[i][j]=t[0]
                
                elif (dtm[0])==0 and (dtm[3])==0: #2-3
                    diagonalMatrix[i][j]=t[1]
                
                elif (dtm[0])==0 and (dtm[1])==0: #3-4
                    diagonalMatrix[i][j]=t[2]
                
                elif (dtm[1])==0 and (dtm[2])==0: #1-4
                    diagonalMatrix[i][j]=t[3]
                
                elif (dtm[1])==0 and (dtm[3])==0: #1-3
                    diagonalMatrix[i][j]=t[4]
                
                elif (dtm[0])==0 and (dtm[2])==0: #2-4
                    diagonalMatrix[i][j]=t[5]
           
            elif all((np.sort((dtm.T)**2)==np.array([0,0,0,1]))):
                if (dtm[0])!=0:
                    diagonalMatrix[i][j]=r[0]
                
                elif (dtm[1])!=0:
                    diagonalMatrix[i][j]=r[1]
                
                elif (dtm[2])!=0:
                    diagonalMatrix[i][j]=r[2]
                
                elif (dtm[3])!=0:
                    diagonalMatrix[i][j]=r[3]
            else:
                diagonalMatrix[i][j]=0
    
    for i in range(0,n**len(dots)):
        diagonalMatrix[i][i]=0

    diagonalMatrix=diagonalMatrix.astype(np.float64)
    
    return diagonalMatrix

def Eprofile(res,P1P2,axis,gate0=None,gate1=None):
    initialise_database()
    new_experiment(name='MOCKTR01test',
                              sample_name="no sample")
    P1P2=np.array(P1P2)
    if gate0==None:
        gate0=simtbpar.v1
    if gate1==None:
        gate1=simtbpar.v2
    
    meas = Measurement()
    meas.register_parameter(gate0)  # register the first independent parameter
    meas.register_parameter(gate1)  # register the second independent parameter
    meas.register_parameter(simtb01.Estate1, setpoints=(gate0,gate1))
    meas.register_parameter(simtb01.Estate2, setpoints=(gate0,gate1))
    meas.register_parameter(simtb01.Estate3, setpoints=(gate0,gate1))
    meas.register_parameter(simtb01.Estate1t, setpoints=(gate0,gate1))
    meas.register_parameter(simtb01.Estate2t, setpoints=(gate0,gate1))
    meas.register_parameter(simtb01.Estate3t, setpoints=(gate0,gate1))# now register the dependent oone
    meas.register_parameter(simtb01.CD, setpoints=(gate0,gate1))
    meas.register_parameter(simtb01.CDt, setpoints=(gate0,gate1))
    meas.write_period = 2
    
    with meas.run() as datasaver:
    
     for set_v1 in np.linspace(axis[0], axis[1], res):
         for set_v2 in np.linspace(axis[2], axis[3], res):
             simtbpar.v1.set(set_v1)
             simtbpar.v2.set(set_v2)
             get_CD = simtb01.CD.get()
             get_CDt = simtb01.CDt.get()
             datasaver.add_result((simtbpar.v1, set_v1),
                                  (simtbpar.v2, set_v2),
                                  (simtb01.CD, get_CD),
                                  (simtb01.CDt, get_CDt))
     dataidMap = datasaver.run_id  # convenient to have for plotting
    
    with meas.run() as datasaver:
       
        for i in range(0,P1P2.shape[0]-1):
            
            if (P1P2[i+1][0]-P1P2[i][0])==0:
                
                set_v1=P1P2[i][0]
                
                for set_v2 in np.linspace(P1P2[i][1], P1P2[i+1][1], res):
                    gate0.set(set_v1)
                    gate1.set(set_v2)
                    get_E1 = simtb01.Estate1.get()
                    get_E2 = simtb01.Estate2.get()
                    get_E3 = simtb01.Estate3.get()
                    get_E1t = simtb01.Estate1t.get()
                    get_E2t = simtb01.Estate2t.get()
                    get_E3t = simtb01.Estate3t.get()
                    datasaver.add_result((gate0, set_v1),
                                         (gate1, set_v2),
                                         (simtb01.Estate1, get_E1),
                                         (simtb01.Estate2, get_E2),
                                         (simtb01.Estate3, get_E3),
                                         (simtb01.Estate1t, get_E1t),
                                         (simtb01.Estate2t, get_E2t),
                                         (simtb01.Estate3t, get_E3t))
                
            elif (P1P2[i+1][1]-P1P2[i][1])==0:
                
                set_v2=P1P2[i+1][1]
                
                for set_v1 in np.linspace(P1P2[i][0], P1P2[i+1][0], res):
                    gate0.set(set_v1)
                    gate1.set(set_v2)
                    get_E1 = simtb01.Estate1.get()
                    get_E2 = simtb01.Estate2.get()
                    get_E3 = simtb01.Estate3.get()
                    get_E1t = simtb01.Estate1t.get()
                    get_E2t = simtb01.Estate2t.get()
                    get_E3t = simtb01.Estate3t.get()
                    datasaver.add_result((gate0, set_v1),
                                         (gate1, set_v2),
                                         (simtb01.Estate1, get_E1),
                                         (simtb01.Estate2, get_E2),
                                         (simtb01.Estate3, get_E3),
                                         (simtb01.Estate1t, get_E1t),
                                         (simtb01.Estate2t, get_E2t),
                                         (simtb01.Estate3t, get_E3t))
                
            else:
            
                slope=(P1P2[i+1][1]-P1P2[i][1])/(P1P2[i+1][0]-P1P2[i][0])
                intersept=P1P2[i][1]-slope*P1P2[i][0]
                    
                for set_v1 in np.linspace(P1P2[i][0], P1P2[i+1][0], res):
                    gate0.set(set_v1)
                    gate1.set(slope*set_v1+intersept)
                    get_E1 = simtb01.Estate1.get()
                    get_E2 = simtb01.Estate2.get()
                    get_E3 = simtb01.Estate3.get()
                    get_E1t = simtb01.Estate1t.get()
                    get_E2t = simtb01.Estate2t.get()
                    get_E3t = simtb01.Estate3t.get()
                    datasaver.add_result((gate0, set_v1),
                                         (gate1, set_v2),
                                         (simtb01.Estate1, get_E1),
                                         (simtb01.Estate2, get_E2),
                                         (simtb01.Estate3, get_E3),
                                         (simtb01.Estate1t, get_E1t),
                                         (simtb01.Estate2t, get_E2t),
                                         (simtb01.Estate3t, get_E3t))
        dataid = datasaver.run_id
        
    
    dsMap = qc.dataset.data_export.load_by_id(dataidMap) #returns only points on sphere
    CDMap=dsMap.get_parameter_data('simtb01_CD')['simtb01_CD']['simtb01_CD']
    CDtMap=dsMap.get_parameter_data('simtb01_CDt')['simtb01_CDt']['simtb01_CDt']
    
    fig1=plt.figure(1)
    ax=fig1.gca()
    plt.imshow(np.rot90(CDMap.reshape(res,res)),cmap='inferno',extent=(axis))
    plt.plot([P1P2[0][0],P1P2[1][0]],[P1P2[0][1],P1P2[1][1]],'bo-')
    plt.xlabel('V1 [V]',size=15)
    plt.ylabel('V2 [V]',size=15)
    plt.title('Charge stability diagram',fontsize=15)
    ax.tick_params(labelsize=15)
#    plt.title('RunID:#%i'%dataidMap+' Charge stability diagram',fontsize=15)
#    plt.savefig('ChargestabilityMapDiagCut',bbox_inches='tight')
    fig1.show()
    
    fig2=plt.figure(2)
    plt.imshow(np.rot90(CDtMap.reshape(res,res)),cmap='inferno',extent=(axis))
    plt.plot([P1P2[0][0],P1P2[1][0]],[P1P2[0][1],P1P2[1][1]],'bo-')
    plt.xlabel('V1 [V]')
    plt.ylabel('V2 [V]')
    plt.title('Charge stability diagram with tunneling',fontsize=15)
#    plt.title('RunID:#%i'%dataidMap+' Charge stability diagram with tunnel',fontsize=15)
    fig2.show()

    ds = qc.dataset.data_export.load_by_id(dataid) #returns only points on sphere
    E1Data=ds.get_parameter_data('simtb01_Estate1')['simtb01_Estate1']['simtb01_Estate1']
    E2Data=ds.get_parameter_data('simtb01_Estate2')['simtb01_Estate2']['simtb01_Estate2']
    E3Data=ds.get_parameter_data('simtb01_Estate3')['simtb01_Estate3']['simtb01_Estate3']
    E1tData=ds.get_parameter_data('simtb01_Estate1t')['simtb01_Estate1t']['simtb01_Estate1t']
    E2tData=ds.get_parameter_data('simtb01_Estate2t')['simtb01_Estate2t']['simtb01_Estate2t']
    E3tData=ds.get_parameter_data('simtb01_Estate3t')['simtb01_Estate3t']['simtb01_Estate3t']
    x=ds.get_parameter_data(gate0.full_name)[gate0.full_name][gate0.full_name]

    fig3=plt.figure(3)
    ax=fig3.gca()
    plt.plot(x,E1Data,'r.',label='(3,1)')
    plt.plot(x,E2Data,'k.',label='(2,2)')
    plt.plot(x,E3Data,'g.',label='(1,3)')
    ax.legend(prop={'size':15})
    ax.grid()
    ax.set_xlabel('V1 (V)',size=15)
    ax.set_ylabel('Energy (eV)',size=15)
    ax.tick_params(labelsize=15)
    plt.title('Energies along cut', fontsize=15)
    plt.xticks(np.arange(min(x), max(x), 0.05))
#    plt.savefig('Energy_cut_no_tun',bbox_inches='tight')
    fig3.show()
    
    fig4=plt.figure(4)
    ax=fig4.gca()
    plt.plot(x,E1tData,'r.',label='(3,1)')
    plt.plot(x,E2tData,'k.',label='(3,1)')
    plt.plot(x,E3tData,'g.',label='(1,3)')
    ax.legend(prop={'size':15})
    ax.grid()
    ax.set_xlabel('V1 (V)',size=15)
    ax.set_ylabel('Energy (eV)',size=15)
    ax.tick_params(labelsize=15)
    plt.title('Energies along cut, with tunneling', fontsize=15)
    plt.xticks(np.arange(min(x), max(x), 0.05))
#    plt.savefig('Energy_cut_with_tun',bbox_inches='tight')
    fig4.show()
    
        

# for Getting difference between two energy states
def Ediff(points,P1P2=None,axis=None,p0=None,p1=None):
    initialise_database()
    new_experiment(name='MOCKTR01test',
                              sample_name="no sample")
    
    if p0==None:
        p0=simtb01.E0
    if p1==None:
        p1=simtb01.E1
    
    meas = Measurement()
    meas.register_parameter(simtbpar.v1)  # register the first independent parameter
    meas.register_parameter(simtbpar.v2)  # register the second independent parameter
    meas.register_parameter(p0, setpoints=(simtbpar.v1,simtbpar.v2))  # now register the dependent oone
    meas.register_parameter(p1, setpoints=(simtbpar.v1,simtbpar.v2))
    meas.register_parameter(simtb01.Ediff, setpoints=(simtbpar.v1,simtbpar.v2))
    meas.register_parameter(simtb01.CD, setpoints=(simtbpar.v1,simtbpar.v2))
    
    meas.write_period = 2
    
    if axis!=None:
        
        with meas.run() as datasaver:
    
            for set_v1 in np.linspace(axis[0], axis[1], points):
                for set_v2 in np.linspace(axis[2], axis[3], points):
                    simtbpar.v1.set(set_v1)
                    simtbpar.v2.set(set_v2)
                    get_CD = simtb01.CD.get()
                    datasaver.add_result((simtbpar.v1, set_v1),
                                         (simtbpar.v2, set_v2),
                                         (simtb01.CD, get_CD))
                    dataidMap = datasaver.run_id  # convenient to have for plotting
     
    P1P2=np.array(P1P2)
        
    with meas.run() as datasaver:
    
        for i in range(0,P1P2.shape[0]-1):
    
            if (P1P2[1][0]-P1P2[0][0])==0:
            
                set_v1=P1P2[1][0]
        
                for set_v2 in np.linspace(P1P2[0][1], P1P2[1][1], points):
                    simtbpar.v1.set(set_v1)
                    simtbpar.v2.set(set_v2)
                    get_E0 = p0.get()
                    get_E1 = p1.get()
                    get_Ediff=get_E1-get_E0
                    datasaver.add_result((simtbpar.v1, set_v1),
                                         (simtbpar.v2, set_v2),
                                         (p0, get_E0),
                                         (p1, get_E1),
                                         (simtb01.Ediff, get_Ediff))

            elif (P1P2[1][1]-P1P2[0][1])==0:
                
                set_v2=P1P2[1][1]
                
                for set_v1 in np.linspace(P1P2[0][0], P1P2[1][0], points):
                    simtbpar.v1.set(set_v1)
                    simtbpar.v2.set(set_v2)
                    get_E0 = p0.get()
                    get_E1 = p1.get()
                    get_Ediff=get_E1-get_E0
                    datasaver.add_result((simtbpar.v1, set_v1),
                                         (simtbpar.v2, set_v2),
                                         (p0, get_E0),
                                         (p1, get_E1),
                                         (simtb01.Ediff, get_Ediff))
        
            else:
                
                slope=(P1P2[1][1]-P1P2[0][1])/(P1P2[1][0]-P1P2[0][0])
                intersept=P1P2[0][1]-slope*P1P2[0][0]
                
                for set_v1 in np.linspace(P1P2[0][0], P1P2[1][0], points):
                    simtbpar.v1.set(set_v1)
                    simtbpar.v2.set(slope*set_v1+intersept)
                    get_E0 = p0.get()
                    get_E1 = p1.get()
                    get_Ediff=get_E1-get_E0
                    datasaver.add_result((simtbpar.v1, set_v1),
                                         (simtbpar.v2, slope*set_v1+intersept),
                                         (p0, get_E0),
                                         (p1, get_E1),
                                         (simtb01.Ediff, get_Ediff))
        dataid = datasaver.run_id  # convenient to have for plotting
            

                
    ds = qc.dataset.data_export.load_by_id(dataid) #returns only points on sphere        
    x=np.linspace(P1P2[0][0],P1P2[1][0],points)
    energy0=ds.get_parameter_data(p0.full_name)[p0.full_name][p0.full_name]
    energy1=ds.get_parameter_data(p1.full_name)[p1.full_name][p1.full_name]
    k=ds.get_parameter_data('simtb01_Ediff')['simtb01_Ediff']['simtb01_Ediff']
    
    if axis:
        dsMap = qc.dataset.data_export.load_by_id(dataidMap) #returns only points on sphere
        kMap=dsMap.get_parameter_data('simtb01_CD')['simtb01_CD']['simtb01_CD']
        
        fig1=plt.figure(1)
        ax=fig1.gca()
        plt.imshow(np.rot90(kMap.reshape(points,points)),extent=(axis))
        plt.plot(P1P2[:,0],P1P2[:,1],'ro-')
        plt.xlabel('V1 [V]')
        plt.ylabel('V2 [V]')
        plt.title('RunID:#%i'%dataidMap+' Charge stability diagram',fontsize=15)
        fig1.show()


    fig2=plt.figure(2)
    ax2=fig2.gca()
    plt.plot(x,energy0,label=p0.full_name[-2:])
    plt.plot(x,energy1,label=p1.full_name[-2:])
    ax2.legend(loc='upper right',prop={'size':15})
    ax2.set_xlabel('V1 [V]',size=15)
    ax2.set_ylabel('Energy [eV]',size=15)
    ax2.tick_params(labelsize=15)
    plt.title('RunID:#%i'%dataid+' Energy cut', fontsize=15)
    plt.xlim(axis[0]-0.01,axis[1]+0.01)
    plt.xticks(np.arange(axis[0], axis[1]+0.05, 0.05))
    plt.grid(b=True, which='major')
    plt.savefig('Energy_cut',bbox_inches='tight')
    plt.show()
    fig2.show()
    
    
    
    fig3=plt.figure(3)
    ax3=fig3.gca()
    plt.plot(x,k,label='Energy difference '+ p1.full_name[-2:]+'-'+p0.full_name[-2:])
    ax3.legend(loc='upper right',prop={'size':15})
    plt.title('RunID:#%i'%dataid+' Energy difference',fontsize=15)
    ax3.set_xlabel('V1 (V)',size=15)
    ax3.set_ylabel('E [eV]',size=15)
    ax3.tick_params(labelsize=15)
    plt.grid(b=True, which='major')
    plt.savefig('Energy_diff',bbox_inches='tight')
    plt.show()
    fig3.show()
    
        
#Expectation value of charge
#ExpQCut(100,P1P2=[[0.1,0.1],[0.3,0.3]],axis=[0,0.5,0,0.5],p0=simtb01.Q2Exp,p1=simtb01.Q3Exp,gate1=simtbpar.v2,gate2=simtbpar.v3)
def ExpQCut(points,P1P2,axis,p0=None,p1=None,gate1=None,gate2=None):
    initialise_database()
    new_experiment(name='MOCKTR01test',
                              sample_name="no sample")
    
    if p0==None:
        p0=simtb01.Q1Exp
    if p1==None:
        p1=simtb01.Q2Exp
    if gate1==None:
        gate1=simtbpar.v1
    if gate2==None:
        gate2=simtbpar.v2
    
    meas = Measurement()
    meas.register_parameter(gate1)  # register the first independent parameter
    meas.register_parameter(gate2)  # register the second independent parameter
    meas.register_parameter(simtb01.Q1Exp, setpoints=(gate1,gate2))  # now register the dependent oone
    meas.register_parameter(simtb01.Q2Exp, setpoints=(gate1,gate2))
    meas.register_parameter(simtb01.Q3Exp, setpoints=(gate1,gate2))
    meas.register_parameter(simtb01.Q4Exp, setpoints=(gate1,gate2))
    meas.register_parameter(simtb01.CD, setpoints=(gate1,gate2))
    
    meas.write_period = 2
        
    with meas.run() as datasaver:

        for set_v1 in np.linspace(axis[0], axis[1], points):
            for set_v2 in np.linspace(axis[2], axis[3], points):
                gate1.set(set_v1)
                gate2.set(set_v2)
                get_CD = simtb01.CD.get()
                get_Q1Exp = p0.get()
                get_Q2Exp = p1.get()
                datasaver.add_result((gate1, set_v1),
                                     (gate2, set_v2),
                                     (simtb01.CD, get_CD),
                                     (p0, get_Q1Exp),
                                     (p1, get_Q2Exp))
                dataidMap = datasaver.run_id  # convenient to have for plotting
    
    P1P2=np.array(P1P2)

    with meas.run() as datasaver:
   
        for i in range(0,P1P2.shape[0]-1):
            
            if (P1P2[i+1][0]-P1P2[i][0])==0:
                
                set_v1=P1P2[i][0]
                
                for set_v2 in np.linspace(P1P2[i][1], P1P2[i+1][1], points):
                    gate1.set(set_v1)
                    gate2.set(set_v2)
                    get_Q1Exp = p0.get()
                    get_Q2Exp = p1.get()
                    datasaver.add_result((gate1, set_v1),
                                         (gate2, set_v2),
                                         (simtb01.CD, get_CD),
                                         (p0, get_Q1Exp),
                                         (p1, get_Q2Exp))
                
            elif (P1P2[i+1][1]-P1P2[i][1])==0:
                
                set_v2=P1P2[i+1][1]
                
                for set_v1 in np.linspace(P1P2[i][0], P1P2[i+1][0], points):
                    gate1.set(set_v1)
                    gate2.set(set_v2)
                    get_Q1Exp = p0.get()
                    get_Q2Exp = p1.get()
                    datasaver.add_result((gate1, set_v1),
                                         (gate2, set_v2),
                                         (simtb01.CD, get_CD),
                                         (p0, get_Q1Exp),
                                         (p1, get_Q2Exp))
                
            else:
            
                slope=(P1P2[i+1][1]-P1P2[i][1])/(P1P2[i+1][0]-P1P2[i][0])
                intersept=P1P2[i][1]-slope*P1P2[i][0]
                    
                for set_v1 in np.linspace(P1P2[i][0], P1P2[i+1][0], points):
                    gate1.set(set_v1)
                    gate2.set(slope*set_v1+intersept)
                    get_Q1Exp = p0.get()
                    get_Q2Exp = p1.get()
                    datasaver.add_result((gate1, set_v1),
                                         (gate2, set_v2),
                                         (simtb01.CD, get_CD),
                                         (p0, get_Q1Exp),
                                         (p1, get_Q2Exp))
        
    dataid = datasaver.run_id  # convenient to have for plotting
    ds = qc.dataset.data_export.load_by_id(dataid) #returns only points on sphere
    
    Q1val=ds.get_parameter_data(p0.full_name)[p0.full_name][p0.full_name]
    Q2val=ds.get_parameter_data(p1.full_name)[p1.full_name][p1.full_name]

    
    if axis:
        dsMap = qc.dataset.data_export.load_by_id(dataidMap) #returns only points on sphere
        CDMap=dsMap.get_parameter_data('simtb01_CD')['simtb01_CD']['simtb01_CD']
        Q1Map=dsMap.get_parameter_data(p0.full_name)[p0.full_name][p0.full_name]
        Q2Map=dsMap.get_parameter_data(p1.full_name)[p1.full_name][p1.full_name]

        fig1=plt.figure()
        plt.imshow(np.rot90(CDMap.reshape(points,points)),cmap='inferno',extent=(axis))
        plt.plot(P1P2[:,0],P1P2[:,1],'r-')
        plt.xlabel('V%s [V]'%gate1.full_name[-1:])
        plt.ylabel('V%s [V]'%gate2.full_name[-1:])
        plt.title('RunID:#%i'%dataidMap+' Charge Stability Diagram')
        fig1.show()
        
        fig2=plt.figure()
        plt.imshow(np.rot90(Q1Map.reshape(points,points)),extent=(axis))
        plt.plot(P1P2[:,0],P1P2[:,1],'r-')
        plt.plot(P1P2[0,0],P1P2[0,1],'ro')
        plt.xlabel('V%s [V]'%gate1.full_name[-1:],fontsize=14)
        plt.ylabel('V%s [V]'%gate2.full_name[-1:],fontsize=14)
        plt.title('CSD of Expected Charge in %s'%p0.full_name[-5:-3],fontsize=14)
        Q2cbar=plt.colorbar()
        Q2cbar.set_label('Expected Charge [q]',size=14)
        plt.savefig('ExpectedChargeQ1Cut',bbox_inches='tight')
        fig2.show()
        
        fig3=plt.figure()
        plt.imshow(np.rot90(Q2Map.reshape(points,points)),extent=(axis))
        plt.plot(P1P2[:,0],P1P2[:,1],'r-')
        plt.plot(P1P2[0,0],P1P2[0,1],'ro')
        plt.xlabel('V%s [V]'%gate1.full_name[-1:],fontsize=14)
        plt.ylabel('V%s [V]'%gate2.full_name[-1:],fontsize=14)
        plt.title('CSD of Expected Charge in %s'%p1.full_name[-5:-3],fontsize=14)
        Q3cbar=plt.colorbar()
        Q3cbar.set_label('Expected Charge [q]',size=14)
        plt.savefig('ExpectedChargeQ2Cut',bbox_inches='tight')
        fig3.show()
        
        fig4=plt.figure()
        plt.imshow(np.rot90((Q2Map+Q1Map).reshape(points,points)),extent=(axis))
        plt.plot(P1P2[:,0],P1P2[:,1],'r-')
        plt.plot(P1P2[0,0],P1P2[0,1],'ro')
        plt.xlabel('V%s [V]'%gate1.full_name[-1:],fontsize=14)
        plt.ylabel('V%s [V]'%gate2.full_name[-1:],fontsize=14)
        plt.title('CSD of total Expected Charge',fontsize=14)
        Q3cbar=plt.colorbar()
        Q3cbar.set_label('Expected Charge [q]',size=14)
        plt.savefig('TotalExpectedChargeCut',bbox_inches='tight')
        fig4.show()

    fig5=plt.figure()
    plt.plot(np.linspace(0,points*(P1P2.shape[0]-1),points*(P1P2.shape[0]-1)).T,Q1val,'r.')
    plt.xlabel('Point along cut',fontsize=14)
    plt.ylabel('Charge [q]',fontsize=14)
    plt.title('Expected Charge in %s'%p0.full_name[-5:-3],fontsize=14)
    plt.savefig('ExpectedChargeQ1Graph',bbox_inches='tight')
    fig5.show()
    
    fig6=plt.figure()
    plt.plot(np.linspace(0,points*(P1P2.shape[0]-1),points*(P1P2.shape[0]-1)).T,Q2val,'b.')
    plt.xlabel('Point along cut',fontsize=14)
    plt.ylabel('Charge [q]',fontsize=14)
    plt.title('Expected Charge in %s'%p1.full_name[-5:-3],fontsize=14)
    plt.savefig('ExpectedChargeQ2Graph',bbox_inches='tight')
    fig6.show()

def Q2Din3D(res,point,direction,axis,p1p2res=None,p0=None,p1=None,p2=None,gate1=None,gate2=None,gate3=None):
    initialise_database()
    new_experiment(name='MOCKTR01test',
                              sample_name="no sample")
    if p1p2res==None:
        p1p2res=100
    if p0==None:
        p0=simtb01.Q1Exp
    if p1==None:
        p1=simtb01.Q2Exp
    if p2==None:
        p2=simtb01.Q3Exp
    if gate1==None:
        gate1=simtbpar.v1
    if gate2==None:
        gate2=simtbpar.v2
    if gate3==None:
        gate3=simtbpar.v3

    
    amInorm=direction[0]**2+direction[1]**2+direction[2]**2
    if amInorm!=1:
        direction=direction/(np.linalg.norm(direction))
    
    perp1=np.cross(direction,np.array([1,0,0]))
    perp2=np.cross(direction,perp1)
    len1=np.linalg.norm(perp1)
    len2=np.linalg.norm(perp2)
    perp1=perp1/len1
    perp2=perp2/len2
    
    cc=np.matrix([direction,perp1,perp2])
    ccinv=cc.I
    
    meas = Measurement()
    meas.register_parameter(gate1)  # register the first independent parameter
    meas.register_parameter(gate2)  # register the second independent parameter
    meas.register_parameter(simtb01.Q1Exp, setpoints=(gate1,gate2))  # now register the dependent oone
    meas.register_parameter(simtb01.Q2Exp, setpoints=(gate1,gate2))
    meas.register_parameter(simtb01.Q3Exp, setpoints=(gate1,gate2))
    meas.register_parameter(simtb01.Q4Exp, setpoints=(gate1,gate2))
    meas.register_parameter(simtb01.CD, setpoints=(gate1,gate2))
    
    meas.write_period = 2
        
    with meas.run() as datasaver:

        for set_v1 in np.linspace(axis[0], axis[1], res):
            for set_v2 in np.linspace(axis[2], axis[3], res):
                
                gate1.set(point[0]+ccinv[0,1]*set_v1+ccinv[0,2]*set_v2)
                gate2.set(point[1]+ccinv[1,1]*set_v1+ccinv[1,2]*set_v2)   
                gate3.set(point[2]+ccinv[2,1]*set_v1+ccinv[2,2]*set_v2)
                get_CD = simtb01.CD.get()
                get_Q1Exp = p0.get()
                get_Q2Exp = p1.get()
                get_Q3Exp = p2.get()
                datasaver.add_result((gate1, set_v1),
                                     (gate2, set_v2),
                                     (simtb01.CD, get_CD),
                                     (p0, get_Q1Exp),
                                     (p1, get_Q2Exp),
                                     (p2, get_Q3Exp))
        dataidMap = datasaver.run_id  # convenient to have for plotting
    
    dsMap = qc.dataset.data_export.load_by_id(dataidMap) #returns only points on sphere
    CDMap=dsMap.get_parameter_data('simtb01_CD')['simtb01_CD']['simtb01_CD']
    Q1Map=dsMap.get_parameter_data(p0.full_name)[p0.full_name][p0.full_name]
    Q2Map=dsMap.get_parameter_data(p1.full_name)[p1.full_name][p1.full_name]
    Q3Map=dsMap.get_parameter_data(p2.full_name)[p2.full_name][p2.full_name]
    
    fig1=plt.figure()
    plt.imshow(np.rot90(CDMap.reshape(res,res)),cmap='inferno',extent=(axis))
    plt.xlabel('V%s [V]'%gate1.full_name[-1:])
    plt.ylabel('V%s [V]'%gate2.full_name[-1:])
    plt.title('RunID:#%i'%dataidMap+' Charge Stability Diagram')
    plt.colorbar()
    fig1.show()
    
    fig2=plt.figure()
    plt.imshow(np.rot90(Q1Map.reshape(res,res)),extent=(axis))
    plt.xlabel('V%s [V]'%gate1.full_name[-1:])
    plt.ylabel('V%s [V]'%gate2.full_name[-1:])
    plt.title('RunID:#%i'%dataidMap+' CSD of Expected Charge in %s'%p0.full_name[-5:-3])
    plt.colorbar()
    fig2.show()
    
    fig3=plt.figure()
    plt.imshow(np.rot90(Q2Map.reshape(res,res)),extent=(axis))
    plt.xlabel('V%s [V]'%gate1.full_name[-1:])
    plt.ylabel('V%s [V]'%gate2.full_name[-1:])
    plt.title('RunID:#%i'%dataidMap+' CSD of Expected Charge in %s'%p1.full_name[-5:-3])
    plt.colorbar()
    fig3.show()
    
    fig4=plt.figure()
    plt.imshow(np.rot90(Q3Map.reshape(res,res)),extent=(axis))
    plt.xlabel('V%s [V]'%gate1.full_name[-1:])
    plt.ylabel('V%s [V]'%gate2.full_name[-1:])
    plt.title('RunID:#%i'%dataidMap+' CSD of Expected Charge in %s'%p2.full_name[-5:-3])
    plt.colorbar()
    fig4.show()

#ExpQ2Din3D(res=70,P1P2=[[0.01,0.01],[0.02,0.02]],point=[0.068877551020408004, 0.067346938775509999, 0.047448979591836749],direction=[1/np.sqrt(3),1/np.sqrt(3),1/np.sqrt(3)],axis=(0.05,0.1,0.05,0.1))
def ExpQ2Din3D(res,P1P2,point,direction,axis,p1p2res=None,p0=None,p1=None,p2=None,gate1=None,gate2=None,gate3=None):
    initialise_database()
    new_experiment(name='MOCKTR01test',
                              sample_name="no sample")
    if p1p2res==None:
        p1p2res=100
    if p0==None:
        p0=simtb01.Q1Exp
    if p1==None:
        p1=simtb01.Q2Exp
    if p2==None:
        p2=simtb01.Q3Exp
    if gate1==None:
        gate1=simtbpar.v1
    if gate2==None:
        gate2=simtbpar.v2
    if gate3==None:
        gate3=simtbpar.v3
    
    amInorm=direction[0]**2+direction[1]**2+direction[2]**2
    if amInorm!=1:
        direction=direction/(np.linalg.norm(direction))
    
    perp1=np.cross(direction,np.array([1,0,0]))
    perp2=np.cross(direction,perp1)
    len1=np.linalg.norm(perp1)
    len2=np.linalg.norm(perp2)
    perp1=perp1/len1
    perp2=perp2/len2
    
    cc=np.matrix([direction,perp1,perp2])
    ccinv=cc.I
    
    meas = Measurement()
    meas.register_parameter(gate1)  # register the first independent parameter
    meas.register_parameter(gate2)  # register the second independent parameter
    meas.register_parameter(simtb01.Q1Exp, setpoints=(gate1,gate2))  # now register the dependent oone
    meas.register_parameter(simtb01.Q2Exp, setpoints=(gate1,gate2))
    meas.register_parameter(simtb01.Q3Exp, setpoints=(gate1,gate2))
    meas.register_parameter(simtb01.Q4Exp, setpoints=(gate1,gate2))
    meas.register_parameter(simtb01.CD, setpoints=(gate1,gate2))
    
    meas.write_period = 2
        
    with meas.run() as datasaver:

        for set_v1 in np.linspace(axis[0], axis[1], res):
            for set_v2 in np.linspace(axis[2], axis[3], res):
                
                gate1.set(point[0]+ccinv[0,1]*set_v1+ccinv[0,2]*set_v2)
                gate2.set(point[1]+ccinv[1,1]*set_v1+ccinv[1,2]*set_v2)   
                gate3.set(point[2]+ccinv[2,1]*set_v1+ccinv[2,2]*set_v2)
                get_CD = simtb01.CD.get()
                get_Q1Exp = p0.get()
                get_Q2Exp = p1.get()
                get_Q3Exp = p2.get()
                datasaver.add_result((gate1, set_v1),
                                     (gate2, set_v2),
                                     (simtb01.CD, get_CD),
                                     (p0, get_Q1Exp),
                                     (p1, get_Q2Exp),
                                     (p2, get_Q3Exp))
        dataidMap = datasaver.run_id  # convenient to have for plotting
    
    P1P2=np.array(P1P2)
    
    with meas.run() as datasaver:
   
        for i in range(0,P1P2.shape[0]-1):
            
            if (P1P2[i+1][0]-P1P2[i][0])==0:
                
                set_v1=P1P2[i][0]
                
                for set_v2 in np.linspace(P1P2[i][1], P1P2[i+1][1], res/10):
                    gate1.set(point[0]+ccinv[0,1]*set_v1+ccinv[0,2]*set_v2)
                    gate2.set(point[1]+ccinv[1,1]*set_v1+ccinv[1,2]*set_v2)   
                    gate3.set(point[2]+ccinv[2,1]*set_v1+ccinv[2,2]*set_v2)
                    get_Q1Exp = p0.get()
                    get_Q2Exp = p1.get()
                    get_Q3Exp = p2.get()
                    datasaver.add_result((gate1, set_v1),
                                         (gate2, set_v2),
                                         (simtb01.CD, get_CD),
                                         (p0, get_Q1Exp),
                                         (p1, get_Q2Exp),
                                         (p2, get_Q3Exp))
                
            elif (P1P2[i+1][1]-P1P2[i][1])==0:
                
                set_v2=P1P2[i+1][1]
                
                for set_v1 in np.linspace(P1P2[i][0], P1P2[i+1][0], res/10):
                    gate1.set(point[0]+ccinv[0,1]*set_v1+ccinv[0,2]*set_v2)
                    gate2.set(point[1]+ccinv[1,1]*set_v1+ccinv[1,2]*set_v2)   
                    gate3.set(point[2]+ccinv[2,1]*set_v1+ccinv[2,2]*set_v2)
                    get_Q1Exp = p0.get()
                    get_Q2Exp = p1.get()
                    get_Q3Exp = p2.get()
                    datasaver.add_result((gate1, set_v1),
                                         (gate2, set_v2),
                                         (simtb01.CD, get_CD),
                                         (p0, get_Q1Exp),
                                         (p1, get_Q2Exp),
                                         (p2, get_Q3Exp))
                
            else:
            
                slope=(P1P2[i+1][1]-P1P2[i][1])/(P1P2[i+1][0]-P1P2[i][0])
                intersept=P1P2[i][1]-slope*P1P2[i][0]
                    
                for set_v1 in np.linspace(P1P2[i][0], P1P2[i+1][0], res/10):
                    set_v2=slope*set_v1+intersept
                    
                    gate1.set(point[0]+ccinv[0,1]*set_v1+ccinv[0,2]*set_v2)
                    gate2.set(point[1]+ccinv[1,1]*set_v1+ccinv[1,2]*set_v2)   
                    gate3.set(point[2]+ccinv[2,1]*set_v1+ccinv[2,2]*set_v2)
                    
                    get_Q1Exp = p0.get()
                    get_Q2Exp = p1.get()
                    get_Q3Exp = p2.get()
                    datasaver.add_result((gate1, set_v1),
                                         (gate2, set_v2),
                                         (simtb01.CD, get_CD),
                                         (p0, get_Q1Exp),
                                         (p1, get_Q2Exp),
                                         (p2, get_Q3Exp))
        dataid = datasaver.run_id  # convenient to have for plotting
    
    
    dsMap = qc.dataset.data_export.load_by_id(dataidMap) #returns only points on sphere
    CDMap=dsMap.get_parameter_data('simtb01_CD')['simtb01_CD']['simtb01_CD']
    Q1Map=dsMap.get_parameter_data(p0.full_name)[p0.full_name][p0.full_name]
    Q2Map=dsMap.get_parameter_data(p1.full_name)[p1.full_name][p1.full_name]
    Q3Map=dsMap.get_parameter_data(p2.full_name)[p2.full_name][p2.full_name]
    
    ds = qc.dataset.data_export.load_by_id(dataid) #returns only points on sphere
    Q1val=ds.get_parameter_data(p0.full_name)[p0.full_name][p0.full_name]
    Q2val=ds.get_parameter_data(p1.full_name)[p1.full_name][p1.full_name]
    Q3val=ds.get_parameter_data(p2.full_name)[p2.full_name][p2.full_name]

    fig1=plt.figure()
    plt.imshow(np.rot90(CDMap.reshape(res,res)),cmap='inferno',extent=(axis))
    plt.plot(P1P2[:,0],P1P2[:,1],'r-')
    plt.xlabel('V%s [V]'%gate1.full_name[-1:],fontsize=14)
    plt.ylabel('V%s [V]'%gate2.full_name[-1:],fontsize=14)
    plt.title('RunID:#%i'%dataidMap+' Charge Stability Diagram')
    plt.colorbar()
    fig1.show()
    
    fig2=plt.figure()
    plt.imshow(np.rot90(Q1Map.reshape(res,res)),extent=(axis))
    plt.plot(P1P2[:,0],P1P2[:,1],'r-')
    plt.xlabel('V%s [V]'%gate1.full_name[-1:],fontsize=14)
    plt.ylabel('V%s [V]'%gate2.full_name[-1:],fontsize=14)
    plt.title('Expected Charge in %s'%p0.full_name[-5:-3],fontsize=14)
    plt.colorbar()
    plt.savefig('MapExpected_Charge_Q1',bbox_inches='tight')
    fig2.show()
    
    fig3=plt.figure()
    plt.imshow(np.rot90(Q2Map.reshape(res,res)),extent=(axis))
    plt.plot(P1P2[:,0],P1P2[:,1],'r-')
    plt.xlabel('V%s [V]'%gate1.full_name[-1:],fontsize=14)
    plt.ylabel('V%s [V]'%gate2.full_name[-1:],fontsize=14)
    plt.title('Expected Charge in %s'%p1.full_name[-5:-3],fontsize=14)
    plt.colorbar()
    plt.savefig('MapExpected_Charge_Q2',bbox_inches='tight')
    fig3.show()
    
    fig4=plt.figure()
    plt.imshow(np.rot90(Q3Map.reshape(res,res)),extent=(axis))
    plt.plot(P1P2[:,0],P1P2[:,1],'r-')
    plt.xlabel('V%s [V]'%gate1.full_name[-1:],fontsize=14)
    plt.ylabel('V%s [V]'%gate2.full_name[-1:],fontsize=14)
    plt.title('Expected Charge in %s'%p2.full_name[-5:-3],fontsize=14)
    Q3cbar=plt.colorbar()
    Q3cbar.set_label('Expected Charge [q]',size=14)
    plt.savefig('MapExpected_Charge_Q3',bbox_inches='tight')
    fig4.show()
    
    fig5=plt.figure()
    plt.plot(np.linspace(0,(res/10)*(P1P2.shape[0]-1),(res/10)*(P1P2.shape[0]-1)).T,Q1val,'r-')
    plt.xlabel('Point along cut',fontsize=14)
    plt.ylabel('Charge [q]',fontsize=14)
    plt.title('RunID:#%i'%dataid+' Expected Charge in %s'%p0.full_name[-5:-3],fontsize=14)
    plt.grid()
    plt.savefig('Expected_Charge_Q1',bbox_inches='tight')
    fig5.show()
    
    fig6=plt.figure()
    plt.plot(np.linspace(0,(res/10)*(P1P2.shape[0]-1),(res/10)*(P1P2.shape[0]-1)).T,Q2val,'b-')
    plt.xlabel('Point along cut',fontsize=14)
    plt.ylabel('Charge [q]',fontsize=14)
    plt.title('RunID:#%i'%dataid+' Expected Charge in %s'%p1.full_name[-5:-3],fontsize=14)
    plt.grid()
    plt.savefig('Expected_Charge_Q2',bbox_inches='tight')
    fig6.show()
    
    fig7=plt.figure()
    plt.plot(np.linspace(0,(res/10)*(P1P2.shape[0]-1),(res/10)*(P1P2.shape[0]-1)).T,Q3val,'g-')
    plt.xlabel('Point along cut',fontsize=14)
    plt.ylabel('Charge [q]',fontsize=14)
    plt.title('RunID:#%i'%dataid+' Expected Charge in %s'%p2.full_name[-5:-3],fontsize=14)
    plt.grid()
    plt.savefig('Expected_Charge_Q3',bbox_inches='tight')
    fig7.show()



def CDplot(dataid,n,dots):
    plt.figure(1)
    #ax,cbar = plot_by_id(dataid,cmap='inferno')
    
    ds = qc.dataset.data_export.load_by_id(dataid) 
    k=ds.get_parameter_data('simtb01_CD')['simtb01_CD']['simtb01_CD']
    x=ds.get_parameter_data('simtbpar_v1')['simtbpar_v1']['simtbpar_v1']
    y=ds.get_parameter_data('simtbpar_v2')['simtbpar_v2']['simtbpar_v2']
    

    vals=[]
    vals=np.append(vals,np.unique(k))
    vals=np.append(vals,max(vals)+0.5)
    
    cmap=plt.cm.get_cmap('inferno')
    norm=mpl.colors.BoundaryNorm(vals,cmap.N)
    
    
    foo=plt.imshow(np.rot90(k.reshape(int(ma.sqrt(len(k))),int(ma.sqrt(len(k))))),cmap=cmap,norm=norm,extent=[min(x),max(x),min(y),max(y)],aspect='auto')#,cmap=discrete_cmap(len(vals),'inferno'))


    ticklocs=[]
    for i in range(len(np.unique(k))):
        ticklocs.append((vals[i+1]+vals[i])/2)
    cbar=plt.colorbar(foo,ticks=np.array(ticklocs),spacing = 'uniform')#,levels=vals) #cbar_kwargs={'ticks' : vals})
    
    #cbar=cbar[0]
    plt.draw()
    
    Electrons=np.arange(n+1)
    emptyMatrix=[[0],[0],[0]]
#    emptyMatrix=[[0],[0]]
    for i in dots:
        emptyMatrix[i-1]=np.array(Electrons)

    stateList=np.array(list(itera.product(emptyMatrix[0],emptyMatrix[1],emptyMatrix[2])))
#    stateList=np.array(list(itera.product(emptyMatrix[0],emptyMatrix[1])))

     
#
#    ticks=np.empty(len(cbar.ax.get_yticklabels()),dtype=object)
#    i=0
    ticks=[]
    for t in np.unique(k): #cbar.ax.get_yticklabels():
#        A.append(int(t.get_text()))
        ticks.append(stateList[t])# int(t.get_text())])
        

    cbar.ax.set_yticklabels(ticks[:])

#    savepath=r'C:\Users\torbj\Google Drev\UNI\Bachelor\gif2/'
#    fname=(savepath+r'test_%i' %dataid+r'.png')
    #plt.savefig(fname=fname)
#    p=findradstart(dataid,31)
#    plt.plot(p[0],p[1],'r+')
    plt.title('Charge stability diagram of triple dot',fontsize=14)
    plt.xlabel('Gate1 [V]',fontsize=14)
    plt.ylabel('Gate2 [V]',fontsize=14)
    plt.plot(trajec[:,0],trajec[:,1],'r-')
    plt.plot(trajec[0,0],trajec[0,1],'ro')
    plt.savefig('ChargestabilityMap2d3d',bbox_inches='tight')
    plt.show()


cc=np.matrix([[1/np.sqrt(3),1/np.sqrt(3),1/np.sqrt(3)],[-1/np.sqrt(2),1/np.sqrt(2),0],[1/np.sqrt(6),1/np.sqrt(6),(-2)/np.sqrt(6)]])
cc_inv=cc.I
def matrixSweep(xstart,xstop,ystart,ystop,points,cc_inv):
    initialise_database()
    new_experiment(name='MOCKTR01test',
                              sample_name="no sample")
    
    meas = Measurement()
#    meas.register_parameter(simtbpar.v1)  # register the first independent parameter
    meas.register_parameter(simtbpar.v2)  # register the second independent parameter
    meas.register_parameter(simtbpar.v3)  # register the second independent parameter
    meas.register_parameter(simtb01.CD, setpoints=(simtbpar.v2,simtbpar.v3))  # now register the dependent oone
    
    meas.write_period = 2
    
    # v1=v17+v12+v43
    # v3=v17+v12-2*v43
    # v2=-v17+v12
    v0=[0.068877551020408004, 0.067346938775509999, 0.047448979591836749]
    
    with meas.run() as datasaver:
#        x=0.1
        for set_v2 in np.linspace(xstart, xstop, points):
            for set_v3 in np.linspace(ystart,ystop,points):
#                for set_v1 in np.linspace(0,0.15,points):
        

#        for set_v2 in np.linspace(-x,x,points):
#            for set_v3 in np.linspace(-x,x,points):
                #for set_v1 in np.linspace(-x,x,points):
                    set_v1=-0.045
                    simtbpar.v1.set(v0[0]+cc_inv[0,0]*set_v1+cc_inv[0,1]*set_v2+cc_inv[0,2]*set_v3)
                    simtbpar.v2.set(v0[0]+cc_inv[1,0]*set_v1+cc_inv[1,1]*set_v2+cc_inv[1,2]*set_v3)
                    simtbpar.v3.set(v0[0]+cc_inv[2,0]*set_v1+cc_inv[2,1]*set_v2+cc_inv[2,2]*set_v3)
                    get_v = simtb01.CD.get()
                    
                    datasaver.add_result((simtbpar.v2, set_v2),
                                         (simtbpar.v3, set_v3),
                                         (simtb01.CD, get_v))
        
        dataid = datasaver.run_id  # convenient to have for plotting
    
    CDplot(dataid,4,[1,2,3])
