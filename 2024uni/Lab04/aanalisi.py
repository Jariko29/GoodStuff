#------------------------------------------------
#------------import libraries--------------------
import numpy as np 
import matplotlib.pyplot as plt 
import numpy.random as rn
from scipy.optimize import curve_fit
#------------------------------------------------
#--------------dedomena--------------------------
#------------------------------------------------
name=['graph1','graph2','graph3','graph4','graph5','graph6','graph7','graph8']
#------------------------------------------------
#--------------insert values here ---------------
#------------------------------------------------
#--------------askisi 1--------------------------
thetar1=[35.61,37.01,38.24,39.51,41.51,43.11,45.31,48.15,51.55,54.02,56.52,59.05,60.69,62.95,65.45,67.52,69.69,71.29,73.29,75.46]# gonies anaklasis meta3i 40-70
l1=[16.2,14.9,13.3,12.3,9.6,8.4,6.7,4.0,1.3,0.4,0.1,1.2,2.7,5.8,12.2,20.8,33.3,40.6,52.7,72.3]# sintelestis anaklastikotitas-parallili sinistosa prosptosis Rp1  #anixneutis1
l2=[93.0,93.2,93.2,92.6,92.9,97.0,96.9,96.9,96.7,96.7,96.8,96.9,96.7,97.0,96.8,96.8,95.3,95.8,95.2,95.5]# sintelestis anaklastikotitas-katheti sinistosa Rp2               #anixneutis 2
ratio1=[i/j for i,j in zip(l1,l2)]
thetar2=[41.14,43.68,46.14,49.01,51.58,53.21,55.52,57.05,58.82,60.95,62.79,64.55,67.39]# gonies anaklasis meta3i 40-70 - askisi 1b
L1=[8.1,7.9,12.1,13.5,12.8,13.8,23.8,30.1,29.7,39.7,47.1,44.5,54.8]#ani3neutis 1 - askisi 1b
L2=[62.4,62.3,62.3,62.4,62.2,62.5,61.9,62.1,62.1,60.6,59.3,54.2,56.1]#ani3neutis 2 - askisi 1b
ratio2=[i/j for i,j in zip(L1,L2)]
#------------------------------------------------
#----------------askisi 2------------------------
thetar3=[40.57,45.51,48.71,50.38,53.45,55.98,58.58,61.12,63.62,65.62,67.79,70.06,72.32]# gonies anaklasis meta3i 40-70
I1=[12.3,6.4,4.8,3.0,1.5,0.4,0,0.5,5.7,9.8,19.6,28.3,50.5]
I2=[53.2,67.2,66.8,68.2,65.9,67.2,64.4,64.6,59.0,60.2,58.6,73.5,74.3]
ratio3=[i/j for i,j in zip(I1,I2)]
#------------------------------------------------
#------------least squares-----------------------
#------------------------------------------------
def lsq (x,y,n):
    sumx=sum(x)
    sumy=sum(y)
    sumx2=sum([i**2 for i in x])
    sumxy=sum(i*j for i,j in zip(x,y))  # Corrected here
    delta=n*sumx2-sumx**2
    slope=(n*sumxy-sumx*sumy)/delta
    ordinate=(sumx2*sumy-sumx*sumxy)/delta
    riza=0
    for i in range (n):
        riza+=(y[i]-ordinate-slope*x[i])**2  
    uncertainty=np.sqrt(1/(n-2)*riza)
    slopeerr=np.sqrt(uncertainty**2*n/delta)
    ordinateerr=np.sqrt(uncertainty**2*sumx2/delta)
    return (slope,ordinate,uncertainty,slopeerr,ordinateerr)
#------------------------------------------------
#--------mean value and STDeviation--------------
#------------------------------------------------
def mean_and_std(x1,x2):
    mean = [i+j for i,j in (zip(x1,x2))]
    mean=[i/2 for i in mean]
    meansqrt=[((i+j)/2)**2 for i,j in (zip(x1,x2))]
    std=[np.sqrt(i-j**2) for i,j in (zip(meansqrt,mean))]
    return mean,std
#------------------------------------------------
#------------plot graph--------------------------
#------------------------------------------------
def plot_graph(x, y,plotname,fig):# slope, ordinate, yerr,
    #----------------------------
    #---creating fit functions---
    #----------------------------
    def func (x,a,b):
       return a*(np.tan(np.radians(x-b)))**2/(np.tan(np.radians(x+b)))**2 # fit , change if something else
    n=len(x)
    params,params_covariance=curve_fit(func,x,y,p0=[8,np.pi/4])
    print("parametri:",params) 
    #coeffs = np.polyfit(x, y, 3)
    #print('parameters:',coeffs)
    #fitted_poly=np.poly1d(coeffs)
    #yfit=fitted_poly(x)
    plt.figure(fig)
    #plt.figure(figsize=(10,8))  
    plt.plot(x,func(x,*params),label='curve fit')
    plt.gca().set_facecolor('0.88')
    plt.grid(True)
    #plt.errorbar(x, y, yerr=yerr, label='σφάλμα',fmt='o', color='black',elinewidth=0.7)
    colors = np.random.rand(n)
    plt.scatter(x, y,label='data')
    #--------------------------
    #---change for each plot---
    #--------------------------
    plt.title('Διάγραμμα ποσοστιαίας έντασης-γωνίας')
    plt.xlabel('Γωνία (μοίρες)')
    plt.ylabel('Ενταση (Ι-%)')
   

    plt.legend(loc='best')
    plt.savefig(plotname, dpi=300)

#------------------------------------------------
#------------function call-----------------------

#------------------------------------------------
#------------plot graph--------------------------
#------------------------------------------------
plot_graph(thetar1,ratio1,'graph1',1)
plot_graph(thetar2,ratio2,'graph2',2)
plot_graph(thetar3,ratio3,'graph3',3)