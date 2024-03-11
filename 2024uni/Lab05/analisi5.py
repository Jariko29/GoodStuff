#------------------------------------------------
#------------import libraries--------------------
import numpy as np 
import matplotlib.pyplot as plt 
import numpy.random as rn
from scipy.optimize import curve_fit
from sympy import symbols, cos,sin,sqrt,tan,sec, asin
#------------------------------------------------
#--------------dedomena--------------------------
#------------------------------------------------
name=['graph1','graph2','graph3','graph4','graph5','graph6','graph7','graph8']
#------------------------------------------------
#--------------insert values here ---------------
#------------------------------------------------

def sfalma():
    return


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
def avg_and_std(x):
    mean = np.mean(x)
    std = np.std(x)
    return mean,std
 #-----------------------------------------------
#---creating fit functions-----------------------
#------------------------------------------------
def func1 (x,a):
    return a/x # fit , change if something else
expression1 = r'$f = \frac{u}{2L}$'
def func2 (x,a):
    return a*np.sqrt(x) # fit , change if something else
expression2 = r'$f = \frac{1}{2L}\sqrt{\frac{T}{S \rho}}$'
def func3 (x,a):
    return 2*L_/(x+1) # fit , change if something else
expression3 = r'$f = \frac{2L}{n+a}$'
#------------------------------------------------
#------------plot graph--------------------------
#------------------------------------------------
def plot_graph(x, y,plotname,fig,fit,fitname,graphtitle,xlab,ylab):# slope, ordinate,yerr
   
   
    n=len(x)
    params,params_covariance=curve_fit(fit,x,y,p0=[0])
    print("parametri:",params) 
    
    plt.figure(fig) 
    fitx=np.linspace(min(x)-10,max(x)+10,1000)
    plt.plot(fitx,fit(fitx,*params),label='curve fit:%s,\n parameters: a=%.2f +-%.2f '%(fitname,params[0],params_covariance[0,0]))
    plt.gca().set_facecolor('0.88')
    plt.grid(True)
    #plt.errorbar(x, y, yerr=yerr, label='σφάλμα',fmt='o', color='black',elinewidth=0.7)
    colors = np.random.rand(n)
    plt.scatter(x, y,label='data')
    #--------------------------
    #---change for each plot---
    #--------------------------
    plt.title(graphtitle)
    plt.xlabel(xlab)
    plt.ylabel(ylab)
    plt.legend(loc='best')
    plt.savefig(plotname, dpi=300)

#--------------askisi 1--------------------------
#------------------------------------------------
L=[30,40,50,60,70,80]#metrisis apo 30-80cm
f30=[228.3,228.2,228.3,228.0,228.1]#sixnotita  sinartisi mikous
f40=[170.7,170.6,170.9,170.2,170.4]
f50=[136.4,136.3,136.6,136.5,136.5]
f60=[113.6,113.7,113.7,113.8,113.6]
f70=[97.6,97.5,97.6,97.3,97.6]  
f80=[85.5,85.5,85.4,85.3,85.4]
#------------------------------------------------
#----------------askisi 2------------------------
T=[12,14,16,18,20,22,25,28,30]#tasi apo 5N-25N
f12=[82.9,82.9,83,83,82.5]#sixnotita talantwsis sinartisi tasis
f14=[94.7,94.5,94.2,94.3,94.3]
f16=[102.8,102.8,102.7,102.9,102.6]
f18=[110.8,110.6,110.7,110.8,110.7]
f20=[119.1,119.2,119.2,119.1,119.1]
f22=[126.7,126.6,125.9,126.0,125.9]
f25=[136.8,136.8,136.7,136.8,136.8]
f28=[145.9,145.8,145.4,144.8,144.9]
f230=[150.9,150.7,150.5,150.4,150.4]
#------------------------------------------------
#----------------askisi 3------------------------
#------------------------------------------------
f = [18.47,26.38,34.13,42.88,54.04,63.26,66.96]
n = [1,2,3,4,5,6,7]
l = [30,21,16,13,9.5,8,7.5] #wave/2 in cm
L_ = 60.5 #length of string in cm

#------------------------------------------------
#------------function call-----------------------
#ask1
ask1= [f30, f40, f50, f60, f70, f80]
favg = [avg_and_std(f)[0] for f in ask1]
fstd = [avg_and_std(f)[1] for f in ask1]  
plot_graph(L,favg, name[0], 1, func1, expression1, 'f=F(L)','Μήκος χορδής (cm)','Συχνότητα (Hz)'	)
#ask2
ask2= [f12, f14, f16, f18, f20, f22, f25, f28, f230]
favg = [avg_and_std(f)[0] for f in ask2]
fstd = [avg_and_std(f)[1] for f in ask2]
plot_graph(T,favg, name[1], 2, func2, expression2, 'f=F(T)','Τάση (N)','Συχνότητα (Hz)'	)
#ask3
plot_graph(n,l, name[2], 3, func3, expression3, 'f=F(n)','Αριθμός συναντήσεων','Συχνότητα (Hz)')


#------------------------------------------------
#------------plot graph--------------------------
#------------------------------------------------
