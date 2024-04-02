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
def avg_and_std(x):
    mean = np.average(x)
    std = np.std(x)
    avger = std/np.sqrt(len(x))
    return mean,std,avger
#-----------------------------------------------
#---creating fit functions-----------------------
#------------------------------------------------
def func1 (x,a,b):
    return 
expression1 = r'$V = \alpha \left[\frac{\ c ^2}{(f - b)^2 + \ c ^2}\right]$'
def func2 (x,a,b):
    return 
expression2 = r'$y = \frac{a}{x + b}$'
#------------------------------------------------
#------------plot graph--------------------------
#------------------------------------------------
def plot_graph(x,y,xerr,yerr,plotname,fig,graphtitle,xlab,ylab,slope,slopeer,ordinate,ordinateer):
    plt.figure(fig) 
    plt.gca().set_facecolor('0.88')
    plt.grid(True)
    plt.errorbar(x, y, xerr=xerr,yerr=yerr,fmt='o',markerfacecolor='none', color='black',elinewidth=0.7,label='data')
    plt.plot(x,slope*np.array(x)+ordinate,label='y=(%.4f+-%.4f) x +(%.3f+-%.3f)'%(slope,slopeer,ordinate,ordinateer),color='red')
    plt.title(graphtitle)
    plt.xlabel(xlab)
    plt.ylabel(ylab)
    plt.legend(loc='best')
    plt.savefig(plotname, dpi=300)
#------------------------------------------------
#------------fit_graph_plot----------------------
#------------------------------------------------  
def fit_plot (x,y,plotname,fig,fit,fitname):
    parm=[1,0]
    params,params_covariance=curve_fit(fit,x,y,p0=parm,maxfev=10000)
    fitx=np.linspace(min(x),max(x),1000)
    fity=fit(fitx,*params)
    a,b=params
    aa,bb=np.sqrt(np.diag(params_covariance))
    print("parametri:a=%.2f, b=%.2f, c=%.2f, d=%.2f"%(a,b))
    plt.figure(fig)
    plt.grid(True)
    plt.plot(fitx,fity,label='curve fit:%s,\n parameters: a=%.2f b=%.2f c=%.2f \n covariance: a=+- %.0f, b=+- %.1f , c=+-%.0f'%(fitname,a,b,aa,bb))
    plt.legend(loc='lower center')
    plt.savefig(plotname, dpi=300)
#------------------------------------------------
#------------fit_2-------------------------------
#------------------------------------------------
def fit_plot2 (x,y,plotname,fig,fit,fitname):
    parm=[1,0]
    params,params_covariance=curve_fit(fit,x,y,p0=parm,maxfev=10000)
    fitx=np.linspace(min(x),max(x),1000)
    fity=fit(fitx,*params)
    a,b=params
    aa,bb=np.sqrt(np.diag(params_covariance))
    print("parametri:a=%.2f, b=%.2f"%(a,b))
    plt.figure(fig)
    plt.grid(True)
    plt.plot(fitx,fity,label='curve fit:%s,\n parameters: a=%.3f b=%.2f \n covariance: a=+- %.0f, b=+- %.1f '%(fitname,a,b,aa,bb))
    plt.legend()
    plt.savefig(plotname, dpi=300)
#------------------------------------------------
#---------log_graph_plot-------------------------
#------------------------------------------------

def log_graph(x,y,ye,slope,slopeer,ordinate,ordinateer,plotname,fig,fit,graphtitle,xlab,ylab):
    plt.figure(fig)
    plt.grid(True)
    plt.errorbar(x,y,yerr=ye,fmt='o',markerfacecolor='none', color='black',elinewidth=0.7)
    plt.plot(x,fit(x,slope,10**ordinate),label='y=(%.2f+-%.2f x +(%.2f+-%.2f)'%(slope,slopeer,ordinate,ordinateer),color='red')
    plt.xscale('log')
    plt.yscale('log')
    plt.title(graphtitle)
    plt.xlabel(xlab)
    plt.ylabel(ylab)    
    plt.legend(loc='best')
    plt.savefig(plotname, dpi=300)

def linear (x,a,b):
    return a*x+b
#------------------------------------------------

def linear_plot(x,y,ye,xe,slope,slopeer,ordinate,ordinateer,plotname,fig,graphtitle,xlab,ylab):
    plt.figure(fig)
    plt.grid(True)
    plt.errorbar(x,y,yerr=ye,xerr=xe,fmt='o',markerfacecolor='none', color='black',elinewidth=0.7)
    plt.plot(x,slope*np.array(x)+ordinate,label='y=(%.0f+-%.0f) x +(%.1f+-%.1f)'%(slope,slopeer,ordinate,ordinateer),color='red')
    plt.title(graphtitle)
    plt.xlabel(xlab)
    plt.ylabel(ylab)    
    plt.legend(loc='best')
    plt.savefig(plotname, dpi=300)

#------------------------------------------------
#------------function call-----------------------
#------------------------------------------------

#-----askisi 1---------------------------------
l = 589*10**-9 #mikos kimatos
n = 1.6 #diktis diathlasis
L = 0.5 #apostasi metaxi pigis kai othonis
aktina = [0,0.01,0.02,0.045,0.07,0.10,0.13,0.17,0.21]
gonia = [-(i/(np.sqrt(i**2+L**2)))**2 for i in aktina]
num_daktilioi = [8,7,6,5,4,3,2,1,0]
slope1,ordinate1,uncertainty1,slopeerr1,ordinateerr1 = lsq(num_daktilioi,gonia,len(num_daktilioi))
plot_graph(num_daktilioi,gonia,0,0,name[0],'graph1','Gonia vs Num_daktilioi','Num_daktilioi','sin^2(gonia)',slope1,slopeerr1,ordinate1,ordinateerr1)
derror = np.sqrt(((l*n)/(slope1**2)*slopeerr1)**2)
d = (l*n)/slope1
print('Paxos imeniou : %.5fm -+ %.5f'%(d,derror))

#-----askisi 2---------------------------------
#------------------------------------------------
aktina_galazio = [0,0.01,0.02,0.045,0.07,0.10,0.13,0.17,0.21]
gonia_galazio = [-(i/(np.sqrt(i**2+L**2)))**2 for i in aktina_galazio]  
num_daktilioi_galazio = [8,7,6,5,4,3,2,1,0]
slope_galazio,ordinate_galazio,uncertainty_galazio,slopeerr_galazio,ordinateerr_galazio = lsq(num_daktilioi_galazio,gonia_galazio,len(num_daktilioi_galazio))
plot_graph(num_daktilioi_galazio,gonia_galazio,0,0,name[1],'graph2','Gonia vs Num_daktilioi','Num_daktilioi','sin^2(gonia)',slope_galazio,slopeerr_galazio,ordinate_galazio,ordinateerr_galazio)  
l_galazio = d*slope_galazio/n
derror_galazio = np.sqrt((((d)/(n))*slopeerr_galazio)**2+(((d)/(n))*derror)**2)
print('Mikos kimatos galazio : %.3fnm -+ %.3f'%(l_galazio*10**9,derror_galazio*10**9))

aktina_prasino = [0,0.01,0.02,0.045,0.07,0.10,0.13,0.17,0.21]
gonia_prasino = [-(i/(np.sqrt(i**2+L**2)))**2 for i in aktina_prasino]
num_daktilioi_prasino = [8,7,6,5,4,3,2,1,0]
slope_prasino,ordinate_prasino,uncertainty_prasino,slopeerr_prasino,ordinateerr_prasino = lsq(num_daktilioi_prasino,gonia_prasino,len(num_daktilioi_prasino))
plot_graph(num_daktilioi_prasino,gonia_prasino,0,0,name[2],'graph3','Gonia vs Num_daktilioi','Num_daktilioi','sin^2(gonia)',slope_prasino,slopeerr_prasino,ordinate_prasino,ordinateerr_prasino)
l_prasino = d*slope_prasino/n
derror_prasino = np.sqrt((((d)/(n))*slopeerr_prasino)**2+(((d)/(n))*derror)**2)
print('Mikos kimatos prasino : %.3fnm -+ %.3f'%(l_prasino*10**9,derror_prasino*10**9))