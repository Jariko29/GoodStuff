#------------------------------------------------
#------------import libraries--------------------
import numpy as np 
import matplotlib.pyplot as plt 
import numpy.random as rn
from scipy.optimize import curve_fit
from sympy import symbols,cos,sin,sqrt,tan,sec,asin
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
def plot_graph(x, y,xerr,yerr,plotname,fig,graphtitle,xlab,ylab):
    plt.figure(fig) 
    plt.gca().set_facecolor('0.88')
    plt.grid(True)
    plt.errorbar(x, y, xerr=xerr,yerr=yerr,fmt='o',markerfacecolor='none', color='black',elinewidth=0.7,label='data')
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

def linear_plot(x,y,ye,xe,slope,slopeer,ordinate,ordinateer,plotname,fig,fit,graphtitle,xlab,ylab):
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
