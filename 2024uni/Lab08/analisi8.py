3#------------------------------------------------
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
def avg_and_std(x):
    mean = np.average(x)
    std = np.std(x)
    avger = std/np.sqrt(len(x))
    return mean,std,avger
#-----------------------------------------------
#---creating fit functions-----------------------
#------------------------------------------------
def func1 (x,a,b,c,d):
    return a*(c**2)/((x-b)**2+c**2)+d # fit , change if something else
expression1 = r'$V = \alpha \left[\frac{\ c ^2}{(f - b)^2 + \ c ^2}\right]$'
def func2 (x,a,b):
    return a/(x+b) # fit , change if something else
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
    parm=[1,0,1,0]
    params,params_covariance=curve_fit(fit,x,y,p0=parm,maxfev=10000)
    fitx=np.linspace(min(x),max(x),1000)
    fity=fit(fitx,*params)
    a,b,c,d=params
    aa,bb,cc,dd=np.sqrt(np.diag(params_covariance))
    print("parametri:a=%.2f, b=%.2f, c=%.2f, d=%.2f"%(a,b,c,d))
    plt.figure(fig)
    plt.grid(True)
    plt.plot(fitx,fity,label='curve fit:%s,\n parameters: a=%.2f b=%.2f c=%.2f \n covariance: a=+- %.0f, b=+- %.1f , c=+-%.0f'%(fitname,a,b,c,aa,bb,cc))
    plt.legend(loc='lower center')
    plt.savefig(plotname, dpi=300)
#------------------------------------------------
#------------fit_2-------------------------------
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
def log_func(x,a,b):
    return b*np.array(x)**a   
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
def linear2 (x,a,b):
    return a*x+b

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
#------------------------------------------------
#prosdiorismos kathisterisis simaton sta sirmata ton siskevon
si=[0.37,0.31,37.9]                           #
sf=[0.32,0.266,0.309]#                         #mpori na min xreiastoun
#ds=[j-i for i,j in zip(si,sf)]  #
ds=[0.05,0.044,0.07]
dt=[320*10**-9,420*10**-9,240*10**-9]                           #
#dsavg,dsstd,dsavger=avg_and_std(ds)
dsavg=sum(ds)/len(ds)
dtavg,dtstd,dtavger=avg_and_std(dt) 

#-----askisi 1---------------------------------
f1a= 60*10**6                           #sixnotites 
f1b=59.9*10**6                            #palmografou
M=f1a/(f1a-f1b)                 #paragontas epivradinsis
Mer=0     #sfalma paragonta epivradinsis
s1=0.424#0.385#[0.35,0.373,0.349,0.33]
s2=[0.393,0.365,0.331,0.306,0.280,0.270,0.251]#[37,33.9,31.1]
ds1=[s1-i for i in s2]
dtprime=[96*10**-9,168*10**-9,264*10**-9,300*10**-9,320*10**-9,356*10**-9,404*10**-9]
dt1=dtprime
slope1,ordinate1,uncertainty1,slopeerr1,ordinateerr1=lsq(dt1,ds1,len(ds1))
linear_plot(dt1,ds1,0.005,50*10**-9,slope1,slopeerr1,ordinate1,ordinateerr1,name[0],1,linear,'Δs=f(Δt)','χρονικό διάστημα(s)','μεταβολή θέσης(m)')
c=slope1*M;cer=np.sqrt((slope1*Mer)**2+(M*slopeerr1)**2)
sigma1=np.abs(3*10**8-c)/cer
akrivia1=np.abs((3*10**8)-c)/(3*10**8)*100# % apoklisi

print('ταχύτητα φωτός: %.1f+-%.1f m/s'%(c,cer),'\n','απόκλιση σ: %.2f'%sigma1,'\n','στατιστικό σφάλμα: %.1f%%\n'%akrivia1)
#print('ταχύτητα φωτός: %.2f+-%.2f m/s'%(c,cer),'\n','απόκλιση σ: %.2f'%sigma1,'\n','στατιστικό σφάλμα: %.2f '%akrivia1)
#------------------------------------------------
#-----askisi 2-----------------------------------
ca=c/1.0028
s2=0.95
s2er=0.05
dt2=[580*10**-9,520*10**-9,500*10**-9,540*10**-9,520*10**-9]
dt2avg,dt2std,dt2avger=avg_and_std(dt2)
nw=1+(ca*dt2avg)/(M*s2)                     #refractive index of water
nwer=np.sqrt((dt2avg/(M*s2)*ca)**2+(ca/(M**s2)*dt2std)**2+((ca*dt2avg)/(M*s2**2*s2er)**2)+((ca*dt2avg)/(M**2*s2)*Mer)**2)# error in refractive index of water
cw=c/nw
cwer=np.sqrt((1/nw*cer)**2+(c/nw**2*nwer)**2)
print('δείκτης διάθλασης νερού: (%.2f +-%.2f)\n'%(nw,nwer))
print('ταχύτητα φωτός στο νερό: (%.0f +-%.0f) m/s\n'%(cw,cwer))
cwater=225000000
nwater=1.33

sigma2=np.abs(cwater-cw)/cwer
akrivia2=np.abs(cwater-cw)/cwater *100# % apoklisi
sigma22=np.abs(nwater-nw)/nwer
akrivia22=np.abs(nwater-nw)/nwater *100# % apoklisi
#print('απόκλιση σ ταχύτητας φωτός στο νερό: %.2f'%sigma2,'\n','στατιστικό σφάλμα: %.2f%'%akrivia2)
#print('απόκλιση σ δείκτη διάθλασης νερού: %.2f'%sigma22,'\n','στατιστικό σφάλμα: %.2f%'%akrivia22)
#------------------------------------------------
#-----askisi 3-----------------------------------
dt3=[780*10**-9,780*10**-9,800*10**-9,720*10**-9,860*10**-9,880*10**-9]
dt3avg,dt3std,dt3avger=avg_and_std(dt3)
s3=0.995
s3er=0.05
nm=1+(ca*dt3avg)/(M*s3)         #refractive index of acrylic
nmer=np.sqrt((dt3avg/(M*s3)*ca)**2+(ca/(M**s3)*dt3std)**2+((ca*dt3avg)/(M*s3**2*s3er)**2)+((ca*dt3avg)/(M**2*s3)*Mer)**2)# error in refractive index
cm=c/nm #speed of light in acrylic
cmer=np.sqrt((1/nm*cer)**2+(c/nm**2*nmer)**2)
print('δείκτης διάθλασης ακρυλικού: (%.2f +-%.2f)\n'%(nm,nmer))
print('ταχύτητα φωτός στο ακρυλικό: (%.2f +-%.2f) m/s\n'%(cm,cmer))
cacrylic=3*10**8/1.49
nacrylic=1.49
sigma3=np.abs(cacrylic-cm)/cmer
akrivia3=np.abs(cacrylic-cm)/cacrylic *100# % apoklisi
sigma33=np.abs(nacrylic-nm)/nmer
akrivia33=np.abs(nacrylic-nm)/nacrylic *100# % apoklisi
print('απόκλιση σ ταχύτητας φωτός στο ακρυλικό: %.2f'%sigma3,'\n','στατιστικό σφάλμα: %.2f%%\n'%akrivia3)
print('απόκλιση σ δείκτη διάθλασης ακρυλικού: %.2f'%sigma33,'\n','στατιστικό σφάλμα: %.2f%%\n'%akrivia33)