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
    plt.figure(fig,figsize=[7,5]) 
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

#-----askisi 1--------------------------------
#---------------------------------------------

pigi = 0.2 #mikos pigis
l = 589*10**-9 #mikos kimatos
n = 1.6 #diktis diathlasis
L = 0.35 #apostasi metaxi mikas kai othonis
aktina = [0.075,0.097,0.115,0.13,0.144,0.158,0.171,0.181,0.190,0.198]
gonia = [-(i/(np.sqrt(i**2+L**2)))**2 for i in aktina]
num_daktilioi = [9,8,7,6,5,4,3,2,1,0]
slope1,ordinate1,uncertainty1,slopeerr1,ordinateerr1 = lsq(num_daktilioi,gonia,len(num_daktilioi))
plot_graph(num_daktilioi,gonia,0,5.37*10**-3,name[0],'graph1',r'$\eta\mu^{2}(\alpha) = f(m)$','Αριθμός Δακτυλίων (m)',r'$\eta\mu^{2}(\alpha)$',slope1,slopeerr1,ordinate1,ordinateerr1)
derror = np.sqrt(((l*n)/(slope1**2)*slopeerr1)**2)
d = (l*n)/slope1
print('Paxos imeniou aristera: %.5fmm -+ %.5f'%(d*10**3,derror*10**3))
aktina_deksia = [0.075,0.095,0.113,0.129,0.143,0.155,0.166,0.176,0.183,0.189]
gonia_deksia = [-(i/(np.sqrt(i**2+L**2)))**2 for i in aktina_deksia]
num_daktilioi_deksia = [9,8,7,6,5,4,3,2,1,0]
slope_deksia,ordinate_deksia,uncertainty_deksia,slopeerr_deksia,ordinateerr_deksia = lsq(num_daktilioi_deksia,gonia_deksia,len(num_daktilioi_deksia))
plot_graph(num_daktilioi_deksia,gonia_deksia,0,5.37*10**-3,name[4],'graph5',r'$\eta\mu^{2}(\alpha) = f(m)$','Αριθμός Δακτυλίων (m)',r'$\eta\mu^{2}(\alpha)$',slope_deksia,slopeerr_deksia,ordinate_deksia,ordinateerr_deksia)
d_deksia = (l*n)/slope_deksia
derror_deksia = np.sqrt(((l*n)/(slope_deksia**2)*slopeerr_deksia)**2)
print('Paxos imeniou deksia : %.5fmm -+ %.5f'%(d_deksia*10**3,derror_deksia*10**3))

#-----askisi 2--------------------------------
#---------------------------------------------
d = 0.000044

#-----galazio---------------------------------
pigi2 = 0.30 #mikos pigis
mika = 0.412 #apostasi mikas othonis
mika2=0.30 #apostasi mikas othonis
aktina_galazio_fake = [0,0.014,0.012,0.010,0.007,0.007,0.006,0.005,0.004,0.004] #315-445 #[0,0.012,0.011,0.01,0.009,0.008,0.008,0.007,0.006,0.005,0.005,0.004]
aktina_galazio = np.cumsum(aktina_galazio_fake) + (0.184/2) #euros485-565
gonia_galazio = [-(i/(np.sqrt(i**2+mika2**2)))**2 for i in aktina_galazio]  
num_daktilioi_galazio = [i for i in range(len(aktina_galazio)-1,-1,-1)]
slope_galazio,ordinate_galazio,uncertainty_galazio,slopeerr_galazio,ordinateerr_galazio = lsq(num_daktilioi_galazio,gonia_galazio,len(num_daktilioi_galazio))
plot_graph(num_daktilioi_galazio,gonia_galazio,0,5.37*10**-3,name[1],'graph2',r'$\eta\mu^{2}(\alpha) = f(m)$','Αριθμός Δακτυλίων (m)',r'$\eta\mu^{2}(\alpha)$',slope_galazio,slopeerr_galazio,ordinate_galazio,ordinateerr_galazio)  
l_galazio = d*slope_galazio/n
derror_galazio = np.sqrt((((d)/(n))*slopeerr_galazio)**2+(((d)/(n))*derror)**2)
print('Mikos kimatos galazio : %.3fnm -+ %.3f'%(l_galazio*10**9,derror_galazio*10**9))
#-----prasino---------------------------------
aktina_prasino_fake = [0,0.02,0.0150,0.014,0.014,0.013,0.013,0.012,0.011,0.01]
aktina_prasino = np.cumsum(aktina_prasino_fake) + (0.222/2) #euros485-565
gonia_prasino = [-(i/(np.sqrt(i**2+mika**2)))**2 for i in aktina_prasino]
num_daktilioi_prasino = [i for i in range(len(aktina_prasino)-1,-1,-1)]
slope_prasino,ordinate_prasino,uncertainty_prasino,slopeerr_prasino,ordinateerr_prasino = lsq(num_daktilioi_prasino,gonia_prasino,len(num_daktilioi_prasino))
plot_graph(num_daktilioi_prasino,gonia_prasino,0,5.37*10**-3,name[2],'graph3',r'$\eta\mu^{2}(\alpha) = f(m)$','Αριθμός Δακτυλίων (m)',r'$\eta\mu^{2}(\alpha)$',slope_prasino,slopeerr_prasino,ordinate_prasino,ordinateerr_prasino)
l_prasino = d*slope_prasino/n
derror_prasino = np.sqrt((((d)/(n))*slopeerr_prasino)**2+(((d)/(n))*derror)**2)
print('Mikos kimatos prasino : %.3fnm -+ %.3f'%(l_prasino*10**9,derror_prasino*10**9))
#-----kitrino---------------------------------
aktina_kitrino_fake = [0,0.021,0.018,0.018,0.016,0.015,0.013,0.012,0.012,0.011]
aktina_kitrino = np.cumsum(aktina_kitrino_fake) + (0.222/2) #euros485-565
gonia_kitrino = [-(i/(np.sqrt(i**2+mika**2)))**2 for i in aktina_kitrino]
num_daktilioi_kitrino = [i for i in range(len(aktina_kitrino)-1,-1,-1)]
slope_kitrino,ordinate_kitrino,uncertainty_kitrino,slopeerr_kitrino,ordinateerr_kitrino = lsq(num_daktilioi_kitrino,gonia_kitrino,len(num_daktilioi_kitrino))
plot_graph(num_daktilioi_kitrino,gonia_kitrino,0,5.37*10**-3,name[3],'graph4',r'$\eta\mu^{2}(\alpha) = f(m)$','Αριθμός Δακτυλίων (m)',r'$\eta\mu^{2}(\alpha)$',slope_kitrino,slopeerr_kitrino,ordinate_kitrino,ordinateerr_kitrino)
l_kitrino = d*slope_kitrino/n
derror_kitrino = np.sqrt((((d)/(n))*slopeerr_kitrino)**2+(((d)/(n))*derror)**2)
print('Mikos kimatos kitrino : %.3fnm -+ %.3f'%(l_kitrino*10**9,derror_kitrino*10**9))
