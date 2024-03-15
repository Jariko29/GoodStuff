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
name=['graph1','graph2','graph3','graph4','graph5','graph6','graph7','graph8','graph9']
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
    plt.legend()
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
    plt.plot(fitx,fity,label='curve fit:%s,\n parameters: a=%.2f b=%.2f \n covariance: a=+- %.0f, b=+- %.1f '%(fitname,a,b,aa,bb))
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
    return a/x+b

def linear_plot(x,y,ye,slope,slopeer,ordinate,ordinateer,plotname,fig,fit,graphtitle,xlab,ylab):
    plt.figure(fig)
    plt.grid(True)
    plt.errorbar(x,y,yerr=ye,fmt='o',markerfacecolor='none', color='black',elinewidth=0.7)
    plt.plot(x,slope*np.array(x)+ordinate,label='y=(%.2f+-%.2f) x +(%.2f+-%.2f)'%(slope,slopeer,ordinate,ordinateer),color='red')
    plt.title(graphtitle)
    plt.xlabel(xlab)
    plt.ylabel(ylab)    
    plt.legend(loc='best')
    plt.savefig(plotname, dpi=300)

#------------------------------------------------
#------------function call-----------------------
#------------------------------------------------
#------------------------------------------------
#-------------askisi 1 a------------------------- προσδιορισμός συχνότητας συντονισμού
V1=[40,55.5,75.5,100,140,190,230,240,250,240,200,190,225,223,180,150,115,90,70]                                             #πλάτος ταλάντωσης mV
f=[37.250,37.500,37.750,38,38.250,38.500,38.73,38.84,39.07,39.45,40.0,40.250,40.750,41.15,41.35,41.50,41.75,42.00,42.25]   
Ve1=1#[np.sqrt((1/f[i]*1)**2+(-16074/f[i]**2*0.001)**2) for i in range(len(f))]                                              #σφάλμα πλάτους ταλάντωσης                                         #συχνότητα	  kHz                                            #συχνότητα                                            #συχνότητα
fe=0.1                                          #σφάλμα συχνότητας
#------------------------------------------------
#-------------askisi 1 b---------------------------ευρέση σχέσης μεταξύ πλάτους ταλάντωσης και απόστασης δέκτη-πομπού
V2=[1100,720,440,400,350,270,240,230,220,170,150,150,140,120]                                             #πλάτος ταλάντωσης
d=[3,4,5,6,7,8,9,10,11,12,13,14,15,16]                                              #απόσταση δέκτη-πομπού
de=0.5                                         #σφάλμα απόστασης δέκτη-πομπού
Ve2=[np.sqrt((1/d[i])**2+(-V2[i]/d[i]**2*de)**2)  for i in range(len(d))]                                       #σφάλμα πλάτους ταλάντωσης
logV2=[np.log10(i) for i in V2]                   #λογαριθμός πλάτους ταλάντωσης
logd=[np.log10(i) for i in d]        
loge=[Ve2[i]/logV2[i] for i in range(len(Ve2))]             #λογαριθμός απόστασης δέκτη-πομπού

#------------------------------------------------
#-------------askisi 2---------------------------υπολογισμός ταχύτητας του ήχουν στον αέρα (κλίση ευθείας)
ds=[8,10,12,14,16,18,20,22,25,28,31,33,38,41.5,45]                                             #απόσταση
ds=[i*10**(-2) for i in ds]
r=[np.sqrt(0.06**2+ds[i]**2) for i in range(len(ds))]
re=[np.sqrt((36/np.sqrt(36+i**2)*0.001)**2+(i/np.sqrt(36+i**2)*0.001)**2) for i in range(len(ds))]
dse=0.001                                            #σφάλμα απόστασης
dx=[ds[i]+r[i] for i in range(len(ds))]
dxe=[i+dse for i in re]
dt=[0.62,0.72,0.84,0.94,1.06,1.18,1.28,1.38,1.54,1.74,1.9,2.02,2.32,2.52,2.74]                                             #χρόνος  
dt=[i*10**(-3) for i in dt]                                           #σφάλμα χρόνου
#------------------------------------------------
#-------------askisi 3---------------------------υπολογισμός μήκους κύματος
N=[10,10,10,10,10,10,10,10,10,10]                         #αριθμός μέγιστων
x1=[34.5,38.6,44.4,37.5,41.7,46.1,50.6,55,31.8,36.1]                                            #θέση μηδέν μεγιστού
x1e=[]                                           #σφάλμα θέσης μηδέν μέγιστου
x2=[38.6,43,49,41.7,46.1,50.6,55,59.5,36.1,40.3]                                            #τελική θέση μεγιστού
x2e=[]                                           #σφάλμα τελικής θέσης μέγιστου

#------------------------------------------------
#-------------askisi 4---------------------------φαινόμενο Doppler ταχύτητα ήχου στον αέρα
v4=[134,180,208,222,218,198,192,200,238,268,252,182,124]
f4=[38.3,38.6,38.85,39.2,39.5,39.75,40,40.2,40.5,40.7,40.96,41.25,41.5]
fi=[400003,400002,400003,400003,400003,400002,400002,400003,400002,400004,400003,400003,400003,400003,400002,400002,400001]                                            #αρχική συχνότητα
fie=0                                             #σφάλμα αρχικής συχνότητας
ff=[400028,400038,400036,400048,400051,400050,400048,400043,400044,400046,400061,400055,400045,400048,400041,400058,400029]                                            #τελική συχνότητα
df=[i-j for i,j in zip(ff,fi)]                                             #διαφορά συχνοτήτων

ffe=0                                             #σφάλμα τελικής συχνότητας
dt2=[0.293,0.232,0.261,0.252,0.226,0.230,0.241,0.249,0.228,0.228,0.181,0.202,0.265,0.212,0.204,0.204,0.307]                                           #χρονικό διάστημα κίνησης 
dt2e=0                                            #σφάλμα χρόνου
u=[0.1*10**(-2)/i for i in dt2]                                             #ταχύτητα αυτοκινήτου
fdf=[i/j  for i,j in zip(df,fi)]
ue=0                                              #σφάλμα ταχύτητας αυτοκινήτου
#------------------------------------------------
#-------------askisi 5---------------------------
#------------------------------------------------
#------------function call-----------------------
#------------------------------------------------

#1a
plot_graph(f,V1,fe,Ve1,name[0],1,'Πλάτος ταλάντωσης-Συχνότητα','f(kHz)','V(mV)')
fit_plot(f,V1,name[0],1,func1,expression1)
#1b
plot_graph(d,V2,None,Ve2,name[1],2,'Πλάτος ταλάντωσης-Απόσταση','d(cm)','V(mV)')
fit_plot2(d,V2,name[1],2,func2,expression2)
slope1,ordinate1,uncertainty1,slopeerr1,ordinateerr1=lsq(logd,logV2,len(logd))
log_graph(d,V2,loge,slope1,slopeerr1,ordinate1,ordinateerr1,name[2],3,log_func,'Πλάτος ταλάντωσης-Απόσταση','d(cm)','V(mV)')
#plot_graph(logd,logV2,None,Ve2,name[8],8,'Πλάτος ταλάντωσης-Απόσταση','log(d)','log(V)')
slope2,ordinate2,uncertainty2,slopeerr2,ordinateerr2=lsq(dt,dx,len(dx))
linear_plot(dt,dx,dxe,slope2,slopeerr2,ordinate2,ordinateerr2,name[3],4,linear,'Απόσταση-Χρόνος','t(s)','Διαφορά δρόμου(m)')
#3
dx=[i-j for i,j in zip(x2,x1)]
wl=[i/5*10**(-2) for i in dx]
wlav,wlstd=avg_and_std(wl)
Us=wlav*39.87*10**3
Usstd=wlstd*39.87*10**3
print('μήκος κύματος λ=(%.4f+-%.4f)m'%(wlav,wlstd))
print('ταχύτητα ήχου στον αέρα U=(%.1f+-%.1f)m/s'%(Us,Usstd))
#4
plot_graph(f4,v4,fe,Ve1,name[5],6,'Συχνότητα-Πλάτος τάσης','f(kHz)','V(mV)')
fit_plot(f4,v4,name[5],6,func1,expression1)
slope3,ordinate3,uncertainty3,slopeerr3,ordinateerr3=lsq(u,fdf,len(u))
plot_graph(u,fdf,None,None,name[6],7,'Διαφορά συχνοτήτων-Ταχύτητα','Δf(Hz)','fa/u(Hz/(m/s))')
#fit_plot(u,fdf,name[6],7,linear2,'Δf/')




plt.figure(figsize=[6,6])
plt.scatter(logd,logV2)
plt.title('Πλάτος ταλάντωσης-Απόσταση') 
plt.xlabel('log(d) (cm)')
plt.ylabel('log(V) (mV)')
plt.grid(True)  
plt.savefig(name[8], dpi=300)

slope1,ordinate1,uncertainty1,slopeerr1,ordinateerr1=lsq(logd,logV2,len(logd))
print('slope1=%.2f+-%.2f, ordinate1=%.2f+-%.2f'%(slope1,slopeerr1,ordinate1,ordinateerr1))