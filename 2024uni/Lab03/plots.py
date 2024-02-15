#------------------------------------------------
#------------import libraries--------------------
import numpy as np 
import matplotlib.pyplot as plt 
import numpy.random as rn
from scipy.optimize import curve_fit
#------------------------------------------------
#--------------dedomena--------------------------
#------------------------------------------------
name=['graph1','graph2','graph3a','graph3b','graph3c','graph3d','graph4a','graph4b']
R=220#antistasi
#--------------insert values here ---------------
#------------------------------------------------
#------------askisi 1----------------------------
gonies=[-90,-75,-60,-45,-30,-15,0,15,30,45,60,75,90] #moires
goniaerr=0.5
tasi=[0.000,0.007,0.019,0.034,0.049,0.059,0.061,0.056,0.042,0.026,0.011,0.002,0.000]#volts thelw omos I=V/R (mV)
tasierr=0.001
entasi=[i/R for i in tasi]
entasierr=1.4e-5
#------------------------------------------------
#------------askisi 2----------------------------
#xrisimopoio times apo askisi ena
gonies2=[-90,-75,-60,-45,-30,-15,0,15,30,45,60,75,90] #moires
goniaerr=0.5
goniaplak2=24
tasi2=[0.000,0.003,0.011,0.023,0.035,0.044,0.048,0.045,0.036,0.024,0.013,0.004,0.001] #volts thelw omos I=V/R
tasierr=0.001
entasi2=[i/R for i in tasi2]
entasierr2=1.4e-5
#------------------------------------------------
#------------askisi3 ----------------------------
gonies3=[-90,-75,-60,-45,-30,-15,0,15,30,45,60,75,90] #
#goniaplak=[-67,-37,-22,-7,8,24,-6,-21,-36,-51,-66]
goniaplakerr=-1
tasi30=[0.017,0.024,0.031,0.035,0.037,0.035,0.029,0.022,0.016,0.011,0.010,0.011,0.018] #volts thelw omos I=V/R
tasi45=[0.024,0.025,0.024,0.025,0.024,0.024,0.023,0.021,0.021,0.021,0.021,0.022,0.023]
tasi60=[0.017,0.014,0.013,0.014,0.019,0.024,0.029,0.032,0.032,0.030,0.027,0.021,0.017]
tasi90=[0.001,0.004,0.011,0.022,0.033,0.041,0.045,0.042,0.034,0.022,0.012,0.005,0.001]
tasierr=0.001
entasi30=[i/R for i in tasi30]
entasi45=[i/R for i in tasi45]
entasi60=[i/R for i in tasi60]
entasi90=[i/R for i in tasi90]
entasierr3=1.4e-5
#------------------------------------------------
#---------------askisi 4------------------------
gonies4=[-90,-75,-60,-45,-30,-15,0,15,30,45,60,75,90] #moires
goniaerr=0
tasi00=[0.000,0.002,0.008,0.016,0.026,0.033,0.037,0.035,0.028,0.019,0.010,0.003,0.001] #volts thelw omos I=V/R
tasi045=[0.020,0.021,0.021,0.020,0.018,0.017,0.015,0.014,0.014,0.014,0.016,0.018,0.020]
tasierr=0.001
entasi00=[i/R for i in tasi00]
entasi045=[i/R for i in tasi045]
entasierr4=1.4e-5
#-----------------------------------------------
#------------least squares----------------------
#-----------------------------------------------
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
def plot_graph(x, y, slope, ordinate, yerr,plotname,fig):
    def func (x,a,b,c):
       return a*np.cos(np.radians(x)+b)**2 +c # fit , change if something else
    n=len(x)
    params,params_covariance=curve_fit(func,x,y,p0=[0,0,0])
    print("parametri:",params) 
    #coeffs = np.polyfit(x, y, 3)
    #print('parameters:',coeffs)
    #fitted_poly=np.poly1d(coeffs)
    #yfit=fitted_poly(x)
    plt.figure(fig)
    plt.figure(figsize=(9,7))  
    plt.plot(x,func(x,*params),label='curve fit')
    plt.gca().set_facecolor('0.88')
    plt.grid(True)
    plt.errorbar(x, y, yerr=yerr, label='σφάλμα',fmt='o', color='black',elinewidth=0.7)
    colors = np.random.rand(n)
    plt.scatter(x, y,label='data')
    #plt.plot(x, yfit, label='fit',color='red')
    #plt.plot(x, [slope*x_i + ordinate for x_i in x])
    #plt.xlim(x[0],x[n-1])
    #plt.ylim([0],y[-1])
    #plt.ylim(ymin,ymax*1.2)
    #change for each plot
    plt.title('Διάγραμμα έντασης-γωνίας')
    plt.xlabel('Γωνία (μοίρες)')
    plt.ylabel('Ἐνταση (Ι)')
    equation = f"y = {slope:.2f}x  {ordinate:.2f}"  #Legend equation

    plt.legend(loc='best')
    plt.savefig(plotname, dpi=300)

#------------------------------------------------
#------------function call-----------------------
# Ensure gonies, entasi, and R are defined and are lists of numbers
klisi1, tetagmeni1, sigmay,klisierr1,teterr1 = lsq(gonies,entasi, len(gonies))
klisi2, tetagmeni2, sigmay,klisierr2,teterr2 = lsq(gonies2,entasi2, len(gonies2))
klisi3a, tetagmeni3a, sigmaya,klisierr3a,teterr3a = lsq(gonies3,entasi30, len(gonies3))
klisi3b, tetagmeni3b, sigmayb,klisierr3b,teterr3b = lsq(gonies3,entasi45, len(gonies3))
klisi3c, tetagmeni3c, sigmayc,klisierr3c,teterr3c = lsq(gonies3,entasi60, len(gonies3))
klisi3d, tetagmeni3d, sigmayd,klisierr3d,teterr3d = lsq(gonies3,entasi90, len(gonies3))
klisi4, tetagmeni4, sigmay,klisierr4,teterr4 = lsq(gonies4,entasi00, len(gonies4))
klisi4a, tetagmeni4a, sigmaya,klisierr4a,teterr4a = lsq(gonies4,entasi045, len(gonies4))
#------------------------------------------------
#------------plot graph--------------------------
#------------------------------------------------
askisi1=plot_graph(gonies,entasi,klisi1,tetagmeni1,entasierr,name[0],1)
askisi2=plot_graph(gonies2,entasi2,klisi2,tetagmeni2,entasierr2,name[1],2)
askisi3a=plot_graph(gonies3,entasi30,klisi3a,tetagmeni3a,entasierr3,name[2],3)
askisi3b=plot_graph(gonies3,entasi45,klisi3b,tetagmeni3b,entasierr3,name[3],4)  
askisi3c=plot_graph(gonies3,entasi60,klisi3c,tetagmeni3c,entasierr3,name[4],5) 
askisi3d=plot_graph(gonies3,entasi90,klisi3d,tetagmeni3d,entasierr3,name[5],6)
askisi4=plot_graph(gonies4,entasi00,klisi4,tetagmeni4,entasierr4,name[6],7)
askisi4a=plot_graph(gonies4,entasi045,klisi4a,tetagmeni4a,entasierr4,name[7],8)