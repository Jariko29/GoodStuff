#------------------------------------------------
#------------import libraries--------------------
import numpy as np 
import matplotlib.pyplot as plt 
from scipy.optimize import curve_fit
#------------------------------------------------
#--------------insert values here ---------------
name=['graph1','graph2','graph3','graph4']
#askisi 1
xvals = [14,14.25,15,17,19.25,20.25,20.5,22]#proti metrisi
xvals1=[14,14.50,15.25,17.5,19.5,20.25,20.5,22.5]#deuteri metrisi
yvals = []
n = len(xvals)
#askisi 2
thetai=58#51.5# aproximately 52 degrees
alpha=(thetai*np.pi/180)
mikosl=[404.6,407.8,438.8,491.6,546.1,577,579.1,690.7]#miki kimmatos mov-mov-mple-prasino-prasino-kitrino-kitrino-kokkino
mikosl=np.array(mikosl)
dm1=[38,38.75,39.75,41.5,41,41.75,42,42.5]
dm=[90-i for i in dm1] #kokkino-kitrino-kitrino-prasino-prasino-mple-mov-mov
dm=np.array(dm) 
print(dm)
dmerr=0.25*np.pi/180 #error in radians
N=[(np.sin((alpha+(i*np.pi/180))/2)/np.sin(alpha/2)) for i in dm]
Nerr=[(np.sin((alpha+(i*np.pi/180))/2)/np.sin(alpha/2))*np.sqrt((dmerr/np.tan(alpha/2))**2+(dmerr/np.tan((alpha+(i*np.pi/180))/2))**2) for i in dm]
print(N)
print(Nerr) 
mik=1/mikosl**2
#colors=['purple','purple','blue','cyan','green','yellow','red','red']
colors=['purple','blue','green','yellow','red']

kima=['λ=404.6','λ=407.8','λ=435.8','λ=491.6','λ=546.1','λ=577','λ=579.1','λ=690.7']
kima2=['1/(404.6)^2','1/(407.8)^2','1/(435.8)^2','1/(491.6)^2','1/(546.1)','1/(577)','1/(579.1)^2','1/(690.7)^2']
#------------------------------------------------
#vale dame da miki kimatos gia to kitrino kai to mple
Ra=((mikosl[6]+mikosl[2])/2)/(mikosl[6]-mikosl[2])
Raa=((mikosl[5]+mikosl[2])/2)/(mikosl[5]-mikosl[2])
print('διακριτική ικανότητα πρίσματος μεταξύ κιτρινο λ=579.1nm και μπλέ λ=435.8:',Ra)
print('διακριτική ικανότητα πρίσματος μεταξύ κιτρινο λ=577nm και μπλέ λ=435.8:',Raa)
#------------------------------------------------

#------------------------------------------------
#------------calculate sums----------------------
def calculate_sums(x, y): 
    sumx = sum(x)
    sumy = sum(y)
    sumx2 = sum([i**2 for i in x])
    sumxy = sum(i*j for i,j in zip(x,y))
    return sumx, sumy, sumx2, sumxy
#------------------------------------------------
#--------mean value and STDeviation--------------
def mean_and_std(x1,x2):
    mean = [i+j for i,j in (zip(x1,x2))]
    mean=[i/2 for i in mean]
    meansqrt=[((i+j)/2)**2 for i,j in (zip(x1,x2))]
    std=[np.sqrt(i-j**2) for i,j in (zip(meansqrt,mean))]
    return mean,std
#------------------------------------------------
#------------calculate parameters----------------
def calculate_parameters(sumx, sumy, sumx2, sumxy, n): 
    delta = n*sumx2 - sumx**2
    slope = (n*sumxy - sumx*sumy) / delta
    ordinate = (sumx2*sumy - sumx*sumxy) / delta
    return slope, ordinate, delta
#------------------------------------------------
#------------calculate errors--------------------
def calculate_errors(x, y, slope, ordinate, delta, n): 
    riza = sum((y[i] - ordinate - slope*x[i])**2 for i in range(n))
    uncertainty = np.sqrt(1/(n-2)*riza)
    slopeerr = np.sqrt(uncertainty**2*n/delta)
    ordinateerr = np.sqrt(uncertainty**2*sumx2/delta)
    return uncertainty, slopeerr, ordinateerr
#------------------------------------------------
#------------plot graph---------------------------
def plot_graph(x, y, slope, ordinate, yerr,plotname):
    plt.figure(figsize=(10,8))
    plt.gca().set_facecolor('0.88')
    plt.grid(True)
    plt.errorbar(x, y, yerr=yerr, fmt='o', color='black',elinewidth=0.7)
    plt.scatter(x, y)
    plt.plot(x, [slope*x_i + ordinate for x_i in x])
    plt.xlim(left=0)
    ymin, ymax = plt.ylim()
    plt.ylim(ymin,ymax*1.2)
    plt.text(0.4, 1.05, r'$\frac{1}{{y_m}^2} = f(\frac{1}{{m}^2})$', fontsize=20, transform=plt.gca().transAxes)
    equation = f"y = {slope:.2f}x  {ordinate:.2f}"  #Legend equation
    plt.text(0.89, 0.15, equation, transform=plt.gca().transAxes, fontsize=13, verticalalignment='bottom',horizontalalignment='right')

    plt.xlabel(r'$\frac{1}{{m}^2}$',fontsize=20)
    plt.text(-0.13, 0.48, r'$\frac{1}{{y_m}^2}$', fontsize=20, rotation=90, va='center', transform=plt.gca().transAxes)
    plt.text(-0.12, 0.55, r'$(\frac{1}{m})$', fontsize=12, rotation=90, va='center', transform=plt.gca().transAxes)

    plt.savefig(plotname, dpi=300)

#------------------------------------------------
#------------fit attempt----------------------
def func(mikosl, a, b):#,c):#curve fit 1
    return a*np.log(mikosl)+b#a*mikosl**2+b*mikosl+c
params, params_covariance = curve_fit(func, mikosl, N)
print("Fitted parameters:", params)
mikoslpikno=np.linspace(min(mikosl-50),max(mikosl+25),500)
#curve fitb
coeffs=np.polyfit(mik,N,2)
fitted=np.poly1d(coeffs)
mik2=np.linspace(min(mik-1e-6),max(mik+.5e-6),500)


#------------------------------------------------
#------------print results------------------------
#------------------------------------------------
#------------plot graph--------------------------
plt.figure(1)
plt.grid(True)
plt.title('Δείκτης διάθλασης(n) συνάρτηση του μήκους κύματος(λ)')
plt.ylabel('δείκτης διάθλασης(n)')
plt.xlabel('μήκος κύματος (nm)')
plt.errorbar(mikosl,N,yerr=Nerr,fmt='o',color='black',elinewidth=0.7,label='error')
plt.scatter(mikosl, N, label='data')
plt.plot(mikoslpikno, func(mikoslpikno, params[0], params[1]), label='fit',color='black')
plt.ylim(1.62,1.7)
colorindex=0
count=0

for i in range(len(mikoslpikno)):
    if count>=100:
        count=0
        colorindex+=1
    color=colors[colorindex]   
    count+=1  
    plt.bar(mikoslpikno[i],func(mikoslpikno[i], params[0], params[1]),width=0.1,color=colors[colorindex])
for i in range(len(mikosl)):
    plt.text(mikosl[i],N[i],kima[i],fontsize=10)
    

plt.legend(loc='best')
plt.savefig('askisi2a.png', dpi=300) 
#plot 2
plt.figure(2)
plt.grid(True)
plt.title('Δείκτης διάθλασης(n) συνάρτηση του 1/λ^2')
plt.ylabel('δείκτης διάθλασης(n)')
plt.xlabel('1/λ^2(1/nm^2)')
plt.scatter(mik,N,label='data')
plt.errorbar(mik,N,yerr=Nerr,fmt='o',color='black',elinewidth=0.7,label='error')
plt.plot(mik2,fitted(mik2), label='fit',color='black')
for i in range(len(mik)):
    plt.text(mik[i],N[i],kima2[i],fontsize=10)
plt.legend(loc='best')
plt.savefig('askisi2b.png', dpi=300)
plt.show()