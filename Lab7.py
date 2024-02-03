
import numpy as np 
import matplotlib.pyplot as plt 
# values gia askisi 1
#ekamamata sosta me tin apostasi pou elalousame (apostasi anixneuti-apostasi sxismis)
xvals = [0.79,0.765,0.74,0.715,0.718,0.665,0.64,0.615,0.59]
yvals = [576,582,527,544,515,504,460,465,421]
yvals = [i*10**(-3)/2 for i in yvals]
n=len(xvals)
#values gia askisi 2
#kai dame ekamata me tin apostasi opos pio pano
xvals2 = [0.79,0.77,0.75,0.73,0.71,0.69,0.67,0.65]
yvals2 = [0.0118,0.01136,0.01136,0.01086,0.01039,0.01058,0.00997,0.00975]
n2=len(xvals2)

#elaxista tetragona
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
        riza+=(y[i]-ordinate-slope*x[i])**2  # Corrected here
    uncertainty=np.sqrt(1/(n-2)*riza)
    slopeerr=np.sqrt(uncertainty**2*n/delta)
    ordinateerr=np.sqrt(uncertainty**2*sumx2/delta)
    return (slope,ordinate,uncertainty,slopeerr,ordinateerr)

klisi, tetagmeni, sigma, sigmaA, sigmaB = lsq(yvals, xvals, n)# call gia askisi 1
klisi2, tetagmeni2, sigma2, sigmaA2, sigmaB2 = lsq(xvals2, yvals2, n2) # call gia askisi 3

#solve askisi 1 kai print apotelesmata
wl=650*10**(-9) # mikos kymatos
mikosd=np.sqrt(klisi**2-1)*1*650*10**(-9) #mikos sxismis
derror=(klisi*1*wl)/(np.sqrt(klisi**2+1))*sigmaA 
apoklisid=abs(mikosd-0.16*10**(-3))/sigma*100#ekatostiaia apoklisi sto mikos sxismis

print('\n')
print("askisi 1")
print('\n')
print('klisi =', klisi, '+-', sigmaA)
print('tetagmeni =', tetagmeni, '+-', sigmaB)
print('mikos sxismis, d=',mikosd,'+-',derror,'m')
print('ekatostiaia apoklisi mikous sxismis =%f%%'%apoklisid)
print('\n')
y1=[((klisi+sigmaA)*y + (tetagmeni+sigmaB)) for y in yvals]
y2=[((klisi-sigmaA)*y + (tetagmeni-sigmaB)) for y in yvals]
#solve askisi 3 kai print apotelesmata
apostasif=wl*(2-0.5)/klisi2 # apostasi f
apostasierr=-(wl*(2-0.5))/(klisi2**2)*sigmaA2
apoklisiapos=abs(apostasif-0.004*10**(-3))/sigma2*100 #ekatostiaia apoklisi stin apostasi f
print('askisi 3 ')
print('klisi =', klisi2, '+-', sigmaA2)
print('tetagmeni =', tetagmeni2, '+-', sigmaB2)
print('apostasi metaksi 2 sxismon , f=',apostasif,'+-',apostasierr,'m')
print('ekatostiaia apoklisi apostasis f = %f%%'%apoklisiapos)
print('\n')

#plotting askisi 1
plt.figure(figsize=(8, 6))
plt.grid(True)

plt.scatter(yvals,xvals,label='data points')
plt.ylim(0.4,1.1)
plt.plot(yvals, [klisi*x + tetagmeni for x in yvals],label='y = %f*x +%f'%(klisi,tetagmeni))  # Corrected here
plt.errorbar(yvals, xvals, yerr=sigma, fmt='o', color='black',elinewidth=0.7,label='error=+-%f'%sigma)
plt.title('x=f(y)')
plt.ylabel('x(m)')
plt.xlabel('y(m)')
plt.fill_between(yvals,y1,y2,label='sfalma klisis & tetagmenis')
plt.legend(loc='best')
plt.savefig('analisi1.png')
#plotting askisi 2
plt.figure(2)
plt.grid(True)
plt.scatter(xvals2,yvals2,label='data points')
plt.plot(xvals2, [klisi2*x + tetagmeni2 for x in xvals2],label='y = %f*x + %f'%(klisi2,tetagmeni2))
plt.errorbar(xvals2, yvals2, yerr=sigma2, fmt='o', color='black',elinewidth=0.7,label='error=+-%f'%sigma2)
plt.title('y=f(x)')
plt.ylabel('y(m)')
plt.xlabel('x(m)')
plt.fill_between(xvals2,[(klisi2+sigmaA2)*x + (tetagmeni2+sigmaB2) for x in xvals2],[(klisi2-sigmaA2)*x + (tetagmeni2-sigmaB2) for x in xvals2],label='sfalma klisis & tetagmenis')
plt.legend(loc='best')
plt.savefig('analisi3.png')
