import numpy as np 
import matplotlib.pyplot as plt 

def calculate_sums(x, y):
    sumx = sum(x)
    sumy = sum(y)
    sumx2 = sum([i**2 for i in x])
    sumxy = sum(i*j for i,j in zip(x,y))
    return sumx, sumy, sumx2, sumxy

def calculate_parameters(sumx, sumy, sumx2, sumxy, n):
    delta = n*sumx2 - sumx**2
    slope = (n*sumxy - sumx*sumy) / delta
    ordinate = (sumx2*sumy - sumx*sumxy) / delta
    return slope, ordinate, delta

def calculate_errors(x, y, slope, ordinate, delta, n):
    riza = sum((y[i] - ordinate - slope*x[i])**2 for i in range(n))
    uncertainty = np.sqrt(1/(n-2)*riza)
    slopeerr = np.sqrt(uncertainty**2*n/delta)
    ordinateerr = np.sqrt(uncertainty**2*sumx2/delta)
    return uncertainty, slopeerr, ordinateerr

def plot_graph(x, y, slope, ordinate,yerr):
    plt.grid(True)
    plt.errorbar(x, y, yerr=yerr, fmt='o',color='black')
    plt.scatter(x, y)
    plt.plot(x, [slope*x_i + ordinate for x_i in x])
    plt.xlim(left=0) # x axis starts from 0
    plt.savefig('analisi.png')

#xvals = [1,2,3,4]
#yvals = [(0.01181/2),(0.02433/2),(0.03553/2),(0.04683/2)]
xvals = [1,1/4,1/9,1/16]
yvals = [1/(0.01181/2)**2,1/(0.02433/2)**2,1/(0.03553/2)**2,1/(0.04683/2)**2]
n = len(xvals)

sumx, sumy, sumx2, sumxy = calculate_sums(xvals, yvals)
slope, ordinate, delta = calculate_parameters(sumx, sumy, sumx2, sumxy, n)
uncertainty, slopeerr, ordinateerr = calculate_errors(xvals, yvals, slope, ordinate, delta, n)

print('klisi =', slope, '+-', slopeerr)
print('tetagmeni =', ordinate, '+-', ordinateerr)
print('sigma =', uncertainty)

plot_graph(xvals, yvals, slope, ordinate,1/(0.01**2))