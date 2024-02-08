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

def plot_graph(x, y, slope, ordinate, yerr,save_path):
    plt.figure(figsize=(10,8))
    plt.gca().set_facecolor('0.93')
    plt.grid(True)
    plt.errorbar(x, y, yerr=yerr, fmt='o', color='black',elinewidth=0.7,label='error=+-%f'%yerr)
    plt.scatter(x, y,label='data')
    plt.plot(x, [slope*x_i + ordinate for x_i in x])
    plt.xlim(left=0)
    plt.fill_between(xvals,[(slope+slopeerr)*x + (ordinate+ordinateerr) for x in xvals],[(slope-slopeerr)*x + (ordinate-ordinateerr) for x in xvals],label='slope & ordinate error')
    equation = f"y = {slope:.2f}x  {ordinate:.2f}"
    plt.text(0.89, 0.15, equation, transform=plt.gca().transAxes, fontsize=13, verticalalignment='bottom',horizontalalignment='right')

    plt.savefig(f'{save_path}',dpi=300)

xvals = [1,2,3]
yvals = [1,2,3]
n = len(xvals)

sumx, sumy, sumx2, sumxy = calculate_sums(xvals, yvals)
slope, ordinate, delta = calculate_parameters(sumx, sumy, sumx2, sumxy, n)
uncertainty, slopeerr, ordinateerr = calculate_errors(xvals, yvals, slope, ordinate, delta, n)


print('klisi =', slope, '+-', slopeerr)
print('tetagmeni =', ordinate, '+-', ordinateerr)
print('sigma =', uncertainty)

plot_graph(xvals, yvals, slope, ordinate, uncertainty,'/2024uni/Lab02/grafiki1.png')