import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import random as rn

def read_spectrum(file_name):
    '''Reads the txt file and return a list'''
    mes = []
    with Path(__file__).with_name(file_name).open() as f:
        for line in f:
            part = line.strip()
            mes.append(float(part))
    return mes

def calc_T(B):
    '''In keV'''
    return float(np.sqrt((16 * B)**2 + 511**2) - 511)

def calc_theoretical_b(B,part='',baros=''):
    E = calc_T(B) + 511
    b = np.sqrt(1 - (511/E)**2)
    if part == 'minus':
        Tmax = 2274
        C = 1.75 * 10**(-12)
        n = (40 * 1.44)/(197.3 * b)
    if part == 'plus':
        if baros == 'high':
            Tmax = 1820
        if baros == 'low':
            Tmax = 546
        C = (5.9/2.75) * 10**(-10)
        #C = 1
        n = - (10 * 1.44)/(197.3 * b)
    p = np.sqrt(E**2 - 511**2)
    F = (2 * np.pi * n)/(1 - np.exp(-2 * np.pi * n))
    if calc_T(B) > Tmax:
        return float(0)
    else:
        N = C * F * (Tmax - calc_T(B))**2 * p * E
        return float(N)

names = ['Ampere.txt','Tesla_21.txt','count rate b minus.txt','count rate b plus.txt']
namespng = ['B os pros I.png','tesla os pros Tkin.png','count rate b minus.png','count rate b plus.png']

x = np.array(read_spectrum(names[1]))
y = np.array(read_spectrum(names[0]))

M = np.column_stack((x, np.ones(len(x))))

coefficients, residuals, rank, singular_values = np.linalg.lstsq(M, y, rcond=None)

slope, intercept = coefficients

if len(y) <= 2 or rank < 2:
    raise ValueError('Need at least 3 data points and a full-rank fit.')

y_fit = M @ coefficients
sum_squared_residuals = np.sum((y - y_fit) ** 2)
residual_variance = sum_squared_residuals / (len(y) - 2)
covariance = residual_variance * np.linalg.inv(M.T @ M)
slope_error, intercept_error = np.sqrt(np.diag(covariance))

print(f'slope: {slope:.4f} ± {slope_error:.4f}, intercept: {intercept:.4f} ± {intercept_error:.4f}')
print(f'residuals: {residuals}')


plt.figure(figsize=[8,6])
equation = f'y = {slope:.4f}x {"+" if intercept >= 0 else "-"} {abs(intercept):.4f}'
plt.scatter(read_spectrum(names[1]),read_spectrum(names[0]),color='red',edgecolor='black',s=27,label=equation)
plt.xlabel('Magnetic Field (mT)')
plt.ylabel('Current (A)')
plt.grid(True,alpha=0.4)
plt.legend()
plt.savefig(Path(__file__).with_name(namespng[0]),dpi=400)
plt.close()

lista_tesla = read_spectrum(names[1])
Tkin = []
for i in range(len(lista_tesla)):
    Tkin.append(calc_T(lista_tesla[i]))
plt.figure(figsize=[8,6])
plt.scatter(Tkin,lista_tesla,color='red',edgecolor='black',s=27)
plt.xlabel('Kinetic Energy (keV)')
plt.ylabel('Magnetic Field (mT)')
plt.grid(True,alpha=0.4)
plt.savefig(Path(__file__).with_name(namespng[1]),dpi=400)

R_bminus = read_spectrum(names[2])
plt.figure(figsize=[8,6])
plt.scatter(Tkin,R_bminus,color='red',edgecolor='black',s=27)
plt.xlabel('Kinetic Energy (keV)')
plt.ylabel('Count Rate (counts/sec)')
plt.grid(True,alpha=0.4)
plt.savefig(Path(__file__).with_name(namespng[2]),dpi=400)

R_bplus = read_spectrum(names[3])
plt.figure(figsize=[8,6])
plt.scatter(Tkin,R_bplus,color='red',edgecolor='black',s=27)
plt.xlabel('Kinetic Energy (keV)')
plt.ylabel('Count Rate (counts/sec)')
plt.grid(True,alpha=0.4)
plt.savefig(Path(__file__).with_name(namespng[3]),dpi=400)

plt.figure(figsize=[8,6])
plt.scatter(Tkin,R_bminus,color='red',edgecolor='black',s=27)
plt.scatter(Tkin,R_bplus,color='green',edgecolor='black',s=27)
plt.xlabel('Kinetic Energy (keV)')
plt.ylabel('Count Rate (counts/sec)')
plt.grid(True,alpha=0.4)
plt.savefig(Path(__file__).with_name('b minus and b plus.png'),dpi=400)

theob = [float(x) for x in np.linspace(7,175,100)]
theo_b_minus = [calc_theoretical_b(theob[i], 'minus') for i in range(len(theob))]
theob = [calc_T(i) for i in theob]
plt.figure(figsize=[8,6])
plt.plot(theob,theo_b_minus,color='red')
plt.xlabel('Kinetic Energy (keV)')
plt.ylabel('Count Rate (counts/sec)')
plt.grid(True,alpha=0.4)
plt.savefig(Path(__file__).with_name('theoretical b minus.png'),dpi=400)

theob = [float(x) for x in np.linspace(7,175,100)]
theo_b_plus_high = [calc_theoretical_b(theob[i], 'plus','high') for i in range(len(theob))]
theo_b_plus_low = [calc_theoretical_b(theob[i], 'plus','low') for i in range(len(theob))]
theo_b_plus = [theo_b_plus_high[i]*0.0005 + theo_b_plus_low[i]*0.9049 for i in range(len(theob))]
theob = [calc_T(i) for i in theob]
plt.figure(figsize=[8,6])
plt.plot(theob,theo_b_plus,color='red')
plt.xlabel('Kinetic Energy (keV)')
plt.ylabel('Count Rate (counts/sec)')
plt.grid(True,alpha=0.4)
plt.savefig(Path(__file__).with_name('theoretical b plus.png'),dpi=400)

plt.figure(figsize=[8,6])
plt.scatter(Tkin,R_bminus,color='red',edgecolor='black',label='Measurements β-',s=27)
plt.plot(theob,theo_b_minus,color='black',label='theoretical β-')
plt.xlabel('Kinetic Energy (keV)')
plt.ylabel('Count Rate (counts/sec)')
plt.grid(True,alpha=0.4)
plt.legend()
plt.savefig(Path(__file__).with_name('Lab 8 - B Minus.png'),dpi=400)

plt.figure(figsize=[8,6])
plt.scatter(Tkin,R_bplus,color='red',edgecolor='black',label='Measurements β+',s=27)
plt.plot(theob,theo_b_plus,color='black',label='theoretical β+')
plt.xlabel('Kinetic Energy (keV)')
plt.ylabel('Count Rate (counts/sec)')
plt.grid(True,alpha=0.4)
plt.legend()
plt.savefig(Path(__file__).with_name('Lab 8 - B Plus.png'),dpi=400)

plt.figure(figsize=[8,6])
plt.scatter(Tkin,R_bminus,color='red',edgecolor='black',label='Measurements β-',s=27)
plt.scatter(Tkin,R_bplus,color='green',edgecolor='black',label='Measurements β+',s=27)
plt.plot(theob,theo_b_minus,color='tomato',label='theoretical β-')
plt.plot(theob,theo_b_plus,color='forestgreen',label='theoretical β+')
plt.xlabel('Kinetic Energy (keV)')
plt.ylabel('Count Rate (counts/sec)')
plt.grid(True,alpha=0.4)
plt.legend()
plt.savefig(Path(__file__).with_name('Lab 8 - B Minus and B Plus.png'),dpi=400)




print('\nfin\n')