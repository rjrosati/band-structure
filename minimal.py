#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
from scipy import linalg as LA
import matplotlib.pyplot as plt
import matplotlib

# unit conversions
eV_to_Ryd = 1/13.6
c = 3.80998 # hbar**2 / (2*m) in  Angstroms^2 * eV

#pseudopotential parameters for Aluminum
d = 0.350 # angstroms 
rc = 0.943 # angstroms
U0 = -31.30 # eV 

# U(K) modified from Marder's (10.38) to give its limit at 0
def U(K):
    Kn = LA.norm(K)
    if Kn > 0:
        return U0 * np.exp(-rc/d) * ((np.sin(rc*Kn) / (d*Kn * ((d*Kn)**2 + 1)) + (np.cos(Kn*rc))/((d*Kn)**2 + 1)))
    else:
        return U0*np.exp(-rc/d)*(rc/d+1)

# Al is an fcc lattice with a spacing of a = 4.05 angstroms
a = 4.05 # angstrom 
(a1,a2,a3) = 0.5 * a * np.array([[1,1,0],[1,0,1],[0,1,1]])

# the lattice's reciprocal vectors are given by Marder's equation 3.24
b1 = 2 * np.pi * np.cross(a2,a3) / a1.dot(np.cross(a2,a3))
b2 = 2 * np.pi * np.cross(a3,a1) / a2.dot(np.cross(a3,a1))
b3 = 2 * np.pi * np.cross(a1,a2) / a3.dot(np.cross(a1,a2))

# index shuffle to ensure we have negative K 
def l(ji,N):
    return ji-N if ji+1 > (N+1)/2 else ji

def generate_lattice(N):
    K = np.empty([N,N,N], dtype=np.ndarray) 
    r = np.empty(K.shape, dtype=np.ndarray) 
    for i,j,k in np.ndindex((N,N,N)):
            K[i,j,k] = l(i,N)*b1 + l(j,N)*b2 + l(k,N)*b3
            r[i,j,k] = (1/N)*(i*a1 + j*a2 + k*a3) 
    return K,r
def generate_potential(K):
    vector_U = np.vectorize(U)
    return vector_U(K)
def get_ek(K,k):
    q=K+k
    return c * q.dot(q)
vector_get_ek = np.vectorize(get_ek,excluded=[1]) # only vectorize zeroth argument
def get_H_block(k,K,U_K):
    # following Marder's Table 7.1
    # eigenvalues of a block-diagonal matrix are simply the eigenvalues of each block
    # can therefore compute only the block for this k
    eks = vector_get_ek(K,k)
    H = np.diagflat(eks)
    m,n = np.indices(H.shape)
    H += U_K.flat[np.abs(m-n)]
    return H
# take advantage of only the diagonal changing under a change of k
def efficient_new_H(H,k,K,U_K):
    # I measure a speedup of order N over get_H_block
    np.fill_diagonal(H,vector_get_ek(K,k).flat + U_K[0,0,0])
    return H
bands=6
# high symmetry points for the reciprocal lattice
L     = 2*np.pi/a*np.array([1/2,1/2,1/2]) 
gamma = 2*np.pi/a*np.array([  0,  0,  0]) 
X     = 2*np.pi/a*np.array([  0,  1,  0]) 
pointU= 2*np.pi/a*np.array([1/4,  1,1/4]) 

path = [L, gamma, X, pointU, gamma]
labels = ['L', r'$\Gamma$', 'X', 'U', r'$\Gamma$']
ksteps = 50
N = 12
energies = np.zeros([ksteps*(len(path)-1),bands])
kscale = np.zeros(ksteps*(len(path)-1))
K,r = generate_lattice(N)
U_K = generate_potential(K)

label_locations = []
H = get_H_block(np.array([0.,0.,0.]),K,U_K)

for i in range(0,len(path)-1):
    label_locations.append(i*ksteps)
    kstep = (path[i+1]-path[i]) / ksteps
    for step in range(ksteps):
        qval = path[i]+step*kstep
        H = efficient_new_H(H,qval,K,U_K)
        energies[step+i*ksteps,:] = LA.eigvalsh(H,eigvals=(0,bands-1))
        kscale[step+i*ksteps] = LA.norm(kstep)

label_locations.append(step+i*ksteps)
kvals = np.cumsum(kscale) # do this to scale the x-axis linearly with k-distance covered

font = {'family' : 'serif', 'weight' : 'bold', 'size' : 18}
matplotlib.rc('text', usetex=True)
matplotlib.rcParams['text.latex.preamble']=[r"\boldmath"]
matplotlib.rc('font', **font)
matplotlib.ticker.ScalarFormatter(useMathText=True)

plt.ylabel('band energy (eV)')
plt.xticks([kvals[x] for x in label_locations], labels)
plt.axes().xaxis.grid(True,which='major')
plt.axes().tick_params(pad=18)
for n in range(bands):
    plt.plot(kvals,energies[:,n])
plt.savefig('al_bands.pdf',bbox_inches='tight')

