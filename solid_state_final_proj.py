#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
from scipy import linalg as LA
import itertools
import matplotlib.pyplot as plt
import matplotlib
import sys
import pdb

#problem 10.4, Marder p. 287

c = np.array([[ 0.25819889+0.51639777j,
                0.77459663+0.25819889j,
                0.00000000+0.j        ,
                0.00000000+0.j        ],
                
              [-0.39235696-0.24968168j,
                 0.32101944-0.07133763j,
                0.10700645+0.74904513j,
                0.32101935+0.j        ],
                
              [ 0.30385175+0.13841607j,
               -0.22113398+0.08271784j,
                0.09764974-0.08722103j,
                0.88951313+0.15619215j],
                
              [ 0.00000000+0.j        ,
                2.00000000+0.j        ,
                1.00000000+0.j        ,
                0.00000000+0.j        ]])
                
def GramSchmidt(c,i):
    # assume columns 0 through i-1 are already mutually orthonormal
    # generate a new orthonormal vector in column i
    # equivalent to generating a basis vector by Gram-Schmidt process, (i.e. http://ltcconline.net/greenl/courses/203/Vectors/orthonormalBases.htm )
    v = c[i]
    for j in range(i):
        v -= (np.vdot(c[i],c[j].transpose()) / np.vdot(c[j],c[j].transpose())) * c[j]
        
    c[i] = v/LA.norm(v)

    #verify that we're orthonormal to within 1e-6 (should avoid any floating point errors)
    assert(np.allclose(0, [abs(np.vdot(c[j],c[k].transpose())) for j,k in itertools.combinations(range(i),2)], atol=1e-6))
    assert(np.allclose(1, [abs(np.vdot(c[j],c[j].transpose())) for j in range(i)], atol=1e-6))
    

GramSchmidt(c,i=3) # indices begin at 0, equivalent to calling with i=4 of problem statement

#print(c)
#print(c[3])

#problem 10.5

#Pseudopotential parameters for Aluminum

d = 0.350 # angstrom 
rc = 0.943 # angstrom
U0 = -31.30 # eV 

def U(K):
    Kn = LA.norm(K)
    return U0 * np.exp(-rc/d) * ((np.sin(rc*Kn) / (d*Kn * ((d*Kn)**2 + 1)) + (np.cos(Kn*rc))/((d*Kn)**2 + 1)))


# a)
#print(U(0))
#this gives an error
#however, the limit does exist:
# U0 * math.exp(-rc/d) * ((math.sin(rc*Kn) / (d*Kn * ((d*Kn)**2 + 1)) + (math.cos(Kn*rc))/((d*Kn)**2 + 1)))
# the cosine term goes to 1, while the sin term goes to rc*Kn / (d*Kn * ((d*Kn)**2 + 1)) == rc / d
# lim_{k -> 0} U(K) == U0 * math.exp(-rc/d)*(rc/d + 1)

#modify U(K) to show this behavior

def U(K):
    Kn = LA.norm(K)
    if Kn > 0:
        return U0 * np.exp(-rc/d) * ((np.sin(rc*Kn) / (d*Kn * ((d*Kn)**2 + 1)) + (np.cos(Kn*rc))/((d*Kn)**2 + 1)))
    else:
        assert(Kn==0)
        return U0*np.exp(-rc/d)*(rc/d+1)


# b)

# Al is an fcc lattice with a spacing of a = 4.05 angstroms
a = 4.05 # angstrom 
(a1,a2,a3) = 0.5 * a * np.array([[1,1,0],[1,0,1],[0,1,1]])

# the lattice's reciprocal vectors are given by equation 3.24
b1 = 2 * np.pi * np.cross(a2,a3) / np.dot(a1,np.cross(a2,a3))
b2 = 2 * np.pi * np.cross(a3,a1) / np.dot(a2,np.cross(a3,a1))
b3 = 2 * np.pi * np.cross(a1,a2) / np.dot(a3,np.cross(a1,a2))

# c)

# such a routine is available in my environment,
# numpy.fft.fftn(array) will compute an n-dimensional fast Fourier transform

# d)
# given N (of part e, below), for what values of r will an fft routine compute U(r) = sum over K ( e^(iK.r) U_K ), and how will they be indexed?

# FFT will generate an NxNxN array of values, from an equal sized array of U(K) ranging over the reciprocal lattice vectors
# given r values will be the corresponding physical lattice vector sums, so that exp(i*K.r) == 1

# e)

def l(ji,N):
    return ji-N if ji+1 > (N+1)/2 else ji
#    return np.ceil(ji-np.ceil(N/2))

def generate_lattice(N):
    K = np.empty([N,N,N], dtype=np.ndarray) 
    r = np.empty(K.shape, dtype=np.ndarray) 
    for i,j,k in np.ndindex((N,N,N)):
            K[i,j,k] = l(i,N)*b1 + l(j,N)*b2 + l(k,N)*b3
            r[i,j,k] = (1/N)*(i*a1 + j*a2 + k*a3) 
    return K,r

def generate_potential(N,K,n):
    vector_U = np.vectorize(U)
    #K_offset = K[list(np.ndindex((N,N,N)))[n]]
    #offsets = np.zeros([N,N,N],dtype=np.ndarray); offsets.fill(K_offset)
    return vector_U(K)

eV_to_Ryd = 1/13.6
'''
#pdb.set_trace()
K, r = generate_lattice(N=5)
U_K = generate_potential(5,K,0)
U_r = np.fft.fftn(U_K)
for i,j,k in np.ndindex((5,5,5)):
    print(r[i,j,k],U_r[i,j,k]*eV_to_Ryd) # convert to angstroms, Rydbergs
'''
# problem 10.6

# a)
c = 3.80998 # hbar**2 / (2*m) in  Angstroms^2 * eV

def convolve(a,b,N):
    return (1/N)**3 * np.fft.fftn(np.multiply(np.fft.ifftn(a),np.fft.ifftn(b)))

def compute_psi(psi,K_sq,Emax,U_K,N):
    return np.multiply(c*K_sq-Emax,psi) + convolve(psi,U_K,N) 

# b)

vectorized_norm = np.vectorize(LA.norm)

def get_closest_index(arr,val,N):
    minm = np.inf
    ind = None
    for i,j,k in np.ndindex((N,N,N)):
        dist = LA.norm(arr[i,j,k] - val)
        if dist < minm:
            minm = dist
            ind = (i,j,k)
    return ind

#find energy, use (7.33)
def energy(psi,q,U_K,K,N):
    qeff = get_closest_index(K,q,N)
    assert(qeff is not None)
    return np.real((c*np.dot(q,q)*psi[qeff] + convolve(psi,U_K,N)[qeff]) / psi[qeff])


def get_converged_psi(n,K,U_K,N): # n selects band, 0-indexed

    psi = np.zeros([N,N,N], dtype='complex128')
    psi[0,0,0] = 1
    #K_offset = K[list(np.ndindex((N,N,N)))[n]]
    offsets = np.zeros([N,N,N])#,dtype=np.ndarray); offsets.fill(K_offset)
    K_sq = vectorized_norm(np.multiply(K+offsets,K+offsets))

    Emax = c * np.max(K_sq) # E at maximum value of K

    for _ in range(1000):
        psi = compute_psi(psi,K_sq,Emax,U_K,N)
        psi = psi / np.sqrt(np.vdot(psi,psi)) 
    
    return psi

def get_ek(K,k):
    a=K+k
    return c * a.dot(a)
vector_get_ek = np.vectorize(get_ek,excluded=[1]) #only vectorize zeroth argument

def get_H_block(k,K,U_K,N):

    eks = vector_get_ek(K,k)

#    H = np.zeros((N**3,N**3), dtype='float64')
    H = np.diagflat(eks)
    m,n = np.indices(H.shape)
    H += U_K.flat[np.abs(m-n)]
#        H[j,j] = (hbar**2 * LA.norm(k)**2)/(2*m) 
    #assert(np.allclose(H.real,H)) # is real
    return H
    
    

'''
psi = get_converged_psi(0,K,U_K,N=5)
ground_energy = LA.eigvals(get_H_block(np.array([0.,0.,0.]),U_K,N=5))[0]
print(ground_energy*eV_to_Ryd)
'''

#10.7
'''
bands = 6
N=8
k = (2*np.pi / a) * np.array([0.4,0.4,0.4])
K, r = generate_lattice(N)
U_K = generate_potential(N,K,0)
#psi = get_converged_psi(0,K,U_K,N) 
H = get_H_block(k,U_K,N)


print(LA.eigvals(H*eV_to_Ryd)[:bands])#,eigvals=(0,bands-1)))
'''
#10.8
bands=6

L     = 2*np.pi/a*np.array([1/2,1/2,1/2]) 
gamma = 2*np.pi/a*np.array([  0,  0,  0]) 
X     = 2*np.pi/a*np.array([  0,  1,  0]) 
pointU= 2*np.pi/a*np.array([1/4,  1,1/4]) 
W     = 2*np.pi/a*np.array([1/2,  1,  0])

path = [L,gamma, X, pointU,gamma]
labels = ['L', r'$\Gamma$', 'X', 'U', r'$\Gamma$']
ksteps = 50
N = 12
energies = np.zeros([ksteps*(len(path)-1),bands])
kscale = np.zeros([ksteps*(len(path)-1)])
K,r = generate_lattice(N)
U_K = generate_potential(N,K,0)

label_locations = []

for i in range(0,len(path)-1):
    label_locations.append(i*ksteps)
    kstep = (path[i+1]-path[i]) / ksteps
    for step in range(ksteps):
        qval = path[i]+step*kstep
        #pdb.set_trace()
        H = get_H_block(qval,K,U_K,N)
        #free particle test
#        for band in range(bands):
#            energies[step+(i)*ksteps,band] = LA.norm(qval+ordered_K[band])**2 * (hbar**2 / (2*m))
#        import timeit
#        t = timeit.Timer('H2 = get_H_block(qval,K,U_K)', setup='from __main__ import get_H_block,efficient_new_H,generate_lattice,generate_potential,U; import numpy as np; import scipy.linalg as LA; qval=np.array([0.,0.,0.]); N=12; K,r = generate_lattice(N); U_K = generate_potential(K)').timeit(number=100)
#        print(t)
#        t = timeit.Timer('H = efficient_new_H(H,qval,K,U_K)', setup='from __main__ import get_H_block,efficient_new_H,generate_lattice,generate_potential,U; import numpy as np; import scipy.linalg as LA; qval=np.array([0.,0.,0.]); N=12; K,r = generate_lattice(N); U_K = generate_potential(K);H = get_H_block(np.array([0.,0.,0.]),K,U_K)').timeit(number=100)
#        print(t)
        '''
        if not np.allclose(H.real,H):
            print("complex H!")
            pdb.set_trace()
        '''    
        energies[step+i*ksteps,:] = LA.eigvalsh(H,eigvals=(0,bands-1))
        kscale[step+i*ksteps] = LA.norm(kstep)

label_locations.append(step+i*ksteps)
kvals = np.cumsum(kscale)

font = {'family' : 'serif',
       'weight' : 'bold',
        'size'   : 18}
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
