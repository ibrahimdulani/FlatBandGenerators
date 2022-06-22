
import math as mt
import cmath as _cmt
import warnings as _wn
#import argparse as ap
#from math import pi

import numpy as np
import scipy.linalg as la
import matplotlib.pyplot as plt
#from progress.bar import Bar
from numpy.linalg import svd

import scipy.sparse as ssp

import flatband as fb


###############
# module for finding nullspace of projector pr

#   Alexei: a better (more stable) method to compute rank using
#           pivoting QR decomposition
#       (Q, R, P) = scipy.linalg.qr(A, pivoting=True)
#       rank = np.sum(np.abs(R.diagonal()) > tol)
def rank(A, atol=1e-13, rtol=0):
    """Estimate the rank (i.e. the dimension of the nullspace) of a matrix.

    The algorithm used by this function is based on the singular value
    decomposition of `A`.

    Parameters
    ----------
    A : ndarray
        A should be at most 2-D.  A 1-D array with length n will be treated
        as a 2-D with shape (1, n)
    atol : float
        The absolute tolerance for a zero singular value.  Singular values
        smaller than `atol` are considered to be zero.
    rtol : float
        The relative tolerance.  Singular values less than rtol*smax are
        considered to be zero, where smax is the largest singular value.

    If both `atol` and `rtol` are positive, the combined tolerance is the
    maximum of the two; that is::
        tol = max(atol, rtol * smax)
    Singular values smaller than `tol` are considered to be zero.

    Return value
    ------------
    r : int
        The estimated rank of the matrix.

    See also
    --------
    numpy.linalg.matrix_rank
        matrix_rank is basically the same as this function, but it does not
        provide the option of the absolute tolerance.
    """

    A = np.atleast_2d(A)
    s = svd(A, compute_uv=False)
    tol = max(atol, rtol * s[0])
    rank = int((s >= tol).sum())
    return rank


def nullspace(A, atol=1e-13, rtol=0):
    """Compute an approximate basis for the nullspace of A.

    The algorithm used by this function is based on the singular value
    decomposition of `A`.

    Parameters
    ----------
    A : ndarray
        A should be at most 2-D.  A 1-D array with length k will be treated
        as a 2-D with shape (1, k)
    atol : float
        The absolute tolerance for a zero singular value.  Singular values
        smaller than `atol` are considered to be zero.
    rtol : float
        The relative tolerance.  Singular values less than rtol*smax are
        considered to be zero, where smax is the largest singular value.

    If both `atol` and `rtol` are positive, the combined tolerance is the
    maximum of the two; that is::
        tol = max(atol, rtol * smax)
    Singular values smaller than `tol` are considered to be zero.

    Return value
    ------------
    ns : ndarray
        If `A` is an array with shape (m, k), then `ns` will be an array
        with shape (k, n), where n is the estimated dimension of the
        nullspace of `A`.  The columns of `ns` are a basis for the
        nullspace; each element in numpy.dot(A, ns) will be approximately
        zero.
    """

    A = np.atleast_2d(A)
    u, s, vh = svd(A)
    tol = max(atol, rtol * s[0])
    nnz = (s >= tol).sum()
    ns = vh[nnz:].conj().T
    return ns

##############
# function for finding orthonormal basis which are also orthonormal to e1 and e2

#   Alexei: you can simplify this as follows (Python3 > 3.4)
#       proj = lambda u, v: u*(v@u)/(u@u)
def proj(u, v):
    # notice: this algrithm assume denominator isn't zero
    return u * np.dot(v,u) / np.dot(u,u)

def GS(V):
    V = 1.0 * V     # to float
    U = np.copy(V)
    for i in range(1, V.shape[1]):
        for j in range(i):
            U[:,i] -= proj(U[:,j], V[:,i])
    # normalize column
    den=(U**2).sum(axis=0) **0.5
    E = U/den
    # assert np.allclose(E.T, np.linalg.inv(E))
    return E

#################################


nu=3
U=2
rng=np.random.RandomState(246)
#l=rng.uniform(-5.,5.)
l=0.5
H0=np.diagflat(np.append([0.,1.],2))
#H0=np.diagflat(np.append([0.,1.],[rng.uniform(1,5,nu-2)]))
##H0 = np.zeros((nu,nu))
##for i in range(nu):                                                # honeycomb chain
##    for j in range(nu):
##        if i==0:
##            H0[i,i+1] = 1
###            H0[i,i+2] = 1
##        elif i==nu-1:
##            H0[i,i-1] = 1
###            H0[i,i-2] = 1
##        else:
##            H0[i,i-1] = 1
##            H0[i,i+1] = 1
ps=np.random.RandomState(47)
#psi1=ps.uniform(-5.,5.,nu)

#H=np.zeros((nu,nu))
#H[0,1]=1
#H[4,3]=1
#psi1=la.eig(H)[1][4]
psi1=np.array([1,-1,1])
#psi1=np.array([1,0,1])

Q1=np.identity(nu)-np.outer(psi1,psi1)/np.dot(psi1,psi1)
e1=psi1/(np.sqrt(np.dot(psi1,psi1)))
if np.all(Q1@H0@psi1 < 1e-8):
    e2=Q1@e1
else:
    e2=Q1@H0@psi1/(np.sqrt(np.dot(Q1@H0@psi1,Q1@H0@psi1)))
pr=np.outer(e1,e1) + np.outer(e2,e2)

#   Alexei: nice idea, but check - perhaps the SVD gives the othonormal
#           set already, i.e. "ns" in nullspace might already be orthonormal.
#           Gram-Schmidt is good in theory, but practically
#           is prone to numerical instabilities or so they say
#basis=GS(nullspace(pr))
#(Q, R, P) = la.qr(pr, pivoting=True)
basis=nullspace(pr)

x1=1/np.dot(psi1,e1)
a=e1@H0@psi1
b=e2@H0@psi1
x2=(l - a*x1)/b
#x2=rng.uniform(-5,5)
#l=(psi1@H0@(x1*e1 + x2*e2))/(psi1@(x1*e1 + x2*e2))

w=psi1@(l*np.identity(nu)-H0)@psi1 - (x1*e1 + x2*e2)@(l*np.identity(nu)-H0)@(x1*e1 + x2*e2)
u=np.zeros(nu-2)
M=np.zeros((nu-2,nu-2))
for i in range(nu-2):
    u[i]=x2*(e2@H0@basis[:,i])
    for j in range(nu-2):
        M[i,j]=basis[:,i]@(l*np.identity(nu)-H0)@basis[:,j]

MM=la.inv(M)
(E , psi)=la.eigh(M)
t=w+u@MM@u

#rd=rng.uniform(0.,5.,nu-2)
rn=np.zeros(E.size)
rn=rng.uniform(0,5,E.size)
for i in range(len(rn)):
    if np.sign(t)!=np.sign(E[i]):
        rn[i]=0
    else:
        pass

#rn=np.array(((w + u@MM@u)/(rn@E))*rn)
#y=psi@np.sqrt(rn)
assert np.allclose(rn,0.)==False, 'no copmatible CLS.'

rnd=(t/np.dot(E,rn))*rn
y=np.zeros(E.size)
for i in range(E.size):
    y+=np.sqrt(rnd[i])*psi[:,i]

xx=y + MM@u
s=np.zeros(nu)
for i in range(nu-2):
    s+=xx[i]*basis[:,i]

psi2=x1*e1 + x2*e2 +s

npsi1 = psi1@psi1
npsi2 = psi2@psi2

#   Projectors
Q1 = np.identity(nu) - np.outer(psi1, psi1)/npsi1
Q2 = np.identity(nu) - np.outer(psi2, psi2)/npsi2

Q21 = np.identity(nu) - np.outer(Q2@psi1, Q2@psi1)/(psi1@Q2@psi1)
Q12 = np.identity(nu) - np.outer(Q1@psi2, Q1@psi2)/(psi2@Q1@psi2)

#   ?
#E = psi2@H0@psi1                                                # flatband energy
EH0 = l*np.identity(nu) - H0                                                  #
x = psi1@EH0@psi1                                               #

#   Sanity check
assert np.allclose(x, psi2@EH0@psi2), 'Incorrect psi2!'

#   Compute the "free" part                                                          #
if nu > 2:                                                  # free part is there                                                  # real "free" part
    K = rng.uniform(size=(nu, nu))                      # the "free" part
    K = Q21@K@Q12
elif nu == 2:                                               # nu = 2
    K = np.zeros((nu, nu))                                 # zero matrix

#   ?
##T = np.outer(EH0@psi1, EH0@psi2)/x                             #
##H1 = Q2@(T + K)@Q1
H11= np.outer(EH0@psi1, EH0@psi2)/x

T=np.zeros((nu*(U+2),nu**2))

for i in range((U+2)*nu):
    if i in range(nu):
        for j in range(nu):
            T[i,(i%nu)*nu+j]=psi2[j]
    elif i in range((U-1)*nu,U*nu):
        for j in range(nu):
            T[i,i%nu+nu*j]=psi1[j]
    elif i in range(U*nu,(U+1)*nu):
        for j in range(nu):
            T[i,(i%nu)*nu+j]=psi1[j]
    else:
        for j in range(nu):
            T[i,(i%nu)+nu*j]=psi2[j]

##   Alexei: I usually prefer np.eye to np.identity
Lambda=np.append((l*np.identity(nu) - H0)@psi1,((l*np.identity(nu) - H0)@psi2,np.zeros(nu),np.zeros(nu)))

##A=ssp.lil_matrix((nu**2,nu**2))
##A[1,1]=1
#A[nu*4+4,nu*4+4]=1

h=la.lstsq(T,Lambda)

##h=la.lstsq(T@A,Lambda)

#sanity check

assert np.allclose(T@h[0],Lambda), ' H1 is an appoximation, not a solution'

##assert np.allclose(T@A@h[0],Lambda), ' H1 is an appoximation, not a solution'

H1=np.zeros((nu,nu))

for i in range(nu):
    for j in range(nu):
        H1[i,j]=h[0][nu
                     *i+j]

H1 = H1# + Q2@K@Q1

k=np.linspace(0.,2*mt.pi,100)

Ev=np.zeros((100,nu))

for i in range(100):
    Hk=np.conjugate(H1.T)*np.exp(k[i]*1j) + H0 + H1*np.exp(-k[i]*1j)
    Ev[i]=np.linalg.eigvalsh(Hk)

print(H0,H1,psi1,psi2,l)

font = {'family': 'DejaVu Sans',
        'color':  'black',
        'weight': 'normal',
        'size': 24,
        }

plt.plot(k,Ev)
plt.xticks([0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi],
           ['$0$', r'$\frac{\pi}{2}$', r'$\pi$', r'$\frac{3\pi}{2}$', r'$2\pi$']
           ,fontsize=22)
plt.yticks(fontsize=22)
plt.xlabel('k',fontdict=font)
plt.ylabel('E(k)',fontdict=font)
plt.show()

H_data = [H0,H1]

np.save('/Users/pcs/Documents/ibrahim/Flatband/data/u2_honeycomb', np.array([H0, H1]), allow_pickle=False)


QM=fb.QuantumNetwork(H_data)
E_fb = psi2@H0@psi1/(psi1@psi2)
if QM.has_fb(E_fb,eps=1e-8):
    print('The model has flatband.')
else:
    print('The model has no flatband.')

#sanity check

#assert np.all([np.any(np.allclose(la.eig(H1)[1][i],psi1)) for i in range(nu)]), 'psi1 is not the zero eigenvector of H1'

assert np.all(H1@psi1 < 1e-8) and np.all(psi2@H1 < 1e-8), 'psi1 is not the zero eigenvector of H1'
