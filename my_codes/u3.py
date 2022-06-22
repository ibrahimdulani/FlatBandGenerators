
import math as mt
import cmath as cmt
import argparse as ap
from math import pi

import numpy as np
import scipy.linalg as la
import matplotlib.pyplot as plt
from progress.bar import Bar
from numpy.linalg import svd


###############
# module for finding nullspace of projector pr

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

###############

l=1
H0=np.diagflat([[0.,1.,2.,3.,4.]])
psi1=np.array([1.,2.,3.,4.,6.])

Q1=np.identity(5)-np.outer(psi1,psi1)/np.dot(psi1,psi1)
e1=(1/np.sqrt(np.dot(psi1,psi1)))*psi1
e2=(1/np.sqrt(np.dot(Q1@H0@psi1,Q1@H0@psi1)))*Q1@H0@psi1
pr=np.outer(e1,e1) + np.outer(e2,e2)

basis=GS(nullspace(pr))

x1=1/np.dot(psi1,e1)
a=e1@H0@psi1
b=e2@H0@psi1
c=(l-a*x1)/b
psi3=x1*e1 + c*e2
u=(l*np.identity(5) - H0)@(psi1 - psi3)
Qu=np.identity(5)-np.outer(u,u)/np.dot(u,u)
M=Qu@(l*np.identity(5) - H0)@Qu
w=psi1@(l*np.identity(5) - H0)@psi1 + psi3@(l*np.identity(5) - H0)@psi3
es=np.linalg.eig(M)

rng = np.random.RandomState(12)

rn=rng.rand(5)
rnd=(w/np.dot(es[0],rn))*rn
x=np.zeros(5)
for i in range(rnd.size):
    x += np.sqrt(rnd[i])*es[1][:,i]

psi2=Qu@x

T=np.zeros((5*(3+2),5**2))

for i in range((3+2)*5):
    if i<=5:
        for j in range(5):
            T[i,(i%5)*5+j]=psi2[j]
    elif i in range(5,(3-1)*5):
        for j in range(5):
            T[i,(i%5)*5+j]=psi3[j]
            T[i,i%5+5*j]+=psi1[j]
    elif i in range((3-1)*5,3*5):
        for j in range(5):
            T[i,i%5+5*j]=psi2[j]
    elif i in range(3*5,(3+1)*5):
        for j in range(5):
            T[i,(i%5)*5+j]=psi1[j]
    else:
        for j in range(5):
            T[i,(i%5)+5*j]=psi3[j]

Lambda=np.append((l*np.identity(5) - H0)@psi1,((l*np.identity(5) - H0)@psi2,(l*np.identity(5) - H0)@psi3,np.zeros(5),np.zeros(5)))

h=np.linalg.lstsq(T,Lambda)

H1=np.zeros((5,5))

for i in range(5):
    for j in range(5):
        H1[i,j]=h[0][5*i+j]

k=np.linspace(0.,2*mt.pi,100)

Ev=np.zeros((100,5))

for i in range(100):
    Hk=np.conjugate(H1.T)*np.exp(k[i]*1j) + H0 + H1*np.exp(-k[i]*1j)
    Ev[i]=np.linalg.eigvalsh(Hk)

print(H0,H1,psi1,psi2,l)

font = {'family': 'Italic',
        'color':  'black',
        'weight': 'normal',
        'size': 16,
        }

plt.plot(k,Ev)

plt.xticks([0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi],
           ['$0$', r'$\frac{\pi}{2}$', r'$\pi$', r'$\frac{3\pi}{2}$', r'$2\pi$'])
plt.xlabel('k',fontdict=font)
plt.ylabel('E(k)',fontdict=font)
plt.show()
    





