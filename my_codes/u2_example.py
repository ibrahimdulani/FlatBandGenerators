
import math as mt
#import cmath as cmt
#import argparse as ap
#from math import pi

import numpy as np
import scipy.linalg as la
import matplotlib.pyplot as plt
#from progress.bar import Bar
from numpy.linalg import svd


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

###############

nu=3
U=2
l=0
#h0=np.random.RandomState(84)
H0=np.array([[0,1,0],[1,0,1],[0,1,0]])
ps=np.random.RandomState(47)
psi1=np.array([1,0,-1])
psi2=np.array([1,0,0])

#Q2=np.identity(nu)-np.outer(psi2,psi2)/np.dot(psi2,psi2)
#h=(1/psi1@(l*np.identity(nu)-H0)@psi1)*np.outer((l*np.identity(nu)-H0)@psi1,(l*np.identity(nu)-H0)@psi2)
#t=np.random.RandomState(473)
#T=t.uniform(-5,5,(nu,nu))
#H1=Q2@h@Q2

T=np.zeros((nu*(U+2),nu**2))

for i in range((U+2)*nu):
    if i in range(nu):
        for j in range(nu):
            T[i,(i%nu)*nu+j]=psi2[j]
#    elif i in range(nu,(U-1)*nu):
#        for j in range(nu):
#            T[i,(i%nu)*nu+j]=psi3[j]
#            T[i,i%nu+nu*j]+=psi1[j]
    elif i in range((U-1)*nu,U*nu):
        for j in range(nu):
            T[i,i%nu+nu*j]=psi1[j]
    elif i in range(U*nu,(U+1)*nu):
        for j in range(nu):
            T[i,(i%nu)*nu+j]=psi1[j]
    else:
        for j in range(nu):
            T[i,(i%nu)+nu*j]=psi2[j]

#   Alexei: I usually prefer np.eye to np.identity
Lambda=np.append((l*np.identity(nu) - H0)@psi1,((l*np.identity(nu) - H0)@psi2,np.zeros(nu),np.zeros(nu)))

h=la.lstsq(T,Lambda)

H1=np.zeros((nu,nu))

for i in range(nu):
    for j in range(nu):
        H1[i,j]=h[0][nu*i+j]

K=np.linspace(0.,2*mt.pi,100)

Ev=np.zeros((100,nu))

for i in range(100):
    Hk=np.conjugate(H1.T)*np.exp(K[i]*1j) + H0 + H1*np.exp(-K[i]*1j)
    Ev[i]=np.linalg.eigvalsh(Hk)

print(H0,H1,psi1,psi2)

plt.plot(Ev)
plt.show()
