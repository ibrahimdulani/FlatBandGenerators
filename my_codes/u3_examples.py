
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
U=3
l=-1.5
h0=np.random.RandomState(84)
H0=np.diagflat(np.append([0.,1.],[h0.uniform(1,5,nu-2)]))
#H0=np.diagflat(np.append([0.,1.],2))
H0 = np.zeros((nu,nu))
for i in range(nu):
    for j in range(nu):
        if i==0:
            H0[i,i+1] = 1
#            H0[i,i+2] = 2
        elif i==nu-1:
            H0[i,i-1] = 1
#            H0[i,i-2] = 2
        else:
            H0[i,i-1] = 1
            H0[i,i+1] = 1
ps=np.random.RandomState(47)
#psi1=np.array(ps.uniform(-5,5,nu))
psi1=np.array([1,-1,1])

Q1=np.identity(nu)-np.outer(psi1,psi1)/np.dot(psi1,psi1)
e1=(1/np.sqrt(np.dot(psi1,psi1)))*psi1
e2=(1/np.sqrt(np.dot(Q1@H0@psi1,Q1@H0@psi1)))*Q1@H0@psi1
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
c=(l-a*x1)/b
psi3=x1*e1 + c*e2
u=(l*np.identity(nu) - H0)@(psi1 - psi3)
Qu=np.identity(nu)-np.outer(u,u)/np.dot(u,u)
M=Qu@(l*np.identity(nu) - H0)@Qu
w=psi1@(l*np.identity(nu) - H0)@psi1 + psi3@(l*np.identity(nu) - H0)@psi3
#   Alexei: perhaps better to write out things explicitely:
#           (E, psi) = la.eigh(M)
#           also note that M is Hermitian, so you can use eigh
#es=np.linalg.eig(M)

rng = np.random.RandomState(1346)

if np.all(M)==0 and w==0:
    psi2=Qu@np.array(ps.uniform(-5,5,nu))
else:
    (E, psi) = la.eigh(M)
    
    assert np.any(np.sign(E)==np.sign(w))==True, 'no copmatible CLS.'
    
    rn=rng.uniform(0,5,nu)
    for i in range(len(rn)):
        if np.sign(w)!=np.sign(E[i]):
            rn[i]=0
        else:
            pass
    
#   assert np.allclose(rn,0.)==False, 'no copmatible CLS.'
    
#   rn=np.array([0.94889, 0.0443666, 0.25151, 0.766195, 0.909334])
    rnd=(w/np.dot(E,rn))*rn
    x=np.zeros(nu)
    for i in range(rnd.size):
        x += np.sqrt(rnd[i])*psi[:,i]
    
    psi2=Qu@x

T=np.zeros((nu*(U+2),nu**2))

for i in range((U+2)*nu):
    if i in range(nu):
        for j in range(nu):
            T[i,(i%nu)*nu+j]=psi2[j]
    elif i in range(nu,(U-1)*nu):
        for j in range(nu):
            T[i,(i%nu)*nu+j]=psi3[j]
            T[i,i%nu+nu*j]+=psi1[j]
    elif i in range((U-1)*nu,U*nu):
        for j in range(nu):
            T[i,i%nu+nu*j]=psi2[j]
    elif i in range(U*nu,(U+1)*nu):
        for j in range(nu):
            T[i,(i%nu)*nu+j]=psi1[j]
    else:
        for j in range(nu):
            T[i,(i%nu)+nu*j]=psi3[j]

#   Alexei: I usually prefer np.eye to np.identity
Lambda=np.append((l*np.identity(nu) - H0)@psi1,((l*np.identity(nu) - H0)@psi2,(l*np.identity(nu) - H0)@psi3,np.zeros(nu),np.zeros(nu)))

h=la.lstsq(T,Lambda)

H1_base=np.zeros((nu,nu))

for i in range(nu):
    for j in range(nu):
        H1_base[i,j] = h[0][nu*i+j]

#   Generate H1_free
    
Q1 = np.identity(nu) - np.outer(psi1,psi1)/np.dot(psi1,psi1)
Q2 = np.identity(nu) - np.outer(psi2,psi2)/np.dot(psi2,psi2)
Q3 = np.identity(nu) - np.outer(psi3,psi3)/np.dot(psi3,psi3)
		
if nu<3:
    H1_free = np.zeros((nu,nu))
else:
    pr1 = np.outer(Q1@psi2,Q1@psi2) + np.outer(Q1@psi3,Q1@psi3)
    pr2 = np.outer(Q3@psi1,Q3@psi1) + np.outer(Q3@psi2,Q3@psi2)
    ns1 = nullspace(pr1)
    ns2 = nullspace(pr2)
    z1 = np.zeros(nu)
    z2 = np.zeros(nu)
    rn1= rng.uniform(-5,5,size=np.shape(ns1)[1])
    rn2= rng.uniform(-5,5,size=np.shape(ns2)[1])
    for i in range(len(rn1)):
        z1+= rn1[i]*ns1[:,i]
        z2+= rn1[i]*ns2[:,i]
    Pz1 = np.outer(z1,z1)/np.dot(z1,z1)
    Pz2 = np.outer(z2,z2)/np.dot(z2,z2)
    MM = rng.uniform(size=(nu,nu))
    H1_free = Pz2@MM@Pz1

H1 = H1_base + Q3@H1_free@Q1

K=np.linspace(0.,2*mt.pi,100)

Ev=np.zeros((100,nu))

for i in range(100):
    Hk=np.conjugate(H1.T)*np.exp(K[i]*1j) + H0 + H1*np.exp(-K[i]*1j)
    Ev[i]=np.linalg.eigvalsh(Hk)

print(H0,H1,psi1,psi2,psi3,l)

font = {'family': 'Italic',
        'color':  'black',
        'weight': 'normal',
        'size': 24,
        }

plt.plot(K,Ev)
plt.xticks([0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi],
           ['$0$', r'$\frac{\pi}{2}$', r'$\pi$', r'$\frac{3\pi}{2}$', r'$2\pi$']
           ,fontsize=22)
plt.yticks(fontsize=22)
plt.xlabel('k',fontdict=font)
plt.ylabel('E(k)',fontdict=font)
plt.show()





