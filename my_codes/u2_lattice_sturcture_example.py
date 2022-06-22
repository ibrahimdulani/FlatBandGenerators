
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

#import flatband as fb


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

#Class QuantumNetwork

class QuantumNetwork():
    """
    Generic d=1 quantum network
    """
    def __init__(self, H_data, eps=1e-8):
        """
        Network constructor

        Parameters
        ----------
        H_data : tuple
            H0, H1, ... : array
                container of all the hopping matrices
                should have the size (1 + mc)xnuxnu
        eps : float, optional
            zero threshold
        canonical : bool, optional
            convert hopping matrices to canonical form
        """

        self.H_all = np.array(H_data, dtype=np.complex128)            # hopping matrices

#   Generic part
        H0 = self.H_all[0]                                              # intracell hopping
        self.nu = H0.shape[0]                                           # number of sites in the cell
        self.mc = self.H_all.shape[0] - 1                               # hopping range
        self.eps = eps                                                  #

#   Sanity checks
        assert self.mc == 1, 'Only n.n. hopping has been implemented!'
        assert np.all(np.abs(H0 - np.conjugate(H0.T)) < self.eps), 'Intracell hopping is not Hermitian!'

    def __getitem__(self, key):
        """
        Dict like access to internal data
        """
        if key in ('nu', 'mc', 'canonical'):                            # number of bands, hopping range,
            key_val = getattr(self, key)
        elif key == 'H0':                                               # intracell hopping
            key_val = self.H_all[0]
        elif key == 'H1':                                               # n.n. hopping to the left
            key_val = self.H_all[1]
        else:
            raise KeyError('Unknown key ' + key)

        return key_val

#   I/O
#    def load

#   ?
    def cls_ham_make(self, U):
        """
        Generate Hamiltonian on U unit cells, assuming wavefunctions are
        zero outside the cells

        Parameters
        ----------
        U : int
            number of the unit cells

        Returns
        -------
        HU : array
            the Hamiltonian matrix
        """
#   Parameters
        H0 = self.H_all[0]                                              # intracell hopping
        H1 = self.H_all[1]                                              # n.n. cell hopping

        HU = [[np.zeros_like(H0) for j in range(U)] for i in range(U)] # empty Hamiltonian matrix

#   Construct the block Hamiltonian
        for i in range(U):                                              # loop over the unit cells
            if i > 0:                                                   # exclude the first row
                HU[i][i-1] = np.conjugate(H1.T)                        # H1^dagger
            HU[i][i] = H0                                               # diagonal element
            if i < U-1:                                                 # exclude the last row
                HU[i][i+1] = H1                                         #

        HU = _np.bmat(HU)                                               # construct array from blocks

#   Sanity check
        assert np.all(np.abs(HU - np.conjugate(HU.T)) < self.eps), 'HU matrix is not Hermitian!'

        return HU

    def bands(self, k_num=50):
        """
        Compute the bands

        Parameters
        ----------
        k_num : int
            (-pi, pi) discretisation density

        Returns
        -------
        k_all : array
            the first Brillouin zone
        Ek_all : array
            the bands
        """
#   Parameters
        H0 = self.H_all[0]                                              # intracell hopping
        H1 = self.H_all[1]                                              # n.n. intercell hopping

        Mk = np.zeros((self.nu, self.nu), dtype=np.complex128)
        k_all = np.linspace(-mt.pi, mt.pi, k_num, endpoint=False)    # first Brillouin zone
        Ek_all = np.zeros((self.nu, k_num))                            # the bands

#   Compute the bands
        i = 0
        for k in k_all:                                                 # loop over the momenta
            Mk[:, :] = _cmt.exp(1j*k)*np.conjugate(H1.T) + H0 + _cmt.exp(-1j*k)*H1
            Ek_all[:, i] = la.eigvalsh(Mk)                             # compute the spectrum

            i += 1

        return (k_all, Ek_all)

    def fb_find(self, eps=1e-8, k_num=50):
        """
        Find flatbands

        Parameters
        ----------
        eps : float
            zero threshold
        k_num : int
            (-pi, pi) discretisation density

        Returns
        -------
        ix : array
            indices of the flatbands
        """
#   Sanity checks
        assert self.H_all.shape[0] == 2, 'Only n.n. hopping has been implemented!'

#   Detect flatbands
        (k_all, Ek_all) = self.bands(k_num=k_num)                       # compute the bands
        Ek_std = Ek_all.std(axis=1)                                     # the bandwidth
        ix = np.where(Ek_std < eps)[0]                                 # flatband indices

        return (np.array(ix), Ek_std)

    def has_fb(self, E_fb=None, eps=1e-8, k_num=50):
        """
        Check whether the network has flatband(s)

        Parameters
        ----------
        E_fb : float
            flatband energy
        eps : float
            zero threshold
        k_num : int
            (-pi, pi) discretisation density

        Returns
        -------
        ix : array
            indices of the flatbands
        """
        if E_fb is None:
            (ix, E_std) = self.fb_find(eps=eps, k_num=k_num)
            r = (ix.size > 0)
        else:
            (k_all, Ek_all) = self.bands(k_num=k_num)					#
#            ix = _np.where(abs(Ek_all[:,] - E_fb) < eps)[0]
#            r = (ix.size >= k_num)
            r = np.all([np.any(np.abs(Ek_all[:, i] - E_fb) < eps) for i in range(k_num)])

        return r

    def cls_find(self, U_max, eps=1e-8):
        """
        Search for possible compact localised states

        Parameters
        ----------
        U_max : int
            maximal CLS size to search for
        eps : float, optional
            zero threshold

        Returns
        -------
        E : list
            list of CLS energies
        Psi : list
            list of CLS
        """
#   Parameters
        nu = self.nu                                                    # number of site in unit cell
        found = False                                                   #

        H0 = self.H_all[0]                                              # intracell hopping
        H1 = self.H_all[1]                                              # n.n. hopping
        H1_hc = np.conjugate(H1.T)                                     # hermitian conjugate of H1
#        Z = _np.zeros_like(H0)                                          # zero matrix

#   Sanity check
        assert self.mc == 1, 'Only n.n. hopping has been implemented so far!'

#   ?
        for U in range(1, U_max):                                       # loop over U class
            HU = self.cls_ham_make(U)                                   # generate U-CLS Hamiltonian from H0 and H1
            (E, psi) = la.eigh(HU)                                     # diagonalise

            adE = np.abs(_np.diff(E))
            mp = np.where(adE < eps)[0]                                # doubley degenerate eigenvalues

            if mp.size > 0:                                             # there are degenerate eigenvalues
                _wn.warn('Degenerate eigenvalue(s) detected: results might be inaccurate!')

#                for i in mp:                                            #
#                    psi = _detangle(psi, i, H1)

#   Psi_1
            u1 = H1@psi[:nu, :]                                         # u1_ak = H1_ab psi_bk
            u1n = np.sqrt(np.sum(np.abs(u1), axis=0)/nu)             # u1n_k = sum_a u1_ak

#   Psi_U
            uu = H1_hc@psi[-nu:, :]
            uun = np.sqrt(np.sum(_p.abs(uu), axis=0)/nu)

#   CLS test
            z = np.array([i for i in range(HU.shape[0]) if u1n[i] < eps and uun[i] < eps])
#            z = _np.array([i for i in ix1 if i in ixu])                 # rows that are zero for both left and right
            if z.size > 0:                                              # there are zero rows, that are left and right at the same time
                found = True                                            #

                break

#   Finalise the output
        if found:                                                       # found a CLS
            r = [(E[i], psi[:, i]) for i in z]
        else:                                                           # no CLS was found
            r = None

        return r
##################################


nu=5
#nu=8
U=2
rng=np.random.RandomState(246)
#l=rng.uniform(-5.,5.)
l=2
H0=np.diagflat(np.append([0.,1.],[rng.uniform(1,5,nu-2)]))
#H0=np.array([[0,1,0,0,0],[1,0,1,0,0],[0,1,0,1,0],[0,0,1,0,1],[0,0,0,1,0]])                           # honeycomb chain, nu=5
#H0=np.array([[0,0,0,0,1,1,0,0],[0,0,0,0,1,0,1,0],[0,0,0,0,0,1,1,1],[0,0,0,0,0,0,1,0],[1,1,0,0,0,0,0,0],[1,0,1,0,0,0,0,0],[0,1,1,1,0,0,0,0],[0,0,1,0,0,0,0,0]])  # honeycomb chain, nu=8
ps=np.random.RandomState(47)

psi1=ps.uniform(-5.,5.,nu)
#psi1=np.array([-1,1,0,1,-1])                       # psi1  for honeycomb chain, nu=5
#psi1=np.array([1,1,1,1])
#(ev,vr) = la.eig(Sp)

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
x2=(l - a*x1)/b
#x2=rng.uniform(0,5)
#l=psi1@H0@(x1*e1 + x2*e2)

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
#T = np.outer(EH0@psi1, EH0@psi2)/x                             #
#H1 = Q2@(T + K)@Q1


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

#   Alexei: I usually prefer np.eye to np.identity
Lambda=np.append((l*np.identity(nu) - H0)@psi1,((l*np.identity(nu) - H0)@psi2,np.zeros(nu),np.zeros(nu)))


##A=np.zeros((nu**2,nu**2))
##A[nu*2,nu*2]=1
##A[nu*2+4,nu*2+4]=1
###Sp[8*7+3,8*7+3]=1
##
##h=la.lstsq(T@A,Lambda)

h=la.lstsq(T,Lambda)

h1=np.zeros((nu,nu))

for i in range(nu):
    for j in range(nu):
        h1[i,j]=h[0][nu*i+j]

H1=h1#+Q2@K@Q1

k=np.linspace(0.,2*mt.pi,100)

Ev=np.zeros((100,nu))

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

H_data = [H0,H1]

(ev, evc2, evc1)=la.eig(H1, left=True)
ix=np.where(np.abs(abs(ev - 0.) < 1e-8))[0]
if ix.size > 0:
    psi1=evc1[:,ix[0]]
    psi2=evc2[:,ix[0]]
else:
    raise ValueError('H1 is non-singular')
QM=QuantumNetwork(H_data)
E_fb = psi2@H0@psi1/(psi2@psi1)
if QM.has_fb(E_fb,eps=1e-8):
    print('The model has flatband.')
else:
    print('The model has no flatband.')
