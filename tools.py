# -*- coding: utf-8 -*-
"""
Created on Wed Apr 11 11:49:33 2018

@author: brian


Miscellaneous tools, useful for working with quspin, and not specific to rydberg chains
"""

import numpy as np
import matplotlib.pyplot as plt
from quspin.basis import boson_basis_1d, boson_basis_general
from quspin.operators import hamiltonian


BC_TYPES = ['open', 'periodic']
SP_TYPES = ["x", "y", "z", "+", "-", "I"]


def check_bc(bc):
    if bc not in BC_TYPES:
        raise TypeError("invalid boundary conditions")

def vprint(v,s):
    if v:
        print(s)
        
def is_hcb_basis(b):
    """Checks whether b is a hard-core boson basis."""
    return (isinstance(b, boson_basis_1d) or isinstance(b, boson_basis_general)) and b.sps==2

def zeta(n,ktrunc):
    return np.sum([1.0/j**n for j in range(1,ktrunc+1)])

def format_state(s):
    """Checks for array type and tries to convert to 1d array.
       Assumes a pure state, i.e. vector, s.
       
       Returns: one-dimensional numpy array.""" 
    s = np.asarray(s)
    if s.ndim == 1:
        return s
    if s.ndim != 2:
        raise ValueError("State array has too many dimensions")
    l,dummy = max(s.shape), min(s.shape)
    if dummy != 1:
        raise ValueError("Conversion into 1d array is ambiguous. For a pure state, one of the array axes ought to be of length 1.")
    return s.reshape(l)
        
        
def overlap(s1,s2):
    """Returns the overlap <s1|s2>.
    s1, s2 = two pure states.
    """
    s1 = format_state(s1)
    s2 = format_state(s2)
    if len(s1)!=len(s2):
        raise ValueError("Input states must be the same shape.")
    return np.sum(s1.conjugate() * s2)   

def norm(s):
    return np.abs(overlap(s,s))

def F(s1, s2):
    """ overlap-squared fidelity"""
    return np.abs(overlap(s1, s2))**2

def identity(basis):
    """identity operator in specified basis"""
    c = get_site_coupling(1.0/basis.L, basis.L)
    static = [ ["I", c]]
    return hamiltonian(static, [], basis=basis)


def is_herm(A):
    """ check if array is (numpy-implemented) hermitian matrix"""
    if not isinstance(A, np.ndarray):
        return False
    s=A.shape
    return (len(s)==2) and (s[1]==s[0]) and (A.conjugate().transpose() == A).all()

def get_state(loc_str,basis,dtype=np.complex128):
    """Given a string of the form '1110101', return the corresponding state vector, in quspin basis labeling.
    
    Returns: numpy 1d array defining the normalized state.
    Only accepts bases which are not symmetry-reduced. If you want a state in the symmetry-reduced basis, form it in the full basis and then project.
    """
    
    ###symmetry reduction has been applied
    if basis.Ns != 2**(basis.N):
        raise ValueError("the basis has been reduced.")
        
    i0 = basis.index(loc_str)
    psi = np.zeros(basis.Ns,dtype=dtype)
    psi[i0] = 1.0
    return psi



##helper functions to get specific product states
def get_zero_str(L):
    return "0"*L

def get_Z2_str(L, which=0):
    if (which==0) or (L%2)!=0:
        base="10"
    elif which==1:
        base="01"
    else:
        raise ValueError
    s = base*(L//2)
    if (L%2)!=0:
        s+= "1"
    return s

def make_z2_str_naive(L, which=0):
    """ z2 string which doesn't notice boundary conditions"""
    if which==0:
        base = "01"
    elif which==1:
        base = "10"
    else:
        raise ValueError
    if L==1:
        return str(which)
    s = (L//2) * base
    if (L%2 !=0):
        s+= str(which)
    return s

def get_checkerboard_str(Lx, Ly, which=0):
    """ Returns the string which defines a checkerboard pattern in a square lattice, 
    unraveled in thread order
        which: indicates the occupation of the first site, i.e. the lower-left corner of the array"""
    s1 = make_z2_str_naive(Lx, which=which)
    s2 = make_z2_str_naive(Lx, which =1-which)
    s = (s1 + s2) * (Ly//2)
    if Ly%2 !=0:
        s += s1
    return s
        
def get_checkerboard(basis, dtype,which=0):
    """ Return a checkerboard state of the specified parity."""
    from ryd_base import HCBBasis2D
    if not isinstance(basis, HCBBasis2D):
        raise ValueError("Expecting a 2D hcb basis")
    s = get_checkerboard_str(basis.Lx, basis.Ly,which=which)
    return get_state(s, basis, dtype=dtype)

def get_Z3_str(L,which=0):
    if which not in range(3):
        raise ValueError
    if (L%3)!=0:
        raise ValueError
    base="0"*which + "1" + "0" * (2-which)
    return base * (L//3)

def get_vac(basis,dtype):
    L=basis.L
    return get_state( get_zero_str(L),basis,dtype=dtype)
def get_Z2(basis,dtype,which=0):
    L=basis.L
    return get_state( get_Z2_str(L,which=which),basis,dtype=dtype)

def get_uniform_state(basis,dtype,which=0):
    base = str(which)
    return get_state( base * basis.N,basis,dtype=dtype)

def get_Z3(basis,dtype,which=0):
    L=basis.L
    return get_state( get_Z3_str(L,which=which),basis,dtype=dtype)

def get_sent(psi, basis, subsys=None, return_rdm=None):
    """Return the entanglement entropy of psi, living in basis <basis>, computed in the reduced subsystem specified by subsys
    subsys = list of site labels [0, 1, ..., k] specifying the subsystem. If subsys=None,  defaults to 0....N/2 -1
    
    return_rdm can be specified as 'A' (the subsystem of interest), 'B', or both; if so a dictionary is returned
    """
    if subsys is None:
        #the default quspin block
        subsys=tuple(range(basis.N//2))

    sdict= basis.ent_entropy(psi, sub_sys_A=subsys,return_rdm=return_rdm)
    # the quspin value is normalized by the subsystem size
    SA= sdict['Sent_A'] * len(subsys) 
    if return_rdm is not None:
        sdict['Sent_A']=SA        
        return sdict
    return SA
      
def get_hcb_basis(N,**symms):
    """returns basis object for N hard-core bosons. The operator strings are 'n', '+', '-'"""
    return boson_basis_1d(N,sps=2,**symms)
 
def get_site_coupling(V, L,verbose=False):
    """list of coupling terms for single-site coupling V.
        If V is iterable, the coupling will be non-uniform: V[i] at site i."""
    try:
        if len(V)!=L:
            raise ValueError("Input coupling list does not match length L")
        
        static= [[V[i],i] for i in range(L)]
        vprint(verbose,"on-site coupling array provided,potentially nonuniform")
    except TypeError:
        static = [[V,i] for i in range(L)] 
        vprint(verbose,"scalar on-site coupling, will be uniform")
    return static

def diagonalize_block(static, gen_basis, L, blockname,dtype=np.complex128,method='eigh',Ne=None,which='SA', other_symm_dict=None):
    """diagonalize static hamiltonian by projecting onto blocks.
        Returns: a dictionary which maps
            k --> (e, s,proj)
        where q is the block label (follows quspin convention) and e,s are the eigenvalues and vectors of the corresponding hamiltonian block. The states are written in the symmetry-reduced basis. proj is the change-of-basis matrix which, applied to those states, 
        returns them expressed in the standard basis of the full space.
        
        static: static parameter list for the full hamiltonian
        gen_basis: produces basis, given L,kblock as inputs
        L: chain length        
        blockname: string name of the symmetry type. Example: "kblock".
        method: how to obtain the eigenstuff (i.e. exact solver or sparse)
        Ne: if using the sparse solver, how many eigenpairs to compute
        which: if using the sparse solver, where to draw the pairs from the spectrum. See Quspin docs for eigsh()
        other_symm_dict: if not None, diagonalizes the hamiltonian in the symmetry sector specified by this dictionary
       """
    methods = {'eigh': lambda h:h.eigh(), 'eigsh': lambda h: h.eigsh(k=Ne,which=which)}
    Nblock_lookup = {'kblock':L, 'pblock':2}
    Eigenvalue_gen_lookup = {'kblock': lambda q:q, 'pblock': lambda q : 2 * (1-q) -1}
    if method not in methods.keys():
        raise ValueError("Not a valid diagonalization method")
    if blockname not in Nblock_lookup.keys():
        raise ValueError("Invalid symmetry type.")
    eigs=dict()
    get_eigs=methods[method]
    #this is the number of symmetry blocks
    Nblock = Nblock_lookup[blockname]
    #this is the function which, given a block label q =0, ..., Nblock-1, returns the eigenvalue of that block as
    #recognized by quspin.
    Eigenvalue_gen = Eigenvalue_gen_lookup[blockname]    
    #this generates the symmetry dictionary (really just one k-v pair) for quspin
    def get_blockdict(k):
        blockdict = dict()
        blockdict[blockname] = k
        return blockdict       
        
    for q in range(Nblock):
        k=Eigenvalue_gen(q)
        blockdict = get_blockdict(k)
        if other_symm_dict is not None:
            for k in other_symm_dict.keys():
                blockdict[k] = other_symm_dict[k]
        basis =gen_basis(L,**blockdict)
        h=hamiltonian(static,[],basis=basis,dtype=dtype)
        e,s=get_eigs(h)
        p=basis.get_proj(dtype)
        eigs[q] = (e,s,p)
    return eigs


def diagonalize_kblock(static, gen_basis, L,dtype=np.complex128,method='eigh',Ne=None,which='SA', other_symm_dict=None):
    """diagonalize static hamiltonian by projecting onto k-blocks.
        Returns: a dictionary which maps
            k --> (e, s,proj)
        where k is the momentum value (=0, ..., L-1) and e,s are the eigenvalues and vectors of the corresponding hamiltonian block. The states are written in the symmetry-reduced basis. proj is the change-of-basis matrix which, applied to those states, 
        returns them expressed in the standard basis of the full space.
        
        static: static parameter list for the full hamiltonian
        gen_basis: produces basis, given L,kblock as inputs
        L: chain length        
        method: how to obtain the eigenstuff (i.e. exact solver or sparse)
        Ne: if using the sparse solver, how many eigenpairs to compute
        which: if using the sparse solver, where to draw the pairs from the spectrum. See Quspin docs for eigsh()
       """
    blockname = 'kblock'
    return diagonalize_block(static, gen_basis, L, blockname,dtype=dtype,method=method,Ne=Ne,which=which,other_symm_dict=other_symm_dict)

def diagonalize_pblock(static, gen_basis, L, dtype=np.complex128, method='eigh', Ne=None, which='SA'):
    blockname='pblock'
    return diagonalize_block(static, gen_basis, L, blockname,dtype=dtype,method=method,Ne=Ne,which=which)


def diagonalize_block_2d(static, gen_basis, Lx,Ly, blockname,dtype=np.complex128,method='eigh',Ne=None,which='SA'):
    """diagonalize static 2d hamiltonian by projecting onto blocks.
        Returns: a dictionary which maps
            (qx,qy) --> (e, s,proj)
        where (qx,qy) are the x/y block labels  and e,s are the eigenvalues and vectors of the corresponding hamiltonian block. The states are written in the symmetry-reduced basis. proj is the change-of-basis matrix which, applied to those states, 
        returns them expressed in the standard basis of the full space.
        
        static: static parameter list for the full hamiltonian
        gen_basis: produces basis, given Lx,Ly, blockdict as inputs
        Lx,Ly: array dimensions.        
        blockname: string name of the symmetry type. Allowed values:
               kblock, pblock
        The blocks are defined by x and y eigenvalues of the same symmetry type.
        method: how to obtain the eigenstuff (i.e. exact solver or sparse)
        Ne: if using the sparse solver, how many eigenpairs to compute
        which: if using the sparse solver, where to draw the pairs from the spectrum. See Quspin docs for eigsh()
       """
    methods = {'eigh': lambda h:h.eigh(), 'eigsh': lambda h: h.eigsh(k=Ne,which=which)}
    Nblock_lookup = {'kblock':(Lx, Ly), 'pblock':(2,2)}
    Eigenvalue_gen_lookup = {'kblock': lambda q:q, 'pblock': lambda q : q}
    
    def get_blocknames(block_type):
        if block_type == 'kblock':
            return ('kxblock', 'kyblock')
        if block_type == 'pblock':
            return ('pxblock', 'pyblock')
        raise ValueError("invalid block type")
        
    if method not in methods.keys():
        raise ValueError("Not a valid diagonalization method")
    if blockname not in Nblock_lookup.keys():
        raise ValueError("Invalid symmetry type.")
    eigs=dict()
    get_eigs=methods[method]
    #this is the number of symmetry blocks
    Nblock = Nblock_lookup[blockname]
    #this is the function which, given a block label q =0, ..., Nblock-1, returns the eigenvalue of that block as
    #recognized by quspin.
    Eigenvalue_gen = Eigenvalue_gen_lookup[blockname]    
    namex, namey = get_blocknames(blockname)
    #this generates the symmetry dictionary (really just one k-v pair) for quspin
    def get_blockdict(kx,ky):
        blockdict = dict()
        blockdict[namex] = kx
        blockdict[namey] = ky
        return blockdict       
        
    for qx in range(Nblock[0]):
        kx=Eigenvalue_gen(qx)
        for qy in range(Nblock[1]):
            
            ky=Eigenvalue_gen(qy)  
            print(kx,ky)
            blockdict = get_blockdict(kx,ky)   
            print(blockdict)
            basis =gen_basis(Lx,Ly,**blockdict)
            h=hamiltonian(static,[],basis=basis,dtype=dtype)
            e,s=get_eigs(h)
            p=basis.get_proj(dtype)
            eigs[(qx,qy)] = (e,s,p)
    return eigs


def get_all(eigdict):
    """ eigdict = dictionary of eigenstuff blocked by k. Return the collection of all eigenvalues and vectors.
            Note that the eigenvalues won't be in ascending order, but will be in the same order as the states."""
    e=[]
    s=[]

    for k in eigdict.keys():
        evals,states,proj=eigdict[k]
        e += list(evals)
        s += [ proj.dot(states[:,i]) for i in range(len(evals))]
    return np.array(e),s


def spectrum_by_k(eigdict,N=2,symm=True, fig=None, ax=None, style='kx', label=''):
    """ Plot the spectrum of a hamiltonian which has been k-diagonalized.
        eigdict = dictionary mapping k --> (e,s,proj)
        N = number of eigvals per k to plot.
        symm: if True, use a symmetric brillouin zone"""
    import matplotlib.pyplot as plt
    if ax is None:
        fig,ax=plt.subplots()

    L=len(eigdict.keys())
    def get_kBZ(k):
        if symm:
            return  k if k <= L//2 else k-L
        else:
            return k
    for k in range(L):
        kBZ= get_kBZ(k)
        e=eigdict[k][0]
        e=e[:N]
        x=kBZ * np.ones(N)
        if k==0:
            ax.plot(x, e, style,label=label)
        else:
            ax.plot(x, e, style)
    ax.set_xlabel('k')
    return fig,ax


class BlockedHamiltonian(object):
    """stores the results of diagonalizing a hamilonian by symmetry sectors."""
    
    def __init__(self, eigdict):
        """eigdict = a dictionary mapping symmetry eigenvalues to (e, s, proj) tuples
                e = list of evals
                s = list of states in the symmetry-reduced basis
                proj = matrix which takes symmetry-reduced states to the full hilbert space"""
        self.eigdict=eigdict
        self.energies, self.states_all = get_all(self.eigdict)
    
    def get_symm_evals(self,k):
        if k not in self.eigdict.keys():
            raise ValueError("not a valid symmetry sector")
        return self.eigdict[k][0]
    
    def evals(self,k=None,sort=True):
        """ return the eigenvalues corresponding to a particular symmetry sector k.
            If none is specified, returns them all in ascending order (unless sort=False)."""
        if k is None:
            if sort:
                return np.sort(self.energies)
            return self.energies
        return self.get_symm_evals(k)
    
    def states(self, k):
        return self.eigdict[k][1]
    def proj(self, k):
        return self.eigdict[k][2]
    
    
    
    
def make_KBlockedHamiltonian(static, gen_basis, L,dtype=np.complex128,method='eigh',Ne=None,which='SA',other_symm_dict=None):
    """ returns block-hamiltonian object, using translation symmetry."""
    eigdict = diagonalize_kblock(static, gen_basis, L, dtype=dtype,method=method,Ne=Ne,which=which,other_symm_dict=other_symm_dict)
    return BlockedHamiltonian(eigdict)
    
def make_PBlockedHamiltonian(static, gen_basis, L,dtype=np.complex128,method='eigh',Ne=None,which='SA'):
    """ returns block-hamiltonian object, using parity symmetry."""
    eigdict = diagonalize_pblock(static, gen_basis, L, dtype=dtype,method=method,Ne=Ne,which=which)
    return BlockedHamiltonian(eigdict)

def get_PDF(x,nbins=100, xlim=None):
    """ Given samples x, returns effective PDF defined by the bin number nbin, as well as
    an array of length nbin giving the left edges of the bins.
    Bins are of uniform size
    
    If xlim is specified: histogram will be computed only within that range; PDF will be renormalized so as to agree with the PDF over the full range."""
    N=len(x)
    if xlim is not None:
        ind_rel =(x<=xlim[1])*(x>=xlim[0]) 
        xrel = x[ind_rel]
    else:
        xrel=x
    h,bin_edges = np.histogram(xrel,bins=nbins)
    Nrel=len(xrel)
    dx = np.diff(bin_edges)[0]
    ptot = Nrel /N
    f = ptot* h/(N*dx)
    return f,bin_edges[:-1]


def barplot(xplot,f,label=''):
    """Make a barplot of a PDF. The bars are given a label (for plt.legend()) if supplied.
        Returns: figure and axes"""
    width=np.diff(xplot)[0]
    bottom=np.min(f)- .1 * np.abs(np.min(f))
    fig,ax = plt.subplots()
    ax.bar(xplot,f-bottom,width=width,align='edge',bottom=bottom,label=label)
    ax.set_ylim([bottom, 1.1*np.max(f)])
    return fig,ax

    
def mean(f, pdf, dx):
    """ dx = bin width"""
    return np.sum(f * pdf * dx )

def mean_x(pdf, xl):
    dx = np.diff(xl)[0]
    xc = xl + dx/2.0
    return mean(xc, pdf, dx)
    











