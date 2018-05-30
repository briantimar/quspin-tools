import numpy as np
import matplotlib.pyplot as plt
from quspin.operators import hamiltonian
from quspin.basis import spin_basis_1d, boson_basis_1d, boson_basis_general
from tools import identity, is_herm, check_bc, vprint, get_hcb_basis
from tools import get_site_coupling
from tools import is_hcb_basis

class PDict(object):
    """A container for hamiltonian parameters and the like. Doesn't do much.
        If a decay rate gamma is requested and none has been assigned, returns None (should be default decay value for Hamiltonian constructors)."""
    
    def __init__(self, **pdict):
        self.pdict = pdict
        
    def __getitem__(self, s):
        return self.pdict.__getitem__(s)
        
    def __setitem__(self, s, x):
        self.pdict.__setitem__(s,x)

    def set_param(self,p,x):
        if p not in self.pdict.keys():
            print("Adding new param ", p)
        self[p] = x
    
    @property
    def Delta(self):
        return self['Delta']
    @property
    def Omega(self):
        return self['Omega']
    @property
    def Vnn(self):
        return self['Vnn']
    @property
    def bc(self):
        return self['bc']
    @property
    def dmax(self):
        return self['dmax']
    @property
    def gamma(self):
        try:
            return self['gamma']
        except KeyError:
            return None
    @property
    def ktrunc(self):
        return self['ktrunc']
    

    def set_Delta(self, x):
        self.set_param('Delta', x)
    def set_Omega(self, x):
        self.set_param('Omega', x)
    def set_Vnn(self, x):
        self.set_param('Vnn', x)
    def set_bc(self, x):
        self.set_param('bc', x)
    def set_dmax(self, x):
        self.set_param('dmax', x)
    def set_gamma(self, x):
        self.set_param('gamma', x)
    def set_ktrunc(self, x):
        self.set_param('ktrunc',x)

    def __repr__(self):
        return self.pdict.__repr__()
    def pop(self, p):
        """ Returns the PDict with key p removed"""
        d= self.pdict.copy()
        try:
            if isinstance(p, str):
                d.pop(p)
            else:
                for key in p:
                    d.pop(key)
        except TypeError:
            print("Provide a string or list of strings")
        except KeyError:
            print("Nothing to remove")
        return PDict(**d)
    
    def unpack_2d_r6(self):
        return self.Delta, self.Omega, self.Vnn, self.dmax, self.bc, self.gamma
    
    def unpack_1d_r6(self):
        try:
            return self.Delta, self.Omega, self.Vnn, self.ktrunc,self.bc,self.gamma
        except AttributeError as e:
            print("Pdict does not have required params")
            raise e
    
    @property
    def infostr(self):
        from EDIO import get_infostr
        return get_infostr(self.pop('dtype').pdict)
    
    @property
    def header(self):
        from EDIO import make_header_str
        return make_header_str(self.pdict)


def get_nth_order_coupling(J, L,n, bc):
    """ returns list of coupling specs
             J, i, i+1, 
             for nth-nearest-neighbor interaction terms. n indexes from 1 -- so n=1 means nearest neighbor, n=2 means next-nearest, etc. 
        Assumes every site has such a coupling"""
    if bc == 'periodic':
        return [ [J, i, (i+n)%L] for i in range(L)]
    elif bc=='open':
        return [ [J, i, i+n] for i in range(L-n)]
        
def get_nn_coupling(J, L, bc):
    """ nearest neigbors"""
    return get_nth_order_coupling(J, L, 1, bc)


def make_ising_tf(Jz, Jx, Jzz, L, bc='open'):      
    n=1
    coup_list = [Jzz]
    return make_nnZ_truncated(n, Jz, Jx, coup_list, bc)


def get_nnZ_couplings(n, Jz, Jx, coup_list, L, bc):
    """ returns tuple [z_list, x_list, zz_list] of lists of coupling terms, that can be
    fed to the quspin hamiltonian constructor."""
    
    coupling_z = get_site_coupling(Jz, L)
    coupling_x = get_site_coupling(Jx, L)
    coupling_zz = []
    for i in range(n):
        coupling_zz = coupling_zz + get_nth_order_coupling(coup_list[i], L, i+1, bc)
    return (coupling_z, coupling_x, coupling_zz)


def get_nnZ_static_spin(n, Jz, Jx, L, coup_list, bc='periodic', pauli=True):
    """ Returns just the static parameter list for the nth-nearest neigbor hamiltonian -- see below.
    Assumes a hamiltonian written in the spin basis"""
    if bc=='open':
        raise TypeError("Warning! For open boundary conditions, this function does not implement edge correction terms -- the Z coupling to each site is uniform.""")
    
    coupling_z, coupling_x, coupling_zz = get_nnZ_couplings(n, Jz, Jx, coup_list, L, bc)
    
    static = [ ["z", coupling_z], ["x", coupling_x], ["zz", coupling_zz] ]
    return static

def get_nnZ_static_hcb(n, Jn, Jx, L, coup_list, bc='periodic'):
    """ Returns the static parameter list for nth-nearest-neighbor hamiltonian, written in hcb basis, 
         where interactions density-density"""

    coupling_n, coupling_x, coupling_nn = get_nnZ_couplings(n, Jn, Jx, coup_list, L, bc)   
    static = [["n", coupling_n], ["+", coupling_x], ["-", coupling_x], ["nn", coupling_nn] ]
    return static

def make_nnZ(n, Jn, Jx, coup_list, L, basis=None,bc='periodic', dtype=np.float64):
    """ return uniform 1d hamiltonian with interactions up to nth-nearest-neighbors. 
        coup_list = list of couplings [V1, ..., Vn] such that each pair of sites (i, i+n) has an interaction term V_n n_i n_i+n, where n are the on-site density operators. The single-site hamiltonian is
        sum_i Jn n_i + Jx (b_i + b_i^dag)
       
        """
    check_bc(bc)
    if basis is None:
        basis = get_hcb_basis(L)
    if not isinstance(basis, boson_basis_1d):
        raise TypeError("this function is only implemented for hard-core bosons")
    static = get_nnZ_static_hcb(n, Jn, Jx, L, coup_list, bc=bc)
    dynamic = []
    return hamiltonian(static, dynamic, basis=basis, dtype=np.float64)
        
def get_fss_static(Delta, Omega, V1, V2,L,bc='periodic'):
    """ static parameter list for the FSS model
        """
    Jn = -Delta
    Jx = - Omega / 2    
    coup_list = [V1, V2]
    return get_nnZ_static_hcb(2, Jn, Jx, L, coup_list, bc=bc)
    
def make_fss(Delta, Omega, V1, V2, basis, bc='periodic', dtype=np.float64):
    """ constructs fendley-sachdev-type 1d hamiltonian written in terms of the laser parameters Delta, Omega """
    if not isinstance(basis, boson_basis_1d):
        raise TypeError("this function is only implemented for hard-core bosons")
    L=basis.L
    static = get_fss_static(Delta, Omega, V1, V2, L, bc=bc)
    dynamic = []
    return hamiltonian(static, dynamic,basis=basis,dtype=dtype)    
    
def make_2nn(Jn, Jx, V1, V2, L, basis=None, bc='periodic'):
    """ spin-1/2 hamiltonian with single-site Z and X couplings $J_Z$ and $J_X$ and 
    ZZ interactions up to second nearest neighbors.
    
    If basis is provided (i.e. to specify k sector) that will be used. 
    Otherwise spin_basis_1d will be generated with no symmetries applied."""
    raise TypeError("this function is deprecated")
    return make_nnZ(2, Jz, Jx, [V1, V2], L, basis=basis, bc=bc, pauli=pauli)


#all of these are in the hcb basis
def get_ryd_coupling_x(Omega, L,verbose=False):
    return get_site_coupling(-Omega/2.0, L,verbose=verbose)
def get_ryd_coupling_n(Delta, L,verbose=False):
    return get_site_coupling(-Delta, L,verbose=verbose)
def get_decay_coupling(gamma,L,verbose=False):
    """ Returns (decay_coup_n, decay_coup_I), two single-site coupling lists which correspond to the n and I operators, respectively, in the hcb basis. 
        gamma = a list of decay rates, for upper and lower states respectively."""
    decay_coup_n = get_site_coupling(-1j*(gamma[0]-gamma[1])/2.0, L)
    decay_coup_I = get_site_coupling(-1j*gamma[1]/2.0, L)
    return decay_coup_n,decay_coup_I

def get_decay_static(gamma,L,verbose=False):
    decay_coup_n, decay_coup_I = get_decay_coupling(gamma,L,verbose=verbose)
    static = []
    static.append(["n", decay_coup_n])
    static.append(["I", decay_coup_I])
    return static

def get_coupling_from_matrix(L, M,verbose=False):
    """ Given hermitian matrix M, return list of coupling terms [[Mij, i, j]] defined by the off-diagonal elements of M.
    L = number of sites"""
    if not isinstance(M,np.ndarray):
        vprint(verbose,"Expecting a numpy array")
    if not is_herm(M):
        raise ValueError("Coupling matrix must be hermitian")
    coupling = []
    if M.shape != (L,L):
        raise ValueError("Coupling matrix should be LxL")
    for i in range(L):
        for j in range(i+1,L):
            coupling.append([M[i,j], i, j] )
    return coupling

def get_power_interaction(V0, alpha):
    """Returns function f which computes power-law interaction, 
    f(r) = V0 / |r|^alpha"""
    def f(r):
        return V0 / np.power(np.abs(r),alpha)
    return f

def get_r6_interaction(V0):
    """A van-der-waals type rydberg interaction. V0 is the interaction strength at unit (lattice) spacing."""
    return get_power_interaction(V0,6.0)

def get_r6_dressed_interaction(V0,Rc):
    """A dressed-state rydberg interaction, with scale Rc. Saturates to V0 at r=0, and decays like r^6 at large r."""
    def f(r):
        return V0 / (np.power(np.abs(r/Rc),6.0) + 1)
    return f

def get_1d_r_coupling_nn(f, L, ktrunc,bc,verbose=False):
    """ Return coupling list [[J, i, i+k], ...] for n-n coupling in a boson model (n being the occupation operator)
        f = some function of r, the inter-site distance, which determines the coupling between sites i and i+r (independent of index i). The form of the corresponding Hamiltonian is
        H = sum (pairs i,j) f(|i-j|) ni nj
        the sum running over all distinct pairs.
        
        If an LxL hermitian matrix is input as f, the off-diagonal elements of that matrix will be used directly to construct the hamiltonian, ie
        H = sum (pairs ij) f_ij ni nj
        In that case, the truncation range will be ignored.
        
        ktrunc = defines the cutoff range for the potential f: sites separated by more than ktrunc will not interact (will not be included in the coupling list)
        """
    check_bc(bc)        
    if isinstance(f,np.ndarray):
        vprint("Direct coupling matrix provided. Truncation length ignored.")
        coupling_nn = get_coupling_from_matrix(L,f,verbose=verbose)
    else:
        vprint(verbose,"Scalar interaction provided. f(r) matrix will be constructed")
        #max number of couplings per site
        if bc=='open':
            nc=min(ktrunc,L-1)
        else:
            nc=min(ktrunc, L//2)
        
        coup_list =np.empty(nc)
        if ktrunc >=L:
            print("Warning--interactions longer than L are discarded.")
        for i in range(1,nc+1):
            v = f(i)
            if i==L/2 and bc=='periodic': #this bond will be overcounted in the final sum
                v = v/2.0
            coup_list[i-1]=v

        coupling_nn = []
        for i in range(nc):
            coupling_nn = coupling_nn + get_nth_order_coupling(coup_list[i], L, i+1,bc) 
    return coupling_nn
        
def get_r6_1d_dressed_coupling_nn(V0,rc, L,ktrunc,bc,verbose=False):
    """ Return coupling list for the following "dressed-state" n-n interaction:
        V(r) = V0 / (r/r)^6 + 1
        """
    check_bc(bc)
    f = get_r6_dressed_interaction(V0,rc)
    return get_1d_r_coupling_nn(f,L,ktrunc,bc,verbose=verbose)    
    
        

def get_r6_1d_coupling_nn(Vnn, L, ktrunc, bc,verbose=False):
    """returns coupling list for n-n coupling up to kth nearest neighbors.
        Vnn = nearest-neighbor strength, if input as scalar.
            Vnn can also be a hermitian matrix in which case the i,j element gives the coupling Vij directly. Note there's no 1/2, only half the matrix is used. Diagonal elements are not used.
        L=number of sites
        k = order of truncated interaction (k=1 means nearest neighbor only)
        """
#    check_bc(bc)
#    if isinstance(Vnn,np.ndarray):
#        vprint("Direct coupling matrix provided. Truncation length ignored.")
#        coupling_nn = get_coupling_from_matrix(L,Vnn,verbose=verbose)
#    else:
#        vprint(verbose,"Scalar interaction provided. r^-6 matrix will be constructed")
#        nc=min(ktrunc,L-1)
#        coup_list =np.empty(nc)
#        if ktrunc >=L:
#            print("Warning--interactions longer than L are discarded.")
#        for i in range(1,nc+1):
#            v = Vnn / np.power(i,6)
#            if i==L/2 and bc=='periodic': #there's only one bond between the 2 sites in this case
#                v = v/2.0
#            if (i>L//2) and bc=='periodic':
#                ## these bonds have already been counted
#                v=0
#            coup_list[i-1]=v
#        coupling_nn = []
#        for i in range(nc):
#            coupling_nn = coupling_nn + get_nth_order_coupling(coup_list[i], L, i+1,bc) 
#            
#    return coupling_nn
    if isinstance(Vnn, np.ndarray):
        f=Vnn
    else:
        f = get_r6_interaction(Vnn)
    return get_1d_r_coupling_nn(f,L,ktrunc,bc,verbose=verbose)


def get_r6_1d_static_nn(Vnn,L,ktrunc,bc,verbose=False):
    """ Returns the static list which just involves n-n interactions """
    coupling_nn=get_r6_1d_coupling_nn(Vnn,L,ktrunc,bc,verbose=verbose)
    return [["nn", coupling_nn]]

def get_r6_1d_dressed_static_nn(V0, Rc,L, ktrunc,bc,verbose=False):
    coupling_nn = get_r6_1d_dressed_coupling_nn(V0,Rc, L,ktrunc,bc,verbose=verbose)
    return [["nn", coupling_nn]]

def get_r6_1d_static(Delta, Omega, Vnn, L, ktrunc,  gamma=None, bc='periodic',verbose=False):
    """Returns just the static parameter list for this hamiltonian. 
        Gamma, if provided, should be a list of decay rates (inverse lifetimes) for the upper and lower states respectively."""

    coupling_x = get_ryd_coupling_x(Omega, L,verbose=verbose)
    coupling_n = get_ryd_coupling_n(Delta, L,verbose=verbose)  
    static_nn = get_r6_1d_static_nn(Vnn, L,ktrunc, bc,verbose=verbose)
    #+ and - are the creation and annihilation ops
    static = [ ["n", coupling_n], ["+", coupling_x], ["-", coupling_x]]
    static += static_nn
    if gamma is not None:
        decay_static = get_decay_static(gamma,L,verbose=verbose)
        static+= decay_static
    return static


def get_r6_1d_dressed_static(Delta, Omega, V0, Rc,L, ktrunc,gamma, bc,verbose=False):
    """ Static parameter list for the 1d dressed hamiltonian."""
    coupling_x = get_ryd_coupling_x(Omega, L,verbose=verbose)
    coupling_n = get_ryd_coupling_n(Delta, L,verbose=verbose) 
    static_nn = get_r6_1d_dressed_static_nn(V0,Rc,L,ktrunc,bc,verbose=verbose)
    static = [ ["n", coupling_n], ["+", coupling_x], ["-", coupling_x]]
    static += static_nn
    if gamma is not None:
        decay_static = get_decay_static(gamma,L,verbose=verbose)
        static+= decay_static
    return static
    
def make_1d_r6_pdict(pdict,basis,dtype=np.float64):
    Delta, Omega, Vnn, ktrunc,bc,gamma = pdict.unpack_1d_r6()
    return make_r6_1d(Delta, Omega, Vnn, ktrunc, basis, gamma=gamma,bc=bc,dtype=dtype)

def make_r6_1d(Delta, Omega, V,  ktrunc, basis, gamma=None, bc='open', dtype=np.float64,verbose=False):
    """returns the hamiltonian operator for a 1d rydberg chain with laser parameters as input. 
        HCB basis.
        
        H = -sum_i Delta_i n_i -  sum_i (omega_i)/2) x_i + sum_(ij) V_ij n_i n_j
        where
        V_ij = C / a^6 (i-j)^6
        for a 1d lattice.
        
        
        Inputs:
            Delta, Omega -- n and x couplings respectively
            V: the nearest-neighbor interaction (|i-j|=1)
            If Delta and Omega are scalars, a uniform hamiltonian is generated with delta_i= delta, etc. If they are iterables, the ith element will specify the coupling (eg delta_i) at the ith site.
            
            If V is input as a single scalar,
                Vij= V / |i-j|^6
                
            V may also be input as a hermitian matrix, in which case the i,j element will specify Vij. Note that in that case you'll have to put in power-law behavior by hand.
            
            gamma (optional, default None): a list of two single-site amplitude-decay rates, specifying upper and lower decay rates respectively.
            If a list of gamma values is specifed, a nonhermitian term
            -1j * (gamma[0]/2) * sum_i n_i - 1j * (gamma[1]/2) * sum_i (1-n_i)
            will be added to the hamiltonian
            
            
            ktrunc = integer, interaction truncation. H will include 1/r^6 interactions of the form shown above up to kth-nearest-neighbors
            
        If verbose=True: some info on hamiltonian construction is given.

    """
    check_bc(bc)
    L=basis.L
    if not isinstance(basis, boson_basis_1d):
        raise TypeError("invalid basis type")
    if not (gamma is None):
        try:
            if len(gamma)!=2:
                raise ValueError("List of decay rates should have length 2.")
        except TypeError:
                raise TypeError("Gamma should be a list")
    
    check_herm=True
    if gamma is not None:
        vprint(verbose,"Decay provided. herm=false, dtype=complex")
        check_herm= False
        dtype=np.complex128
        
    static = get_r6_1d_static(Delta, Omega, V, L,ktrunc,gamma, bc=bc,verbose=verbose)    
    dynamic = []
    return hamiltonian(static, dynamic, basis=basis,dtype=dtype,check_herm=check_herm)
   
    

def make_dressed_r6_1d(Delta, Omega, V0, Rc, ktrunc,basis,gamma=None,bc='periodic',dtype=np.float64,verbose=False):
    """ Returns quspin hamiltonian:
        
        H = -sum_i Delta_i n_i -  sum_i (omega_i)/2) x_i + sum_(pairs ij) V_ij n_i n_j
        where
        V_ij =  V0 / ( (r / Rc)^6 + 1)
        for a 1d lattice.
     gamma (optional, default None): a list of two single-site amplitude-decay rates, specifying upper and lower decay rates respectively.
                If a list of gamma values is specifed, a nonhermitian term
                -1j * (gamma[0]/2) * sum_i n_i - 1j * (gamma[1]/2) * sum_i (1-n_i)
                will be added to the hamiltonian
                
                
                ktrunc = integer, interaction truncation. H will include 1/r^6 interactions of the form shown above up to kth-nearest-neighbors
                
            If verbose=True: some info on hamiltonian construction is given.
                """
    check_bc(bc)
    L=basis.L
    if not isinstance(basis, boson_basis_1d):
        raise TypeError("invalid basis type")
    if not (gamma is None):
        try:
            if len(gamma)!=2:
                raise ValueError("List of decay rates should have length 2.")
        except TypeError:
                raise TypeError("Gamma should be a list")
    check_herm=True
    if gamma is not None:
        vprint(verbose,"Decay provided. herm=false, dtype=complex")
        check_herm= False
        dtype=np.complex128
    static = get_r6_1d_dressed_static(Delta, Omega, V0, Rc, L, ktrunc,gamma, bc,verbose=verbose)
    dynamic=[]
    return hamiltonian(static, dynamic, basis=basis, dtype=dtype,check_herm=check_herm)
                

#############################################################
    ##### stuff involving ladder hamiltonians
    ######################################################
    
def get_all_pairs(s):
    """all pairs of distinct members of list s"""
    pairs = []
    for i in range(len(s)):
        pairs = pairs + [ [s[i], s[i+k]] for k in range(1, len(s)-i)]
    return pairs

def get_site_labels_kladder(k, L):
    """ Returns 1d array listing the sites of the ladder in thread order. """
    return np.arange(k*L)

def get_transverse_shifted_site_labels_kladder(k, L, l):
    """ Returns 1d array listing the result of applying a transverse shift on rung l to the list of sites, in thread order (i.e. the kth element is the location of site k after the shift is applied) """
    sites = get_site_labels_kladder(k, L)
    indicators = ((sites - (l+1))%L) // (L-1)   #nonzero at the sites which get translated
    sites_translated = (sites + L * indicators) % (k *L)
    return sites_translated

def make_kladder_transverse_symm_dict(k, L, qlist):
    """ Return dictionary which has keys 'rungj', j = 1, .... L. The
    corresponding elements are (T_j, q_j), T_j where k*L-length 1d arrays which list the result of applying transverse translation on rung j, in thread order.
   q_j are the corresponding quantum numbers under this transformation."""
    symm = dict()
    for j in range(L):
        symm['rung{0}'.format(j)] = (get_transverse_shifted_site_labels_kladder(k, L, j), qlist[j] )
    return symm

def get_half_block_sites_kladder(k, L):
    """Return list of sites which constitute the left half of a k-ladder"""
    s = get_site_labels_kladder(k, L)
    return list(filter(lambda x:  x%L < L//2, s))



def make_kladder_symmetric(Delta, Omega, V1, V2, k,L,basis=None, bc='periodic',dtype=np.float64):
    """ Returns the following hamiltonian:
        H = - Delta sum_i n_i - (Omega/2) sum_i X_i + V1 sum_e (nn)_e
        where the last term joins any two sites which are the same or adjacent rungs.
        
        Assumes periodic boundary conditions (in the sense that the last rung joins to the first)
        L = number of rungs
        k = width (number of sites) of each rung
        
      Site ordering: The index i of the individual sites always increases left-to-right. At the end of the ladder, it moves down one step and keeps increasing.
      For a example, for a 2-ladder of length 8 the sites are as follows:
          0  1  2  3  4  5  6  7
          8  9  10 11 12 13 14 15
          
     This is 'thread' ordering, as opposed to 'snake' ordering where sites i, i+1 are always adjacent.  
     
        
    """    

    check_bc(bc)
    if bc=='open':
        raise TypeError("Ladder with open BC's is not implemented!")
    if basis is None:
        print("Generating basis")
        basis = get_hcb_basis(k*L)
    if not (isinstance(basis, boson_basis_1d) or isinstance(basis, boson_basis_general)):
        raise TypeError("Only implemented for hcb basis")
  
   
    #special case, the next-nearest will overcount by factor 2 
    if L==4:
        V2 = V2 / 2.0
    Jx = - (Omega / 2.0)
    Jn = -Delta
    Jnn_nn1 = V1
    Jnn_nn2 = V2
    #total number of sites
    N=k*L
    coupling_x = get_site_coupling(Jx, N)
    coupling_n = get_site_coupling(Jn, N)
    
    #matrix which stores the thread-order labels
    site_labels = np.empty((k, L), dtype=int)
    for i in range(k):
        site_labels[i, :] = np.array(range(L)) + i * L
    
    #all terms proportional to V1, ie nearest-neighbor sites
    coupling_nn_nn1 = []
    
    for i in range(L):
        #this adds the interactions between sites on the same rung
        prs = get_all_pairs(site_labels[:,i])
        coupling_nn_nn1 += [[Jnn_nn1] + p for p in prs]
        #this adds the interactions between sites on neighboring rungs
        for j in range(k):
            coupling_nn_nn1 += [ [Jnn_nn1, site_labels[j, i], site_labels[m, (i+1)%L] ] for m in range(k) ]
     
    #all terms proportional to V2, i.e. next-nearest-neighbor rungs   
    coupling_nn_nn2 = []
    for i in range(L):
        for j in range(k):
            coupling_nn_nn2 += [ [Jnn_nn2, site_labels[j, i], site_labels[m, (i+2)%L] ] for m in range(k) ]

    static = [ ["n", coupling_n], ["+", coupling_x], ["-", coupling_x], ["nn", coupling_nn_nn1 + coupling_nn_nn2]]
    dynamic = []

    return hamiltonian(static, dynamic, basis=basis,dtype=dtype)



def make_kladder_symmetric_SPIN(Delta, Omega, V1, V2, k,L,basis=None, bc='periodic'):
    """ Returns the following hamiltonian:
        H = - Delta sum_i n_i - (Omega/2) sum_i X_i + V1 sum_e (nn)_e
        where the last term joins any two sites which are the same or adjacent rungs.
        
        Assumes periodic boundary conditions (in the sense that the last rung joins to the first)
        L = number of rungs
        k = width (number of sites) of each rung
        
      Site ordering: The index i of the individual sites always increases left-to-right. At the end of the ladder, it moves down one step and keeps increasing.
      For a example, for a 2-ladder of length 8 the sites are as follows:
          0  1  2  3  4  5  6  7
          8  9  10 11 12 13 14 15
          
     This is 'thread' ordering, as opposed to 'snake' ordering where sites i, i+1 are always adjacent.  
     
        
    """    
    raise TypeError("deprecated")
    check_bc(bc)
    if bc=='open':
        raise TypeError("Ladder with open BC's is not implemented!")
    if basis is None:
        print("Generating basis")
        basis = spin_basis_1d(k*L, pauli=True)
  
    #coupling strengths in the Pauli basis
    #special case, the next-nearest will overcount by factor 2 
    if L==4:
        V2 = V2 / 2.0
    Jx_pauli = - (Omega / 2.0)
    Jz_pauli = - Delta/2.0 + (V1/4.0) * (3*k-1) + (V2/4.0) * 2*k
    Jzz_pauli_1nn =  V1/4.0   
    Jzz_pauli_2nn = V2/4.0
    
    N=k*L
    coupling_x = get_site_coupling(Jx_pauli, N)
    coupling_z = get_site_coupling(Jz_pauli, N)
    
    #matrix which stores the thread-order labels
    site_labels = np.empty((k, L), dtype=int)
    for i in range(k):
        site_labels[i, :] = np.array(range(L)) + i * L
    
    #all terms proportional to V1, ie nearest-neighbor sites
    coupling_zz_1nn = []
    
    for i in range(L):
        #this adds the interactions between sites on the same rung
        prs = get_all_pairs(site_labels[:,i])
        coupling_zz_1nn += [[Jzz_pauli_1nn] + p for p in prs]
        #this adds the interactions between sites on neighboring rungs
        for j in range(k):
            coupling_zz_1nn += [ [Jzz_pauli_1nn, site_labels[j, i], site_labels[m, (i+1)%L] ] for m in range(k) ]
     
    #all terms proportional to V2, i.e. next-nearest-neighbor rungs   
    coupling_zz_2nn = []
    for i in range(L):
        for j in range(k):
            coupling_zz_2nn += [ [Jzz_pauli_2nn, site_labels[j, i], site_labels[m, (i+2)%L] ] for m in range(k) ]

    static = [ ["z", coupling_z], ["x", coupling_x], ["zz", coupling_zz_1nn + coupling_zz_2nn]]
    dynamic = []

    return hamiltonian(static, dynamic, basis=basis)

def make_1d_TFI_static(J, Omega,L, bc='periodic',dtype=np.float64):
    coupling_x = get_site_coupling(-Omega, L)
    coupling_zz = get_nth_order_coupling(-J, L, 1, bc)
    static= [["x", coupling_x], ["zz", coupling_zz]]
    return static

def make_1d_TFI_spin(J, Omega, basis, bc='periodic',dtype=np.float64):
    """Returns a pure transverse-field Ising model in the spin -1/2 basis
    H = -Omega sum(x) - J sum(Z_i Z_(i+1))
    """
    L=basis.L
    static=make_1d_TFI_static(J,Omega,L,bc=bc,dtype=dtype)
    dynamic = []
    return hamiltonian(static, dynamic,basis=basis, dtype=dtype)


### tools for making spin-spin correlators, order parameters...


def sigma(i, alpha, L, basis=None):
    """ Returns the sigma-alpha operator at site i, written out in 
    basis <basis> (if not specified, uses spin_basis_1d with no symmetries applied)
    
    i = site index 0, ..., L-1
    Allowed values for alpha: "x", "y", "z", "+", "-"
    """
    
    if alpha not in SP_TYPES:
        raise TypeError("Invalid Spin index")        
    if basis is None:
        basis = spin_basis_1d(L, pauli=True)
    static = [ [alpha, [[1, i]] ]]
    dynamic = []
    return hamiltonian(static, dynamic, basis=basis)
    

def sumSigma(alpha, sitelist, basis, check_symm=True):
    """ returns the sum of sigma_alpha over sites specified in sitelist.
        Example: sitelist = [0, 2, 4, ...] picks out every other spin
        L = chain length
        If basis is not provided, spin_basis_1d is generated """
    if alpha not in SP_TYPES:
        raise TypeError("Invalid Spin index")        
    
    coupling_alpha = [ [1, i] for i in sitelist]
    static = [[alpha, coupling_alpha]]
    dynamic = []
    return hamiltonian(static, dynamic, basis=basis, check_symm=check_symm)
        



def sumn( basis):
    """ sum of on-site number operators. hcb basis"""
    
    N=basis.N
    coupling_n = [ [1, i] for i in range(N)]
    static = [["n", coupling_n]]
    return hamiltonian(static, [], basis=basis)


def sum_Z2_n_kladder(k, L, basis=None, check_symm=True):
    """Return the sum of the excitations on even-numbered ladder rungs.
        Assumes thread-order
    """
    sites = get_site_labels_kladder(k, L)
    sitelist= list(filter(lambda x: (x %L)%2==0, sites) )
    return  0.5*(sumSigma("z", sitelist,  basis=basis, check_symm=check_symm) + sumSigma("I", sitelist,  basis=basis, check_symm=check_symm))
 
#########################
### for defining 2D geometries
### 'thread-ordering'
def get_sitelist(Lx, Ly):
    """returns array labeling the sites"""
    return np.array(list(range(Lx*Ly)))

def get_x(s, Lx,Ly):
    """ x = sitelist. return array of x-coordinates"""
    return s%Lx

def get_y(s,Lx,Ly):
    """ array of y coordinates"""
    return s//Lx

def get_Tx(Lx,Ly):
    """array defining x-translation"""
    s=get_sitelist(Lx,Ly)
    x, y = get_x(s,Lx, Ly), get_y(s, Lx, Ly)
    return (x+1)%Lx + y * Lx
    

def get_Txdag(Lx,Ly):
    """array defining reverse x-translation"""
    s=get_sitelist(Lx,Ly)
    x, y = get_x(s,Lx, Ly), get_y(s, Lx, Ly)
    return (x-1)%Lx + y * Lx
    

def get_Ty(Lx,Ly):
    """array defining x-translation"""
    s=get_sitelist(Lx,Ly)
    x, y = get_x(s,Lx, Ly), get_y(s, Lx, Ly)
    return ((y+1)%Ly)*Lx + x

def get_Tydag(Lx,Ly):
    """array defining reverse-y translation"""
    s=get_sitelist(Lx,Ly)
    x, y = get_x(s,Lx, Ly), get_y(s, Lx, Ly)
    return ((y-1)%Ly)*Lx + x

def get_Px(Lx,Ly):
    """ defines reflection about x-axis"""
    s=get_sitelist(Lx,Ly)
    x, y = get_x(s,Lx, Ly), get_y(s, Lx, Ly)
    return x + Lx * (Ly-1-y)

def get_Py(Lx,Ly):
    """ defines reflection about y-axis"""
    s=get_sitelist(Lx,Ly)
    x, y = get_x(s,Lx, Ly), get_y(s, Lx, Ly)
    return Lx*y + (Lx-1-x)

class HCBBasis2D(boson_basis_general):
    """ defines a basis for hard-core bosons living on a grid"""
    def __init__(self, Lx, Ly, **symmdict):
        N=Lx*Ly
        sps=2
        self.Lx=Lx
        self.Ly=Ly
        self.Tx = get_Tx(Lx,Ly)
        self.Ty = get_Ty(Lx,Ly)
        boson_basis_general.__init__(self,N,sps=sps,**symmdict)
        
    def __repr__(self):
        return boson_basis_general.__repr__(self)
    
def get_hcb_basis_2d(Lx,Ly,kxblock=None,kyblock=None,pxblock=None,pyblock=None):
    """ Returns a HCBBasis2D of size Lx by Ly, with sites in thread (x-first) order."""
    blocks = dict()
    if kxblock is not None:
        if kxblock not in range(Lx):
            raise ValueError("Invalid kx value.")
        Tx=get_Tx(Lx,Ly)
        kxpair = (Tx,kxblock)
        blocks['kxblock']=kxpair
    if kyblock is not None:
        if kyblock not in range(Ly):
            raise ValueError("Invalid ky value.")
        Ty=get_Ty(Lx,Ly)
        kypair = (Ty,kyblock)
        blocks['kyblock']=kypair
    if pxblock is not None:
        if pxblock not in [0,1]:
            raise ValueError("The quspin basis_general parity eigenvalues are specified by 0 or 1.")
        Px = get_Px(Lx,Ly)
        pxpair = (Px,pxblock)
        blocks['pxblock']=pxpair
    if pyblock is not None:
        if pyblock not in [0,1]:
            raise ValueError("The quspin basis_general parity eigenvalues are specified by 0 or 1.")
        Py = get_Py(Lx,Ly)
        pypair = (Py,pyblock)
        blocks['pyblock']=pypair    
    return HCBBasis2D(Lx,Ly,**blocks)
        
def min_lin_dist_pbc(dx,L):
    dx = np.abs(dx % L)
    return min(dx, L-dx)
    
def get_min_dist(x1,x2,y1,y2,Lx,Ly,bc):
    check_bc(bc)
    if bc=='periodic': 
        dx=min_lin_dist_pbc(x2-x1,Lx)
        dy=min_lin_dist_pbc(y2-y1,Ly)
    else:
        dx = np.abs(x2-x1)
        dy = np.abs(y2-y1)
    return np.sqrt(dx**2 + dy**2)

def get_2d_nn_coupling(f, Lx,Ly,dmax,bc):
    """ list of coupling terms for a 2d lattice; sites interact through radial function f.
         nshell is the number of shells to include around one site"""
        
    sitelist=get_sitelist(Lx,Ly)
    x,y=get_x(sitelist,Lx,Ly), get_y(sitelist,Lx,Ly)
    N=len(sitelist)
    couplings=[]
    for i in range(N):
        for j in range(i+1, N):
            s1,s2 = sitelist[i],sitelist[j]
            x1,x2 = x[i],x[j]
            y1,y2 = y[i],y[j]
            d = get_min_dist(x1,x2,y1,y2, Lx,Ly,bc)
            if d<=dmax:
                couplings += [[ f(d), s1, s2 ]]
    return couplings

def get_2d_radial_static(Delta, Omega, f, dmax, Lx, Ly,bc='periodic', gamma=None):
    N=Lx*Ly
    n_coupling = get_site_coupling(-Delta, N)
    x_coupling = get_site_coupling(-Omega/2,N)
    nn_coupling = get_2d_nn_coupling(f, Lx,Ly,dmax,bc)

    static = [ ['n', n_coupling], ['+', x_coupling], ['-', x_coupling], ['nn', nn_coupling]]
    if gamma is not None:
        static += get_decay_static(gamma, N)
    return static


def make_2d_radial(Delta, Omega, f, dmax, basis, bc='periodic', gamma=None):
    """ Returns a hamiltonian on 2d square lattice (defined by basis)
          h = -Delta sumi ni - (Omega/2) sumi Xi + sumi<j f(rij) ninj
          """
    
    Lx, Ly = basis.Lx, basis.Ly
    static = get_2d_radial_static(Delta, Omega, f,dmax, Lx,Ly,bc=bc, gamma=gamma)
    check_herm = True
    if gamma is not None:
        check_herm = False
    dynamic=[]
    return hamiltonian(static,dynamic,basis=basis,check_herm=check_herm)
    
def get_2d_r6_static(Delta, Omega, Vnn, dmax, Lx, Ly, bc='periodic', gamma=None):
    """Returns just the static list"""
    f=lambda r : Vnn/r**6
    return get_2d_radial_static(Delta, Omega, f, dmax, Lx, Ly, bc=bc, gamma=gamma)

def get_2d_r6_interaction_static(pdict, Lx, Ly):
    """ Returns a static list which only includes interaction terms.
    Used only for setting up dynamic hamiltonians."""
    f = lambda r: pdict.Vnn / (r**6)
    nn_coupling = get_2d_nn_coupling(f, Lx, Ly, pdict.dmax, pdict.bc)
    return [['nn', nn_coupling]]

def make_2d_r6(pdict,basis):
    """hamiltonain for r6 coupling in the plane."""
    Delta, Omega,Vnn,dmax,bc,gamma= pdict.unpack_2d_r6()
    Vint = lambda r: Vnn / r**6
    return make_2d_radial(Delta, Omega, Vint, dmax, basis,bc=bc,gamma=gamma)    
    
def get_summed_interaction_1d(V, ktrunc):
    """ The sum of all interaction links entering any node on a 1d chain"""
    return 2 *np.sum([V(r) for r in range(1,ktrunc+1)] )

def get_summed_r6_interaction_1d(Vnn,ktrunc):
    V = lambda r: Vnn / r**6
    return get_summed_interaction_1d(V,ktrunc)

def get_induced_delta_r6_1d(Vnn, ktrunc):
    """ the value of Delta which will cancel out all longitudinal-field terms"""
    return 0.5 * get_summed_r6_interaction_1d(Vnn,ktrunc)

def get_summed_interaction_2d(V,dmax):
    """ on an infinite 2d graph, the sum of the interaction links touching any one site.
         V = some function of radial distance."""
    Vsum=0
    for x in range(0, int(dmax)+1):
        for y in range(0,int(dmax)+1):
            d = np.sqrt(x**2 + y**2)
            
            if d>0 and d <=dmax:
                if (x==0 or y==0):
                    mult=2
                else:
                    mult = 4
                
                Vsum += mult * V(d)
    return Vsum

def get_summed_r6_interaction_2d(Vnn,dmax):
    V = lambda r: Vnn / r**6
    return get_summed_interaction_2d(V, dmax)

def get_induced_delta_r6_2d(Vnn,dmax):
    return 0.5 * get_summed_r6_interaction_2d(Vnn,dmax)
    

### for the hard-core boson basis

def get_sf_op(m,L,basis=None, check_symm=True, usez=False):
    """ Returns the 'structure factor' operator, on L sites, for period m
    (in units of lattice spacing).
        Implemented as hamiltonian() object.
        
        returns norm of the fourier transform of the on-site density ops, 
        evaluated at momentum component q = 2 pi/m :
            S = (1/L^2) * |sum_x e^(i k x) n_x |^2
            
            If usez = True, n_x is replaced by (nx - 1/2)
                  """
    if basis is None:
        basis = get_hcb_basis(L)
        
    q = 2 * np.pi/m
    ampl = lambda i, j: np.exp(1j * q * (i-j))/(L**2)
    coupling_nn=[]
    for i in range(L):
        for j in range(L):
            coupling_nn += [[ampl(i,j), i, j]]
    if usez:
        static = [["zz", coupling_nn]]
    else:
        static = [ ["nn", coupling_nn] ]
    return hamiltonian(static, [], basis=basis,check_symm=check_symm)

def get_psi_op(m, basis,check_symm=True,usez=True):
    q = 2 * np.pi/m
    coupling_z = [ [np.exp(1j * q * x)/basis.L,x] for x in range(basis.L)]
    static = [["z", coupling_z]]
    return hamiltonian(static, [], basis=basis, check_symm=check_symm, check_herm=False)
    

def n_op(i, basis,check_symm=True,dtype=np.float64):
    """Returns the on-site boson number operator at site i """
    if not isinstance(basis, boson_basis_1d):
        raise TypeError("Expecting 1d boson basis input")
    static = [["n", [[1, i] ]]]
    dynamic=[]
    return hamiltonian(static, dynamic, basis=basis,dtype=dtype,check_symm=check_symm)


def n2_op(i, j, basis, check_symm=True,dtype=np.float64):
    """ product of the n operators at sites i, j """
    if not isinstance(basis, boson_basis_1d):
        raise TypeError("Expecting 1d boson basis input")
    static = [["nn", [[1, i, j]]]]
    dynamic = []
    return hamiltonian(static, dynamic, basis=basis, dtype=dtype, check_symm=check_symm)

def z_op(i,basis,check_symm=True,dtype=np.float64):
    """ returns the z operator for hcb basis"""
    if not is_hcb_basis(basis):
        raise TypeError("expecting hcb basis")
    #the quspin z operator is defined as a spin, i.e n - 1/2
    static = [["z", [ [2.0, i] ]]]
    dynamic=[]
    return hamiltonian(static,dynamic,basis=basis,dtype=dtype,check_symm=check_symm)


def get_z2corr(i,j,basis,psi,check_symm=True):
    """ Returns the 2-point correlator
         <Zi Zj> - <Zi><Zj>
         for state in hcb basis. Z are the pauli matrices.
         """
    dtype=psi.dtype
    z1 = z_op(i,basis,check_symm=check_symm,dtype=dtype)
    z2 = z_op(j,basis,check_symm=check_symm,dtype=dtype)
    z1z2=z1*z2
    return z1z2.expt_value(psi) - (z1.expt_value(psi)) * (z2.expt_value(psi))

def get_summed_Z(sitelist, basis,dtype=np.float64):
    """ Returns the sum of pauli-Z operators (defined in hcb basis as 2n -1) over the sitelist indicated"""
    return sum([z_op(i, basis, dtype=dtype) for i in sitelist])

def get_string_Z(sitelist, basis, dtype=np.float64):
    """ Product of the string ops in specified sitelist"""
    from functools import reduce
    return reduce(lambda x, y: x*y, [z_op(i, basis, dtype=dtype) for i in sitelist])

def proj_blockade_loc(i, basis,bc='periodic'):
    """ Returns a local blockade projector defined on link i, i.e. between sites i, i+1.
        Nonzero when at most one of those sites is excited.
        
        Note that it's defined on the global hilbert space"""
        
    if not isinstance(basis, boson_basis_1d):
        raise TypeError("Expecting 1d boson basis input")
    L = basis.L
    #for open conditions, there's no constraint between 1st and last links
    if bc=='open' and i==(L-1):
        return identity(basis)
    return identity(basis) - n2_op(i, (i+1)%L, L, basis=basis, check_symm=False)

def proj_blockade(basis, bc='periodic'):
    """ Returns a global projector onto the blockade subspace.
        (a next-nearest neighbor blockade)
        basis = hcb basis
        """
    if not isinstance(basis, boson_basis_1d):
        raise TypeError("Expecting 1d boson basis input")
    Pi = identity(basis)
    L = basis.L
    for i in range(L):
        Pi = Pi * proj_blockade_loc(i, basis, bc=bc)
    return Pi

def domain_wall_indicator(i,basis,bc,check_symm=True,dtype=np.float64):
    """Local operator d which indicates (0 or 1) whether there is a domain wall (assuming a Z2 crytstal) to the right of site i.
        d = (1 - n_i)(1-n_i+1) + n_i n_i+1"""
    
    if not isinstance(basis, boson_basis_1d):
        raise TypeError("Expecting 1d boson basis")
    L=basis.L
    if bc=='open' and i==(L-1):
        return 0 * identity(basis)
    n1=n_op(i,basis,check_symm=check_symm,dtype=dtype)
    n2 = n_op((i+1)%L,basis,check_symm=check_symm,dtype=dtype)
    return identity(basis) - (n1 + n2) + 2*n1*n2


def domain_wall_Z3_indicator(i,basis,bc,check_symm=True,dtype=np.float64):
    """Local operator d which indicates (0 or 1) whether there is a domain wall (assuming a Z2 crytstal) to the right of site i.
        d = (1 - n_i)(1-n_i+1)"""
    
    if not is_hcb_basis(basis):
        raise TypeError("Expecting 1d boson basis")
    L=basis.L
    if bc=='open' and (i==(L-1) or i==0):
        return 0 * identity(basis)
    n0=n_op((i-1)%L,basis,check_symm=check_symm,dtype=dtype)
    n1=n_op(i,basis,check_symm=check_symm,dtype=dtype)
    n2 = n_op((i+1)%L,basis,check_symm=check_symm,dtype=dtype)
    I=identity(basis)
    return (I-n0)*(I-n1)*(I-n2)


def domain_wall_population(basis,bc,check_symm=True,dtype=np.float64):
    if not isinstance(basis, boson_basis_1d):
        raise TypeError("Expecting 1d boson basis")
    L=basis.L
    Nd = 0 * identity(basis)
    for ii in range(L):
        Nd = Nd + domain_wall_indicator(ii,basis,bc,check_symm=check_symm,dtype=dtype)
    return Nd

def domain_wall_Z3_population(basis,bc,check_symm=True,dtype=np.float64):
    if not is_hcb_basis(basis):
        raise TypeError("Expecting 1d boson basis")
    L=basis.L
    Nd = 0 * identity(basis)
    for ii in range(L):
        Nd = Nd + domain_wall_Z3_indicator(ii,basis,bc,check_symm=check_symm,dtype=dtype)
    return Nd



