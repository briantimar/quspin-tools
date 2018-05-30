#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 24 11:47:21 2018

@author: brian

The 'block quenchers' are based around block_ops from quspin

"""
import numpy as np
import matplotlib.pyplot as plt
from ryd_base import make_r6_1d
from tools import get_hcb_basis
from quspin.operators import exp_op
from quspin.tools.block_tools import block_ops
from tools import overlap, get_Z2, norm, get_Z3, get_sent
from ryd_base import get_r6_1d_static, get_2d_r6_static, get_hcb_basis_2d

def get_block_quencher(static, gen_basis,basis_args,blocks, njobs=3+1,dtype=np.complex128):
    """Returns a 'quencher' function. Given an initial state, and desired sample times, the quencher will return evolved state
    at those times. """
    dynamic=[]
    def quencher(psi0, ti, tf, Nt):
        H_block = block_ops(blocks,static,dynamic,gen_basis, basis_args,dtype)
        return H_block.expm(psi0,iterate=False,n_jobs=njobs,a=-1j,start=ti,stop=tf,endpoint=True,num=Nt)
    return quencher

def get_kblock_quencher(static, gen_basis, L ,njobs):
    """Returns a quencher using momentum blocks for chain of length L. Datatype is complex """
    blocks = [dict(kblock=k) for k in range(L)]
    basis_args =(L,)
    return get_block_quencher(static, gen_basis, basis_args,blocks,njobs=njobs,dtype=np.complex128)

def get_pblock_quencher(static, gen_basis,L,njobs):
    """ Returns quencher using parity blocks for 1d chain of length L."""
    blocks = [dict(pblock=p) for p in [-1,1]]
    basis_args = (L,)
    return get_block_quencher(static, gen_basis, basis_args,blocks,njobs=njobs,dtype=np.complex128)

def get_2d_pblock_quencher(static, gen_basis,Lx,Ly,njobs):
    blocks = [dict(pxblock=p) for p in [-1,1]]
    basis_args = (Lx,Ly)
    return get_block_quencher(static, gen_basis, basis_args,blocks,njobs=njobs,dtype=np.complex128)

def get_r6_1d_pbc_quencher(Delta, Omega, Vnn, L, ktrunc,bc='periodic',block='momentum',gamma=None):
    """Returns quencher for 1d r6 chain, periodic bc's, blocking by momentum."""
    if block not in ["momentum","parity"]:
        raise ValueError("Unknown block type")
    quencher_types = dict(momentum=get_kblock_quencher, parity=get_pblock_quencher)
    static = get_r6_1d_static(Delta, Omega, Vnn, L, ktrunc, gamma=gamma,bc=bc)
    gen_basis = get_hcb_basis
    gen_quencher = quencher_types[block]
    return gen_quencher(static,gen_basis,L,njobs=3+1)

def get_r6_2d_quencher(Delta, Omega, Vnn,dmax,Lx,Ly,bc='periodic',block='px'):
    if block not in ["px"]:
        raise ValueError("Unknown block type")
    quencher_types = dict(px=get_2d_pblock_quencher)
    static = get_2d_r6_static(Delta, Omega, Vnn,dmax,Lx,Ly,bc=bc)
    gen_basis = get_hcb_basis_2d
    gen_quencher = quencher_types[block]
    return gen_quencher(static,gen_basis,Lx,Ly,njobs=3+1)

def Scl(p,tol=1E-12):
    """ The von-neumann entropy of a classical probability distribution (input as a 1d numpy array)"""
    assert np.all(p>=0)
    if np.abs(np.sum(p)-1.0)>tol:
        print("Warning -- probability sum violated, 1-sum(p) = {0:.3e}".format(np.abs(np.sum(p)-1.0)))
    ptrunc= p[p>tol] 
    return -np.sum(ptrunc * np.log(ptrunc))

def cl_mutual_information(rho1, rho2, psi):
    """ mutual information defined by distributions in the diagonal bases"""
    pglobal = np.abs(psi)**2
    p1 = np.diagonal(rho1)
    p2 = np.diagonal(rho2)
    Stot = Scl(pglobal)
    S1 = Scl(p1)
    S2 = Scl(p2)
    return np.real(S1 + S2 - Stot)
    
def get_prob(psi):
    """returns list of probabilties corresponding to psi in some basis"""
    return np.abs(psi)**2


from tools import get_PDF

def get_leveldist(e,nbins=100):
    """ Get the probability density function that describes the distribution of spacings between adjacent levels.
       e= (sorted) list of levels
       
       Returns: f, x
       f being the PDF and x defining the left edges of the corresponding bins
       """
    de = np.diff(e)
    return get_PDF(de,nbins=nbins)

def get_level_spacing(h,nbins=100):
    e=h.eigvalsh()
    return get_leveldist(e,nbins=nbins)
    
def get_pdist(psi,basis,xmax=np.inf,nbins=100):    
    N=len(psi)
    #check that the state has not been symmetry-reduced
    if N != np.power(basis.sps, basis.N):
        raise ValueError("state has been symmetry-reduced")
    p=get_prob(psi) 
    x = N * p
    x=x[x<=xmax]
    f,xplot = get_PDF(x,nbins=nbins)
    return f,xplot


    
def do_z_density_plot(time_normalized, z2corr,dcorr):
    fig,ax=plt.subplots()
    x1,x2,y1,y2=np.min(time_normalized), np.max(time_normalized), np.min(dcorr), np.max(dcorr)
    plt.imshow(np.transpose(z2corr[:,:]),origin='lower',cmap='gist_heat', extent=[x1,x2,y1,y2],aspect=(x2-x1)/(y2-y1))
    plt.colorbar()
    plt.xlabel("$|\Omega| t$")
    plt.ylabel('Site distance $d$')
    plt.title(r"$\left< Z_0 Z_d \right>_c$")
    return fig,ax


def do_z_plot(t,z2corr,dcorr):
    fig,ax=plt.subplots()
    dmax=len(dcorr)
    for j in range(dmax):
        plt.plot(t, z2corr[:,j],label='d={0}'.format(dcorr[j]))
    plt.xlabel(r"$|\Omega| t$")
    plt.ylabel(r"$\left< Z_0 Z_d \right >_c$")
    plt.legend()
    return fig,ax

def do_z_normalized_plot(t,z2corr,dcorr):
    """Plot absolute value of the two-point correlation, normalized to its maximum value"""
    fig,ax=plt.subplots()
    dmax=len(dcorr)
    for j in range(dmax):
        z=np.abs(z2corr[:,j])
        plt.plot(t, z/np.max(z),label='d={0}'.format(dcorr[j]))
    plt.xlabel(r"$|\Omega| t$")
    plt.ylabel(r"$|\left< Z_0 Z_d \right >_c|$")
    plt.legend()
    return fig,ax


def do_S_plot(S,t,subsystem_lengths,xstr='',ystr='$S_{vn}(l)$'):
    fig,ax=plt.subplots()
    NL = len(subsystem_lengths)
    for l in range(NL):
        plt.plot(t, S[:,l], label='l={0}'.format(subsystem_lengths[l]))
    plt.xlabel(xstr)
    plt.ylabel(ystr)
    plt.legend()
    return fig,ax
    


