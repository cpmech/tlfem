# Copyright 2012 Dorival de Moraes Pedroso. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

import sys                                                     # to load modules and classes
from   itertools           import chain                        # to flatten a list of lists
from   numpy               import arange, array, delete, zeros # some functions from numpy
from   numpy               import transpose, hstack, ones      # more functions from numpy
from   scipy.linalg        import eigvals, eig, norm           # solvers from scipy
from   scipy.sparse        import lil_matrix                   # sparse matrix from scipy
from   scipy.sparse.linalg import spsolve, eigs                # sparse solvers from scipy
from   scipy.integrate     import odeint                       # ODE integrators from scipy
from   tlfem.output        import Output                       # output class
from   tlfem.vtu           import Vtu                          # Vtu class for ParaView files

from numpy import ndarray

class Solver:
    def __init__(self, mesh, params):
        """
        FEM: Solver
        ===========
        INPUT:
            mesh   : an instance of FEMmesh
            params : a dictionary connecting elements tags to a dictionary of element parameters
                     Example:
                        p = { -1:{'type':Ediffusion1D', 'rho':1.0, 'beta':0.0, 'kx':1.0},
                              -2:{'type':Ediffusion1D', 'rho':2.0, 'beta':0.5, 'kx':1.5} }
        STORED:
            m         : the mesh
            elems     : list with all allocated elements
            beams     : ids of beam elements
            rods      : ids of rod elements
            sovs      : list of solution variables or each node, ex: [('ux','uy'),('ux','uy','pw')]
            eqs       : equation numbers for each sov of each node, ex: [(0,1), (2,3,4)]
            amap      : assembly map : local equations => global eqs (location array)
            neqs      : total number of equations
            ukeys     : all ukeys, ex: ['ux', 'uy', 'pw']
            f2ukey    : converts f to u key, ex: {'fx':'ux', 'Qw','pw'}
            skeys     : all secondary variables keys, ex: 'sx', 'sy' [for extrapolation]
                        does not include beam/rod variables
            porous    : has at least one porous media element
            pu_dec    : pw/u decopling in vtu (for Babuska-Brezzi elements)
            flat_sovs : flatten version of sovs
            flat_eqs  : flatten version of eqs
        RETURNS:
            None
        """
        # store data
        self.m = mesh

        # load element files
        type2class = {}                                # maps element type to its class object
        for p in params.values():                      # for each set of parameters
            etyp = p['type']                           # element name/type, e.g 'EelasticTri'
            emod = 'tlfem.' + etyp                     # element module to be imported
            __import__(emod)                           # load module by its name, e.g 'EelasticTri'
            type2class[etyp] = getattr(sys.modules[emod], etyp) # store class object

        # allocate all elements and set assembly map (amap)
        self.elems  = []                               # all elements in mesh
        self.beams  = []                               # ids of beams
        self.rods   = []                               # ids of rods
        self.sovs   = [[] for _ in self.m.V]           # solution variables @ each vertex
        self.eqs    = [[] for _ in self.m.V]           # equation numbers @ each vertex
        self.amap   = []                               # assembly map: list of location arrays
        self.neqs   = 0                                # total number of equations
        self.f2ukey = {}                               # converts u2fkeys
        self.skeys  = set()                            # skeys set
        self.porous = False                            # has porous element?
        self.pu_dec = False                            # pw/u decoupling in vtu
        for c in self.m.C:                             # for each cell in the mesh

            # allocate element
            verts = [self.m.V[i] for i in c[2]]        # collect vertices of this cell/element
            if c[1] not in params:                     # check if params has this cell's tag
                raise Exception('cannot find parameters for element tagged with %d' % c[1])
            p = params[c[1]]                           # parameters for this element
            self.elems.append(type2class[p['type']](verts, p)) # allocate element

            # location array, solution variables and assembly map
            loc = []                                   # location array of this element
            esovs, f2u, sks = self.elems[-1].info()    # get element information
            self.f2ukey.update(f2u)                    # update map f2ukey
            for i, I in enumerate(c[2]):               # for each vertex: i:local, I:global
                for sov in esovs[i]:                   # for each sov key. ex: 'ux', 'uy', 'pw'
                    if sov in self.sovs[I]:            # sov already has an equation num assigned
                        idx = self.sovs[I].index(sov)  # index of sov in sublist of sovs[I]
                        eq  = self.eqs [I][idx]        # equation number of sov of vertex I
                        loc.append(eq)                 # append equation to location array
                    else:                              # assign new eq number to sov of vertex I
                        self.sovs[I].append(sov)       # add new sov to vertex I
                        self.eqs [I].append(self.neqs) # set equation number to new sov
                        loc.append(self.neqs)          # set location array with new equation
                        self.neqs += 1                 # increment next equation number
            self.amap.append(loc)                      # set assembly map for this element

            # beam/rod element?
            if p['type'] == 'EelasticBeam':            # beam element
                self.beams.append(c[0])                # add to list of linear cells
            elif p['type'] == 'EelasticRod':           # rod element
                self.rods.append(c[0])                 # add to list of linear cells
            else:                                      # other elements
                self.skeys.update(sks)                 # update skeys

            # check for porous elements
            if p['type'] == 'EelasticPorous2D':        # porous element?
                self.porous = True                     # indicate porous element
                if not isinstance(p['geom'], str):     # not single geometry type
                    if p['geom'][0] != p['geom'][1]:   # Babuska-Brezzi elements
                        self.pu_dec = True             # pw/u decoupling (in vtu)
                #TODO: change to (?):
                #      hasattr(p['geom'], "__iter__")

        # collect all u keys, such as 'ux', 'uy', 'pw'
        self.flat_sovs = array(list(chain.from_iterable(self.sovs)))
        self.flat_eqs  = array(list(chain.from_iterable(self.eqs)))
        self.ukeys     = set(self.flat_sovs.tolist()) # unique

    def set_bcs(self, vb={}, eb={}):
        """
        Set boundary conditions
        =======================
        INPUT:
            vb : a dictionary with bry cond info at vertices [optional]
            eb : a dictionary with bry cond info at edges [optional]
        STORED:
            presc_U_eqs : prescribed U equations
            presc_U_vls : prescribed U values
            presc_F_eqs : prescribed F equations
            presc_F_vls : prescribed F values
            eq2         : prescribed equations
            eq1         : equations to be solved for
        RETURNS:
            None
        """
        # set auxiliary lists
        self.presc_U_eqs, self.presc_F_eqs = [], []
        self.presc_U_vls, self.presc_F_vls,= [], []

        # clear (natural) bcs (if any)
        for e in self.elems:                                   # for each element
            if hasattr(e, 'clear_bcs'): e.clear_bcs()          # clear boundary conditions data

        # edge boundary conditions
        for edtag, bc in eb.items():                           # edges bry conds: edge tag, bry c.
            cids = self.m.get_cells_with_etag(edtag)           # get ids of cells with tagged edges
            for ielem, iedge in cids:                          # index of elem, index of edge
                for k, v in bc.items():                        # ex: for k, v in {'conv':(0.5,2.0)}
                    if k in self.ukeys:                        # u bry cond => set nodes directly
                        vids = self.m.get_edge_verts(ielem, iedge) # verts on edge
                        for n in vids:                         # for each node on edge with u spec.
                            if not k in self.sovs[n]: continue # node does not have k, ex: 'pw'
                            eq = self.get_eq(n, k)             # eq number of specified u
                            if not eq in self.presc_U_eqs:     # not set yet
                                val = self.calc_val_vert(n, v) # compute value @ node
                                self.presc_U_eqs.append(eq)    # new presc U equation
                                self.presc_U_vls.append(val)   # prescribed U value
                    else:                                      # natural (or mixed) bry condition
                        self.elems[ielem].set_nat_bcs(iedge, k, v) # set natural bcs

        # vertex boundary conditions
        for vtag, bc in vb.items():                            # vertices bry conds: vertex tag, bc
            vids = self.m.get_verts(vtag)                      # get vertices by tag
            for n in vids:                                     # for each node number n
                for k, v in bc.items():                        # ex: for k, v in {'ux':0.}
                    if k in self.ukeys:                        # u (essential) bry condition
                        if not k in self.sovs[n]: continue     # node does not have k, ex: 'pw'
                        eq  = self.get_eq(n, k)                # get equation number
                        val = self.calc_val_vert(n, v)         # compute value @ node
                        if eq in self.presc_U_eqs:             # already set earlier => override
                            self.presc_U_vls[self.presc_U_eqs.index(eq)] = val
                        else:                                  # new value to be set
                            self.presc_U_eqs.append(eq)        # new prescribed U equation
                            self.presc_U_vls.append(val)       # new prescribed U value
                    else:                                      # natural bry condition
                        uk = self.f2ukey[k]                    # convert fkey to ukey: 'fx' => 'ux'
                        eq = self.get_eq(n, uk)                # get equation number
                        self.presc_F_eqs.append(eq)            # new prescribed F equation
                        self.presc_F_vls.append(v)             # new prescribed F value

        # set lists with equation numbers
        all_eqs  = arange(self.neqs)                           # all equation numbers
        self.eq2 = array(self.presc_U_eqs)                     # prescribed equations
        self.eq1 = delete(all_eqs, self.eq2)                   # eqs to be solved for (free)

        # maps for porous media simulations
        if self.porous:
            self.p2g = self.flat_eqs[self.flat_sovs=='pw']     # p eq => global eq
            self.u2g = delete(all_eqs, self.p2g)               # u eq => global eq
            self.g2u = -ones(self.neqs, dtype=int)             # global eq => u eq
            self.g2p = -ones(self.neqs, dtype=int)             # global eq => p eq
            for u, g in enumerate(self.u2g): self.g2u[g] = u   # set map
            for p, g in enumerate(self.p2g): self.g2p[g] = p   # set map
            self.neqsP = len(self.p2g)                         # number of p equations
            self.neqsU = len(self.u2g)                         # number of u equations
            self.eqs_b, self.eqs_d = [], []                    # presc U / presc P equations
            self.uvs_b, self.pvs_d = [], []                    # vals U / vals P of presc equations
            for i, I in enumerate(self.presc_U_eqs):           # for each prescribed UP equation
                if I in self.u2g:                              # => equation is u type
                    r = self.g2u[I]                            # u equation number
                    self.eqs_b.append(r)                       # equations b
                    self.uvs_b.append(self.presc_U_vls[i])     # u values b
                else:                                          # => equations is p type
                    r = self.g2p[I]                            # p equation number
                    self.eqs_d.append(r)                       # equations d
                    self.pvs_d.append(self.presc_U_vls[i])     # p values d
            eqsU = arange(self.neqsU)                          # all U equations
            eqsP = arange(self.neqsP)                          # all P equations
            self.eqs_a = delete(eqsU, self.eqs_b)              # free U equations
            self.eqs_c = delete(eqsP, self.eqs_d)              # free P equations

    def set_out_nodes(self, tags_or_ids=[]):
        """
        Set a list of nodes for complete output
        =======================================
        INPUT:
            tags_or_ids : a list with vertex tags or vertex ids (or mixed)
        STORED:
            outn : a list with nodes ids for complete output
        RETURNS:
            None
        Notes:
            1) By default, all nodes will have full output
        """
        self.outn = []
        for tag_or_id in tags_or_ids:
            if tag_or_id < 0:
                vids = self.m.get_verts(tag_or_id)
                self.outn.update(vids)
            else: self.outn.append(tag_or_id)

    def solve_steady(self, reactions=False, extrap=False, emethod=2, vtu_fnkey=None, output_F=False):
        """
        Solve steady-state problem
        ==========================
        INPUT:
            reactions : calculate reaction forces?
            extrap    : extrapolate element values to nodes?
            emethod   : extrapolation method. see Output for details.
            vtu_fnkey : filename key to be used when writing vtu files
            output_F  : outputs the right-hand side vector: K * U = F
        STORED:
            None
        RETURNS:
            An instance of Output with results
        """
        # check
        if len(self.presc_U_vls) < 1:
            raise Exception('at least one u-value must be prescribed (in set_bcs)')

        # assemble K and F
        F   = zeros(self.neqs)
        K11 = lil_matrix((self.neqs,self.neqs))
        K12 = lil_matrix((self.neqs,self.neqs))
        if reactions:
            K21 = lil_matrix((self.neqs,self.neqs))
            K22 = lil_matrix((self.neqs,self.neqs))
        for ie, e in enumerate(self.elems):
            Ke, Fe = e.calc_K(), e.calc_F()
            for i, I in enumerate(self.amap[ie]):
                for j, J in enumerate(self.amap[ie]):
                    if   (I in self.eq1) and (J in self.eq1): K11[I,J] += Ke[i,j]
                    elif (I in self.eq1) and (J in self.eq2): K12[I,J] += Ke[i,j]
                    if reactions:
                        if   (I in self.eq2) and (J in self.eq1): K21[I,J] += Ke[i,j]
                        elif (I in self.eq2) and (J in self.eq2): K22[I,J] += Ke[i,j]
                F[I] += Fe[i]

        # set prescribed F values
        if len(self.presc_F_eqs) > 0:
            F[self.presc_F_eqs] += self.presc_F_vls

        # modify/augment K11
        for eq in self.eq2: K11[eq,eq] = 1.0

        # convert K11 and K12 to compressed-row format (for efficiency)
        K11 = K11.tocsr()
        K12 = K12.tocsr()

        # prescribed values
        U2 = zeros(self.neqs)
        U2[self.eq2] = self.presc_U_vls

        # solve
        W           = F - K12*U2       # workspace
        W[self.eq2] = self.presc_U_vls # copy presc vals to W2
        U           = spsolve(K11, W)  # solve: U = inv(K11)*W

        # output object
        fo = Output(self, 0., extrap=extrap, emethod=emethod, vtu_fnkey=vtu_fnkey)

        # calculate reaction forces
        if reactions or output_F:
            K21 = K21.tocsr()          # convert to sparse matrix
            K22 = K22.tocsr()          # convert to sparse matrix
            W   = K21 * U + K22 * U    # workspace
            if reactions:
                R   = W - F            # reaction forces
                fo.out_reactions(R)    # store in fo
            if output_F:
                F[self.eq2] = W[self.eq2]
                fo.out_F(F)

        # output results
        fo.out(1., U)
        fo.stop()
        return fo

    def solve_eigen(self, dyn=True, nevals=None, high=False, evecs=False, nrm=False, dense=False,
                    vtu_fnkey=None):
        """
        Solve eigenvalue problem
        ========================
        INPUT:
            dyn       : solve dynamics problem? otherwise solves first order transient problem
            nevals    : number of desired eigenvalues. None => only max
                        must be smaller than neq1 (neq1=len(self.eq1))
            high      : if nevals != None, compute the highest nevals eigenvalues?
            evecs     : also calculate and store eigenvectors?
            nrm       : normalise eigenvectors?
            dense     : use dense version (slow)?
            vtu_fnkey : filename key for vtu files (==> each time corresponds to one shape-mode)
        STORED:
            L  : array of eigenvalues (lambdas)
            Lv : Modes (eigenvectors) corresponding to L [optional]
                 the column Lv[:,k] corresponds to L[k]
                 Note:
                   the eigenvectors components are placed in the eq1 positions;
                   hence, the columns of Lv are similar to U
        RETURNS:
            None
        """
        # assemble (A = M or C) and K
        A11 = lil_matrix((self.neqs,self.neqs))
        K11 = lil_matrix((self.neqs,self.neqs))
        for ie, e in enumerate(self.elems):
            Ke = e.calc_K()
            if dyn: Ae = e.calc_M()
            else:   Ae = e.calc_C()
            for i, I in enumerate(self.amap[ie]):
                for j, J in enumerate(self.amap[ie]):
                    if (I in self.eq1) and (J in self.eq1):
                        A11[I,J] += Ae[i,j]
                        K11[I,J] += Ke[i,j]

        # remove extra rows and columns
        A1_ = A11[self.eq1,:]
        A11 = A1_[:,self.eq1]
        K1_ = K11[self.eq1,:]
        K11 = K1_[:,self.eq1]

        # just one equation
        if len(self.eq1) == 1:
            self.L = array([K11.todense()[0,0] / A11.todense()[0,0]])
            return

        # number of eigenvalues
        nev = 1 if nevals==None else nevals
        if nev > K11.shape[0]: nev = K11.shape[0]

        # compute eigenvalues and eigenvectors
        if nev >= K11.shape[0]-1: dense = True
        if dense: # using dense version (might be slow)
            if evecs: l, v  = eig(K11.todense(), A11.todense())
            else:     l = eigvals(K11.todense(), A11.todense())
        else: # sparse version
            # convert to compressed-column format (for efficiency)
            A11 = A11.tocsc()
            K11 = K11.tocsc()
            sig = None if nevals==None else 0
            if high: sig = None
            if evecs:
                l, v = eigs(K11, nev, A11, sigma=sig, which='LM', return_eigenvectors=True)
            else:
                l = eigs(K11, nev, A11, sigma=sig, which='LM', return_eigenvectors=False)

        # check for complex eigenvalues
        if abs(l.imag.max()) > 0.0 or abs(l.imag.min()) > 0.0:
            raise Exception('found complex eigenvalues')

        # store storted eigenvalues and corresponding eigenvectors
        idx    = l.real.argsort()                  # indices of sorted items
        self.L = l.real[idx]                       # collected sorted items
        if evecs:                                  # has eigenvectors
            nmodes = v.shape[1]                    # number of modes
            if nrm:                                # normalise eigenvectors
                for k in range(nmodes):            # for each mode
                    v[:,k] = v[:,k] / norm(v[:,k]) # normalise
            self.Lv = zeros((self.neqs,nmodes))    # U for each mode
            self.Lv[self.eq1,:] = v[:,idx].real    # mode solution

            # write vtu file with shape-modes
            if vtu_fnkey != None:
                vtu = Vtu(self.pu_dec)                         # allocate vtu structure
                vtu.start(vtu_fnkey)                           # start vtu object
                for k in range(nmodes):                        # for each mode
                    Umod = {}                                  # shape-modes
                    for key in self.ukeys:                     # for each 'ux', 'uy', etc.
                        Umod[key] = {}                         # initialise dictionary of u values
                        for n in range(self.m.nv):             # for each node
                            if key in self.sovs[n]:            # node has solution variable
                                eq = self.get_eq(n, key)       # get corresponding equation number
                                Umod[key][n] = [self.Lv[eq,k]] # save u
                            else:                              # node does not have sol var
                                Umod[key][n] = 1.0e+30         # flag non-existent value
                    vtu.write(float(k), self.m, Umod)          # write vtu file
                vtu.stop()                                     # stop vtu object

    def solve_transient(self, t0, tf, dt, dtout=None, U0=None, theta=0.5,
                        extrap=False, emethod=2, ext_dtout=None, vtu_fnkey=None, vtu_dtout=None):
        """
        Solve transient problem
        =======================
        INPUT:
            t0        : initial time
            tf        : final time
            dt        : time step increment
            dtout     : time step increment for output of results
            U0        : initial values [optional]
            theta     : 0.0   => Euler/forward (cond/stable) O1
                        0.5   => Crank/Nicolson (stable) O2
                        2./3. => Galerkin method (stable) O2
                        1.0   => Euler/backward (stable) O1
            extrap    : extrapolate secondary values to nodes?
            emethod   : extrapolation method. see Output for details.
            ext_dtout : time step increment to calculate extrapolated values
            vtu_fnkey : filename key to be used when writing vtu files
            vtu_dtout : time step increment to write vtu files [for ParaView]
        STORED:
            None
        RETURNS:
            An instance of Output with results
        """
        # check
        if len(self.presc_U_vls) < 1:
            raise Exception('at least one u-value must be prescribed (in set_bcs)')

        # assemble C, K and F
        C11 = lil_matrix((self.neqs,self.neqs))
        C12 = lil_matrix((self.neqs,self.neqs))
        K11 = lil_matrix((self.neqs,self.neqs))
        K12 = lil_matrix((self.neqs,self.neqs))
        F   = zeros(self.neqs)
        for ie, e in enumerate(self.elems):
            Ce, Ke, Fe = e.calc_C(), e.calc_K(), e.calc_F()
            for i, I in enumerate(self.amap[ie]):
                for j, J in enumerate(self.amap[ie]):
                    if (I in self.eq1) and (J in self.eq1):
                        C11[I,J] += Ce[i,j]
                        K11[I,J] += Ke[i,j]
                    elif (I in self.eq1) and (J in self.eq2):
                        C12[I,J] += Ce[i,j]
                        K12[I,J] += Ke[i,j]
                F[I] += Fe[i]

        # set prescribed F values
        if len(self.presc_F_eqs) > 0:
            F[self.presc_F_eqs] += self.presc_F_vls

        # modify/augment C11
        for eq in self.eq2: C11[eq,eq] = 1.0

        # convert matrices to compressed-row format (for efficiency)
        C11 = C11.tocsr()
        C12 = C12.tocsr()
        K11 = K11.tocsr()
        K12 = K12.tocsr()

        # prescribed values
        U2 = zeros(self.neqs)           # all U, but will set only presc ones
        V2 = zeros(self.neqs)           # all V, but will set only presc ones
        U2[self.eq2] = self.presc_U_vls # set prescribed u values

        # right-hand-side of split system
        G = F - C12*V2 - K12*U2

        # initial values
        t = t0
        U = U0 if isinstance(U0, ndarray) else zeros(self.neqs) 

        # output object
        fo = Output(self, t, dtout, extrap, emethod, ext_dtout, vtu_fnkey, vtu_dtout)

        # theta-method
        fo.out(t, U) # first output
        if theta<0.0 or theta>1.0: raise Exception('theta must be between 0.0 and 1.0')
        a = theta * dt
        b = (1.0 - theta) * dt
        A = C11 + a*K11
        B = C11 - b*K11
        while t < tf:                       # loop for all time steps
            W = B*U + dt*G                  # set workspace
            W[self.eq2] = self.presc_U_vls  # copy presc vals to W2
            U  = spsolve(A, W)              # solve: U = inv(A) * W
            t += dt                         # update time
            fo.out(t, U)                    # output results

        # results
        fo.stop()
        return fo

    def solve_dynamics(self, t0, tf, dt, dtout=None, U0=None, V0=None, theta1=0.5, theta2=0.5,
                       rayleigh=True, rayM=0.0, rayK=0.0, HHT=False, alpha=-1.0/3.0,
                       extrap=False, emethod=2, ext_dtout=None, vtu_fnkey=None, vtu_dtout=None,
                       load_mult=None):
        """
        Solve dynamics problem
        ======================
        INPUT:
            t0        : initial time
            tf        : final time
            dt        : time step increment
            dtout     : time step increment for output of results
            U0        : initial displacements [optional]
            V0        : initial velocities [optional]
            theta1    : Newmark parameter gamma
            theta2    : Newmark parameter 2*beta
            rayleigh  : use Rayleigh damping? => C = rayM * M + rayC * C
            rayM      : coefficient of M when using Rayleigh damping
            rayK      : coefficient of K when using Rayleigh damping
            HHT       : use Hilber-Hughes-Taylor method ?
            alpha     : Hilber-Hughes-Taylor parameter [-1/3 <= alpha <= 0]
            extrap    : extrapolate secondary values to nodes? ==> generate Eout
            emethod   : extrapolation method. see calc_secondary for details.
            ext_dtout : time step increment to calculate extrapolated values
            vtu_fnkey : filename key to be used when writing vtu files
            vtu_dtout : time step increment to write vtu files [for ParaView]
            load_mult : distributed load multiplier: a function  such as 'lambda t:cos(pi*t)'
                        to be multiplied to the correspondig load vector during the simulation
                        [this applies to disbributed loads/natural bcs only]
        STORED:
            None
        RETURNS:
            An instance of Output with results
        Notes:
            1) With HHT=True, theta1 and theta2 are calculated in order to
               obtain the unconditionally stable O2 method; otherwise, if:
               th1=1/2 and th2=1/2 => o2/stable
        """
        # check
        if len(self.presc_U_vls) < 1:
            raise Exception('at least one u-value must be prescribed (in set_bcs)')

        # auxiliary variable
        neq = self.neqs

        # dynamics coefficients
        a1, a2, a3, a4, a5, a6, a7, a8 = self.dyn_coefficients(dt, theta1, theta2, HHT, alpha)
        B11 = lil_matrix((neq,neq))

        # assemble B, M, C, K and F
        M11 = lil_matrix((neq,neq)); M12 = lil_matrix((neq,neq))
        C11 = lil_matrix((neq,neq)); C12 = lil_matrix((neq,neq))
        K11 = lil_matrix((neq,neq)); K12 = lil_matrix((neq,neq))
        F   = zeros(neq)
        for ie, e in enumerate(self.elems):
            Me, Ke, Fe = e.calc_M(), e.calc_K(), e.calc_F()
            if rayleigh: Ce = rayM * Me + rayK * Ke
            else:        Ce = e.calc_C()
            for i, I in enumerate(self.amap[ie]):
                for j, J in enumerate(self.amap[ie]):
                    if (I in self.eq1) and (J in self.eq1):
                        B11[I,J] += a1 * Me[i,j] + a7 * Ce[i,j] + a8 * Ke[i,j]
                        M11[I,J] += Me[i,j]
                        C11[I,J] += Ce[i,j]
                        K11[I,J] += Ke[i,j]
                    elif (I in self.eq1) and (J in self.eq2):
                        M12[I,J] += Me[i,j]
                        C12[I,J] += Ce[i,j]
                        K12[I,J] += Ke[i,j]
                F[I] += Fe[i]

        # modify/augment B11 and M11
        for eq in self.eq2:
            B11[eq,eq] = 1.0
            M11[eq,eq] = 1.0
        B11 = B11.tocsr()

        # convert matrices to compressed-row format (for efficiency)
        M11 = M11.tocsr(); M12 = M12.tocsr()
        C11 = C11.tocsr(); C12 = C12.tocsr()
        K11 = K11.tocsr(); K12 = K12.tocsr()

        # prescribed values
        U2 = zeros(neq)                 # all U, but will set only presc ones
        V2 = zeros(neq)                 # all V, but will set only presc ones
        A2 = zeros(neq)                 # all A, but will set only presc ones
        U2[self.eq2] = self.presc_U_vls # set prescribed u values

        # right-hand-side of split system
        def G(t):
            g = F.copy()
            if load_mult != None:
                g *= load_mult(t)
            for k, eq in enumerate(self.presc_F_eqs):
                if isinstance(self.presc_F_vls[k], float): g[eq] += self.presc_F_vls[k]
                else:                                      g[eq] += self.presc_F_vls[k](t)
            return g - M12*A2 - C12*V2 - K12*U2

        # initial values
        t = t0
        U = zeros(neq) if U0==None else U0
        V = zeros(neq) if V0==None else V0

        # output object
        fo = Output(self, t, dtout, extrap, emethod, ext_dtout, vtu_fnkey, vtu_dtout)

        # initial accelerations
        W = G(t0) - C11*V - K11*U
        A = spsolve(M11, W)
        A[self.eq2] = A2[self.eq2] # set prescribed accelerations
        # time loop
        fo.out(t, U)                                     # first output
        while t < tf:                                    # loop for all time steps
            m = a1*U + a2*V + a3*A                       # auxiliary variable
            n = a4*U + a5*V + a6*A                       # auxiliary variable
            if HHT:                                      # HHT method
                ta = t + (1.0+alpha)*dt                  # time at alpha+1
                qb = a8*n + alpha*V                      # auxiliary qb
                W = G(ta) + M11*m + C11*qb + alpha*K11*U # set workspace
            else:                                        # standard Newmark method
                W = G(t+dt) + M11*m + C11*n              # set workspace
            W[self.eq2] = self.presc_U_vls               # copy presc vals to W2
            U = spsolve(B11, W)                          # solve: U = inv(A) * W
            V = a4*U - n                                 # new V
            A = a1*U - m                                 # new A
            t += dt                                      # update time
            fo.out(t, U)                                 # output results

        # results
        fo.stop()
        return fo

    def solve_porous(self, t0, tf, dt, dtout=None, U0=None, V0=None, P0=None, theta=0.5,
                     theta1=0.5, theta2=0.5, rayleigh=True, rayM=0.0, rayK=0.0,
                     HHT=False, alpha=-1.0/3.0, extrap=False, emethod=2, ext_dtout=None,
                     vtu_fnkey=None, vtu_dtout=None, load_mult=None):
        """
        Solve coupled porous mechanics problem
        ======================================
        INPUT:
            t0        : initial time
            tf        : final time
            dt        : time step increment
            dtout     : time step increment for output of results
            U0        : initial displacements (size must be equal to self.neqsU) [optional]
            V0        : initial velocities (size must be equal to self.neqsU) [optional]
            P0        : initial pore-water pressures (size must be self.neqsP) [optional]
            theta     : theta value for theta-method
            theta1    : Newmark parameter gamma
            theta2    : Newmark parameter 2*beta
            rayleigh  : use Rayleigh damping? => C = rayM * M + rayC * C
            rayM      : coefficient of M when using Rayleigh damping
            rayK      : coefficient of K when using Rayleigh damping
            HHT       : use Hilber-Hughes-Taylor method?
            alpha     : Hilber-Hughes-Taylor parameter [-1/3 <= alpha <= 0]
            extrap    : extrapolate secondary values to nodes? ==> generate Eout
            emethod   : extrapolation method. see calc_secondary for details.
            ext_dtout : time step increment to calculate extrapolated values
            vtu_fnkey : filename key to be used when writing vtu files
            vtu_dtout : time step increment to write vtu files [for ParaView]
            load_mult : qn multiplier: a function such as 'lambda t:cos(pi*t)' to be multiplied
                        to the correspondig load vector during the simulation
                        [this applies to disbributed loads/natural bcs only]
        STORED:
            None
        RETURNS:
            An instance of Output with results
        Notes:
            'None' values of U0, V0, P0 will initialise these arrays with zeros
        """
        # check
        if len(self.presc_U_vls) < 1:
            raise Exception('at least one u-value must be prescribed (in set_bcs)')

        # auxiliary variables
        nu, np, neq = self.neqsU, self.neqsP, self.neqs

        # dynamics coefficients
        a1, a2, a3, a4, a5, a6, a7, a8 = self.dyn_coefficients(dt, theta1, theta2, HHT, alpha)
        b1, b2 = 1.0/(theta*dt), (1.0-theta)/theta # for the continuity equation (theta-method)
        B11 = lil_matrix((neq, neq))

        # assemble matrices
        Maa = lil_matrix((nu,nu));  Oca = lil_matrix((np,nu))
        Caa = lil_matrix((nu,nu));  Nca = lil_matrix((np,nu))
        Kaa = lil_matrix((nu,nu));  Lcc = lil_matrix((np,np))
        Mab = lil_matrix((nu,nu));  Ocb = lil_matrix((np,nu))
        Cab = lil_matrix((nu,nu));  Ncb = lil_matrix((np,nu))
        Kab = lil_matrix((nu,nu));  Lcd = lil_matrix((np,np))
        Qad = lil_matrix((nu,np));  Hcd = lil_matrix((np,np))
        F   = zeros(neq)
        for ie, e in enumerate(self.elems):
            Me, Ke, Fe   = e.calc_M(), e.calc_K(), e.calc_F()
            Oe, Qe       = e.calc_O(), e.calc_Q()
            Le, He, Fbe  = e.calc_L(), e.calc_H(), e.calc_Fb()
            amapU, amapP = e.get_amaps(self.amap[ie])
            Ne           = transpose(Qe)
            if rayleigh: Ce = rayM * Me + rayK * Ke
            else:        Ce = e.calc_C()
            for i, I in enumerate(amapU): # [a,b]
                r = self.g2u[I]
                for j, J in enumerate(amapU): # [a,b]
                    s = self.g2u[J]
                    if (r in self.eqs_a) and (s in self.eqs_a): # aa => I in eq1 and J in eq1
                        B11[I,J] += a1*Me[i,j] + a7*Ce[i,j] + a8*Ke[i,j]
                        Maa[r,s] += Me[i,j]
                        Caa[r,s] += Ce[i,j]
                        Kaa[r,s] += Ke[i,j]
                    elif (r in self.eqs_a) and (s in self.eqs_b): # ab
                        Mab[r,s] += Me[i,j]
                        Cab[r,s] += Ce[i,j]
                        Kab[r,s] += Ke[i,j]
                for j, J in enumerate(amapP): # [c,d]
                    s = self.g2p[J]
                    if (r in self.eqs_a) and (s in self.eqs_c): # ac => I in eq1 and J in eq1
                        B11[I,J] += (-Qe[i,j])
                        B11[J,I] += a1*Oe[j,i] + a4*Ne[j,i]
                        Oca[s,r] += Oe[j,i]
                        Nca[s,r] += Ne[j,i]
                    elif (r in self.eqs_b) and (s in self.eqs_c): # bc
                        Ocb[s,r] += Oe[j,i]
                        Ncb[s,r] += Ne[j,i]
                    elif (r in self.eqs_a) and (s in self.eqs_d): # ad
                        Qad[r,s] += Qe[i,j]
                F[I] += Fe[i]
            for i, I in enumerate(amapP): # [c,d]
                r = self.g2p[I]
                for j, J in enumerate(amapP): # [c,d]
                    s = self.g2p[J]
                    if (r in self.eqs_c) and (s in self.eqs_c): # cc => I in eq1 and J in eq1
                        B11[I,J] += b1*Le[i,j] + He[i,j]
                        Lcc[r,s] += Le[i,j]
                    elif (r in self.eqs_c) and (s in self.eqs_d): # cd
                        Lcd[r,s] += Le[i,j]
                        Hcd[r,s] += He[i,j]
                F[I] += Fbe[i]

        # modify/augment matrices (and convert to csr)
        for r in self.eqs_b: Maa[r,r] = 1.0
        for I in self.eq2:   B11[I,I] = 1.0
        B11 = B11.tocsr()

        # convert matrices to compressed-row format (for efficiency)
        Maa = Maa.tocsr();  Oca = Oca.tocsr()
        Caa = Caa.tocsr();  Nca = Nca.tocsr()
        Kaa = Kaa.tocsr();  Lcc = Lcc.tocsr()
        Mab = Mab.tocsr();  Ocb = Ocb.tocsr()
        Cab = Cab.tocsr();  Ncb = Ncb.tocsr()
        Kab = Kab.tocsr();  Lcd = Lcd.tocsr()
        Qad = Qad.tocsr();  Hcd = Hcd.tocsr()

        # prescribed values
        U_b = zeros(nu) # size of U, but will set only presc ones
        V_b = zeros(nu) # size of V, but will set only presc ones
        A_b = zeros(nu) # size of A, but will set only presc ones
        P_d = zeros(np) # size of P, but will set only presc ones
        Z_d = zeros(np) # size of dPdt, but will set only presc ones
        U_b[self.eqs_b] = self.uvs_b
        P_d[self.eqs_d] = self.pvs_d

        # right-hand-side of split system
        dgU = - Mab*A_b - Cab*V_b - Kab*U_b + Qad*P_d
        dgP = - Ocb*A_b - Ncb*V_b - Lcd*Z_d - Hcd*P_d
        def G(t):
            g = F.copy()
            if load_mult != None:
                g[self.u2g] *= load_mult(t)
            for k, I in enumerate(self.presc_F_eqs):
                if isinstance(self.presc_F_vls[k], float): g[I] += self.presc_F_vls[k]
                else:                                      g[I] += self.presc_F_vls[k](t)
            g[self.u2g] += dgU
            g[self.p2g] += dgP
            return g

        # initial values
        t = t0
        U = zeros(nu) if U0==None else U0
        V = zeros(nu) if V0==None else V0
        P = zeros(np) if P0==None else P0

        # output object
        fo = Output(self, t, dtout, extrap, emethod, ext_dtout, vtu_fnkey, vtu_dtout)

        # initial accelerations and Z (rates of pw)
        W = G(t0)[self.u2g] - Caa*V - Kaa*U       # workspace
        A = spsolve(Maa, W)                       # accelerations
        Z = zeros(np)                             # rate of P
        A[self.eqs_b] = 0.0                       # prescribed values
        Z[self.eqs_d] = 0.0                       # prescribed values
        # time loop
        UP = zeros(neq)                           # augmented solution vector
        UP[self.u2g] = U                          # set U
        UP[self.p2g] = P                          # set P
        fo.out(t, UP)                             # first output
        W = zeros(neq)                            # augmented (UP) workspace
        while t < tf:                             # loop for all time steps
            m = a1*U + a2*V + a3*A                # auxiliary variable
            n = a4*U + a5*V + a6*A                # auxiliary variable
            o = b1*P + b2*Z                       # auxiliary variable
            if HHT:                               # HHT method
                ta = t + (1.0+alpha)*dt           # alpha-time
                nb = a8*n + alpha*V               # modified n variable
                W[self.u2g] = G(ta)  [self.u2g] + Maa*m + Caa*nb + alpha*Kaa*U
                W[self.p2g] = G(t+dt)[self.p2g] + Oca*m + Nca*n  + Lcc*o
            else:                                 # standard Newmark method
                g1 = G(t+dt)                      # new g
                W[self.u2g] = g1[self.u2g] + Maa*m + Caa*n
                W[self.p2g] = g1[self.p2g] + Oca*m + Nca*n + Lcc*o
            W[self.eq2] = self.presc_U_vls        # copy presc vals to W2
            UP = spsolve(B11, W)                  # solve: U = inv(A) * W
            U  = UP[self.u2g]                     # new U
            P  = UP[self.p2g]                     # new P
            V  = a4*U - n                         # new V
            A  = a1*U - m                         # new A
            Z  = b1*P - o                         # new dPdt
            t += dt                               # update time
            fo.out(t, UP)                         # output results

        # results
        fo.stop()
        return fo

    # Auxiliary methods ---------------------------------------------------------------------------

    def get_eq(self, vid, key):
        """
        Get equation number
        ===================
        INPUT:
            vid : vertex id
            key : u key such as 'ux', 'uy', 'pw'
        STORED:
            None
        RETURNS:
            The equation number
        """
        return self.eqs[vid][self.sovs[vid].index(key)]

    def calc_val_vert(self, vid, val):
        """
        Calculate value @ vertex
        ========================
        INPUT:
            vid : vertex id
            val : constant value or callback function, examples:
                    1D: lambda x: x**2.0
                    2D: lambda x,y: x+y
        STORED:
            None
        RETURNS:
            Either a constant value or the result of the 'lambda' function
        """
        if isinstance(val,int) or isinstance(val,float): # constant value specified
            v = float(val)                               # set v equal to val
        else:                                            # callback function specified
            x = self.m.V[vid][2]                         # node x-coordinate
            if self.m.ndim > 1:                          # 2D mesh
                y = self.m.V[vid][3]                     # node y-coordinate
                v = val(x, y)                            # compute value
            else:                                        # 1D mesh
                v = val(x)                               # compute value
        return v

    def dyn_coefficients(self, dt, theta1, theta2, HHT, alpha):
        """
        Get dynamics coefficients
        =========================
        INPUT:
            dt     : time step increment
            theta1 : Newmark parameter (gamma)  [0 <= theta1 <= 1]
            theta2 : Newmark parameter (2*beta) [0 <= theta2 <= 1]
            HHT    : use Hilber-Hughes-Taylor method ?
            alpha  : Hilber-Hughes-Taylor parameter [-1/3 <= alpha <= 0]
        STORED:
            None
        RETURNS:
            a1, a2, a3, a4, a5, a6, a7, a8 : Newmark/HHT coefficients
        Note:
            if HHT==True, theta1 and theta2 are automatically calculated
            for unconditional stability
        """
        # check
        if theta1 < 0.0 or theta1 > 1.0:
            raise Exception('theta1 must be between 0.0 and 1.0')
        if theta2 < 0.0 or theta2 > 1.0:
            raise Exception('theta2 must be between 0.0 and 1.0')

        # HHT method
        if HHT:
            if alpha < -1./3. or alpha > 0.:
                raise Exception('HHT method requires: -1/3 <= alpha <= 0 (%g is incorrect)'%alpha)
            theta1 = (1.0-2.0*alpha) / 2.0
            theta2 = ((1.0-alpha)**2.0) / 2.0

        # auxiliary variables
        h,  H      = dt,                  (dt**2.0)/2.0
        a1, a2, a3 = 1.0/(theta2*H),      h/(theta2*H),          1.0/theta2-1.0
        a4, a5, a6 = theta1*h/(theta2*H), 2.0*theta1/theta2-1.0, (theta1/theta2-1.0)*h

        # HHT method
        a7, a8 = a4, 1.0
        if HHT: a7, a8 = (1.0+alpha)*a4, (1.0+alpha)

        # return coefficients
        return a1, a2, a3, a4, a5, a6, a7, a8
