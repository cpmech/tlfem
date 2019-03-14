# Copyright 2012 Dorival de Moraes Pedroso. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

from numpy      import array, zeros, dot, transpose, outer, sqrt, ones
from quadrature import get_ips, ip_coords
from shape      import get_shape_fcn, shape_derivs, face_integ, face_coords

class EelasticPorous2D:
    def __init__(self, verts, params):
        """
        2D elasticity problem with porous media
        =======================================
            Example of input:
                      global_id  tag   x    y
               verts = [[3,    -100, 0.0, 0.0],
                        [4,    -100, 1.0, 0.0],
                        [7,    -100, 1.0, 1.0],
                        [1,    -100, 0.0, 1.0]]
               params = {'E':1., 'nu':1., 'pstress':True, 'thick':1.,
                         'geom':'tri3', 'ipe':'QuaIp4', 'rho':1.0}
               pstress : plane-stress instead of plane-strain? [optional]
               thick   : thickness for plane-stress only [optional]
               ipe     : [optional] integration points
               geom types:
                   tri3, tri6, qua4, qua8
        INPUT:
            verts  : list of vertices
            params : dictionary of parameters
        STORED:
            geom     : geometry key pair, ex: (tri6,tri3), (qua8,qua4)
            fce, fcf : element and face shape/deriv functions
            nne, nnf : element and face number of nodes
            ipe, ipf : integration points of element and edge/face
            rho      : density [optional]
            E        : Young modulus
            nu       : Poisson's coefficient
            pstress  : is plane-stress problem instead of plane-strain?
            thick    : thickness for plane-stress problems
            has_load : has applied distributed load to any side
            xy       : matrix of coordinates of nodes
            D        : constitutive modulus (matrix)
            K        : stiffness matrix
        """
        # set geometry
        if isinstance(params['geom'],str):
            self.geoU = params['geom']
            self.geoP = params['geom']
        else:
            self.geoU = params['geom'][0]
            self.geoP = params['geom'][1]
        self.fceU, self.nneU, self.fcfU, self.nnfU = get_shape_fcn(self.geoU)
        self.fceP, self.nneP, self.fcfP, self.nnfP = get_shape_fcn(self.geoP)
        self.ipeU, self.ipfU = get_ips(self.geoU)
        self.ipeP, self.ipfP = get_ips(self.geoP)
        if 'ipeU' in params: self.ipeU = params['ipeU']
        if 'ipeP' in params: self.ipeP = params['ipeP']

        # check
        if len(verts) != self.nneU:
            raise Exception('this element needs %d vertices exactly'%self.nneU)

        # set data
        self.E        = params['E']    # Young modulus
        self.nu       = params['nu']   # Poisson's coefficient
        self.kx       = params['kx']   # x-conductivity
        self.ky       = params['ky']   # y-conductivity
        self.has_load = False          # has distributed loads
        self.has_flux = False          # has flux specified

        # other parameters
        rhoS = params['rhoS'] # density of solid grains
        rhoW = params['rhoW'] # water real density
        gamW = params['gamW'] # water unit weight of reference
        eta  = params['eta']  # porosity
        Kw   = params['Kw']   # water bulk modulus
        rho  = (1.0-eta)*rhoS + eta*rhoW # density of mixture
        Cwb  = eta / Kw

        # matrix of nodal coordinates
        self.xyU = zeros((2,self.nneU)) # 2 => 2D
        self.xyP = zeros((2,self.nneP)) # 2 => 2D
        for n, v in enumerate(verts):
            self.xyU[0][n] = v[2] # x-coordinates
            self.xyU[1][n] = v[3] # y-coordinates
            if n < self.nneP:
                self.xyP[0][n] = v[2] # x-coordinates
                self.xyP[1][n] = v[3] # y-coordinates

        # constitutive matrix (plane strain)
        nu = self.nu # Poisson coef
        cf = self.E / ((1.0 + nu) * (1.0 - 2.0 * nu))
        self.D = cf * array([[1.-nu , nu    , nu    , 0.      ],
                             [nu    , 1.-nu , nu    , 0.      ],
                             [nu    , nu    , 1.-nu , 0.      ],
                             [0.    , 0.    , 0.    , 1.-2.*nu]])

        # conductivity matrix
        self.kap = array([[self.kx / gamW,  0.0],
                          [0.0,  self.ky / gamW]])

        # iota tensor
        self.iota = array([1., 1., 1., 0.])

        # K, M, Q and O matrices
        self.K = zeros((self.nneU*2,self.nneU*2))
        self.M = zeros((self.nneU*2,self.nneU*2))
        self.Q = zeros((self.nneU*2,self.nneP))
        self.O = zeros((self.nneP,self.nneU*2))
        for ip in self.ipeU:
            # K and M
            S, G, detJ = shape_derivs(self.xyU, self.fceU, ip)
            B  = self.calc_B(G)
            N  = self.calc_N(S)
            cf = detJ * ip[2]
            self.K += cf * dot(transpose(B), dot(self.D, B))
            self.M += (cf * rho) * dot(transpose(N), N)
            # Q and O
            Sb, Gb, _ = shape_derivs(self.xyP, self.fceP, ip)
            self.Q += cf * dot(transpose(B), outer(self.iota, Sb))
            self.O += (cf * rhoW) * dot(Gb, dot(self.kap, N))
            #print detJ, detJb, detJ-detJb

        # L and H matrices
        self.L = zeros((self.nneP,self.nneP))
        self.H = zeros((self.nneP,self.nneP))
        for ip in self.ipeP:
            Sb, Gb, detJb = shape_derivs(self.xyP, self.fceP, ip)
            cf      = detJb * ip[2]
            self.L += (cf * Cwb) * outer(Sb, Sb)
            self.H += cf * dot(Gb, dot(self.kap, transpose(Gb)))

        # local equation numbers
        neqs      = self.nneU*2 + self.nneP
        self.eqsP = [2+n*3 for n in range(self.nneP)]
        self.eqsU = [i for i in range(neqs) if i not in self.eqsP]

    def info(self):
        """
        Get Solution Variables
        ======================
        """
        sovs = [('ux','uy','pw') for _ in range(self.nneP)]
        sovs.extend([('ux','uy') for _ in range(self.nneP,self.nneU)])
        return sovs, {'fx':'ux', 'fy':'uy'}, \
               ['sxE','syE','szE','sxyE', 'ex','ey','ez','exy']

    def get_amaps(self, amap):
        """
        Get separated assembly maps
        ===========================
        INPUT:
            amap : global assembly map corresponding to 'sovs' returned by info
        RETURNS:
            amapU : assembly map for U sovs
            amapP : assembly map for P sovs
        """
        return [amap[i] for i in self.eqsU], [amap[i] for i in self.eqsP]

    def clear_bcs(self):
        """
        Clear boundary conditions
        =========================
        """
        self.has_load = False
        self.has_flux = False

    def set_nat_bcs(self, edge_num, bc_type, values):
        """
        Set natural (or mixed) boundary conditions
        ==========================================
        INPUT:
            edge_num : edge number => side of triangle
            bc_type  : 'qxqy', 'qnqt' distributed loads
            values   : list with 2 items => values of distributed loads
        STORED:
            Fq       : force vector
            has_load : has loads applied to any side?
        RETURNS:
            None
        """
        # check
        if not bc_type in ['qxqy', 'qnqt']:
            raise Exception('boundary condition type == %s is not available' % bc_type)

        # coordinates and indices of edge/face nodes
        xyfU, fnoU = face_coords(self.xyU, edge_num, self.nneU)

        # calc loads
        if not self.has_load: self.Fq = zeros(self.nneU*2)
        for ip in self.ipfU:
            Sf, detJf, dxydr = face_integ(xyfU, self.fcfU, ip)
            # qn and qt => find qx and qy
            if bc_type == 'qnqt':
                nx, ny = dxydr[1], -dxydr[0] # normal multiplied by detJf
                qn, qt = values[0], values[1]
                qx = nx*qn - ny*qt
                qy = ny*qn + nx*qt
            # qx and qy
            else:
                qx = values[0] * detJf
                qy = values[1] * detJf
            for i, n in enumerate(fnoU):
                self.Fq[0+n*2] += ip[2] * qx * Sf[i]
                self.Fq[1+n*2] += ip[2] * qy * Sf[i]

        # set flag
        self.has_load = True

    def calc_M(self): return self.M
    def calc_K(self): return self.K
    def calc_O(self): return self.O
    def calc_Q(self): return self.Q
    def calc_L(self): return self.L
    def calc_H(self): return self.H

    def calc_F(self):
        """
        Calculate vector F = Fe
        =======================
        """
        if self.has_load: return self.Fq
        else:             return zeros(self.nneU*2)

    def calc_Fb(self):
        """
        Calculate vector Fb = Fbe
        =========================
        """
        if self.has_flux: return self.Fb
        else:             return zeros(self.nneP)

    def secondary(self, UPe):
        """
        Calculate secondary variables
        =============================
        INPUT:
            UPe : vector of primary variables at each node
        STORED:
            None
        RETURNS:
            svs : dictionary with secondary values
        """
        Ue  = array([UPe[i] for i in self.eqsU])
        Pe  = array([UPe[i] for i in self.eqsP])
        nip = len(self.ipeU)
        svs = {'sxE':zeros(nip), 'syE':zeros(nip), 'szE':zeros(nip), 'sxyE':zeros(nip), # effective values
               'ex' :zeros(nip), 'ey' :zeros(nip), 'ez' :zeros(nip), 'exy' :zeros(nip)}
        sq2 = sqrt(2.0)
        for k, ip in enumerate(self.ipeU):
            S, G, _ = shape_derivs(self.xyU, self.fceU, ip)
            B   = self.calc_B(G)
            eps = dot(B, Ue)       # strains
            sig = dot(self.D, eps) # stresses
            svs['sxE' ][k] = sig[0]
            svs['syE' ][k] = sig[1]
            svs['szE' ][k] = sig[2]
            svs['sxyE'][k] = sig[3]/sq2
            svs['ex'  ][k] = eps[0]
            svs['ey'  ][k] = eps[1]
            svs['ez'  ][k] = eps[2]
            svs['exy' ][k] = eps[3]/sq2
        return svs

    def get_ips(self):
        """
        Get integration points
        ======================
        """
        ips = []
        for ip in self.ipeU:
            S, _ = self.fceU(ip[0], ip[1])
            ips.append(ip_coords(S, self.xy))
        return ips

    def calc_B(self, G):
        """
        Calculate B matrix
        ==================
        INPUT:
            G : gradient of shape functions at a selected ip
        RETURNS:
            B : B matrix
        """
        sq2 = sqrt(2.0)
        B   = zeros((4,2*self.nneU)) # 4 stress components, 2D*nne
        for n in range(self.nneU):
            B[0,0+n*2] = G[n,0]
            B[1,1+n*2] = G[n,1]
            B[3,0+n*2] = G[n,1]/sq2
            B[3,1+n*2] = G[n,0]/sq2
        return B

    def calc_N(self, S):
        """
        Calculate N matrix
        ==================
        INPUT:
            S : shape functions at a selected ip
        RETURNS:
            N : N matrix
        """
        N = zeros((2,2*self.nneU)) # 2D, 2D*nne
        for n in range(self.nneU):
            N[0,0+n*2] = S[n]
            N[1,1+n*2] = S[n]
        return N
