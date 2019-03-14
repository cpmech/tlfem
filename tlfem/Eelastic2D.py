# Copyright 2012 Dorival de Moraes Pedroso. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

from numpy      import array, zeros, dot, transpose, outer, sqrt
from quadrature import get_ips, ip_coords
from shape      import get_shape_fcn, shape_derivs, face_integ, face_coords

class Eelastic2D:
    def __init__(self, verts, params):
        """
        2D elasticity problem
        =====================
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
            geom     : geometry key [tri3,tri6,qua4,qua8]
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
        self.geom = params['geom']
        self.fce, self.nne, self.fcf, self.nnf = get_shape_fcn(self.geom)
        self.ipe, self.ipf = get_ips(self.geom)
        if 'ipe' in params: self.ipe = params['ipe']

        # check
        if len(verts) != self.nne:
            raise Exception('this element needs %d vertices exactly'%self.nne)

        # set data
        self.rho      = 0.0          # density
        self.E        = params['E']  # Young modulus
        self.nu       = params['nu'] # Poisson's coefficient
        self.pstress  = False        # plane stress
        self.thick    = 1.0          # thickness of element
        self.has_load = False        # has distributed loads
        if 'rho'     in params: self.rho = params['rho']
        if 'pstress' in params:
            self.pstress = params['pstress'] # plane-stress?
            self.thick   = params['thick']   # thickness

        # matrix of nodal coordinates
        self.xy = zeros((2,self.nne)) # 2 => 2D
        for n, v in enumerate(verts):
            self.xy[0][n] = v[2] # x-coordinates
            self.xy[1][n] = v[3] # y-coordinates

        # constitutive matrix
        nu = self.nu # Poisson coef
        if self.pstress:
            cf = self.E / (1.0 - nu**2.0)
            self.D = cf * array([[1., nu, 0., 0.   ],
                                 [nu, 1., 0., 0.   ],
                                 [0., 0., 0., 0.   ],
                                 [0., 0., 0., 1.-nu]])
        else: # plane strain
            cf = self.E / ((1.0 + nu) * (1.0 - 2.0 * nu))
            self.D = cf * array([[1.-nu , nu    , nu    , 0.      ],
                                 [nu    , 1.-nu , nu    , 0.      ],
                                 [nu    , nu    , 1.-nu , 0.      ],
                                 [0.    , 0.    , 0.    , 1.-2.*nu]])

        # K and M matrices
        self.K = zeros((self.nne*2,self.nne*2))
        self.M = zeros((self.nne*2,self.nne*2))
        for i, ip in enumerate(self.ipe):
            S, G, detJ = shape_derivs(self.xy, self.fce, ip)
            B  = self.calc_B(G)
            N  = self.calc_N(S)
            cf = detJ * ip[2] * self.thick
            self.K += cf * dot(transpose(B), dot(self.D, B))
            self.M += (cf * self.rho) * dot(transpose(N), N)

    def info(self):
        """
        Get Solution Variables
        ======================
        """
        return [('ux','uy') for i in range(self.nne)], {'fx':'ux', 'fy':'uy'}, \
               ['sx','sy','sz','sxy', 'ex','ey','ez','exy']

    def clear_bcs(self):
        """
        Clear boundary conditions
        =========================
        """
        self.has_load = False

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
        xyf, fno = face_coords(self.xy, edge_num, self.nne)

        # calc loads
        if not self.has_load: self.Fq = zeros(self.nne*2)
        for ip in self.ipf:
            Sf, detJf, dxydr = face_integ(xyf, self.fcf, ip)
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
            for i, n in enumerate(fno):
                self.Fq[0+n*2] += ip[2] * qx * Sf[i] * self.thick
                self.Fq[1+n*2] += ip[2] * qy * Sf[i] * self.thick

        # set flag
        self.has_load = True

    def calc_M(self):
        """
        Calculate matrix M = Me
        =======================
        """
        if abs(self.rho) < 1.0e-10:
            raise Exception('to use M matrix, rho must positive')
        return self.M

    def calc_K(self):
        """
        Calculate matrix K = Ke
        =======================
        """
        return self.K

    def calc_F(self):
        """
        Calculate vector F = Fe
        =======================
        """
        if self.has_load: return self.Fq
        else:             return zeros(self.nne*2)

    def secondary(self, Ue):
        """
        Calculate secondary variables
        =============================
        INPUT:
            Ue : vector of primary variables at each node
        STORED:
            None
        RETURNS:
            svs : dictionary with secondary values
        """
        nip = len(self.ipe)
        svs = {'sx':zeros(nip), 'sy':zeros(nip), 'sz':zeros(nip), 'sxy':zeros(nip),
               'ex':zeros(nip), 'ey':zeros(nip), 'ez':zeros(nip), 'exy':zeros(nip)}
        sq2 = sqrt(2.0)
        for k, ip in enumerate(self.ipe):
            S, G, _ = shape_derivs(self.xy, self.fce, ip)
            B   = self.calc_B(G)
            eps = dot(B, Ue)       # strains
            sig = dot(self.D, eps) # stresses
            svs['sx' ][k] = sig[0]
            svs['sy' ][k] = sig[1]
            svs['sz' ][k] = sig[2]
            svs['sxy'][k] = sig[3]/sq2
            svs['ex' ][k] = eps[0]
            svs['ey' ][k] = eps[1]
            svs['ez' ][k] = eps[2]
            svs['exy'][k] = eps[3]/sq2
        return svs

    def get_ips(self):
        """
        Get integration points
        ======================
        """
        ips = []
        for ip in self.ipe:
            S, _ = self.fce(ip[0], ip[1])
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
        B   = zeros((4,2*self.nne)) # 4 stress components, 2D*nne
        for n in range(self.nne):
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
        N = zeros((2,2*self.nne)) # 2D, 2D*nne
        for n in range(self.nne):
            N[0,0+n*2] = S[n]
            N[1,1+n*2] = S[n]
        return N
