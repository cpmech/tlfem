# Copyright 2012 Dorival de Moraes Pedroso. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

from numpy import array, sqrt, zeros, dot, transpose, cross

class EelasticTri:
    def __init__(self, verts, params):
        """
        Elastic triangle
        ================
            Example of input:
                     global_id  tag   x    y
               verts = [[3,    -100, 0.0, 0.0],
                        [4,    -100, 1.0, 0.0],
                        [1,    -100, 0.0, 1.0]]
               params = {'E':1., 'nu':1., 'pstress':True, 'thick':1., 'rho':1.}
               pstress : plane-stress instead of plane-strain? [optional]
               thick   : thickness for plane-stress only [optional]
        INPUT:
            verts  : list of vertices
            params : dictionary of parameters
        STORED:
            E        : Young modulus
            nu       : Poisson's coefficient
            pstress  : is plane-stress problem instead of plane-strain?
            thick    : thickness for plane-stress problems
            has_load : has applied distributed load to any side
            D        : constitutive modulus (matrix)
            K        : stiffness matrix
            l        : lengths of edges
            n        : outward normals of each edge
            xc, yc   : coordinates of centroid
        """
        # check
        if len(verts) != 3:
            raise Exception('this element needs exactly 3 vertices')

        # set data
        self.rho      = 0.0           # density
        self.E        = params['E']   # Young modulus
        self.nu       = params['nu']  # Poisson's coefficient
        self.pstress  = False         # plane stress
        self.thick    = 1.0           # thickness of element
        self.has_load = False         # has distributed loads
        if 'rho'     in params: self.rho = params['rho']
        if 'pstress' in params:
            self.pstress = params['pstress'] # plane-stress?
            self.thick   = params['thick']   # thickness

        # nodal coordinates
        x0, y0 = verts[0][2], verts[0][3]
        x1, y1 = verts[1][2], verts[1][3]
        x2, y2 = verts[2][2], verts[2][3]

        # derived variables
        b0 = y1-y2;         b1 = y2-y0;         b2 = y0-y1;
        c0 = x2-x1;         c1 = x0-x2;         c2 = x1-x0;
        f0 = x1*y2-x2*y1;   f1 = x2*y0-x0*y2;   f2 = x0*y1-x1*y0

        # area
        self.A = (f0 + f1 + f2) / 2.0

        # check
        if self.A < 0.0:
            raise Exception('the area of element must be positive (nodes must be counter-clockwise ordered)')

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

        # store B matrix
        cf = 1.0 / (2.0 * self.A)
        sq = sqrt(2.0)
        self.B = cf * array([[b0    , 0.    , b1    , 0.    , b2    , 0.   ],
                             [0.    , c0    , 0.    , c1    , 0.    , c2   ],
                             [0.    , 0.    , 0.    , 0.    , 0.    , 0.   ],
                             [c0/sq , b0/sq , c1/sq , b1/sq , c2/sq , b2/sq]])

        # store K matrix
        cf = self.thick * self.A
        self.K = cf * dot(transpose(self.B), dot(self.D, self.B))

        # store M matrix
        cf = self.rho * self.thick * self.A / 12.0
        self.M = cf * array([[2., 0., 1., 0., 1., 0.],
                             [0., 2., 0., 1., 0., 1.],
                             [1., 0., 2., 0., 1., 0.],
                             [0., 1., 0., 2., 0., 1.],
                             [1., 0., 1., 0., 2., 0.],
                             [0., 1., 0., 1., 0., 2.]])

        # edges lengths
        self.l = [sqrt((x0-x1)**2.0 + (y0-y1)**2.0),
                  sqrt((x1-x2)**2.0 + (y1-y2)**2.0),
                  sqrt((x2-x0)**2.0 + (y2-y0)**2.0)]

        # outward normals
        self.n = [[(y1-y0)/self.l[0], (x0-x1)/self.l[0]],
                  [(y2-y1)/self.l[1], (x1-x2)/self.l[1]],
                  [(y0-y2)/self.l[2], (x2-x0)/self.l[2]]]

        # centroid
        self.xc = (x0 + x1 + x2) / 3.0
        self.yc = (y0 + y1 + y2) / 3.0

    def info(self):
        """
        Get Solution Variables
        ======================
        """
        return [('ux','uy'), ('ux','uy'), ('ux','uy')], {'fx':'ux', 'fy':'uy'}, \
               ['sx','sy','sz','sxy', 'ex','ey','ez','exy']

    def clear_bcs(self):
        """
        Clear boundary conditions
        =========================
        """
        self.has_load = False

    def set_nat_bcs(self, edge_num, bc_type, values):
        """
        Set natural boundary conditions
        ===============================
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

        # auxiliary variables
        e  = edge_num
        cf = self.thick * self.l[e] / 2.0

        # qn and qt => find qx and qy
        if bc_type == 'qnqt':
            qn = values[0]
            qt = values[1]
            nx = self.n[e][0]
            ny = self.n[e][1]
            qx = nx*qn - ny*qt
            qy = ny*qn + nx*qt

        # qx and qy
        else:
            qx = values[0]
            qy = values[1]

        # set Fq
        if not self.has_load: self.Fq = zeros(6)
        if   e==0: self.Fq += cf * array([qx, qy, qx, qy, 0., 0.])
        elif e==1: self.Fq += cf * array([0., 0., qx, qy, qx, qy])
        elif e==2: self.Fq += cf * array([qx, qy, 0., 0., qx, qy])
        self.has_load = True

    def calc_M(self):
        """
        Calculate matrix M = Me
        =======================
        """
        if abs(self.rho) < 1.0e-10:
            raise Exception('to use M matrix, rho must be positive')
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
        else:             return zeros(6)

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
        # strains and stresses
        eps = dot(self.B, Ue ) # strains
        sig = dot(self.D, eps) # stresses

        # calculate ez for plane-stress
        if self.pstress:
            eps[2] = -self.nu*(sig[0]+sig[1])/self.E

        # output data
        sq2 = sqrt(2.0)
        svs = {
            'sx' :[sig[0]    ],
            'sy' :[sig[1]    ],
            'sz' :[sig[2]    ],
            'sxy':[sig[3]/sq2],
            'ex' :[eps[0]    ],
            'ey' :[eps[1]    ],
            'ez' :[eps[2]    ],
            'exy':[eps[3]/sq2],
        }
        return svs

    def get_ips(self):
        """
        Get integration points
        ======================
        """
        return [(self.xc, self.yc)] # centroid
