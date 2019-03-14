# Copyright 2012 Dorival de Moraes Pedroso. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

from numpy import array, sqrt, zeros, dot, transpose

class EelasticRod:
    def __init__(self, verts, params):
        """
        Elastic rod problem
        ===================
            Example of input:
                     global_id  tag   x     y
                verts = [[3,   -100, 0.75, 0.0],
                         [4,   -100, 1.0,  0.0]]
               params = {'E':1.0, 'A':1.0, 'rho':1.0}
        STORED:
            x0, y0 : left node coordinates
            x1, y1 : right node coordinates
            l      : length of element
            E      : Young modulus
            A      : cross-sectional area
            rho    : density [optional]
            T      : transformation matrix (rotation)
            K      : stiffness matrix
            M      : mass matrix
        """
        # set data
        self.x0  = verts[0][2]         # left node x-coord
        self.y0  = verts[0][3]         # left node y-coord
        self.x1  = verts[1][2]         # right node x-coord
        self.y1  = verts[1][3]         # right node y-coord
        dx       = self.x1 - self.x0   # delta x
        dy       = self.y1 - self.y0   # delta y
        self.l   = sqrt(dx**2.+dy**2.) # length of element
        self.E   = params['E']         # Young modulus
        self.A   = params['A']         # cross-sectional area
        self.rho = 0.0                 # density
        if 'rho' in params: self.rho = params['rho']

        # global-to-local transformation matrix
        c = dx / self.l
        s = dy / self.l
        self.T = array([[c, s, 0, 0],
                        [0, 0, c, s]])

        # K and M matrices
        ck = self.E   * self.A / self.l
        cm = self.rho * self.A * self.l / 6.0
        self.K = ck * array([[ c*c,  c*s, -c*c, -c*s],
                             [ c*s,  s*s, -c*s, -s*s],
                             [-c*c, -c*s,  c*c,  c*s],
                             [-c*s, -s*s,  c*s,  s*s]])
        self.M = cm * array([[2., 0., 1., 0.],
                             [0., 2., 0., 1.],
                             [1., 0., 2., 0.],
                             [0., 1., 0., 2.]])

    def info(self):
        """
        Get Solution Variables
        ======================
        """
        return [('ux','uy'), ('ux','uy')], {'fx':'ux', 'fy':'uy'}, ['N']

    def calc_M(self):
        """
        Calculate matrix M = Me
        =======================
        """
        if abs(self.rho) < 1.0e-10:
            raise Exception('to use M matrix, rho must be positive')
        return self.M

    def calc_C(self):
        """
        Calculate matrix C = Ce
        =======================
        """
        pass

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
        return zeros(4)

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
        # output data
        ua  = dot(self.T, Ue)      # axial displacements
        ea  = (ua[1]-ua[0])/self.l # axial strain
        sa  = self.E * ea          # axial stress
        N   = self.A * sa          # axial normal force
        svs = {'N':[N]}            # secondary values @ centroid
        return svs

    def get_ips(self):
        """
        Get integration points
        ======================
        """
        return [((self.x0 + self.x1)/2.0, (self.y0 + self.y1)/2.0)] # centroid
