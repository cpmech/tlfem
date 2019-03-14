# Copyright 2012 Dorival de Moraes Pedroso. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

from numpy import array, sqrt, zeros, ones, dot, transpose

class EelasticBeam:
    def __init__(self, verts, params):
        """
        Elastic beam problem (Euler/Bernoulli)
        ======================================
            Example of input:
                     global_id  tag   x     y
                verts = [[3,   -100, 0.75, 0.0],
                         [4,   -100, 1.0,  0.0]]
               params = {'E':1.0, 'A':1.0, 'I':1.0, 'rho':1.0,
                         'qnqt':(qnl,qnr,qt), 'nop':11}
               Notes: [optional]
                   qnqt is a distributed load along the Beam
                   qnl  is the left value of this load and
                   qnr  is the right value
                   qt   is a tangential distributed load
                   nop  number of points for output of N, V, M
        STORED:
            x0, y0 : left node coordinates
            x1, y1 : right node coordinates
            l      : length of element
            E      : Young modulus
            A      : cross-sectional area
            I      : inertia coefficient (Izz)
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
        self.I   = params['I']         # Inertia
        self.rho = 0.0                 # density
        if 'rho' in params: self.rho = params['rho']

        # distributed load
        self.qnqt     = (0.,0.,0.) # distributed load: (qnL,qnR,qt)
        self.has_load = False      # has distributed load?
        if 'qnqt' in params: self.qnqt, self.has_load = params['qnqt'], True

        # output
        self.nop = 11 # number points for output of N, V and M
        if 'nop' in params: self.nop = params['nop']

        # global-to-local transformation matrix
        c = dx / self.l
        s = dy / self.l
        self.c = c
        self.s = s
        self.T = array([[  c,  s,  0.,  0.,  0.,  0.],
                        [ -s,  c,  0.,  0.,  0.,  0.],
                        [ 0., 0.,  1.,  0.,  0.,  0.],
                        [ 0., 0.,  0.,   c,   s,  0.],
                        [ 0., 0.,  0.,  -s,   c,  0.],
                        [ 0., 0.,  0.,  0.,  0.,  1.]])

        # K matrix
        l  = self.l
        ll = self.l * self.l
        m  = self.E * self.A / self.l
        n  = self.E * self.I / (ll*self.l)
        Kl = array([[  m,        0.,       0.,   -m,        0.,       0. ],
                    [ 0.,   12.  *n,  6.*l *n,   0.,  -12.  *n,  6.*l *n ],
                    [ 0.,    6.*l*n,  4.*ll*n,   0.,   -6.*l*n,  2.*ll*n ],
                    [ -m,        0.,       0.,    m,        0.,       0. ],
                    [ 0.,  -12.  *n, -6.*l *n,   0.,   12.  *n, -6.*l *n ],
                    [ 0.,    6.*l*n,  2.*ll*n,   0.,   -6.*l*n,  4.*ll*n ]])
        self.K = dot(transpose(self.T), dot(Kl, self.T))

        # M matrix
        m  = self.rho * self.A * self.l / 420.
        Ml = array([[ 140.*m ,    0.     ,   0.      ,   70.*m ,    0.     ,    0.      ],
                    [   0.   ,  156.*m   ,  22.*l*m  ,    0.   ,   54.*m   ,  -13.*l*m  ],
                    [   0.   ,   22.*l*m ,   4.*ll*m ,    0.   ,   13.*l*m ,   -3.*ll*m ],
                    [  70.*m ,    0.     ,   0.      ,  140.*m ,    0.     ,    0.      ],
                    [   0.   ,   54.*m   ,  13.*l*m  ,    0.   ,  156.*m   ,  -22.*l*m  ],
                    [   0.   ,  -13.*l*m ,  -3.*ll*m ,    0.   ,  -22.*l*m ,    4.*ll*m ]])
        self.M = dot(transpose(self.T), dot(Ml, self.T))

        # F vector
        if self.has_load:
            qnL, qnR, qt = self.qnqt
            # local Fq vector
            Fql = array([qt*l/2.0, l*(7.0*qnL+3.0*qnR)/20.0,  l*l*(3.0*qnL+2.0*qnR)/60.0,
                         qt*l/2.0, l*(3.0*qnL+7.0*qnR)/20.0, -l*l*(2.0*qnL+3.0*qnR)/60.0])
            # rotated Fq vector
            self.Fq = dot(transpose(self.T), Fql)
        else:
            self.Fq = zeros(6)

    def info(self):
        """
        Get Solution Variables
        ======================
        """
        return [('ux','uy','rz'), ('ux','uy','rz')], {'fx':'ux', 'fy':'uy', 'mz':'rz'}, \
               ['N','V','M']

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
        return self.Fq

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
        # displacements in local coordinates
        Ul = dot(self.T, Ue)

        # axial force
        N = self.E * self.A * (Ul[3] - Ul[0]) / self.l

        # results
        svs = {'N':N*ones(self.nop),
               'V':zeros(self.nop),
               'M':zeros(self.nop)}

        # calculate V and M
        if self.nop == 1:
            r = 0.5 * self.l # local coordinate
            svs['V'][0], svs['M'][0] = self.calc_V_M(Ul, r)
        else:
            dr = self.l / float(self.nop-1) # delta r
            for i in range(self.nop):       # for each point
                r = i * dr                  # local coordinate
                svs['V'][i], svs['M'][i] = self.calc_V_M(Ul, r)
        return svs

    def get_ips(self):
        """
        Get integration points (output stations)
        ========================================
        """
        sta = [] # output stations
        if self.nop == 1:
            r = 0.5 * self.l                                 # local coordinate
            sta.append((self.x0+r*self.c, self.y0+r*self.s)) # global coordinate
        else:
            dr = self.l / float(self.nop-1)                      # delta r
            for i in range(self.nop):                            # for each point
                r = i * dr                                       # local coordinate
                sta.append((self.x0+r*self.c, self.y0+r*self.s)) # global coordinate
        return sta

    def calc_V_M(self, Ul, r):
        """
        Calculate results
        =================
        INPUT:
            Ul : displacements in local coordinates
            r  : natural coordinate   0 <= r <= self.l
        RETURNS:
            V : shear force
            M : bending moment
        """
        # auxiliary variables
        l   = self.l
        ll  = l**2.0
        lll = l**3.0

        # shear force
        V = self.E * self.I * ((12.*Ul[1])/lll + (6.*Ul[2])/ll - (12.*Ul[4])/lll + (6.*Ul[5])/ll)

        # bending moment
        M = self.E * self.I * (Ul[1]*((12.*r)/lll-6./ll) + Ul[2]*((6.*r)/ll-4./l) + Ul[4]*(6./ll-(12.*r)/lll) + Ul[5]*((6.*r)/ll-2./l))

        # corrections due to applied loads
        if self.has_load:
            qnL, qnR, _ = self.qnqt
            rr  = r**2.0
            rrr = r**3.0
            V += -(3.*qnR*ll +7.*qnL*ll-20.*qnL*r*l -10.*qnR*rr  +10.*qnL*rr)/(20.*l)
            M +=  (2.*qnR*lll+3.*qnL*lll-9.*qnR*r*ll-21.*qnL*r*ll+30.*qnL*rr*l+10.*qnR*rrr-10.*qnL*rrr)/(60.*l)
            if qnL>0.0: M = -M  # swap the sign of M

        return V, M
