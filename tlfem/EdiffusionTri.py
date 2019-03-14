# Copyright 2012 Dorival de Moraes Pedroso. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

from numpy           import array, sqrt, zeros, dot, cross
from scipy.integrate import dblquad

class EdiffusionTri:
    def __init__(self, verts, params):
        """
        2D diffusion problem (triangle)
        ===============================
            Solving:         2         2
                    du      d u       d u
                rho == - kx ==== - ky ==== = s(x)
                    dt      d x2      d y2
            Example of input:
                    global_id  tag   x    y
               verts = [[3,   -100, 0.0, 0.0],
                        [4,   -100, 1.0, 0.0],
                        [1,   -100, 0.0, 1.0]]
               params = {'rho':1., 'kx':1., 'ky':1.,
                         'source':src_val_or_fcn}
            Note:
                src_val_or_fcn: can be a constant value or a
                                callback function such as
                                lambda x,y: x+y
        INPUT:
            verts  : list of vertices
            params : dictionary of parameters
        STORED:
            rho        : rho parameter [optional]
            kx         : x-conductivity
            ky         : y-conductivity
            source     : the source term [optional]
            has_flux   : has flux applied to any side?
            has_conv   : has convection applied to any side?
            has_source : has source term
            A          : the area of triangle
            C          : Ce matrix [rate of u]
            Kk         : Kke matrix [conductivity]
            l          : length of edges
            G          : matrix gradient of shape functions
            xc, yc     : coordinates of centroid
            Fs         : Fs vector [source]
        """
        # check
        if len(verts) != 3:
            raise Exception('this element needs exactly 3 vertices')

        # set data
        self.rho        = 0.0           # rho parameter
        self.kx         = params['kx']  # x-conductivity
        self.ky         = params['ky']  # y-conductivity
        self.has_flux   = False         # has flux bry conditions
        self.has_conv   = False         # has convection bry conds
        self.has_source = False         # has source term
        if 'rho'    in params: self.rho = params['rho']
        if 'source' in params:
            self.source     = params['source']
            self.has_source = True

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

        # check area
        if self.A < 0.0:
            raise Exception('the area of element must be positive (nodes must be counter-clockwise ordered)')

        # store C matrix
        cf = self.rho * self.A / 12.0
        self.C = cf * array([[2.0, 1.0, 1.0],
                             [1.0, 2.0, 1.0],
                             [1.0, 1.0, 2.0]])

        # store Kk matrix
        k00 = (self.kx*b0*b0 + self.ky*c0*c0) / (4.0*self.A)
        k01 = (self.kx*b0*b1 + self.ky*c0*c1) / (4.0*self.A)
        k02 = (self.kx*b0*b2 + self.ky*c0*c2) / (4.0*self.A)
        k11 = (self.kx*b1*b1 + self.ky*c1*c1) / (4.0*self.A)
        k12 = (self.kx*b1*b2 + self.ky*c1*c2) / (4.0*self.A)
        k22 = (self.kx*b2*b2 + self.ky*c2*c2) / (4.0*self.A)
        self.Kk = array([[k00, k01, k02],
                         [k01, k11, k12],
                         [k02, k12, k22]])

        # gradient of shape functions matrix
        cf = 1.0 / (2.0 * self.A)
        self.G = cf * array([[b0,b1,b2],
                             [c0,c1,c2]])

        # edges lengths
        self.l = [sqrt((x0-x1)**2.0 + (y0-y1)**2.0),
                  sqrt((x1-x2)**2.0 + (y1-y2)**2.0),
                  sqrt((x2-x0)**2.0 + (y2-y0)**2.0)]

        # centroid
        self.xc = (x0 + x1 + x2) / 3.0
        self.yc = (y0 + y1 + y2) / 3.0

        # source term
        self.Fs = zeros(3)
        if self.has_source:
            if isinstance(self.source, float): # constant value
                cf = self.source * self.A / 3.0
                self.Fs = cf * array([1.0, 1.0, 1.0])
            else: # function(x)

                # shape functions
                S = [lambda x,y: 0.5 * (b0*x + c0*y + f0) / self.A,
                     lambda x,y: 0.5 * (b1*x + c1*y + f1) / self.A,
                     lambda x,y: 0.5 * (b2*x + c2*y + f2) / self.A]

                # find correct order of nodes for integration
                n = array([0, 1, 2])
                for _ in range(3):
                    if verts[n[0]][2] < verts[n[1]][2]: break
                    n += 1
                    n = n % 3
                else:
                    raise Exception('cannot find x0 < x1 for numerical integration of Fs')

                # nodes coordinates
                X = [x0, x1, x2]
                Y = [y0, y1, y2]

                # functions for each line drawing the triangle's edges
                L = [lambda x: Y[n[0]] + (x-X[n[0]]) * (Y[n[1]]-Y[n[0]]) / (X[n[1]]-X[n[0]]), # l01
                     lambda x: Y[n[0]] + (x-X[n[0]]) * (Y[n[2]]-Y[n[0]]) / (X[n[2]]-X[n[0]]), # l02
                     lambda x: Y[n[2]] + (x-X[n[2]]) * (Y[n[1]]-Y[n[2]]) / (X[n[1]]-X[n[2]])] # l21

                # integrand functions and limits
                fcn = [lambda y,x: S[0](x,y) * self.source(x,y),
                       lambda y,x: S[1](x,y) * self.source(x,y),
                       lambda y,x: S[2](x,y) * self.source(x,y)]
                I0 = [[None, 0., 0., None, None], # fcnA, xmin,xmax, ymin(x),ymax(x)
                      [None, 0., 0., None, None]] # fcnB, xmin,xmax, ymin(x),ymax(x)
                I1 = [[None, 0., 0., None, None],
                      [None, 0., 0., None, None]]
                I2 = [[None, 0., 0., None, None],
                      [None, 0., 0., None, None]]
                if abs(X[n[2]]-X[n[0]]) > 1.0e-8:
                    I0[0] = [fcn[0], X[n[0]], X[n[2]], L[0], L[1]]
                    I1[0] = [fcn[1], X[n[0]], X[n[2]], L[0], L[1]]
                    I2[0] = [fcn[2], X[n[0]], X[n[2]], L[0], L[1]]
                if abs(X[n[1]]-X[n[2]]) > 1.0e-8:
                    I0[1] = [fcn[0], X[n[2]], X[n[1]], L[0], L[2]]
                    I1[1] = [fcn[1], X[n[2]], X[n[1]], L[0], L[2]]
                    I2[1] = [fcn[2], X[n[2]], X[n[1]], L[0], L[2]]

                # perform integration
                for k in range(2):
                    if I0[k][0]!=None: self.Fs[0] += dblquad(I0[k][0], I0[k][1], I0[k][2], I0[k][3], I0[k][4])[0]
                    if I1[k][0]!=None: self.Fs[1] += dblquad(I1[k][0], I1[k][1], I1[k][2], I1[k][3], I1[k][4])[0]
                    if I2[k][0]!=None: self.Fs[2] += dblquad(I2[k][0], I2[k][1], I2[k][2], I2[k][3], I2[k][4])[0]

    def info(self):
        """
        Get Solution Variables
        ======================
        """
        return [('u'), ('u'), ('u')], {'Q':'u'}, ['wx','wy']

    def clear_bcs(self):
        """
        Clear boundary conditions
        =========================
        """
        self.has_flux = False
        self.has_conv = False

    def set_nat_bcs(self, edge_num, bc_type, values):
        """
        Set natural (or mixed) boundary conditions
        ==========================================
            Example:
                if bc_type=='q': flux_value = values
                if bc_type=='c':
                    conv_coef    = values[0]
                    environ_temp = values[1]
        INPUT:
            edge_num : edge number => side of triangle
            bc_type  : 'flux' or 'conv' (convection)
            values   : list with 1 or 2 values
        STORED:
            Ff       : flux vector
            Fc       : convection vector
            Kc       : convection matrix
            has_flux : has flux applied to any side?
            has_conv : has convection applied to any side?
        RETURNS:
            None
        """
        # flux term
        if bc_type=='q':
            e  = edge_num
            cf = values * self.l[e] / 2.0
            if not self.has_flux:
                self.Ff = zeros(3)
            if   e==0: self.Ff += cf * array([1.0,1.0,0.0])
            elif e==1: self.Ff += cf * array([0.0,1.0,1.0])
            elif e==2: self.Ff += cf * array([1.0,0.0,1.0])
            self.has_flux = True

        # convection term
        elif bc_type == 'c':
            e    = edge_num
            cc   = values[0] # convection coefficient
            Tinf = values[1] # environment temperature
            cfK  = cc * self.l[e]        / 6.0
            cfF  = cc * self.l[e] * Tinf / 2.0
            if not self.has_conv:
                self.Kc = zeros((3,3))
                self.Fc = zeros(3)
            if e==0:
                self.Kc += cfK * array([[2.0, 1.0, 0.0],
                                        [1.0, 2.0, 0.0],
                                        [0.0, 0.0, 0.0]])
                self.Fc += cfF * array( [1.0, 1.0, 0.0])
            elif e==1:
                self.Kc += cfK * array([[0.0, 0.0, 0.0],
                                        [0.0, 2.0, 1.0],
                                        [0.0, 1.0, 2.0]])
                self.Fc += cfF * array( [0.0, 1.0, 1.0])
            elif e==2:
                self.Kc += cfK * array([[2.0, 0.0, 1.0],
                                        [0.0, 0.0, 0.0],
                                        [1.0, 0.0, 2.0]])
                self.Fc += cfF * array( [1.0, 0.0, 1.0])
            self.has_conv = True
        else:
            raise Exception('boundary condition type == %s is not available' % bc_type)

    def calc_C(self):
        """
        Calculate matrix C = Ce
        =======================
        """
        if abs(self.rho) < 1.0e-10:
            raise Exception('to use C matrix, rho must positive')
        return self.C

    def calc_K(self):
        """
        Calculate matrix K = Ke = Kk + Kc
        =================================
        """
        if self.has_conv: return self.Kk + self.Kc
        else:             return self.Kk

    def calc_F(self):
        """
        Calculate vector F = Fe
        =======================
        """
        F = zeros(3)
        if self.has_source: F += self.Fs
        if self.has_flux:   F += self.Ff
        if self.has_conv:   F += self.Fc
        return F

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
        gra = dot(self.G, Ue) # gradients
        svs = {               # secondary values at each ip
            'wx': [-self.kx * gra[0]],
            'wy': [-self.ky * gra[1]]
        }
        return svs

    def get_ips(self):
        """
        Get integration points
        ======================
        """
        return [(self.xc, self.yc)] # integration points == centroid
