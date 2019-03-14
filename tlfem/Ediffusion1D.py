# Copyright 2012 Dorival de Moraes Pedroso. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

from numpy           import array, zeros
from scipy.integrate import quad

class Ediffusion1D:
    def __init__(self, verts, params):
        """
        1D diffusion problem
        ====================
            Solving:         2
                    du      d u
                rho == - kx ==== + beta*u = s(x)
                    dt      d x2
            Example of input:
                     global_id  tag   x
                verts = [[3,   -100, 0.75],
                         [4,   -100, 1.0 ]]
                params = {'rho':1.0, 'beta':1.0, 'kx':1.0,
                          'source':src_val_or_fcn}
            Note:
                src_val_or_fcn: can be a constant value or a
                                callback function such as
                                lambda x: x**2.
        INPUT:
            verts  : list of vertices
            params : dictionary of parameters
        STORED:
            x0, x1     : nodal coordinates
            l          : length of element
            rho        : rho parameter [optional]
            beta       : beta parameter [optional]
            kx         : x-conductivity
            source     : the source term [optional]
            has_source : has source term
            C          : Ce matrix [rate of u]
            K          : Ke matrix [conductivity]
            Fs         : Fs vector [source]
        """
        # check
        if len(verts) != 2:
            raise Exception('this element needs exactly 2 vertices')

        # set data
        self.x0         = verts[0][2]       # left node coord
        self.x1         = verts[1][2]       # right node coord
        self.l          = self.x1 - self.x0 # length of element
        self.rho        = 0.0               # rho parameter
        self.beta       = 0.0               # beta parameter
        self.kx         = params['kx']      # kx parameter
        self.has_source = False             # has source term
        if 'rho'    in params: self.rho  = params['rho']
        if 'beta'   in params: self.beta = params['beta']
        if 'source' in params:
            self.source = params['source']
            self.has_source = True

        # Ce
        cf = self.rho * self.l / 6.0 # coefficient of C
        self.C = cf * array([[2., 1.],
                             [1., 2.]])
        # Ke
        cfb = self.beta * self.l / 6.0 # coefficient of Kb
        cfk = self.kx / self.l         # coefficient of Kk
        self.K = cfb * array([[ 2., 1.],
                              [ 1., 2.]]) \
               + cfk * array([[ 1.,-1.],
                              [-1., 1.]])
        # source term
        self.Fs = zeros(2)
        if self.has_source:
            if isinstance(self.source, float): # constant value
                cf = self.source * self.l / 2.0
                self.Fs = cf * array([1.0, 1.0])
            else: # function(x)
                i0 = lambda x: ((self.x1-x)/self.l) * self.source(x)
                i1 = lambda x: ((x-self.x0)/self.l) * self.source(x)
                self.Fs = array([quad(i0, self.x0, self.x1)[0],
                                 quad(i1, self.x0, self.x1)[0]])

    def info(self):
        """
        Get information
        ===============
        """
        return [('u'), ('u')], {'Q':'u'}, ['wx']

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
        Calculate matrix K = Ke = Keb + Kek
        ===================================
        """
        return self.K

    def calc_F(self):
        """
        Calculate vector F = Fe
        =======================
        """
        return self.Fs

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
        return {'wx': [ - self.kx * (Ue[1] - Ue[0]) / self.l ]}

    def get_ips(self):
        """
        Get integration points
        ======================
        """
        return [((self.x0 + self.x1)/2.0 , )] # the last comma is required to force a tuple
