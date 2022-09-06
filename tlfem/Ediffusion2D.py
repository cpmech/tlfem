# Copyright 2012 Dorival de Moraes Pedroso. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

from numpy      import array, zeros, dot, transpose, outer
from tlfem.quadrature import get_ips, ip_coords
from tlfem.shape      import get_shape_fcn, shape_derivs, face_integ, face_coords

class Ediffusion2D:
    def __init__(self, verts, params):
        """
        2D diffusion problem
        ====================
            Solving:         2         2
                    du      d u       d u
                rho == - kx ==== - ky ==== = s(x)
                    dt      d x2      d y2
            Example of input:
                      global_id  tag   x    y
               verts = [[3,    -100, 0.0, 0.0],
                        [4,    -100, 1.0, 0.0],
                        [7,    -100, 1.0, 1.0],
                        [1,    -100, 0.0, 1.0]]
               params = {'rho':1., 'kx':1., 'ky':1.,
                         'source':src_val_or_fcn, 'geom':qua4}
            Notes:
               src_val_or_fcn: can be a constant value or a
                               callback function such as
                               lambda x,y: x+y
               geom types:
                   tri3, tri6, qua4, qua8
        INPUT:
            verts  : list of vertices
            params : dictionary of parameters
        STORED:
            geom       : geometry key [tri3,tri6,qua4,qua8]
            fce, fcf   : element and face shape/deriv functions
            nne, nnf   : element and face number of nodes
            ipe, ipf   : integration points of element and edge/face
            rho        : rho parameter [optional]
            kx         : x-conductivity
            ky         : y-conductivity
            source     : the source term [optional]
            has_flux   : has flux applied to any side?
            has_conv   : has convection applied to any side?
            has_source : has source term
            xy         : matrix of coordinates of nodes
            D          : matrix with kx and ky
            C          : Ce matrix [rate of u]
            Kk         : Kke matrix [conductivity]
            Fs         : Fs vector [source]
        """
        # set geometry
        self.geom = params['geom']
        self.fce, self.nne, self.fcf, self.nnf = get_shape_fcn(self.geom)
        self.ipe, self.ipf = get_ips(self.geom)

        # check
        if len(verts) != self.nne:
            raise Exception('this element needs %d vertices exactly'%self.nne)

        # set data
        self.rho        = 0.0          # rho parameter
        self.kx         = params['kx'] # x-conductivity
        self.ky         = params['ky'] # y-conductivity
        self.has_flux   = False        # has flux bry conditions
        self.has_conv   = False        # has convection bry conds
        self.has_source = False        # has source term
        if 'rho'    in params: self.rho = params['rho']
        if 'source' in params:
            self.source     = params['source']
            self.has_source = True

        # matrix of nodal coordinates
        self.xy = zeros((2,self.nne)) # 2 => 2D
        for n, v in enumerate(verts):
            self.xy[0][n] = v[2] # x-coordinates
            self.xy[1][n] = v[3] # y-coordinates

        # conductivity matrix
        self.D = array([[self.kx,  0.0],
                        [0.0,  self.ky]])

        # Ce, Kke and source
        self.C  = zeros((self.nne,self.nne))
        self.Kk = zeros((self.nne,self.nne))
        self.Fs = zeros(self.nne)
        for ip in self.ipe:
            S, G, detJ = shape_derivs(self.xy, self.fce, ip)
            cf         = detJ * ip[2]
            self.C    += cf * self.rho * outer(S, S)
            self.Kk   += cf * dot(G, dot(self.D, transpose(G)))
            if self.has_source:
                if isinstance(self.source, float): # constant value
                    self.Fs += cf * self.source * S
                else:                              # function(x,y)
                    xip, yip = ip_coords(S, self.xy)
                    self.Fs += cf * self.source(xip, yip) * S

    def info(self):
        """
        Get Solution Variables
        ======================
        """
        return [('u') for i in range(self.nne)], {'Q':'u'}, ['wx','wy']

    def clear_bcs(self):
        """
        Clear boundary conditions
        =========================
        """
        self.has_flux = False # has flux bry conditions
        self.has_conv = False # has convection bry conds

    def set_nat_bcs(self, edge_num, bc_type, values):
        """
        Set natural (or mixed) boundary conditions
        ==========================================
            Example:
                if bc_type=='flux': flux_value = values
                if bc_type=='conv':
                    conv_coef    = values[0]
                    environ_temp = values[1]
        INPUT:
            edge_num : edge number => side of element
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
        # coordinates and indices of edge/face nodes
        xyf, fno = face_coords(self.xy, edge_num, self.nne)

        # flux term
        if bc_type=='q':
            if not self.has_flux: self.Ff = zeros(self.nne)
            for ip in self.ipf:
                Sf, detJf, _ = face_integ(xyf, self.fcf, ip)
                for i, n in enumerate(fno):
                    self.Ff[n] += detJf * ip[2] * values * Sf[i]
            self.has_flux = True

        # convection term
        elif bc_type == 'c':
            cc   = values[0]
            Tinf = values[1]
            if not self.has_conv:
                self.Kc = zeros((self.nne,self.nne))
                self.Fc = zeros(self.nne)
            for ip in self.ipf:
                Sf, detJf, _ = face_integ(xyf, self.fcf, ip)
                for i, n in enumerate(fno):
                    self.Fc[n] += detJf * ip[2] * cc * Tinf * Sf[i]
                    for j, m in enumerate(fno):
                        self.Kc[n,m] += detJf * ip[2] * cc * Sf[i] * Sf[j]
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
        F = zeros(self.nne)
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
        nip = len(self.ipe)
        svs = {'wx':zeros(nip), 'wy':zeros(nip)}
        for k, ip in enumerate(self.ipe):
            S, G, _ = shape_derivs(self.xy, self.fce, ip)
            gra     = dot(Ue, G)
            w       = -dot(self.D, gra)
            svs['wx'][k] = w[0]
            svs['wy'][k] = w[1]
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
