# Copyright 2012 Dorival de Moraes Pedroso. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

from numpy        import zeros, dot, sqrt
from scipy.linalg import inv, det
from tlfem.mesh   import Edg2Vids

def lin2_fcn(r, s):
    """
         -1     0    +1
          0-----------1-->r
    """
    S, dSdrs = zeros(2), zeros(2)

    S[0] = 0.5 * (1.0 - r)
    S[1] = 0.5 * (1.0 + r)

    dSdrs[0] = -0.5
    dSdrs[1] =  0.5

    return S, dSdrs

def lin3_fcn(r, s):
    """
         -1     0    +1
          0-----2-----1-->r
    """
    S, dSdrs = zeros(3), zeros(3)

    S[0] = 0.5 * (r * r - r)
    S[1] = 0.5 * (r * r + r)
    S[2] = 1.0 -  r * r

    dSdrs[0] = r - 0.5
    dSdrs[1] = r + 0.5
    dSdrs[2] = -2.0 * r

    return S, dSdrs

def tri3_fcn(r, s):
    """      s
            |
            2, (0,1)
            | ',
            |   ',
            |     ',
            |       ',
            |         ',
            |           ',
            |             ',
            |               ',
            | (0,0)           ', (1,0)
            0-------------------1 ---- r
    """
    S, dSdrs = zeros(3), zeros((3,2))

    S[0] = 1.0 - r - s
    S[1] = r
    S[2] = s

    dSdrs[0,0] = -1.0
    dSdrs[1,0] =  1.0
    dSdrs[2,0] =  0.0

    dSdrs[0,1] = -1.0
    dSdrs[1,1] =  0.0
    dSdrs[2,1] =  1.0

    return S, dSdrs

def tri6_fcn(r, s):
    """      s
            |
            2, (0,1)
            | ',
            |   ',
            |     ',
            |       ',
            5         '4
            |           ',
            |             ',
            |               ',
            | (0,0)           ', (1,0)
            0---------3---------1 ---- r
    """
    S, dSdrs = zeros(6), zeros((6,2))

    S[0] = 1.0 - (r + s) * (3.0 - 2.0 * (r + s))
    S[1] = r * (2.0 * r - 1.0)
    S[2] = s * (2.0 * s - 1.0)
    S[3] = 4.0 * r * (1.0 - (r + s))
    S[4] = 4.0 * r * s
    S[5] = 4.0 * s * (1.0 - (r + s))

    dSdrs[0,0] = -3.0 + 4.0 * (r + s)
    dSdrs[1,0] =  4.0 * r - 1.0
    dSdrs[2,0] =  0.0
    dSdrs[3,0] =  4.0 - 8.0 * r - 4.0 * s
    dSdrs[4,0] =  4.0 * s
    dSdrs[5,0] = -4.0 * s

    dSdrs[0,1] = -3.0 + 4.0*(r + s)
    dSdrs[1,1] =  0.0
    dSdrs[2,1] =  4.0 * s - 1.0
    dSdrs[3,1] = -4.0 * r
    dSdrs[4,1] =  4.0 * r
    dSdrs[5,1] =  4.0 - 4.0 * r - 8.0*s

    return S, dSdrs

def qua4_fcn(r, s):
    """
          3-----------2
          |     s     |
          |     |     |
          |     +--r  |
          |           |
          |           |
          0-----------1
    """
    S, dSdrs = zeros(4), zeros((4,2))

    S[0] = (1.0 - r - s + r * s) / 4.0
    S[1] = (1.0 + r - s - r * s) / 4.0
    S[2] = (1.0 + r + s + r * s) / 4.0
    S[3] = (1.0 - r + s - r * s) / 4.0

    dSdrs[0,0] = (-1.0 + s) / 4.0
    dSdrs[0,1] = (-1.0 + r) / 4.0
    dSdrs[1,0] = (+1.0 - s) / 4.0
    dSdrs[1,1] = (-1.0 - r) / 4.0
    dSdrs[2,0] = (+1.0 + s) / 4.0
    dSdrs[2,1] = (+1.0 + r) / 4.0
    dSdrs[3,0] = (-1.0 - s) / 4.0
    dSdrs[3,1] = (+1.0 - r) / 4.0

    return S, dSdrs

def qua8_fcn(r, s):
    """
          3-----6-----2
          |     s     |
          |     |     |
          7     +--r  5
          |           |
          |           |
          0-----4-----1
    """
    S, dSdrs = zeros(8), zeros((8,2))

    S[0] = (1.0 - r) * (1.0 - s) * (- r - s - 1.0) / 4.0
    S[1] = (1.0 + r) * (1.0 - s) * (  r - s - 1.0) / 4.0
    S[2] = (1.0 + r) * (1.0 + s) * (  r + s - 1.0) / 4.0
    S[3] = (1.0 - r) * (1.0 + s) * (- r + s - 1.0) / 4.0
    S[4] = (1.0 - s) * (1.0 - r  * r) / 2.0
    S[5] = (1.0 + r) * (1.0 - s  * s) / 2.0
    S[6] = (1.0 + s) * (1.0 - r  * r) / 2.0
    S[7] = (1.0 - r) * (1.0 - s  * s) / 2.0

    dSdrs[0,0] = - (1.0 - s) * (- r - r - s) / 4.0
    dSdrs[1,0] =   (1.0 - s) * (  r + r - s) / 4.0
    dSdrs[2,0] =   (1.0 + s) * (  r + r + s) / 4.0
    dSdrs[3,0] = - (1.0 + s) * (- r - r + s) / 4.0
    dSdrs[4,0] = - (1.0 - s) * r
    dSdrs[5,0] =   (1.0 - s  * s) / 2.0
    dSdrs[6,0] = - (1.0 + s) * r
    dSdrs[7,0] = - (1.0 - s  * s) / 2.0

    dSdrs[0,1] = - (1.0 - r) * (- s - s - r) / 4.0
    dSdrs[1,1] = - (1.0 + r) * (- s - s + r) / 4.0
    dSdrs[2,1] =   (1.0 + r) * (  s + s + r) / 4.0
    dSdrs[3,1] =   (1.0 - r) * (  s + s - r) / 4.0
    dSdrs[4,1] = - (1.0 - r  * r) / 2.0
    dSdrs[5,1] = - (1.0 + r) * s
    dSdrs[6,1] =   (1.0 - r  * r) / 2.0
    dSdrs[7,1] = - (1.0 - r) * s

    return S, dSdrs

def qua9_fcn(r, s):
    """
          3-----6-----2
          |     s     |
          |     |     |
          7     8--r  5
          |           |
          |           |
          0-----4-----1
    """
    S, dSdrs = zeros(9), zeros((9,2))

    S[0] = r * (r - 1.0) * s * (s - 1.0) / 4.0
    S[1] = r * (r + 1.0) * s * (s - 1.0) / 4.0
    S[2] = r * (r + 1.0) * s * (s + 1.0) / 4.0
    S[3] = r * (r - 1.0) * s * (s + 1.0) / 4.0

    S[4] = - (r *  r - 1.0) *  s * (s - 1.0) / 2.0
    S[5] = -  r * (r + 1.0) * (s *  s - 1.0) / 2.0
    S[6] = - (r *  r - 1.0) *  s * (s + 1.0) / 2.0
    S[7] = -  r * (r - 1.0) * (s *  s - 1.0) / 2.0

    S[8] = (r * r - 1.0) * (s * s - 1.0)

    dSdrs[0,0] = (r + r - 1.0) * s * (s - 1.0) / 4.0
    dSdrs[1,0] = (r + r + 1.0) * s * (s - 1.0) / 4.0
    dSdrs[2,0] = (r + r + 1.0) * s * (s + 1.0) / 4.0
    dSdrs[3,0] = (r + r - 1.0) * s * (s + 1.0) / 4.0

    dSdrs[0,1] = r * (r - 1.0) * (s + s - 1.0) / 4.0
    dSdrs[1,1] = r * (r + 1.0) * (s + s - 1.0) / 4.0
    dSdrs[2,1] = r * (r + 1.0) * (s + s + 1.0) / 4.0
    dSdrs[3,1] = r * (r - 1.0) * (s + s + 1.0) / 4.0

    dSdrs[4,0] = - (r + r)       *  s * (s - 1.0) / 2.0
    dSdrs[5,0] = - (r + r + 1.0) * (s *  s - 1.0) / 2.0
    dSdrs[6,0] = - (r + r)       *  s * (s + 1.0) / 2.0
    dSdrs[7,0] = - (r + r - 1.0) * (s *  s - 1.0) / 2.0

    dSdrs[4,1] = - (r *  r - 1.0) * (s + s - 1.0) / 2.0
    dSdrs[5,1] = -  r * (r + 1.0) * (s + s)       / 2.0
    dSdrs[6,1] = - (r *  r - 1.0) * (s + s + 1.0) / 2.0
    dSdrs[7,1] = -  r * (r - 1.0) * (s + s)       / 2.0

    dSdrs[8,0] = 2.0 * r * (s * s - 1.0)
    dSdrs[8,1] = 2.0 * s * (r * r - 1.0)

    return S, dSdrs

def get_shape_fcn(geom):
    """
    Get functions that calculate shape and derivatives
    ==================================================
    INPUT:
        geom : a key representing the element geometry
    RETURNS:
        fce : shape/derivs function
        nne : number of nodes in element
        fcf : shape/derivs function of edge/face
        nnf : number of nodes of edge/face
    """
    if   geom == 'lin2': fce, nne, fcf, nnf = lin2_fcn, 2, None,     0
    elif geom == 'lin3': fce, nne, fcf, nnf = lin3_fcn, 3, None,     0
    elif geom == 'tri3': fce, nne, fcf, nnf = tri3_fcn, 3, lin2_fcn, 2
    elif geom == 'tri6': fce, nne, fcf, nnf = tri6_fcn, 6, lin3_fcn, 3
    elif geom == 'qua4': fce, nne, fcf, nnf = qua4_fcn, 4, lin2_fcn, 2
    elif geom == 'qua8': fce, nne, fcf, nnf = qua8_fcn, 8, lin3_fcn, 3
    elif geom == 'qua9': fce, nne, fcf, nnf = qua9_fcn, 9, lin3_fcn, 3
    else: raise Exception('geometry %s is not available'%geom.__str__())
    return fce, nne, fcf, nnf

def shape_derivs(xy, fcn, ip):
    """
    Calculate shape and derivatives of shape functions at integration point
    =======================================================================
    INPUT:
        xy  : the element matrix of coordinates
        fcn : the function that calculates shape/derivatives functions
        ip  : integration point
    RETURNS:
        S     : shape functions at (r,s)
        dSdxy : derivative of shape functions w.r.t real coordinates
        detJ  : determinant of J=dxydr
    """
    S, dSdrs = fcn(ip[0], ip[1])
    dxydr    = dot(xy, dSdrs)
    dSdxy    = dot(dSdrs, inv(dxydr))
    return S, dSdxy, det(dxydr)

def face_integ(xyf, fcn, ip):
    """
    Face integration
    ================
    INPUT:
        xyf : the element matrix of coordinates [face]
        fcn : the function that calculates shape/derivatives functions [face]
        ip  : integration point [face]
    RETURNS:
        S     : face shape functions at (r,s)
        detJ  : face determinant of J=dxydr
        dxydr : Jacobian
    """
    S, dSdrs = fcn(ip[0], ip[1])
    dxydr    = dot(xyf, dSdrs)
    return S, sqrt(dxydr[0]**2.0 + dxydr[1]**2.0), dxydr

def face_coords(xy, edge_num, nne):
    """
    Coordinates of face nodes
    =========================
        Builds a matrix of coordinates corresponding to an edge/face of a 2D element
    INPUT:
        xy       : the element matrix of coordinates
        edge_num : index of edge/face
        nne      : number of nodes in element
    RETURNS:
        xyf : coordinates of face nodes
        fno : local indices of nodes on edge/face
    """
    fno = Edg2Vids[(2,nne)][edge_num] # edge/face nodes (local indices)
    nnf = len(fno)                    # number of nodes in face
    xyf = zeros((2,nnf))
    for i, n in enumerate(fno):
        xyf[0][i] = xy[0,n]
        xyf[1][i] = xy[1,n]
    return xyf, fno
