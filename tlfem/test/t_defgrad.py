# Copyright 2012 Dorival de Moraes Pedroso. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

from __future__ import print_function # for Python 3

from numpy       import array, sqrt, zeros, tensordot, dot, transpose
from pylab       import plot, axis, show, arrow
from tlfem.shape import tri3_fcn, qua4_fcn

def calc_F(X, x):
    """
    Calc deformation gradient F
    ===========================
    """
    if   len(X)==3: fcn, RS = tri3_fcn, [(0.,0.), (1.,0.), (0.,1.)]
    elif len(X)==4: fcn, RS = qua4_fcn, [(-1.,-1.), (1.,-1.), (1.,1.), (-1.,1.)]
    else: raise Exception('number of points must be 3 or 4')
    dxdr = zeros((2,2))
    dXdr = zeros((2,2))
    for k, rs in enumerate(RS):
        _, G  = fcn(rs[0], rs[1])
        dxdr += tensordot(x[k], G[k], axes=0)
        dXdr += tensordot(X[k], G[k], axes=0)
    drdX = zeros((2,2))
    det  = dXdr[0][0]*dXdr[1][1] - dXdr[0][1]*dXdr[1][0]
    if abs(det) < 1.0e-10: raise Exception('null determinant')
    drdX[0][0] =  dXdr[1][1] / det
    drdX[0][1] = -dXdr[0][1] / det
    drdX[1][0] = -dXdr[1][0] / det
    drdX[1][1] =  dXdr[0][0] / det
    return dot(dxdr, drdX)

def calc_eps(F):
    """
    Green-Lagrange strain tensor
    ============================
    """
    return 0.5*(dot(transpose(F), F) - array([[1., 0.],[0., 1.]]))

def calc_D(E, nu, pstress=True):
    """
    Calc linear elasticity matrix
    =============================
    """
    if pstress:
        cf = E / (1.0 - nu**2.0)
        D = cf * array([[1., nu, 0.   ],
                        [nu, 1., 0.   ],
                        [0., 0., 1.-nu]])
    else: # plane strain
        cf = E / ((1.0 + nu) * (1.0 - 2.0 * nu))
        D  = cf * array([[1.-nu , nu    , 0.      ],
                         [nu    , 1.-nu , 0.      ],
                         [0.    , 0.    , 1.-2.*nu]])
    return D

def calc_B(x):
    """
    Calc B matrix
    =============
    """
    if not len(x)==3: raise Exception('number of points must be 3')
    # nodal coordinates
    x0, y0 = x[0][0], x[0][1]
    x1, y1 = x[1][0], x[1][1]
    x2, y2 = x[2][0], x[2][1]

    # derived variables
    b0 = y1-y2;         b1 = y2-y0;         b2 = y0-y1;
    c0 = x2-x1;         c1 = x0-x2;         c2 = x1-x0;
    f0 = x1*y2-x2*y1;   f1 = x2*y0-x0*y2;   f2 = x0*y1-x1*y0

    # area
    A = (f0 + f1 + f2) / 2.0

    cf = 1.0 / (2.0 * A)
    sq = sqrt(2.0)
    B = cf * array([[b0    , 0.    , b1    , 0.    , b2    , 0.   ],
                    [0.    , c0    , 0.    , c1    , 0.    , c2   ],
                    [c0/sq , b0/sq , c1/sq , b1/sq , c2/sq , b2/sq]])
    return B

def draw_poly(p, clr='k'):
    N = len(p)
    for i in range(N): plot([p[i][0], p[(i+1)%N][0]],
                            [p[i][1], p[(i+1)%N][1]], color=clr)
def draw_forces(P, F):
    arrow(P[0][0],P[0][1],F[0],F[1], width=0.005)
    arrow(P[1][0],P[1][1],F[2],F[3], width=0.005)
    arrow(P[2][0],P[2][1],F[4],F[5], width=0.005)

if __name__=="__main__":

    prob = 5

    if prob==1:
        P = array([[0.,0.], [1.,0.], [1.,1.], [0.,1.]])
        p = array([[1.0            ,  1.0              ],
                   [1.0+sqrt(3.)/2.,  3./2.            ],
                   [0.5+sqrt(3.)/2.,  3./2.+sqrt(3.)/2.],
                   [0.5            ,  1.+sqrt(3.)/2.   ]])
        print('F =\n', calc_F(P, p))
        print('F correct =\n', array([[sqrt(3.)/2., -0.5], [0.5, sqrt(3.)/2.]]))

    if prob==2:
        P = array([[0., 0.], [1., 0.], [1., 1.], [0., 1.]])
        p = array([[0.25,     0.5    ],
                   [1.12469,  1.005  ],
                   [0.629686, 1.86237],
                   [-0.245,   1.35737]])
        F = calc_F(P, p)
        print('F =\n', F)
        print('eps =\n', calc_eps(F))

    if prob==3:
        P = array([[0., 0.], [1., 0.], [1., 1.], [0., 1.]])
        p = array([[0.25,     0.5    ],
                   [1.33253,  1.125  ],
                   [0.982532, 1.73122],
                   [-0.1,     1.10622]])
        F = calc_F(P, p)
        print('F =\n', F)
        print('eps =\n', calc_eps(F))

    if prob==4:
        P = array([[0., 0.], [1., 0.], [0.5, sqrt(3.)/2.]])
        p = array([[0., 0.], [1., 0.], [0.5, 0.5]])

        F   = calc_F(P, p)
        eps = calc_eps(F)
        D   = calc_D(0.1, 0.5)
        B   = calc_B(p)
        sig = dot(D, array([eps[0,0], eps[1,1], sqrt(2.)*eps[0,1]]))
        Fe  = dot(transpose(B), sig)

        print('F =\n', F)
        print('D =\n', D)
        print('B =\n', B)
        print('eps =\n', eps)
        print('sig =\n', sig)
        print('Fe =\n', Fe)

        draw_poly(P)
        draw_poly(p,'b')
        draw_forces(P, Fe)
        axis('equal')
        show()

    if prob==5:
        P = array([[0., 0.], [1., 0.], [0.5, sqrt(3.)/2.]])
        p = array([[0., 0.], [1., 0.], [0.5, 0.5]])

        Ue = array([p[0][0]-P[0][0], p[0][1]-P[0][1],
                    p[1][0]-P[1][0], p[1][1]-P[1][1],
                    p[2][0]-P[2][0], p[2][1]-P[2][1]])

        B   = calc_B(p)
        D   = calc_D(0.1, 0.5)
        eps = dot(B, Ue)
        sig = dot(D, eps)
        Fe  = dot(transpose(B), sig)
        print('eps =\n', eps)
        print('sig =\n', sig)
        print('Fe  =\n', Fe)

        draw_poly(P)
        draw_poly(p,'b')
        draw_forces(P, Fe)
        axis('equal')
        show()
