# Copyright 2012 Dorival de Moraes Pedroso. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

from numpy import sqrt, array, dot

#                  r               s                w
LinIp2 = array([[ -sqrt(3.0)/3.0 , 0.0            , 1.0        ],
                [  sqrt(3.0)/3.0 , 0.0            , 1.0        ]])

LinIp3 = array([[ -sqrt(3.0/5.0) , 0.0            , 5.0/9.0    ],
                [            0.0 , 0.0            , 8.0/9.0    ],
                [  sqrt(3.0/5.0) , 0.0            , 5.0/9.0    ]])

TriIp1 = array([[  1.0/3.0       , 1.0/3.0        , 1.0/2.0    ]])

TriIp3 = array([[  1.0/6.0       , 1.0/6.0        , 1.0/6.0    ],
                [  2.0/3.0       , 1.0/6.0        , 1.0/6.0    ],
                [  1.0/6.0       , 2.0/3.0        , 1.0/6.0    ]])

QuaIp4 = array([[ -sqrt(3.0)/3.0 , -sqrt(3.0)/3.0 , 1.0        ],
                [  sqrt(3.0)/3.0 , -sqrt(3.0)/3.0 , 1.0        ],
                [ -sqrt(3.0)/3.0 ,  sqrt(3.0)/3.0 , 1.0        ],
                [  sqrt(3.0)/3.0 ,  sqrt(3.0)/3.0 , 1.0        ]])

QuaIp9 = array([[ -sqrt(3.0/5.0) , -sqrt(3.0/5.0) , 25.0/81.0 ],
                [            0.0 , -sqrt(3.0/5.0) , 40.0/81.0 ],
                [  sqrt(3.0/5.0) , -sqrt(3.0/5.0) , 25.0/81.0 ],
                [ -sqrt(3.0/5.0) ,            0.0 , 40.0/81.0 ],
                [            0.0 ,            0.0 , 64.0/81.0 ],
                [  sqrt(3.0/5.0) ,            0.0 , 40.0/81.0 ],
                [ -sqrt(3.0/5.0) ,  sqrt(3.0/5.0) , 25.0/81.0 ],
                [            0.0 ,  sqrt(3.0/5.0) , 40.0/81.0 ],
                [  sqrt(3.0/5.0) ,  sqrt(3.0/5.0) , 25.0/81.0 ]])

def get_ips(geom):
    """
    Get integration points
    ======================
    INPUT:
        geom : a key representing the element geometry
    RETURNS:
        ipe : integration points of element
        ipf : integration points of edge/face
    """
    if   geom == 'lin2': ipe, ipf = LinIp2, None
    elif geom == 'lin3': ipe, ipf = LinIp3, None
    elif geom == 'tri3': ipe, ipf = TriIp3, LinIp2
    elif geom == 'tri6': ipe, ipf = TriIp3, LinIp3
    elif geom == 'qua4': ipe, ipf = QuaIp4, LinIp2
    elif geom == 'qua8': ipe, ipf = QuaIp9, LinIp3
    elif geom == 'qua9': ipe, ipf = QuaIp9, LinIp3
    else: raise Exception('geometry %s is not available'%geom)
    return ipe, ipf

def ip_coords(S, XY):
    """
    Coordinates of integration point
    ================================
    INPUT:
        S  : shape functions evaluated at the desired ip
        XY : matrix of coordinates
    RETURNS
        xip, yip : (real) coordinates of ip
    """
    xip, yip = 0.0, 0.0
    for n in range(len(S)):
        xip += S[n] * XY[0][n]
        yip += S[n] * XY[1][n]
    return xip, yip
