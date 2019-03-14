# Copyright 2012 Dorival de Moraes Pedroso. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

from __future__ import print_function # for Python 3

import sys
from   numpy         import linspace, zeros, sin, pi
from   pylab         import axis, show
from   tlfem.genmesh import Gen2Dregion, JoinAlongEdge
from   tlfem.solver  import Solver
from   tlfem.mesh    import Mesh

def runtest(prob):
    if prob==1:

        tri = 0

        if 1:
            nxa = 3 #5
            nxb = 5 #9
            nya = 3
            nyb = 3
        else:
            nxa = 11
            nxb = nxa + (nxa-1)
            nya = 7
            nyb = 4

        x = linspace(0.,4.,nxa)
        y = linspace(0.,2.,nya)
        m = Gen2Dregion(x, y, triangle=tri)
        m.gen_mid_verts()

        x = linspace(0.,4., nxb)
        y = linspace(2.,2.5,nyb)
        b = Gen2Dregion(x, y, ctag=-2, triangle=tri, etags=[-10,-11,-14,-13])

        m.join_along_edge(b, -12, -10)
        m.check_overlap()
        m.draw(pr=1); m.show()

        E, nu = 10.0, 0.25
        if tri:
            p = {-1:{'type':'Eelastic2D', 'E':E, 'nu':nu, 'geom':'tri6'},
                 -2:{'type':'Eelastic2D', 'E':E, 'nu':nu, 'geom':'tri3'}}
        else:
            p = {-1:{'type':'Eelastic2D', 'E':E, 'nu':nu, 'geom':'qua8'},
                 -2:{'type':'Eelastic2D', 'E':E, 'nu':nu, 'geom':'qua4'}}

        s = Solver(m, p)
        eb = {-10:{'uy':0.}, -13:{'ux':0.}, -14:{'qnqt':(-1.,0.)}}
        s.set_bcs(eb=eb)
        s.solve_steady(extrap=True)#, vtu_fnkey='patchrecov1')

        #PrintTable(s.Uout)
        #PrintTable(s.Eout)

        if 0:
            U0, V0 = zeros(s.neqs), zeros(s.neqs)
            s.solve_dynamics(0., U0, V0, 1., 0.01)#, vtu_fnkey='patchrecov_dyn1')

    if prob==2:

        tri = 0
        o2  = 1

        x = linspace(0.,4.,5)
        y = linspace(0.,2.,3)
        m = Gen2Dregion(x,y, triangle=tri)
        m.tag_vert(-100, 1., 2.)
        m.tag_vert(-101, 3., 2.)
        m.tag_edges_on_line(-14, 0.,1.,0.)
        if o2: m.gen_mid_verts()
        m.edges_to_lincells(-12, -2)
        vid0 = m.get_verts(-100)[0]
        vid1 = m.get_verts(-101)[0]
        V = [[m.nv,   -200, 1., 3.],
             [m.nv+1, -201, 2., 3.],
             [m.nv+2, -202, 3., 3.]]
        C = [[m.nc,   -3, [vid0,   m.nv  ]],
             [m.nc+1, -4, [m.nv,   m.nv+1]],
             [m.nc+2, -4, [m.nv+1, m.nv+2]],
             [m.nc+3, -3, [m.nv+2, vid1  ]]]
        m.extend(V, C)
        m.check_overlap()
        m.draw(pr=1)
        #m.show()

        if tri: geom = 'tri6' if o2 else 'tri3'
        else:   geom = 'qua8' if o2 else 'qua4'

        E, nu    = 10.0, 0.25
        Eb, A, I = 100.0, 1., 1.
        p = {-1:{'type':'Eelastic2D',   'E':E,  'nu':nu, 'geom':geom},
             -2:{'type':'EelasticBeam', 'E':Eb, 'A':A,   'I':I},
             -3:{'type':'EelasticBeam', 'E':Eb, 'A':A,   'I':I},
             -4:{'type':'EelasticBeam', 'E':Eb, 'A':A,   'I':I, 'qnqt':(-1.,-1.,0.)}}

        s = Solver(m, p)
        eb = {-10:{'uy':0.}, -11:{'ux':0.}, -13:{'ux':0.}}
        if 0:
            s.set_bcs(eb=eb, vb={-101:{'fy':lambda t:sin(pi*t)}})
            o = s.solve_dynamics(0., 1., 0.01, dtout=0.5, extrap=True)#, vtu_fnkey='patchrecov_dyn2')
        else:
            s.set_bcs(eb=eb)
            o = s.solve_steady(extrap=1)#, vtu_fnkey='patchrecov2')

        o.beam_moments()
        o.beam_print()
        axis('equal')
        show()

# run tests
prob = int(sys.argv[1]) if len(sys.argv)>1 else -1
if prob < 0:
    for p in range(1,3):
        print()
        print('[1;33m####################################### %d #######################################[0m'%p)
        print()
        runtest(p)
else: runtest(prob)
