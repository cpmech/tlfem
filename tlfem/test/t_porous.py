# Copyright 2012 Dorival de Moraes Pedroso. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

from __future__ import print_function # for Python 3

import sys
import copy
from   numpy         import linspace, zeros, sin, pi, array
from   pylab         import subplot, plot, show
from   tlfem.genmesh import Gen2Dregion, NlSpace
from   tlfem.solver  import Solver
from   tlfem.util    import GetITout
from   tlfem.fig     import column_nodes, Gll

def runtest(prob):
    if prob==1: # Zienk & Shiomi 1984, Example 1

        # input
        bb = int(sys.argv[2]) if len(sys.argv)>2 else 1

        # generate mesh
        W  = 3.0
        H  = 30.0
        ny = 11
        x  = linspace(0.,W,2)
        y  = linspace(0.,H,ny)
        m  = Gen2Dregion(x,y, triangle=False)
        if bb: m.gen_mid_verts()
        m.tag_vert(-101, 0.,H/(ny-1.))
        m.tag_vert(-102, 0.,H/2.)
        m.tag_vert(-103, 0.,H-H/(ny-1.))
        #m.draw()
        #m.show()
        #m.write_json('zs-01')

        # parameters
        E,    nu,   rhoS = 30.0,   0.2,    2.0e-3 # [MPa],   [-],     [Gg/m3]
        rhoW, gamW, Kw   = 1.0e-3, 1.0e-2, 100.0  # [Gg/m3], [MN/m3], [MPa]
        eta,  kx,   ky   = 0.3,    1.0e-2, 1.0e-2 # [-],     [m/s],   [m/s]
        qn,   tsw        = -1.e-3, 0.1            # [MPa],   [s]

        #Kw *= 100.
        #Kw = 1e+30
        #nu = 0.4999999

        # solver
        geom = ('qua8','qua4') if bb else 'qua4'
        p = {-1:{'type':'EelasticPorous2D', 'E':E, 'nu':nu, 'rhoS':rhoS, 'rhoW':rhoW,
                 'gamW':gamW, 'Kw':Kw, 'eta':eta, 'kx':kx, 'ky':ky, 'geom':geom}}
        s = Solver(m, p)

        # boundary conditions
        eb = {-10:{'uy':0.}, -11:{'ux':0.}, -13:{'ux':0.}, -12:{'qnqt':(qn,0.), 'pw':0.0}}
        s.set_bcs(eb=eb)

        # run simulation
        t0, tf     = 0.0, 30.0
        dt, dtout  = 0.05, 0.05
        o = s.solve_porous(t0, tf, dt, dtout,# vtu_fnkey='zienkshiomi',
                           theta1=0.6, theta2=0.65, vtu_dtout=10*dtout,
                           load_mult=lambda t:1.0 if t>tsw else sin(0.5*pi*t/tsw))

        # plot
        subplot(2,1,1)
        left, _ = column_nodes(nc=ny-1, o2=False)
        left_Y  = [m.V[vid][3] for vid in left]
        Tout    = [2., 5.2, 9., 24.5]
        ITout   = GetITout(o.Tout, Tout, 0.5)
        for iout, tout in ITout:
            pw = [o.Uout['pw'][vid][iout]*1000. for vid in left]
            plot(pw, left_Y, label='t=%g'%tout)
        Gll('pw [kPa]','y')

        subplot(2,1,2)
        vids = m.get_verts([-101,-102,-103])
        for vid in vids:
            plot(o.Tout, array(o.Uout['pw'][vid])*1000., label='%d @ y=%g'%(m.V[vid][1],m.V[vid][3]))
        Gll('t','pw [kPa]')
        show()

    if prob==2:

        tri = 0

        if 0:
            H  = 1.0
            ny = 3
        else:
            H  = 10.0
            #ny = 21
            ny = 11
            #ny = 4
        x = linspace(0.,1.,2)
        y = linspace(0.,H,ny)
        #y = NlSpace(0.,H,ny,0.1,rev=True)
        m = Gen2Dregion(x,y, triangle=tri)
        m.tag_vert(-100, 0.,H)
        m.tag_vert(-100, 1.,H)
        m.gen_mid_verts()
        #m.draw()
        #m.show()

        geom = ('tri6','tri3') if tri else ('qua8','qua4')
        #geom = ('qua4','qua4')
        #geom = ('qua8','qua8')

        E,    nu,  rhoS = 3000.,  0.25, 2.7e-3 # [MPa],   [-],   [Gg/m3]
        kx,   ky,  rhoW = 1.e-5, 1.e-5, 1.0e-3 # [m/s],   [m/s], [Gg/m3]
        gamW, eta, Kw   = 1.e-2,   0.3, 1.e+6  # [MN/m3], [-],   [MPa]
        qn,   tsw       = -0.1,   0.05         # [MPa],   [s]

        p = {-1:{'type':'EelasticPorous2D', 'E':E, 'nu':nu, 'rhoS':rhoS,
                 'kx':kx, 'ky':ky, 'rhoW':rhoW, 'gamW':gamW, 'eta':eta, 'Kw':Kw,
                 'geom':geom}}
        s = Solver(m, p)
        vb = {-100:{'pw':0.}}
        eb = {-10:{'uy':0.}, -11:{'ux':0.}, -13:{'ux':0.}, -12:{'qnqt':(qn,0.)}}
        s.set_bcs(vb=vb, eb=eb)

        t0, tf     = 0.0, 10.0
        dt, dtout  = 0.05, 0.1
        o = s.solve_porous(t0, tf, dt, dtout,# vtu_fnkey='porous1',
                           load_mult=lambda t:1.0 if t>tsw else t/tsw)

        left, _ = column_nodes(nc=ny-1, o2=False)
        left_Y  = [m.V[vid][3] for vid in left]
        ITout   = GetITout(o.Tout, [0., dtout, 10.*dtout, 1.0, 2., 3., 4., 5.], dtout/2.0)
        print(left)
        for iout, tout in ITout:
            pw = [o.Uout['pw'][vid][iout] for vid in left]
            plot(pw, left_Y, label='t=%g'%tout)
        Gll('pw','y')
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
