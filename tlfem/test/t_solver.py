# Copyright 2012 Dorival de Moraes Pedroso. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

from __future__ import print_function # for Python 3

import sys
from   numpy         import sqrt, sin, cos, tan, sinh, cosh, array, meshgrid
from   numpy         import zeros, pi, exp, vectorize
from   scipy.special import erfc
from   pylab         import linspace, plot, show, subplot, contour, axis, suptitle
from   tlfem.mesh    import Mesh
from   tlfem.solver  import Solver
from   tlfem.genmesh import Gen1Dmesh, Gen2Dregion
from   tlfem.util    import CompareArrays, GetITout, PrintDiff
from   tlfem.fig     import Contour, PlotSurf, Cross, Read, Gll

ge2tri = {'tri3':True,  'tri6':True,  'qua4':False, 'qua8':False, 'qua9':False}
ge2o2  = {'tri3':False, 'tri6':True,  'qua4':False, 'qua8':True,  'qua9':True }
ge2cen = {'tri3':False, 'tri6':False, 'qua4':False, 'qua8':False, 'qua9':True }

def runtest(prob):
    if prob == 0:
        from tlfem.EdiffusionTri import EdiffusionTri
        for case in range(0,6):
            if case == 0:
                V = [[0, 0, 0.0, 0.0],
                     [1, 0, 1.0, 0.0],
                     [2, 0, 0.0, 1.0]]
            elif case == 1:
                V = [[0, 0, 0.0, 1.0],
                     [1, 0, 0.0, 0.0],
                     [2, 0, 1.0, 0.0]]
            elif case == 2:
                V = [[0, 0, 1.0, 0.0],
                     [1, 0, 0.0, 1.0],
                     [2, 0, 0.0, 0.0]]
            elif case == 3:
                V = [[0, 0, 0.0,          0.0        ],
                     [1, 0, sqrt(2.)/2., -sqrt(2.)/2.],
                     [2, 0, sqrt(2.),     0.0        ]]
            elif case == 4:
                V = [[0, 0, sqrt(2.)/2., -sqrt(2.)/2.],
                     [1, 0, sqrt(2.),     0.0        ],
                     [2, 0, 0.0,          0.0        ]]
            elif case == 5:
                V = [[0, 0, sqrt(2.),     0.0        ],
                     [1, 0, 0.0,          0.0        ],
                     [2, 0, sqrt(2.)/2., -sqrt(2.)/2.]]
            C = [[0, -1, [0,1,2], {}]]
            m = Mesh(V, C)
            m.draw();  m.show()
            ep = {'rho':0.,'beta':0.,'kx':1.,'ky':1.,
                  'source':lambda x,y:1.}
            e = EdiffusionTri(V, ep)
            print(e.calc_F())

    if prob==1:
        # generate 1D mesh
        Lx = 3.0
        nx = 11
        m  = Gen1Dmesh(Lx, nx)
        #m.draw()
        #m.show()

        # create dictionary with all parameters
        p = {-1:{'type':'Ediffusion1D', 'rho':1.0, 'beta':0.0, 'kx':1.0, 'source':lambda x:x}}

        # allocate fem solver object
        s = Solver(m, p)

        # set boundary conditions
        #vbcs = {-101:{'u':0.}, -102:{'u':0.}}
        vbcs = {-101:{'u':lambda x:0.}, -102:{'u':0.}}
        s.set_bcs(vb=vbcs)

        # solve steady
        o = s.solve_steady(reactions=True, extrap=True, emethod=2)

        # output
        o.print_res('U')
        o.print_res('R')
        o.print_res('E')

        # plot
        X = [m.V[i][2] for i in range(m.nv)]
        U = [o.Uout['u'][n][-1] for n in range(m.nv)]
        x = linspace(0.,Lx,101)
        y = (x*Lx**2.-x**3.)/6.
        plot(X, U, 'ro', ls='-', label='fem')
        plot(x, y, 'g-', label='solution')
        Gll(r'$x$',r'$u$')
        show()

    if prob == 2:

        dt    = float(sys.argv[2]) if len(sys.argv)>2 else 0.02
        theta = float(sys.argv[3]) if len(sys.argv)>3 else 0.5

        # generate 1D mesh
        Lx = 1.0
        nx = 6
        m  = Gen1Dmesh(Lx, nx)
        #m.draw(); m.show()

        # create dictionary with all parameters
        p = {-1:{'type':'Ediffusion1D', 'rho':1.0, 'beta':0.0, 'kx':1.0}}

        # allocate fem solver object
        s = Solver(m, p)

        # set boundary conditions
        vbcs = {-101:{'u':0.}, -102:{'u':0.}}
        s.set_bcs(vb=vbcs)

        # solve transient
        tf = 0.2
        X  = array([v[2] for v in m.V]) # grid coords
        U0 = 4.0*X-4.0*X**2.0           # initial values
        o = s.solve_transient(0.0, tf, dt, U0=U0, theta=theta)

        # plot
        if 1:
            tt,xx = meshgrid(o.Tout, X)
            U     = [o.Uout['u'][vid] for vid in range(m.nv)]
            ax    = PlotSurf(tt,xx,U, r'$t$',r'$x$',r'$U$',0.,1.)
            ax.view_init(20.,30.)
            show()

    if prob == 3: # bh 1.4
        # input mesh
        # vertices
        V = [[0, -100,  0.0,  0.0],
             [1, -200, 1500, 3500],
             [2,    0,  0.0, 5000],
             [3, -300, 5000, 5000]]
        # cells/bars
        C = [[0, -1, [0,1]],
             [1, -1, [1,3]],
             [2, -2, [0,2]],
             [3, -2, [2,3]],
             [4, -3, [1,2]]]
        m = Mesh(V, C)
        #m.draw()
        #m.show()

        # create dictionary with all parameters
        p = {-1:{'type':'EelasticRod', 'E':200000,'A':4000},
             -2:{'type':'EelasticRod', 'E':200000,'A':3000},
             -3:{'type':'EelasticRod', 'E': 70000,'A':2000}}

        # allocate fem solver object
        s = Solver(m, p)

        # set boundary conditions
        vbcs = {-100:{'ux':0.0,'uy':0.0},
                -200:{'fy':-150000},
                -300:{'ux':0.0,'uy':0.0}}
        s.set_bcs(vb=vbcs)

        # solve steady
        o = s.solve_steady(reactions=True)

        # output
        o.print_res('U')
        o.print_res('R', True)

        # bhatti's solution
        Ubh = array([[0.000000000000000e+00,  0.000000000000000e+00],
                     [5.389536380057676e-01, -9.530613006371175e-01],
                     [2.647036149579491e-01, -2.647036149579490e-01],
                     [0.000000000000000e+00,  0.000000000000000e+00]])
        ux_fem = [o.Uout['ux'][n][-1] for n in range(m.nv)]
        uy_fem = [o.Uout['uy'][n][-1] for n in range(m.nv)]
        CompareArrays(ux_fem, Ubh[:,0])

    if prob == 4: # bh 1.5
        # input mesh
        V = [[0, -100, 0.0, 0.0],
             [1,    0, 0.2, 0.0],
             [2,    0, 0.2, 0.3],
             [3, -100, 0.0, 0.1],
             [4,    0, 0.1, 0.1]]
        C = [[0, -1, [0,1,4], {}     ],
             [1, -1, [1,2,4], {0:-10}],
             [2, -1, [2,3,4], {}     ],
             [3, -1, [0,4,3], {}     ]]
        m = Mesh(V, C)
        #m.draw(pr=1); m.show()

        # parameters
        p = {-1:{'type':'EdiffusionTri', 'kx':1.4, 'ky':1.4}}

        # allocate fem solver object
        s = Solver(m, p)

        # set boundary conditions
        #vbcs = {-100:{'u':300.0}}
        vb = {-100:{'u':lambda x,y:300.0}}
        eb = {-10:{'c':(27.0, 20.0)}}
        s.set_bcs(vb=vb, eb=eb)

        # solve steady
        o = s.solve_steady(reactions=True, extrap=True)

        # output
        o.print_res('U')
        o.print_res('R', True)

        # bhatti's solution
        Ubh = array([3.000000000000000e+02,
                     9.354661202985511e+01,
                     2.384369969266794e+01,
                     3.000000000000000e+02,
                     1.828327235474901e+02])

        wxbh = -1.4 * array([-1.032266939850724e+03,
                             -1.125204156300308e+03,
                             -1.171672764525099e+03,
                             -1.171672764525099e+03])

        wybh = -1.4 * array([-1.394058246743742e+02,
                             -2.323430411239573e+02,
                             -2.091087370115617e+02,
                              0.000000000000000e+00])

        u_fem = [o.Uout['u'][n][-1] for n in range(m.nv)]
        CompareArrays(u_fem, Ubh)

    if prob == 5: # bh 1.6
        # input mesh
        V = [[0, -100, 0.0, 0.0],
             [1, -100, 0.0, 2.0],
             [2,    0, 2.0, 0.0],
             [3,    0, 2.0, 1.5],
             [4,    0, 4.0, 0.0],
             [5,    0, 4.0, 1.0]]
        C = [[0, -1, [0,2,3], {}     ],
             [1, -1, [3,1,0], {0:-10}],
             [2, -1, [2,4,5], {}     ],
             [3, -1, [5,3,2], {0:-10}]]
        m = Mesh(V, C)
        #m.draw(pr=1); m.show()

        # parameters
        #p = {-1:{'type':'EelasticTri', 'E':1.0e+4, 'nu':0.2, 'pstress':True, 'thick':0.25}}
        p = {-1:{'type':'Eelastic2D', 'geom':'tri3', 'E':1.0e+4, 'nu':0.2, 'pstress':True, 'thick':0.25}}

        # allocate fem solver object
        s = Solver(m, p)

        # boundary conditions
        vbcs = {-100:{'ux':0.0, 'uy':0.0}}
        ebcs = {-10:{'qnqt':[-20.0, 0.0]}}
        s.set_bcs(eb=ebcs, vb=vbcs)

        # solve steady
        o = s.solve_steady(reactions=True)

        # output
        o.print_res('U')
        o.print_res('R', True)

        # bhatti's solution
        Ubh = array([[ 0.000000000000000e+00,   0.000000000000000e+00],
                     [ 0.000000000000000e+00,   0.000000000000000e+00],
                     [-1.035527877607004e-02,  -2.552969847657423e-02],
                     [ 4.727650463081949e-03,  -2.473565538172127e-02],
                     [-1.313941349422282e-02,  -5.549310752960183e-02],
                     [ 8.389015766816341e-05,  -5.556637423271112e-02]])
        ux_fem = [o.Uout['ux'][n][-1] for n in range(m.nv)]
        uy_fem = [o.Uout['uy'][n][-1] for n in range(m.nv)]
        CompareArrays(ux_fem, Ubh[:,0], table=False, namea='ux', nameb='Bhatti')
        CompareArrays(uy_fem, Ubh[:,1], table=False, namea='uy', nameb='Bhatti')

    if prob == 6: # ZZ.1992 p1339
        # input mesh
        Lx = 1.0
        nx = 9
        m  = Gen1Dmesh(Lx, nx)
        #m.draw(1,1,1,1,1,1); m.show()

        # parameters
        p = {-1:{'type':'Ediffusion1D', 'rho':0., 'beta':1., 'kx':1.,
                 'source':lambda x: 15.*sinh(4.*x)/sinh(4.)+x**2.-2.}}

        # allocate fem solver object
        s = Solver(m, p)

        # boundary conditions
        vbcs = {-101:{'u':0.0}, -102:{'u':0.0}}
        s.set_bcs(vb=vbcs)

        # solve steady
        o = s.solve_steady(reactions=True, extrap=True, emethod=2)
        o.print_res('U')
        o.print_res('R')
        o.print_res('E')

        u_fem  = array([o.Uout['u' ][n][-1] for n in range(m.nv)])
        wx_fem = array([o.Eout['wx'][n][-1] for n in range(m.nv)])

        # solution
        u    = lambda x: x**2.-sinh(4.*x)/sinh(4.)
        dudx = lambda x: 2.*x-(4.*cosh(4.*x))/sinh(4.)

        # error
        X = array([v[2] for v in m.V])
        dudx_sol = dudx(X)
        CompareArrays(dudx_sol, -wx_fem, tol=0.3,  table=True, namea='wx:Solution', nameb='wx:SPR')

        # plot
        x = linspace(0.,Lx,101)
        subplot(2,1,1)
        plot(X, u_fem, 'ro', ls='-', label='fem')
        plot(x, u(x),  'g-', label='solution')
        Gll(r'$x$',r'$u$')
        subplot(2,1,2)
        plot(x, dudx(x), 'g-', label='solution')
        plot(X, -wx_fem, 'ro', label='fem (spr)', ls=':')
        Gll(r'$x$',r'$\sigma = \frac{\partial u}{\partial x}$')
        show()

    if prob == 7: # Reddy: p136
        # input mesh
        Lx = 1.0
        nx = 5
        m  = Gen1Dmesh(Lx, nx)

        # parameters
        p = {-1:{'type':'Ediffusion1D', 'rho':0., 'beta':-1., 'kx':1., 'source':lambda x: -x**2.}}

        # allocate fem solver object
        s = Solver(m, p)

        # boundary conditions
        vbcs = {-101:{'u':0.0}, -102:{'u':0.0}}
        s.set_bcs(vb=vbcs)

        # solve steady
        o = s.solve_steady()

        # plot
        u_fem = [o.Uout['u'][n][-1] for n in range(m.nv)]
        X = [v[2] for v in m.V]
        x = linspace(0.,Lx,101)
        u = x**2.0 - 2.0 + (sin(x)+2.*sin(1.-x)) / sin(1.)
        plot(X, u_fem, 'ro', ls='-', label='fem')
        plot(x, u, 'g-', label='solution')
        Gll(r'$x$',r'$u$')
        show()

    if prob == 8: # ZZ p1342 Example 2

        # input
        nx       = int(sys.argv[2]) if len(sys.argv)>2 else 5
        triangle = int(sys.argv[3]) if len(sys.argv)>3 else 1
        o2       = int(sys.argv[4]) if len(sys.argv)>4 else 0
        centre   = int(sys.argv[5]) if len(sys.argv)>5 else 0
        show_msh = int(sys.argv[6]) if len(sys.argv)>6 else 0

        # mesh
        xcoord = linspace(0.,1.,nx)
        m = Gen2Dregion(xcoord, xcoord, triangle=triangle)
        if o2: m.gen_mid_verts(centre)
        if show_msh:
            m.draw(pr=1)
            plot(0.25,0.25,'ro',zorder=20)
            m.show()

        # parameters
        geom = 'tri3' if triangle else 'qua4'
        if o2:
            geom = 'tri6' if triangle else 'qua8'
            if not triangle and centre:
                geom = 'qua9'
        p = {-1:{'type':'Ediffusion2D', 'rho':0., 'beta':0., 'kx':1., 'ky':1., 'geom':geom,
                 'source':lambda x,y:-14.*y**3.+(16.-12.*x)*y**2.+(-42.*x**2.+54.*x-2.)*y-4.*x**3.+16.*x**2.-12.*x}}

        # allocate fem solver object
        s = Solver(m, p)

        # set boundary conditions
        ebcs = {-10:{'u':0.0}, -11:{'u':0.0}, -12:{'u':0.0}, -13:{'u':0.0}}
        s.set_bcs(eb=ebcs)

        # solve steady
        o = s.solve_steady(extrap=True, emethod=2)
        o.print_res('E')

        # interpolate to a grid and plot using contour
        xg, yg, gv = o.mesh_to_grid()
        Contour(xg, yg, gv['u'], nlevels=11)

        # analytical  solution
        def usol (x,y): return x*(1.-x)*y*(1.-y)*(1.+2.*x+7.*y)
        def gxsol(x,y): return -x*(1.-y)*y*(7.*y+2.*x+1.)+(1.-x)*(1.-y)*y*(7.*y+2.*x+1.)+2.*(1.-x)*x*(1.-y)*y
        def gysol(x,y): return -(1.-x)*x*y*(7.*y+2.*x+1.)+(1.-x)*x*(1.-y)*(7.*y+2.*x+1.)+7.*(1.-x)*x*(1.-y)*y

        # comparison
        print('\n\nComparison:')
        ns = 2*6 + 4*10 + 2*13 + 10
        l  = '='*ns + '\n'
        l += '%6s '%'x' + '%6s  '%'y' + '%10s '%'gxsol' + '%10s  '%'gysol' + '%10s '%'gxnum' + '%10s  '%'gynum'
        l += '%13s '%'err_gx' + '%13s\n'%'err_gy'
        l += '-'*ns + '\n'
        for iv, v in enumerate(m.V):
            x, y = v[2], v[3]
            l += '%6.3f %6.3f  %10.6f %10.6f  %10.6f %10.6f  %13.6e %13.6e\n' % (v[2], v[3],
                 gxsol(x,y), gysol(x,y), -o.Eout['wx'][iv][-1], -o.Eout['wy'][iv][-1],
                 abs(gxsol(x,y) - (-o.Eout['wx'][iv][-1])),
                 abs(gysol(x,y) - (-o.Eout['wy'][iv][-1])))
        l += '='*ns + '\n'
        print(l)

        # plot
        if 1:
            x, y = meshgrid(linspace(0.,1.,31),linspace(0.,1.,31))
            us   = usol(x,y)
            contour(x, y, us,levels=linspace(us.min(),us.max(),11),linestyles='dashed',linewidths=(4),colors=('y'))
            axis('equal')
            show()

    if prob == 9: # Reddy p443 and p494
        tri      = int(sys.argv[2]) if len(sys.argv)>2 else True
        msh      = int(sys.argv[3]) if len(sys.argv)>3 else 2
        show_msh = int(sys.argv[4]) if len(sys.argv)>4 else 0

        if tri: etype = 'EdiffusionTri'
        else:   etype = 'Ediffusion2D'

        p       = {-1:{'type':etype, 'rho':1.,'kx':1.,'ky':1.}}
        msh2nx  = {1:2    , 2:3    , 3:5     , 4:9}
        msh2nev = {1:1    , 2:4    , 3:6     , 4:6}
        nx      = msh2nx [msh]
        nev     = msh2nev[msh]

        # Reddy's results
        tri_msh2lam = {1:[6.0],
                       2:[5.415, 32.0, 38.2, 76.390],
                       3:[5.068, 27.250, 28.920, 58.220, 85.350, 86.790],
                       4:[4.969, 25.340, 25.730, 48.080, 69.780, 69.830]}

        qua_msh2lam = {1:[6.0],
                       2:[5.193, 34.290, 34.290, 63.380],
                       3:[4.999, 27.370, 27.370, 49.740, 84.570, 84.570],
                       4:[4.951, 25.330, 25.330, 45.710, 69.260, 69.260]}

        if nx == 2:
            V = [[0, 0, 0.0, 0.0],
                 [1, 0, 1.0, 0.0],
                 [2, 0, 1.0, 1.0]]
            C = [[0, -1, [0,1,2], {1:-11}]]
            m = Mesh(V, C)
            if show_msh:
                m.draw(); m.show()
            s  = Solver(m, p)
            eb = {-11:{'u':0.0}}
            s.set_bcs(eb=eb)
            s.solve_eigen(dyn=False)
            rl = tri_msh2lam[msh]
        else:
            xc = linspace(0.,1,nx)
            m  = Gen2Dregion(xc, xc, triangle=tri, rotated=True)
            if show_msh:
                m.draw(); m.show()
            if tri:
                s  = Solver(m, p)
                rl = tri_msh2lam[msh]
            else:
                p[-1].update({'geom':'qua4'})
                s  = Solver(m, p)
                rl = qua_msh2lam[msh]
            eb = {-11:{'u':0.0}, -12:{'u':0.0}}
            s.set_bcs(eb=eb)
            s.solve_eigen(dyn=False, nevals=nev, evecs=True)#, vtu_fnkey='prob9_eigen')
        CompareArrays(rl, s.L, tol=1.0e-2, table=True, namea='Reddy', nameb='PyFEM')

    if prob == 10: # Reddy p495
        triangle = int(sys.argv[2]) if len(sys.argv)>2 else True
        mapped   = int(sys.argv[3]) if len(sys.argv)>3 else False
        o2       = int(sys.argv[4]) if len(sys.argv)>4 else False
        nx       = int(sys.argv[5]) if len(sys.argv)>5 else 5
        show_msh = int(sys.argv[6]) if len(sys.argv)>6 else 0
        xcoord   = linspace(0.,1.,nx)
        m = Gen2Dregion(xcoord, xcoord, triangle=triangle, rotated=True)
        if not triangle: mapped = True
        if triangle: geom = 'tri3'
        else:        geom = 'qua4'
        if o2:
            mapped = True
            m.gen_mid_verts()
            if triangle: geom = 'tri6'
            else:        geom = 'qua8'
        m.tag_verts_on_line(-100, 0.,0.,0.)
        if show_msh:
            m.draw(); m.show()

        kx, src = 1.0, 1.0
        etype = 'Ediffusion2D' if mapped else 'EdiffusionTri'
        p = {-1:{'type':etype, 'rho':1.,'kx':kx,'ky':kx, 'source':src}}
        if mapped: p[-1].update({'geom':geom})
        s  = Solver(m, p)
        eb = {-11:{'u':0.0}, -12:{'u':0.0}}

        # find lmax
        dense = True if o2 else False
        s.set_bcs(eb=eb)
        s.solve_eigen(dyn=False, dense=dense)
        lmax = max(s.L)
        dtcrit = 2.0 / lmax
        if mapped: print('. . . . . . . '+geom+' element . . . . . . .')
        else:      print('. . . . . . . triangle element . . . . . . .')
        print('lmax   =', lmax)
        print('dtcrit =', dtcrit)

        # plot u @ node 0 versus time ------------------------------
        subplot(2,1,1)

        # solve transient (Euler/forward)
        dt = dtcrit
        o = s.solve_transient(0.0, 0.1, dt, theta=0.0)
        plot(o.Tout, o.Uout['u'][0], 'k-', label=r'$\theta=0$, $\Delta t=%g$'%dt)

        if triangle and not mapped:
            dt = 0.01
            o = s.solve_transient(0.0, 0.1, dt, theta=0.0)
            plot(o.Tout, o.Uout['u'][0], 'ro', ls='-', label=r'$\theta=0$, $\Delta t=%g$'%dt)

        # solve transient (Crank/Nicolson)
        dt = 0.01
        o = s.solve_transient(0.0, 0.1, dt, theta=0.5)
        plot(o.Tout, o.Uout['u'][0], 'g+', ls='-', label=r'$\theta=0.5$, $\Delta t=%g$'%dt, ms=10)

        # labels, grid, and legend
        Gll(r'$t$', r'$u(x=0,y=0)$')

        # plot x versus a number of times ------------------------------

        # steady solution
        o = s.solve_steady()

        # steady-state analytical solution
        def usteady(x,y):
            res = 0.0
            for i in range(1,21):
                a = 0.5*pi*(2.*i-1.)
                res += ((-1.)**float(i)) * cos(a*y) * cosh(a*x) / ((a**3.) * cosh(a))
            return (0.5*src/kx) * ((1.-y**2.) + 4.0*res)
        subplot(2,1,2)

        # collect bottom nodes
        Be = m.get_verts(-100, xsorted=True)
        Bx = [m.V[n][2] for n in Be]
        Bu = [o.Uout['u'][n][-1] for n in Be]

        # plot solution and numerical steady solution
        x = linspace(0.,1.,101)
        plot(x,usteady(x,0.),'k-',lw=2,label='steady solution')
        plot(Bx, Bu, 'yo',ls=':',label='fem: steady')

        # plot transient solution
        dt = 0.005
        o  = s.solve_transient(0.0, 1.0, dt, theta=0.5)
        ITout = GetITout(o.Tout, [0.005, 0.1, 0.2, 0.5, 1.0])
        for iout, tout in ITout:
            U = [o.Uout['u'][vid][iout] for vid in Be]
            plot(Bx, U, label=r'$t=%g$'%tout)

        # labels, grid, and legend
        Gll(r'$x$', r'$u(x,0)$')
        show()

    if prob == 11: # Bhatti p445
        V = [[0, 0, 0.00, 0.030],
             [1, 0, 0.03, 0.030],
             [2, 0, 0.03, 0.015],
             [3, 0, 0.06, 0.015],
             [4, 0, 0.06, 0.000],
             [5, 0, 0.00, 0.000]]
        C = [[0, -1, [5, 2, 1, 0], {1:-11, 2:-11, 3:-13}],
             [1, -1, [5, 4, 3, 2], {0:-10, 2:-11, 1:-14}]]
        m = Mesh(V, C)

        o2 = int(sys.argv[2]) if len(sys.argv)>2 else False
        geom = 'qua4'
        if o2:
            m.gen_mid_verts()
            geom = 'qua8'
        #m.draw()
        #m.show()

        p = {-1:{'type':'Ediffusion2D', 'kx':45., 'ky':45., 'source':5.e+6, 'geom':geom}}
        s = Solver(m, p)

        cc, Tinf = 55., 20.
        eb = {-10:{'u':110.}, -11:{'c':(cc,Tinf)}, -13:{'q':8000.}}
        s.set_bcs(eb=eb)

        o = s.solve_steady()
        u_fem = [o.Uout['u'][n][-1] for n in range(m.nv)]

        # bhatti's solution
        if geom == 'qua4':
            Ubh = array([1.533936036487368e+02,
                         1.429067064467641e+02,
                         1.328533310855348e+02,
                         1.245394204969134e+02,
                         1.100000000000000e+02,
                         1.100000000000000e+02])
            CompareArrays(u_fem, Ubh)
        elif geom == 'qua8':
            Ubh = array([1.564405024662022e+02,  #  0
                         1.491964629456369e+02,  #  1
                         1.338432701060948e+02,  #  2
                         1.217463572762217e+02,  #  3
                         1.100000000000000e+02,  #  4
                         1.100000000000000e+02,  #  5
                         1.442245542836665e+02,  #  6
                         1.507560541872988e+02,  #  7
                         1.100000000000000e+02,  #  8
                         1.446754222244303e+02,  #  9
                         1.240019529443107e+02,  # 10
                         1.291320079882028e+02,  # 11
                         1.191481315065258e+02]) # 12
            CompareArrays(u_fem, Ubh)

    if prob == 12: # Reddy p622
        triangle = int(sys.argv[2]) if len(sys.argv)>2 else True
        mapped   = int(sys.argv[3]) if len(sys.argv)>3 else False
        o2       = int(sys.argv[4]) if len(sys.argv)>4 else False
        nx       = int(sys.argv[5]) if len(sys.argv)>5 else 2
        xcoord   = linspace(0.,120.,nx)
        ycoord   = linspace(0.,160.,nx)
        m = Gen2Dregion(xcoord, ycoord, triangle=triangle, rotated=True)
        if not triangle: mapped = True
        if triangle: geom = 'tri3'
        else:        geom = 'qua4'
        if o2:
            mapped = True
            m.gen_mid_verts()
            if triangle: geom = 'tri6'
            else:        geom = 'qua8'
        #m.draw()
        #m.show()

        if mapped: print('. . . . . . . '+geom+' element . . . . . . .')
        else:      print('. . . . . . . triangle element . . . . . . .')

        etype = 'Eelastic2D' if mapped else 'EelasticTri'
        thick = 0.036
        p = {-1:{'type':etype, 'E':30.e+6, 'nu':0.25, 'pstress':True, 'thick':thick}}

        if mapped: p[-1].update({'geom':geom})
        s = Solver(m, p)

        eb = {-13:{'ux':0.0, 'uy':0.0}, -11:{'qnqt':(10.0/thick,0.)}}
        s.set_bcs(eb=eb)
        o = s.solve_steady()

        ux_fem = [o.Uout['ux'][n][-1] for n in range(m.nv)]
        uy_fem = [o.Uout['uy'][n][-1] for n in range(m.nv)]

        if triangle and nx == 2 and not o2:
            Urd = array([[0.0,          0.0       ],
                         [11.291*1e-4,  1.964*1e-4],
                         [0.0,          0.0       ],
                         [10.113*1e-4, -1.080*1e-4]])
            CompareArrays(ux_fem, Urd[:,0], tol=1.e-7)
            CompareArrays(uy_fem, Urd[:,1], tol=1.e-7)

    if prob == 13: # SG p181
        V = [[ 0,  0,  0.0,   0.0],
             [ 1,  0,  0.0,  -3.0],
             [ 2,  0,  0.0,  -6.0],
             [ 3,  0,  0.0,  -9.0],
             [ 4,  0,  3.0,   0.0],
             [ 5,  0,  3.0,  -3.0],
             [ 6,  0,  3.0,  -6.0],
             [ 7,  0,  3.0,  -9.0],
             [ 8,  0,  6.0,   0.0],
             [ 9,  0,  6.0,  -3.0],
             [10,  0,  6.0,  -6.0],
             [11,  0,  6.0,  -9.0]]
        C = [[ 0, -1, [0,1, 5, 4], {0:-10, 3:-30}],
             [ 1, -1, [1,2, 6, 5], {0:-10}       ],
             [ 2, -1, [2,3, 7, 6], {0:-10, 1:-20}],
             [ 3, -1, [4,5, 9, 8], {2:-10}       ],
             [ 4, -1, [5,6,10, 9], {2:-10}       ],
             [ 5, -1, [6,7,11,10], {1:-20, 2:-10}]]
        m = Mesh(V, C)
        m.gen_mid_verts()
        #m.draw()
        #m.show()

        from tlfem.quadrature import QuaIp4
        p = {-1:{'type':'Eelastic2D', 'E':1.0e+6, 'nu':0.3, 'geom':'qua8', 'ipe':QuaIp4}}
        s = Solver(m, p)

        eb = {-10:{'ux':0.}, -20:{'ux':0., 'uy':0.}, -30:{'qnqt':(-1.0,0.)}}
        s.set_bcs(eb=eb)
        o = s.solve_steady()

        ux_fem = [o.Uout['ux'][n][-1] for n in range(m.nv)]
        uy_fem = [o.Uout['uy'][n][-1] for n in range(m.nv)]

        Usg = array([[ 0.000000000000000E+00, -5.310671749739340E-06],  #  0
                     [ 0.000000000000000E+00, -3.243122913926113E-06],  #  1
                     [ 0.000000000000000E+00, -1.378773528167206E-06],  #  2
                     [ 0.000000000000000E+00,  0.000000000000000E+00],  #  3
                     [-7.221882229919595E-07, -3.342856970059000E-06],  #  4
                     [ 3.669995660268174E-07, -2.228571333618901E-06],  #  5
                     [ 1.912334843273510E-07, -1.114285662470972E-06],  #  6
                     [ 0.000000000000000E+00,  0.000000000000000E+00],  #  7
                     [ 0.000000000000000E+00, -1.375042190378655E-06],  #  8
                     [ 0.000000000000000E+00, -1.214019753311685E-06],  #  9
                     [ 0.000000000000000E+00, -8.497977967747343E-07],  # 10
                     [ 0.000000000000000E+00,  0.000000000000000E+00],  # 11
                     [ 0.000000000000000E+00, -4.288483503711751E-06],  # 12
                     [ 0.000000000000000E+00, -2.217378283206796E-06],  # 14 => 13
                     [ 2.996313570447692E-07, -1.671428489956403E-06],  # 21 => 14
                     [ 1.370485290061350E-07, -1.298706937331023E-06],  # 17 => 15
                     [ 1.121994376475524E-07, -5.571428285393078E-07],  # 23 => 16
                     [ 2.708147610339123E-07, -1.583976767620262E-06],  # 22 => 17
                     [ 3.774202249726477E-07, -2.785714138358060E-06],  # 19 => 18
                     [-4.211153193865630E-07, -1.644482664256252E-06],  # 20 => 19
                     [ 0.000000000000000E+00,  0.000000000000000E+00],  # 25 => 20
                     [ 0.000000000000000E+00, -1.282944773004366E-06],  # 26 => 21
                     [ 2.708147610339125E-07, -2.873165860694200E-06],  # 15 => 22
                     [ 1.370485290061355E-07, -9.298643811646859E-07],  # 24 => 23
                     [ 0.000000000000000E+00, -6.453528854650004E-07],  # 16 => 24
                     [-4.211153193865630E-07, -5.041231308584792E-06],  # 13 => 25
                     [ 0.000000000000000E+00,  0.000000000000000E+00],  # 18 => 26
                     [ 0.000000000000000E+00, -4.689327716136146E-07],  # 28 => 27
                     [ 0.000000000000000E+00, -1.125478696706005E-06]]) # 27 => 28
        CompareArrays(ux_fem, Usg[:,0])
        CompareArrays(uy_fem, Usg[:,1])

    if prob == 14: # lewis/nithi/seetha (lns) p159

        subprob = int(sys.argv[2]) if len(sys.argv)>2 else 1
        th_type = int(sys.argv[3]) if len(sys.argv)>3 else 2

        if subprob == 1:
            m = Gen2Dregion(linspace(0.,20.,41), linspace(0.,1.,3), triangle=True)
            m.tag_verts_on_line(-111,0.,0.5,0.)
            #m.draw(); m.show()
            p = {-1:{'type':'EdiffusionTri', 'rho':1.,'kx':1.,'ky':1.}}
            s = Solver(m, p)

        if subprob == 2:
            m = Gen2Dregion(linspace(0.,20.,41), linspace(0.,1.,2), triangle=False)
            m.tag_verts_on_line(-111,0.,0.0,0.)
            m.draw(); m.show()
            p = {-1:{'type':'Ediffusion2D', 'rho':1.,'kx':1.,'ky':1.,'geom':'qua4'}}
            s = Solver(m, p)

        eb = {-13:{'q':1.}, -11:{'u':0.}}
        s.set_bcs(eb=eb)

        if th_type == 1:
            s.solve_eigen(dyn=False, dense=True)
            lmin, lmax = s.L.min(), s.L.max()
            dtcrit = 2.0 / lmax
            print('lmin =', lmin, '  lmax =', lmax)
            print('dtcrit =', dtcrit)
            theta = 0.0
            #
            #
            # try dt = 0.018 for example of instabilities
            #
            #
            dt = float(sys.argv[4]) if len(sys.argv)>4 else dtcrit
            t0, U0, tf = 0., zeros(m.nv), 1.

        if th_type == 2:
            #theta = 0.878
            theta = 0.5
            t0, tf, dt = 0., 1., 0.1

        # solve
        o = s.solve_transient(t0, tf, dt, theta=theta)

        def usol(t,x): return 2.0*sqrt(t/pi)*(exp(-0.25*x*x/t)-0.5*x*sqrt(pi/t)*erfc(0.5*x/sqrt(t)))

        # plot values at times -----------------------------------------------------
        subplot(3, 1, 1)
        nods  = m.get_verts(-111, xsorted=True)
        X     = [m.V[n][2] for n in nods]
        x     = linspace(0.,20.,101)
        ITout = GetITout(o.Tout, [dt, 0.1, 0.2, 0.5, 1.], 0.9*dt)
        for iout, tout in ITout:
            U = [o.Uout['u'][n][iout] for n in nods]
            plot(x, usol(tout,x), 'g-', lw=2, label='sol @ t=%g'%tout)
            plot(X, U, 'ro', ls='-', label='fem @ t=%g'%tout)
        Gll(r'$x$',r'$u$')
        Cross()
        axis([-0.5,4.5,axis()[2],axis()[3]])

        # plot values at nodes ----------------------------------------------------
        subplot(3, 1, 2)
        t = linspace(0.001,tf,101)
        for i in [0, 1, 2, -1]:
            n  = nods[i]
            xn = m.V[n][2]
            plot(o.Tout, o.Uout['u'][n], label='fem @ $x=%g$ ($u_{max}=%g$)'%(xn,max(o.Uout['u'][n])))
            plot(t, usol(t,xn), 'k:')
        Gll('r$t$',r'$u$')

        # plot error --------------------------------------------------------------
        subplot(3, 1, 3)
        ini    = 2
        maxerr = 0.0
        for n in [0, 1, 2, m.nv-1]: # for each node index
            xn  = s.m.V[n][2]      # x-coordinate of node
            uc  = usol(array(o.Tout[ini:]), xn)  # u-correct
            err = abs(o.Uout['u'][n][ini:] - uc)
            maxerr = max(maxerr, max(err))
            plot(o.Tout[ini:], err, label='error @ x=%g'%(xn))
        Gll('t','error')
        print('maxerr =', maxerr)

        # title and show
        suptitle('theta-method: theta=%g, dt=%g' % (theta, dt))
        show()

    if prob == 15: # zienk p602
        xcoord = linspace(0.,4.,11)
        ycoord = linspace(0.,0.4,2)
        m = Gen2Dregion(xcoord, ycoord, triangle=False)
        m.gen_mid_verts()
        #m.draw()
        #m.show()
        p = {-1:{'type':'Ediffusion2D', 'rho':1.,'kx':1.,'ky':1.,'geom':'qua8'}}
        s = Solver(m, p)
        eb = {-13:{'u':1.}}
        s.set_bcs(eb=eb)
        t0, tf, dt = 0., 30., 2.
        # theta-method
        for theta in [0.5, 2./3., 0.878]:
            o = s.solve_transient(t0, tf, dt, theta=theta)
            #for k, n in enumerate([46, 16, 49]):
            for k, n in enumerate([34, 16, 49]):
                subplot(3,1,k+1)
                plot(o.Tout, o.Uout['u'][n], label=r'fem 2D @ x=%g $\theta=%g$'%(m.V[n][2],theta))
        show()

    if prob == 16: # zienk p602
        m = Gen1Dmesh(4.,11)
        #m.draw()
        #m.show()
        p = {-1:{'type':'Ediffusion1D', 'rho':1.,'kx':1.}}
        s = Solver(m, p)
        vb = {-101:{'u':1.}}
        s.set_bcs(vb=vb)
        t0, tf, dt = 0., 30., 2.
        for theta in [0.5, 2./3., 0.878]:
            o = s.solve_transient(t0, tf, dt, theta=theta)
            for k, n in enumerate([2,5,10]):
                subplot(3,1,k+1)
                plot(o.Tout, o.Uout['u'][n], label=r'fem 1D @ x=%g $\theta=%g$'%(m.V[n][2],theta))
                Gll(r'$t$',r'$u$')
        show()

    if prob == 17: # reddy p77 Set 2
        m = Gen1Dmesh(1.,11)
        #m.draw()
        #m.show()
        p = {-1:{'type':'Ediffusion1D', 'kx':1.,'beta':-1.,'source':lambda x: -x**2.}}
        s = Solver(m, p)
        if 0:
            vb = {-101:{'u':0.}, -102:{'u':0.}}
            def usol(x): return x**2.0 - 2.0 + (sin(x)+2.*sin(1.-x)) / sin(1.)
        else:
            vb = {-101:{'u':0.}, -102:{'Q':1.}}
            def usol(x): return (2.*cos(1.-x)-sin(x))/cos(1.) + x**2. - 2.
        s.set_bcs(vb=vb)
        o = s.solve_steady()
        X = [v[2] for v in m.V]
        x = linspace(0.,1.,101)
        u_fem = [o.Uout['u'][n][-1] for n in range(m.nv)]
        plot(x,usol(x),'g-',label='solution')
        plot(X,u_fem,'ro',label='fem')
        Gll(r'$x$',r'$u$')
        show()

    if prob==18: # reddy p340 q6.18
        nn = 11
        m1d = Gen1Dmesh(1.,nn)
        V, C = [], []
        for v in m1d.V: v.append(0.)
        m = Mesh(m1d.V, m1d.C)
        #m.draw(); m.show()
        p = {-1:{'type':'EelasticRod', 'rho':1., 'E':1., 'A':1.}}
        s = Solver(m, p)
        #vb = {-101:{'ux':0.,'uy':0.},-100:{'uy':0.0},-102:{'uy':0.,'fx':1.}}
        if nn > 2: vb = {-101:{'ux':0.,'uy':0.},-100:{'uy':0.0},-102:{'uy':0.,'fx': lambda t: 1.}}
        else:      vb = {-101:{'ux':0.,'uy':0.},                -102:{'uy':0.,'fx': lambda t: 1.}}
        s.set_bcs(vb=vb)
        t0, tf, dt     = 0., 2.1, 0.1
        theta1, theta2 = 0.5, 0.5

        rk_prms={'silent':False, 'PredCtrl':True, 'Rtol':1e-1, 'MaxSS':1000}

        o = s.solve_dynamics(t0, tf, dt, theta1=theta1, theta2=theta2)
        #print('uy =', o.Uout['uy'])
        #print()
        #print('ux =', o.Uout['ux'])
        def u(t,x,rho,E,A,L,P):
            a = sqrt(E/rho)
            s = 0.0
            for j in range(20):
                i = float(j)*2.+1.
                m = i*pi/(2.*L)
                print(i, j)
                s += ((-1.)**float(j)) * sin(m*x) * (1.-cos(m*a*t)) / (i**2.)
            return s * 8.*L*P/(pi*pi*a*a*rho*A)
        def uu(t):
            s = 0.0
            for j in range(40):
                i = float(j)*2.+1.
                s += (1.-cos(i*pi*t/2.)) / (i**2.)
            return s * 8./(pi**2.)

        d = Read('../data/rod1d_dyn_nod_%d.cmp'%(nn-1))
        plot(d['Time'],d['ux'],'b+',label='mechsys',ms=10)

        #uv = vectorize(u)
        #usol = uv(o.Tout, 1., 1.,1.,1.,1.,1.)
        uv = vectorize(uu)
        T  = linspace(t0, tf, 201)
        plot(T, uv(T), 'g-', label='solution')
        plot(o.Tout, o.Uout['ux'][nn-1], 'r.',ls='-',label='fem')
        Gll('t','ux')
        show()

    if prob==19:
        H = 1.0   # height
        h = 0.5   # half height
        L = 4.0   # length
        l = L/3.  # element length
        m = l/2.  # half length (for mid nodes)
        V = [[ 0,    0,  0.0,    H],
             [ 1,    0,  0.0,    h],
             [ 2,    0,  0.0,  0.0],
             [ 3,    0,    m,    H],
             [ 4,    0,    m,  0.0],
             [ 5,    0,    l,    H],
             [ 6,    0,    l,    h],
             [ 7,    0,    l,  0.0],
             [ 8,    0,  l+m,    H],
             [ 9,    0,  l+m,  0.0],
             [10,    0,  l+l,    H],
             [11,    0,  l+l,    h],
             [12,    0,  l+l,  0.0],
             [13,    0,  L-m,    H],
             [14,    0,  L-m,  0.0],
             [15,    0,    L,    H],
             [16,    0,    L,    h],
             [17, -100,    L,  0.0]]
        C = [[ 0,   -1, [ 2, 7, 5, 0,  4, 6, 3, 1], {3:-10}],
             [ 1,   -1, [ 7,12,10, 5,  9,11, 8, 6]],
             [ 2,   -1, [12,17,15,10, 14,16,13,11]]]
        m = Mesh(V, C)
        #m.draw();  m.show()
        from tlfem.quadrature import QuaIp4
        #p = {-1:{'type':'Eelastic2D', 'rho':1., 'E':1., 'nu':0.3, 'geom':'qua8', 'ipe':QuaIp4}}
        p = {-1:{'type':'Eelastic2D', 'rho':1., 'E':1., 'nu':0.3, 'geom':'qua8'}}
        s = Solver(m, p)
        vb = {-100:{'fy':lambda t: cos(0.3*t)}}
        eb = {-10:{'ux':0., 'uy':0.}}
        s.set_bcs(eb=eb, vb=vb)
        t0, tf, dt     = 0., 100., 1.0
        theta1, theta2 = 0.5, 0.5
        rayM,   rayK   = 0.005, 0.272
        alpha          = -0.1
        o = s.solve_dynamics(t0, tf, dt, theta1=theta1, theta2=theta2, rayM=rayM, rayK=rayK, HHT=True, alpha=alpha)
        d = Read('../data/fig_11_04_nod_17.cmp')
        plot(d['Time'],d['uy'],'b+',label='mechsys',ms=10)
        plot(o.Tout, o.Uout['uy'][17], 'ro', ls='-',label='fem')
        Gll('t','uy')
        show()

    if prob==20: # Yang, Bingen Stress, Strain and structural dynamics
                 # Fig 15.1.1 p740
        show_msh = int(sys.argv[2]) if len(sys.argv)>2 else 0
        xcoord = linspace(0.,3.,5)
        ycoord = linspace(0.,2.,5)
        m = Gen2Dregion(xcoord, ycoord, cross=True)
        m.tag_verts(-111, [4])
        if show_msh:
            m.draw(); m.show()
        p = {-1:{'type':'EelasticTri', 'E':5.e4, 'nu':0.3, 'pstress':True, 'thick':0.1, 'rho':1.}}
        s = Solver(m, p)
        vb = {-111:{'fy':-50.0}}
        eb = {-13:{'ux':0., 'uy':0.}}
        s.set_bcs(vb=vb, eb=eb)
        s.solve_steady()

    if prob==21: # Reddy: Example 11.7.2 p625
        trans    = int(sys.argv[2]) if len(sys.argv)>2 else 1
        use_tri  = int(sys.argv[3]) if len(sys.argv)>3 else 0
        show_msh = int(sys.argv[4]) if len(sys.argv)>4 else 0
        msh2nx = [5, 9, 17,
                  5, 9, 17,
                  3, 5, 8]
        msh2ny = [3, 3, 3,
                  3, 3, 3,
                  2, 2, 2]
        msh2ge = ['tri3', 'tri3', 'tri3',
                  'qua4', 'qua4', 'qua4',
                  'qua9', 'qua9', 'qua9']
        # Reddy's results
        msh2uy = [-0.001611, -0.002662, -0.003166,
                  -0.003134, -0.004388, -0.004878,
                  -0.005031, -0.005129, -0.005137]
        ge_steady = ['tri3', 'qua4', 'qua9']
        ge_trans  = ['tri3', 'qua4', 'qua9']
        #ge_trans  = ['qua4','qua9']
        for msh in range(9):
            nx, ny, ge = msh2nx[msh], msh2ny[msh], msh2ge[msh]
            tri, o2    = ge2tri[ge], ge2o2[ge]
            a, b, h    = 10., 2., 1.
            xc, yc     = linspace(0.,a,nx), linspace(0.,b,ny)
            m = Gen2Dregion(xc, yc, triangle=tri, rotated=True, vtags=False)
            if o2: m.gen_mid_verts(ge2cen[ge])
            m.tag_vert(-111,a,b/2.)
            m.tag_vert(-222,0,b/2.)
            if show_msh:
                m.draw(); m.show()
            etype = 'Eelastic2D'
            if ge == 'tri3' and use_tri:
                print('[1;35m>>>>>>>>>>>>>>>>>>> using EelasticTri <<<<<<<<<<<<<<<<<<<<[0m')
                etype = 'EelasticTri'
            p  = {-1:{'type':etype, 'E':30.e6, 'nu':0.25, 'rho':8.8255e-3, 'pstress':True, 'thick':h, 'geom':ge}}
            s  = Solver(m, p)
            vb = {-111:{'ux':0., 'uy':0.}}
            eb = {-13:{'qnqt':(0.,150.)}, -11:{'ux':0.}}
            s.set_bcs(vb=vb, eb=eb)
            key = '%d %s (%d,%d)' % (m.nc, ge, nx-1, ny-1)
            if ge in ge_steady:
                o = s.solve_steady()
                ux_fem = [o.Uout['ux'][n][-1] for n in range(m.nv)]
                uy_fem = [o.Uout['uy'][n][-1] for n in range(m.nv)]
                vid = m.get_verts(-222)[0]
                print(key)
                print('  vid =', vid, '  uy =', uy_fem[vid],  ' uy(Reddy) =', msh2uy[msh])
                PrintDiff('  diff (uy@(0,1)) = %17.10e', uy_fem[vid], msh2uy[msh], 1.e-6)
                P, E, I, nu, L = 150.*h*b, 30.e6, 2./3., 0.25, a
                uy01 = -(P*L**3.)*(1.+3.*(1.+nu)/(L**2.))/(3.*E*I)
                print('uy (elasticity) =', uy01)
            if trans and ge in ge_trans:
                theta1, theta2, dt = 0.5, 0.5, 0.25e-3
                o  = s.solve_dynamics(0., 0.005, dt, theta1=theta1, theta2=theta2)
                mk = None if msh < 5 else '+'
                plot(o.Tout, o.Uout['uy'][0], marker=mk, label=key)
                #print(s.elems[0].calc_M())
                #print()
        if trans:
            Gll('t','uy @ 0')
            show()

    if prob==22: # Reddy: Example 11.7.3 p627
        show_msh = int(sys.argv[2]) if len(sys.argv)>2 else 0
        withtri6 = int(sys.argv[2]) if len(sys.argv)>2 else 0
        msh2nx = [5, 9, 3, 5, # triangles
                  5, 9, 3, 5] # quads
        msh2ny = [3, 3, 2, 2,
                  3, 3, 2, 2]
        msh2ge = ['tri3', 'tri3', 'tri6', 'tri6',
                  'qua4', 'qua4', 'qua9', 'qua9']
        # Reddy's results
        msh2oms = [[2019.4, 9207.4, 10449.6, 25339.2, 29193.2, 42363.4, 52937.0, 67964.6, 76833.2, 79443.0], # tri3
                   [1583.0, 8264.0,  9177.7, 19540.5, 27843.9, 32727.8, 46840.4, 48014.4, 61560.4, 68257.4], # tri3
                   [1186.4, 7896.6,  9158.2, 18369.1, 27805.3, 40399.2, 50469.6, 66260.9, 74582.1, 79241.8], # tri6
                   [1156.7, 6496.5,  9156.0, 16219.9, 27441.7, 28696.9, 39762.6, 45815.6, 57429.5, 64867.4], # tri6
                   [1465.5, 8457.9,  9218.4, 22334.0, 29113.3, 40309.7, 52991.9, 66842.5, 74523.3, 76515.5], # qua4
                   [1242.3, 6845.8,  9171.7, 16887.7, 27836.8, 29433.6, 44231.1, 47441.0, 60078.3, 67813.3], # qua4
                   [1169.9, 7197.7,  9158.2, 17890.8, 27869.8, 39583.7, 50964.4, 67015.3, 74064.6, 80029.3], # qua9
                   [1151.8, 6341.4,  9156.0, 15572.7, 27226.3, 27442.2, 39302.3, 45839.9, 56949.9, 64636.0]] # qua9
        tests = range(8)
        if not withtri6: tests = [0,1, 4,5, 6,7]
        for msh in tests:
            nx, ny, ge = msh2nx[msh], msh2ny[msh], msh2ge[msh]
            tri, o2    = ge2tri[ge], ge2o2[ge]
            a, b, h    = 10., 2., 1.
            xc, yc     = linspace(0.,a,nx), linspace(0.,b,ny)
            m = Gen2Dregion(xc, yc, triangle=tri, rotated=True, vtags=False)
            if o2: m.gen_mid_verts(ge2cen[ge])
            m.tag_vert(-111,a,b/2.)
            m.tag_vert(-222,0,b/2.)
            p  = {-1:{'type':'Eelastic2D', 'E':30.e6, 'nu':0.25, 'rho':8.8255e-3, 'pstress':True, 'thick':h, 'geom':ge}}
            s  = Solver(m, p)
            vb = {-111:{'ux':0., 'uy':0.}}
            eb = {-11:{'ux':0.}}
            s.set_bcs(vb=vb, eb=eb)

            print('[1;33m', ge, nx, ny, '[0m')

            # critical time-step corresponding to Newmark parameters
            s.solve_eigen() # just the maximum eigenvalue
            th1, th2 = 0.5, 1./3. # Newmark parameters
            lmax     = max(s.L)   # largest lambda
            om_max   = sqrt(lmax) # largest natural frequency
            dtcrit   = sqrt(2.0)/(om_max*sqrt(th1-th2)) # critical dt
            print('lambdas =', s.L)
            print('lmax    =', lmax)
            print('om_max  =', om_max)
            print('dtcrit  =', dtcrit)

            s.solve_eigen(nevals=10)
            oms = sqrt(s.L)
            CompareArrays(msh2oms[msh], oms, tol=0.07, table=True, namea='Reddy', nameb='PyFEM')
            print()
            if show_msh:
                m.draw(); m.show()

    if prob==23:
        from tlfem.Eelastic2D import Eelastic2D
        from tlfem.shape      import shape_derivs
        show_msh = int(sys.argv[2]) if len(sys.argv)>2 else 0
        #for msh in [3]:
        for msh in range(2):
            if msh==0:
                tg60 = tan(60.*pi/180.)
                V = [[0, 0, 0.0,          0.0],
                     [1, 0, 10.0,         0.0],
                     [2, 0, 10.+10.*tg60, 10.],
                     [3, 0, 10*tg60,      10.]]
                C = [[0, -1, [0,1,2,3]]]
                g = 'qua4'
                def dJ(r,s): return 25.0
            if msh==1:
                V = [[0, 0, 0.0, 0.0],
                     [1, 0, 5.0, 0.0],
                     [2, 0, 6.5, 7.0],
                     [3, 0, 0.0, 5.0]]
                C = [[0, -1, [0,1,2,3]]]
                g = 'qua4'
                def dJ(r,s): return (67.5 + 7.5*s + 10.*r)/8.0
            if msh==2:
                V = [[0, 0,  0.0, 0.0],
                     [1, 0,  6.0, 0.0],
                     [2, 0, 10.0, 6.0],
                     [3, 0,  0.0, 6.0],
                     [4, 0,  3.0, 0.0],
                     [5, 0,  6.5, 1.5],
                     [6, 0,  5.0, 6.0],
                     [7, 0,  0.0, 3.0]]
                C = [[0, -1, [0,1,2,3,4,5,6,7]]]
                g = 'qua8'
                def dJ(r,s):
                    return 10.5 + 0.75*r + 9.*s + 6.*r*s + 3.*s**2. + 0.75*r*s**2.
            if msh==3:
                V = [[0, 0,  0.0, 0.0],
                     [1, 0,  4.0, 0.0],
                     [2, 0,  6.0, 4.0],
                     [3, 0,  0.0, 4.0],
                     [4, 0,  2.0, 0.0],
                     [5, 0,  4.5, 1.0],
                     [6, 0,  3.0, 4.0],
                     [7, 0,  0.0, 2.0],
                     [8, 0,  2.5, 2.0]]
                C = [[0, -1, [0,1,2,3,4,5,6,7]]]
                g = 'qua9'
                def dJ(r,s):
                    return 4.75 - 3.25*r - 1.5*s - 8.5*r*s - 3.5*r**2. - 2.25*s**2. \
                           - 5.25*r*s**2. - 5.*s*r**2. - r**3. - 2.*s*r**3. \
                           - 1.5*(r**2.)*(s**2.) - (r**3.)*(s**2.)
            m = Mesh(V, C)
            P = {'E':1., 'nu':0.25, 'geom':g}
            e = Eelastic2D(V, P)
            print(msh)
            for i, ip in enumerate(e.ipe):
                S, G, detJ = shape_derivs(e.xy, e.fce, ip)
                PrintDiff('  diff = %g', detJ, dJ(ip[0],ip[1]), 1.0e-14)
            if show_msh:
                m.draw(); m.show()

    if prob==24: # Reddy: Problem 13.60, p708
        show_msh = int(sys.argv[2]) if len(sys.argv)>2 else 0
        m = Gen2Dregion([0.,0.125,0.25,0.5,0.75,1.25,2.,2.75,3.75,4.75,6.0], linspace(0.,2.,5), triangle=False, vtags=False)
        m.tag_vert(-100,0.,1.0)
        m.tag_vert(-111,6.,0.0)
        m.tag_vert(-122,6.,0.5)
        m.tag_vert(-133,6.,1.0)
        m.tag_vert(-144,6.,1.5)
        m.tag_vert(-155,6.,2.0)
        if show_msh: m.draw(); m.show()
        p = {-1:{'type':'Eelastic2D', 'E':3.e7, 'nu':0.3, 'pstress':True, 'thick':1., 'rho':0.0088, 'geom':'qua4'}}
        s = Solver(m, p)
        vb = {-100:{'ux':0., 'uy':0.},
              -111:{'fx':-187.5},
              -122:{'fx':-225.0},
              -144:{'fx': 225.0},
              -155:{'fx': 187.5}}
        eb = {-13:{'ux':0.}}
        s.set_bcs(vb=vb, eb=eb)
        o = s.solve_steady()
        vid = m.get_verts(-133)[0]
        ux_fem = [o.Uout['ux'][n][-1] for n in range(m.nv)]
        uy_fem = [o.Uout['uy'][n][-1] for n in range(m.nv)]
        print('  vid =', vid, '  uy =', uy_fem[vid])#,  ' uy(Reddy) =', msh2uy[msh]

        rux_0_21 = array([0.00000E+00, -0.37352E-05, -0.74728E-05, -0.14897E-04, -0.22359E-04, -0.36942E-04,
                         -0.58041E-04, -0.79386E-04, -0.10646E-03, -0.13396E-03, -0.16520E-03,  0.00000E+00,
                         -0.18461E-05, -0.36899E-05, -0.73777E-05, -0.11029E-04, -0.18282E-04, -0.29024E-04,
                         -0.39502E-04, -0.53253E-04, -0.66617E-04, -0.84050E-04])

        rux_34_54 = array([0.18461E-05,  0.36899E-05,  0.73777E-05,  0.11029E-04,  0.18282E-04,  0.29024E-04,
                           0.39502E-04,  0.53253E-04,  0.66617E-04,  0.84050E-04,  0.00000E+00,  0.37352E-05,
                           0.74728E-05,  0.14897E-04,  0.22359E-04,  0.36942E-04,  0.58041E-04,  0.79386E-04,
                           0.10646E-03,  0.13396E-03,  0.16520E-03])

        ruy_0_21 = array([-0.44216E-05, -0.46542E-05, -0.53515E-05, -0.81369E-05, -0.12776E-04, -0.27563E-04,
                          -0.63135E-04, -0.11462E-03, -0.20746E-03, -0.32777E-03, -0.51440E-03, -0.10905E-05,
                          -0.13241E-05, -0.20262E-05, -0.48223E-05, -0.94826E-05, -0.24330E-04, -0.59949E-04,
                          -0.11150E-03, -0.20443E-03, -0.32466E-03, -0.51167E-03])

        ruy_34_54 = array([-0.13241E-05, -0.20262E-05, -0.48223E-05, -0.94826E-05, -0.24330E-04, -0.59949E-04,
                           -0.11150E-03, -0.20443E-03, -0.32466E-03, -0.51167E-03, -0.44216E-05, -0.46542E-05,
                           -0.53515E-05, -0.81369E-05, -0.12776E-04, -0.27563E-04, -0.63135E-04, -0.11462E-03,
                           -0.20746E-03, -0.32777E-03, -0.51440E-03])

        uxa = [o.Uout['ux'][n][-1] for n in range(22)]
        uxb = [o.Uout['ux'][n][-1] for n in range(34,55)]
        uya = [o.Uout['uy'][n][-1] for n in range(22)]
        uyb = [o.Uout['uy'][n][-1] for n in range(34,55)]
        CompareArrays(rux_0_21,  uxa, tol=1.0e-8, table=True, namea='ux:Reddy', nameb='ux:PyFEM', stride=0)
        CompareArrays(rux_34_54, uxb, tol=1.0e-8, table=True, namea='ux:Reddy', nameb='ux:PyFEM', stride=34)
        CompareArrays(ruy_0_21,  uya, tol=1.0e-8, table=True, namea='uy:Reddy', nameb='uy:PyFEM', stride=0)
        CompareArrays(ruy_34_54, uyb, tol=1.0e-8, table=True, namea='uy:Reddy', nameb='uy:PyFEM', stride=34)


        roms = array([0.11229E+07, 0.11133E+07, 0.11039E+07, 0.10958E+07, 0.10919E+07, 0.94200E+06,
                      0.93590E+06, 0.87998E+06, 0.87925E+06, 0.72894E+06])
        roms.sort()

        s.solve_eigen(dyn=True, nevals=10, high=True)
        #print('lambdas = ', s.L)
        #print('omegas  = ', sqrt(s.L))
        CompareArrays(roms, sqrt(s.L), tol=46., table=True, namea='eigenv:Reddy', nameb='eigenv:PyFEM', stride=34)


# run tests
prob = int(sys.argv[1]) if len(sys.argv)>1 else -1
if prob < 0:
    for p in range(1,25):
        print()
        print('[1;33m####################################### %d #######################################[0m'%p)
        print()
        runtest(p)
else: runtest(prob)
