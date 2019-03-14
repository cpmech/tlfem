# Copyright 2012 Dorival de Moraes Pedroso. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

from __future__ import print_function # for Python 3

from pylab         import quiver
from tlfem.mesh    import *
from tlfem.solver  import *
from tlfem.fig     import *
from tlfem.genmesh import Gen1Dmesh
from tlfem.util    import PrintDiff, CompareArrays

def gen_sym_frame():
    nx, ny = 13,  2
    dx, dy = 0.3, 0.2
    V,  C  = [], []
    for i in range(nx):
        for j in range(ny):
            x, y = -dx*(nx-1)/2. + i*dx, j*dy
            V.append([len(V), 0, x, y])
        k = len(V)-1
        if i > 0 and i < 7:
            C.append([len(C), -1, [k-3,k-1]])
            C.append([len(C), -1, [k-2,k-1]])
            C.append([len(C), -1, [k-2,k  ]])
            C.append([len(C), -1, [k-1,k  ]])
        elif i > 6:
            C.append([len(C), -1, [k-3,k-1]])
            C.append([len(C), -1, [k-3,k  ]])
            C.append([len(C), -1, [k-2,k  ]])
            C.append([len(C), -1, [k-1,k  ]])
        else:
            C.append([len(C), -1, [0,1]])
    m = Mesh(V, C)
    m.tag_verts(-101, [0,1,m.nv-2,m.nv-1])
    return m

def runtest(prob):
    wid = 1.0   # m
    hei = 0.3   # m
    rho = 2.5   # g/m3
    E   = 1000. # kN/m2
    A   = 0.3   # m2
    I   = wid*(hei**3.0)/12.0 # m4

    if prob==1:
        V = [[0, -101, 0.0, 0.0],
             [1, -102, 1.0, 0.0]]
        C = [[0, -1, [0,1]]]
        m = Mesh(V, C)
        p = {-1:{'type':'EelasticBeam', 'E':E, 'A':A, 'I':I, 'qnqt':(-1.,-1.,0), 'nop':11}}
        s = Solver(m, p)
        vb = {-101:{'ux':0., 'uy':0.}, -102:{'uy':0.}}
        s.set_bcs(vb=vb)
        o = s.solve_steady(True, True)
        o.beam_moments()
        m.show()
        PrintDiff('diff = %17.10e', o.Mout[0][-1][5], 1./8., 1.0e-15)

    if prob==2:
        V = [[0, -101, 0.0, 0.0],
             [1, -102, 1.0, 0.0]]
        C = [[0, -1, [0,1]]]
        m = Mesh(V, C)
        p = {-1:{'type':'EelasticBeam', 'E':E, 'A':A, 'I':I, 'qnqt':(-1.,-1.,0), 'nop':11}}
        s = Solver(m, p)
        vb = {-101:{'ux':0., 'uy':0., 'rz':0.}}
        s.set_bcs(vb=vb)
        o = s.solve_steady(True, True)
        o.beam_moments()
        m.show()
        PrintDiff('diff = %17.10e', o.Mout[0][-1][0], -1./2., 1.0e-15)

    if prob==3:
        L = 1.0
        V = [[0, -101,  0.0, 0.0],
             [1,    0, L/2., 0.0],
             [2, -102,    L, 0.0]]
        C = [[0,   -1, [0,1]],
             [1,   -1, [1,2]]]
        m = Mesh(V, C)
        p = {-1:{'type':'EelasticBeam', 'E':E, 'A':A, 'I':I, 'qnqt':(-1.,-1.,0), 'nop':11}}
        s = Solver(m, p)
        vb = {-101:{'ux':0., 'uy':0.}, -102:{'uy':0.}}
        s.set_bcs(vb=vb)
        o = s.solve_steady(True, True)
        o.beam_moments()
        m.show()
        PrintDiff('diff = %17.10e', o.Mout[0][-1][-1], 1./8., 1.0e-15)
        PrintDiff('diff = %17.10e', o.Mout[1][-1][0],  1./8., 1.0e-15)

    if prob==4:
        L = 1.0
        V = [[0, -101,  0.0, 0.0],
             [1,    0, L/2., 0.0],
             [2, -102,    L, 0.0]]
        C = [[0,   -1, [0,1]],
             [1,   -1, [1,2]]]
        m = Mesh(V, C)
        p = {-1:{'type':'EelasticBeam', 'E':E, 'A':A, 'I':I, 'qnqt':(-1.,-1.,0), 'nop':11}}
        s = Solver(m, p)
        vb = {-101:{'ux':0., 'uy':0., 'rz':0.}}
        s.set_bcs(vb=vb)
        o = s.solve_steady(True, True)
        o.beam_moments()
        m.show()
        PrintDiff('diff = %17.10e', o.Mout[0][-1][0], -1./2., 1.0e-15)
        PrintDiff('diff = %17.10e', o.Mout[1][-1][0], -1./8., 1.0e-15)

    if prob==5:
        V = [[0, -100,  0.0, 2.5],
             [1, -100,  2.5, 0.0],
             [2, -100, 10.0, 5.0],
             [3,    0,  7.5, 2.5],
             [4,    0,  2.5, 2.5],
             [5, -200,  5.0, 2.5]]
        C = [[0,   -3, [0,4]],
             [1,   -1, [4,1]],
             [2,   -1, [3,2]],
             [3,   -2, [5,3]],
             [4,   -2, [4,5]]]
        m = Mesh(V, C)
        #m.draw(); m.show()
        p = {-1:{'type':'EelasticBeam', 'E':E, 'A':A, 'I':I, 'nop':11},
             -2:{'type':'EelasticBeam', 'E':E, 'A':A, 'I':I, 'nop':11, 'qnqt':(-9.2,-9.2,0)},
             -3:{'type':'EelasticBeam', 'E':E, 'A':A, 'I':I, 'nop':11}}
        s = Solver(m, p)
        vb = {-100:{'ux':0., 'uy':0.}}
        s.set_bcs(vb=vb)
        o = s.solve_steady(True, True)
        o.beam_moments()
        Ma = [0.0000, -9.3832, 9.3724, 0.0000, -10.8887, 0.0000, 13.9279, -10.8887, -18.7555, 13.9279]
        Mb = [o.Mout[i//2][-1][-(i%2)] for i in range(2*5)]
        CompareArrays(Ma, Mb, 1.0e-4, table=True, namea='reference', nameb='PyFEM')
        m.show()

    if prob==6:
        V = [[0, -100,  4.0,  0.0],
             [1, -400,  4.0, 10.0],
             [2, -200, 12.0, 16.0],
             [3, -300,  0.0, 10.0]]
        C = [[0,   -1, [0,1]],
             [1,   -2, [1,2]],
             [2,   -3, [3,1]]]
        m = Mesh(V, C)
        #m.draw(); m.show()
        E = 518400.0 # kip/ft2
        A = 1.0      # ft2
        I = 1.0/12.0 # ft4

        for load_case in range(1,5):

            # 1) global uniform distributed load on frame element 2, plus concentrated load on node 3
            if load_case==1:
                p = {-1:{'type':'EelasticBeam', 'E':E, 'A':A, 'I':I},
                     -2:{'type':'EelasticBeam', 'E':E, 'A':A, 'I':I},
                     -3:{'type':'EelasticBeam', 'E':E, 'A':A, 'I':I, 'qnqt':(-1.8,-1.8,0)}}
                s = Solver(m, p)
                vb = {-100:{'ux':0., 'uy':0.}, -200:{'ux':0.}, -300:{'fy':-10.}}
                Ma = [0., 34., -20.4, 0., 0., -54.4]

            # 2) global joint force and moment at node 1
            if load_case==2:
                p = {-1:{'type':'EelasticBeam', 'E':E, 'A':A, 'I':I},
                     -2:{'type':'EelasticBeam', 'E':E, 'A':A, 'I':I},
                     -3:{'type':'EelasticBeam', 'E':E, 'A':A, 'I':I}}
                s = Solver(m, p)
                vb = {-100:{'ux':0., 'uy':0.}, -200:{'ux':0.}, -400:{'fy':-17.2, 'mz':54.4}}
                Ma = [0., 34., -20.4, 0., 0., 0.]

            # 3) uniformly distributed load on elements 0 and 1
            if load_case==3:
                p = {-1:{'type':'EelasticBeam', 'E':E, 'A':A, 'I':I, 'qnqt':(-2.,-2.,0.)},
                     -2:{'type':'EelasticBeam', 'E':E, 'A':A, 'I':I, 'qnqt':(-2.,-2.,0.)},
                     -3:{'type':'EelasticBeam', 'E':E, 'A':A, 'I':I}}
                s = Solver(m, p)
                vb = {-100:{'ux':0., 'uy':0.}, -200:{'ux':0.}}
                Ma = [0., 20., 20, 0., 0., 0.]

            # 4) trapezoidal load on elements 0 and 1
            if load_case==4:
                p = {-1:{'type':'EelasticBeam', 'E':E, 'A':A, 'I':I, 'qnqt':(-0.9984, -0.3744, 0.)},
                     -2:{'type':'EelasticBeam', 'E':E, 'A':A, 'I':I, 'qnqt':(-0.3744, 0.0, 0.)},
                     -3:{'type':'EelasticBeam', 'E':E, 'A':A, 'I':I}}
                s = Solver(m, p)
                vb = {-100:{'ux':0., 'uy':0.}, -200:{'ux':0.}}
                def e0M(x): return 5.0752*x - 0.9984*x*x/3.0 - (16.0-x)*0.0624*x*x/6.0
                def e1M(x): return 1.7472*(10.-x) - 0.6*0.0624*((10.-x)**3.)/6.0
                Ma = [e0M(0.), e0M(10.), e1M(0.), e1M(10.), 0., 0.]

            s.set_bcs(vb=vb)
            o = s.solve_steady(True, True)
            o.beam_moments(txt=1)
            Mb = [o.Mout[i//2][-1][-(i%2)] for i in range(2*m.nc)]
            CompareArrays(Ma, Mb, 1.0e-4, table=True, namea='reference', nameb='PyFEM')
            m.show()

    if prob==7: # SG p470 fig 11.1
        V = [[0, -101, 0.0, 0.0],
             [1, -102, 1.0, 0.0]]
        C = [[0, -1, [0,1]]]
        m = Mesh(V, C)
        p = {-1:{'type':'EelasticBeam', 'rho':1., 'E':3.194, 'A':1., 'I':1.}}
        s = Solver(m, p)
        vb = {-101:{'ux':0., 'uy':0., 'rz':0.}, -102:{'fy':lambda t: 3.194*sin(pi*t) if t<1. else 0.}}
        s.set_bcs(vb=vb)
        t0, tf, dt     = 0., 5., 0.05
        U0, V0         = zeros(s.neqs), zeros(s.neqs)
        theta1, theta2 = 0.5, 0.5
        o = s.solve_dynamics(t0, tf, dt, theta1=theta1, theta2=theta2)

        tsw = 1.0
        def us(t):
            if (t<tsw): return  0.441*sin(pi*t/tsw)-0.216*sin(2.0*pi*t/tsw);
            else:       return -0.432*sin(6.284*(t-tsw))
        u = vectorize(us)

        t = linspace(0.,5.,101)
        plot(t,u(t),'g-',label='solution')
        plot(o.Tout, o.Uout['uy'][1], 'ro',label='fem @ 1')
        Gll('t','uy')
        show()

    if prob==8: # CK: Fundamentals of structural dynamics, Craig, Roy R; Kurdila, Andrew J
                # 14. Intro FE modelling of structures. Example 14.10 p456
        V = [[0, -101, 0.0, 0.0],
             [1, -102, 1.0, 0.0]]
        C = [[0, -1, [0,1]]]
        m = Mesh(V, C)
        #m.draw(); m.show()
        p = {-1:{'type':'EelasticBeam', 'E':1., 'A':1., 'I':1., 'rho':1.}}
        s = Solver(m, p)
        vb = {-101:{'ux':0., 'uy':0., 'rz':0.}}
        s.set_bcs(vb=vb)
        s.solve_eigen()
        lmax, lmaxCK = s.L[0], 1211.52
        PrintDiff('diff(lmax,lmaxCK) = %17.10e', lmax, lmaxCK, 1.0e-3)

    if prob==9: # CK: Fundamentals of structural dynamics, Craig, Roy R; Kurdila, Andrew J
                # 15.6 Numerical Case Study p496
        m = gen_sym_frame()
        p = {-1:{'type':'EelasticBeam', 'E':69.0e+9, 'A':21.2e-4, 'I':349.0e-8, 'rho':2.7e+3 }} # Pa, m2, m4, kg/m3
        s = Solver(m, p)
        vb = {-101:{'ux':0., 'uy':0.}}
        s.set_bcs(vb=vb)
        s.solve_eigen(nevals=8, evecs=True)
        print('lambdas =', s.L)
        print('omegas  =', sqrt(s.L))
        if 0:
            #res = s.get_mode(0)
            #m.draw()
            #X = [v[2] for v in m.V]
            #Y = [v[3] for v in m.V]
            #quiver(X, Y, res['ux'], res['uy'],zorder=20)
            d = read_table('/home/dorival/23.figs/craigkurdila-fig15.3.dat')
            plot(d['x'],d['y'],lw=2,zorder=30)
            m.show()

    if prob==10: # CK: Fundamentals of structural dynamics, Craig, Roy R; Kurdila, Andrew J
                 # 16.5 Numerical Case Study p525
        m = gen_sym_frame()
        m.tag_verts(-102, [12])
        #m.draw(); m.show()
        p = {-1:{'type':'EelasticBeam', 'E':69.0e+9, 'A':21.2e-4, 'I':349.0e-8, 'rho':2.7e+3 }} # Pa, m2, m4, kg/m3
        s = Solver(m, p)
        t0, tf = 0.0, 0.001
        Dt     = tf - t0
        Om     = 2.*pi*20./Dt
        vb = {-101:{'ux':0., 'uy':0.}, -102:{'fy':lambda t: sin(Om*t) if t<Dt/20. else 0.}}
        s.set_bcs(vb=vb)
        U0, V0 = zeros(s.neqs), zeros(s.neqs)
        dt = 1.0e-6
        theta1, theta2 = 0.5, 0.5
        #o = s.solve_dynamics(t0, tf, dt, theta1=theta1, theta2=theta2)
        #plot(o.Tout, o.Uout['uy'][12])
        #show()

    if prob==11: # Inman: q8.45 p609

        for mesh in range(3):

            if mesh==0:
                V = [[0, -101, 0.0, 0.0],
                     [1,    0, 2.0, 1.0],
                     [2, -101, 0.0, 2.0]]
                C = [[0, -1, [0,1]],
                     [1, -1, [1,2]]]

                om_inman = [377.5, 8763.7, 10951.2]
                dense    = True

            if mesh==1:
                V = [[0, -101, 0.0, 0.0],
                     [1,    0, 1.0, 0.5],
                     [2,    0, 2.0, 1.0],
                     [3,    0, 1.0, 1.5],
                     [4, -101, 0.0, 2.0]]
                C = [[0, -1, [0,1]],
                     [1, -1, [1,2]],
                     [2, -1, [2,3]],
                     [3, -1, [3,4]]]

                om_inman = [286.8, 419.1, 1074.5, 1510.8, 2838.9]
                dense    = False

            if mesh==2:
                V = [[0, -101, 0.0, 0.0 ],
                     [1,    0, 0.5, 0.25],
                     [2,    0, 1.0, 0.5 ],
                     [3,    0, 1.5, 0.75],
                     [4,    0, 2.0, 1.0 ],
                     [5,    0, 1.5, 1.25],
                     [6,    0, 1.0, 1.5 ],
                     [7,    0, 0.5, 1.75],
                     [8, -101, 0.0, 2.0 ]]
                C = [[0, -1, [0,1]],
                     [1, -1, [1,2]],
                     [2, -1, [2,3]],
                     [3, -1, [3,4]],
                     [4, -1, [4,5]],
                     [5, -1, [5,6]],
                     [6, -1, [6,7]],
                     [7, -1, [7,8]]]

                om_inman = [284.3, 413.0 ,925.6, 1147.3, 1959.7]
                dense    = False

            m = Mesh(V, C)
            #m.draw(); m.show()
            p = {-1:{'type':'EelasticBeam', 'E':69.e10, 'A':0.0004, 'I':1.33e-8, 'rho':2700.}}
            s = Solver(m, p)
            vb = {-101:{'ux':0., 'uy':0., 'rz':0.}}
            s.set_bcs(vb=vb)
            s.solve_eigen(nevals=len(om_inman), dense=dense)
            CompareArrays(om_inman, sqrt(s.L), 0.1, table=True, namea='Inman', nameb='PyFEM')

    if prob==12: # Nageim et al p307. Structural Mechanics
        Lx  = 5.       # [m]
        E   = 205000.0 # [MPa]
        A   = 6.9e-3   # [m2]
        Izz = 1.872e-4 # [m4]
        qn  = 0.06     # [MN/m]
        np  = 3
        tmp = Gen1Dmesh(5., 3)
        for k in range(tmp.nv): tmp.V[k].append(0.)
        m   = Mesh(tmp.V, tmp.C)
        #m.draw(); m.show()
        p = {-1:{'type':'EelasticBeam', 'E':E, 'A':A, 'I':Izz, 'qnqt':(-qn,-qn,0.)}}
        s = Solver(m, p)
        vb = {-101:{'ux':0., 'uy':0.}, -102:{'uy':0.}}
        s.set_bcs(vb=vb)
        o = s.solve_steady(True, True)
        Mmax  = qn*(Lx**2.)/8.
        W     = qn*Lx
        uymin = -(5./384.) * W * (Lx**3.) / (E*Izz)
        Uy    = [o.Uout['uy'][iv][-1] for iv in range(m.nv)]
        uynum = min(Uy)
        Mnum  = [max(o.Mout[ic][-1]) for ic in range(m.nc)]
        Mnummax = max(Mnum)
        print('Mmax    = ', Mmax,  '  numerical =', Mnummax)
        print('min(uy) = ', uymin, '  numerical =', uynum)
        PrintDiff('diff(Mmax,Mnum)   = %17.10e', Mmax,  Mnummax, 1.0e-14)
        PrintDiff('diff(uysol,uynum) = %17.10e', uymin, uynum,   1.0e-14)
        o.beam_moments()
        m.show()


# run tests
prob = int(sys.argv[1]) if len(sys.argv)>1 else -1
if prob < 0:
    for p in range(1,13):
        print()
        print('[1;33m####################################### %d #######################################[0m'%p)
        print()
        runtest(p)
else: runtest(prob)
