# Copyright 2012 Dorival de Moraes Pedroso. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

#from matplotlib    import collections as collections
from numpy import transpose, vstack, zeros, ones, arange  # some functions from numpy
from numpy import cos, sin, pi, sqrt, vectorize, meshgrid # more functions from numpy
from numpy import linspace, sort, array, cross, log10     # more functions from numpy
from numpy import sum as npsum                            # import sum function
from tlfem.mesh import Mesh, Edg2Vids                     # import fem mesh and Edg2Vids
from tlfem.fig  import column_nodes                       # calc nodes in a vertical column
import json                                               # to read json files (.msh)

# additional imports
from scipy.spatial import Delaunay # for Delaunay triangulation

def Gen1Dmesh(Lx, nx=None, dx=None, vtag=-100, ctag=-1):
    """
    Generate 1D Mesh/Grid
    =====================
    INPUT:
        Lx   : length of bar
        nx   : number of points/nodes [optional]
        dx   : increments [optional]
        vtag : default tag for all vertices
        ctag : default tag for all cells
    RETURNS:
        a Mesh object
    ----
    Notes:
       1) One or another among nx or dx must be specified
       2) By default, the first vertex is tagged with -101
          and the last one with -102
    """
    if dx==None:
        dx = Lx/float(nx-1)
    else:
        nx = int(Lx/dx)
        dx = Lx/float(nx-1)
    V = [[i, vtag,  i * dx ] for i in range(nx)]
    C = [[i, ctag, [i, i+1]] for i in range(nx-1)]
    V[0   ][1] = -101
    V[nx-1][1] = -102
    return Mesh(V, C)

def Gen1Dlayered(H, N=11):
    """
    Generate 1D layred mesh
    =======================
    INPUT:
        H : a list with the thickness of each layer
        N : the number of division within each layer.
            Can be a list with the number of divisions of each layer
    RETURNS:
        an instance of Mesh
    """
    nlay = len(H)          # number of layers
    if isinstance(N, int):
        N = N * ones(nlay) # same division for all layers
    V = [[0, -101, 0.]]    # first vertex
    C = []                 # cells
    for i in range(nlay):
        x0 = sum(H[:i])             # left
        x1 = x0 + H[i]              # right
        L  = linspace(x0, x1, N[i]) # divide layer
        for l in L[1:]:
            V.append([len(V), 0, l])
            C.append([len(C), -(i+1), [len(V)-2,len(V)-1]])
    V[-1][1] = -102 # tag last vertex
    return Mesh(V, C)

def NlSpace(xmin, xmax, nx, n=2.0, rev=False):
    """
    Non-linear space
    ================
    INPUT:
        xmin : min x-coordinate
        xmax : max x-coordinate
        nx   : number of points
        n    : power of nonlinear term
        rev  : reversed?
    RETURNS:
        a non-linear sequence
    """
    fcn = vectorize(lambda i: xmin + (xmax-xmin)*(float(i)/float(nx-1))**n)
    return fcn(arange(nx))

def Gen2Dregion(X, Y, ctag=-1, triangle=True, rotated=False, cross=False, vtags=False,
                etags=[-10, -11, -12, -13]):
    """
    Generate 2D region
    ==================
    INPUT:
        X        : the X coordinates. ex: linspace(0.,4.,5)
        Y        : the Y coordinates. ex: linspace(0.,3.,4)
        ctag     : cell tag to be attached to all cells
        triangle : generate triangles instead of rectangles
        rotated  : rotate triangles
        cross    : rotate every second cell => cross-like shape
        vtags    : tag boundary vertices
        etags    : edge tags for [bottom, right, top, left]
    RETURNS:
        a Mesh object
    """
    nx, ny = len(X), len(Y)
    if nx < 2: raise Exception('minimum number of columns (nx) is 2')
    if ny < 2: raise Exception('minimum number of rows (ny) is 2')
    V  = []
    for j in range(ny):
        for i in range(nx):
            n = i + j*nx
            tag = 0
            if vtags:
                if j==0:    tag = -100
                if i==nx-1: tag = -101
                if j==ny-1: tag = -102
                if i==0:    tag = -103
            V.append([n, tag, X[i], Y[j]])
    C = []
    if triangle:
        if cross: rotated = True
        for je in range(ny-1):
            for ie in range(nx-1):
                btag0 = {}
                btag1 = {}
                if rotated:
                    vert0 = [ie+je*nx, ie+je*nx+1,     ie+(je+1)*nx+1]
                    vert1 = [ie+je*nx, ie+(je+1)*nx+1, ie+(je+1)*nx  ]
                    if je == 0:    btag0[0] = etags[0]
                    if ie == nx-2: btag0[1] = etags[1]
                    if je == ny-2: btag1[1] = etags[2]
                    if ie == 0:    btag1[2] = etags[3]
                else:
                    vert0 = [ie+je*nx,     ie+je*nx+1, ie+(je+1)*nx  ]
                    vert1 = [ie+(je+1)*nx, ie+je*nx+1, ie+(je+1)*nx+1]
                    if je == 0:    btag0[0] = etags[0]
                    if je == ny-2: btag1[2] = etags[2]
                    if ie == nx-2: btag1[1] = etags[1]
                    if ie == 0:    btag0[2] = etags[3]
                C.append([len(C), ctag, vert0, btag0])
                C.append([len(C), ctag, vert1, btag1])
                if cross:
                    rotated = not rotated
                    if ie == nx-2:
                        if je%2 == 0: rotated = False
                        else:         rotated = True
    else:
        for je in range(ny-1):
            for ie in range(nx-1):
                verts = [ie+je*nx, ie+je*nx+1, ie+(je+1)*nx+1, ie+(je+1)*nx]
                btags = {}
                if je == 0:    btags[0] = etags[0]
                if je == ny-2: btags[2] = etags[2]
                if ie == nx-2: btags[1] = etags[1]
                if ie == 0:    btags[3] = etags[3]
                C.append([len(C), ctag, verts, btags])
    return Mesh(V, C)

def GenColumn(nc, w, h):
    """
    Generate column
    ===============
    """
    dy = h / float(nc)
    l, c, r, L, R = column_nodes(nc, True)
    nn = R[-2] + 1
    V  = []
    for n in range(nn):
        V.append([n, 0, 0.0, 0.0])
    y = 0.0
    for i in range(len(L)):
        V[L[i]][3], V[R[i]][3], V[R[i]][2] = y, y, w
        y += dy/2.0
    y = 0.0
    for n in c:
        V[n][3], V[n][2] = y, w/2.0
        y += dy
    C = []
    for i in range(nc):
        con = [l[i],r[i],r[i+1],l[i+1],c[i],R[i*2+1],c[i+1],L[i*2+1]]
        btg = {1:-11, 3:-13}
        if i == 0:    btg = {0:-10, 1:-11, 3:-13}
        if i == nc-1: btg = {1:-11, 2:-12, 3:-13}
        C.append([i, -1, con, btg, 0])
    return Mesh(V, C)

def JoinAlongEdge(a, b, eta, etb, tol=0.001):
    """
    Join two meshes along common edge
    =================================
    INPUT:
        a   : first Mesh object
        b   : second Mesh object
        eta : tag of edges in a to be merged
        etb : tag of edges in b to be merged
    """
    if a.ndim != 2: raise Exception('this method is for 2D meshes only')
    if b.ndim != 2: raise Exception('this method is for 2D meshes only')

    # collect vertices along edge of a
    Va = set()
    for ca in a.etag2cids[eta]:
        ia, ed = ca[0], ca[1]
        la = Edg2Vids[(a.ndim, len(a.C[ia][2]))][ed]
        Va.update([a.C[ia][2][iv] for iv in la])

    # new vertices and cells list
    V, C = a.V[:], a.C[:]

    # remap vertices of b
    v2v = zeros(b.nv, dtype=int)                              # map verts in b to new verts in a
    k   = a.nv                                                # new vertex index
    for vb in b.V:                                            # for each vertex in b
        found = -1                                            # found vb in Va (-1 => not found)
        for ia in Va:                                         # for each vertex in Va
            va = a.V[ia]                                      # vertex of a
            dd = sqrt((vb[2]-va[2])**2.0+(vb[3]-va[3])**2.0)  # distance between vertices
            if dd < tol:                                      # overlapping vertex
                found = ia                                    # found coincident vertex in a
                break                                         # no need to check other verts in Va
        if found < 0:                                         # not found
            iv = k                                            # index of vertex to be added
            V.append([iv, vb[1], vb[2], vb[3]])               # add new vertex
            k += 1                                            # next new vertex index
        else:                                                 # use vertex of a
            iv = found                                        # index of vertex to be added
        v2v[vb[0]] = iv                                       # set map

    # remap cells of b
    k = a.nc                                                  # new cell index
    for c in b.C:                                             # for each cell in b
        if len(c) < 4:                                        # if there is no edge tags data
            etags = {}                                        # empty etags dictionary
        else:                                                 # there is etags data
            etags = {k:v for k,v in c[3].items() if v!=etb}   # filter etb out
        C.append([k, c[1], [v2v[i] for i in c[2]], etags])    # add new cell
        k += 1                                                # next cell number

    # create and return mesh
    return Mesh(V, C)

def GenQplateHole(Lx, Ly, r, nx=5, ny=5, nc=5, tag_hole=True, uniform=True, nit=10):
    """
    Generate one quarter of a plate with hole
    =========================================
        y
        ______Lx______
        |             |
        |__           |
           '-.        Ly
              \       |
        r ---> |______| __ x
    INPUT:
        Lx       : x length
        Ly       : y length
        r        : radius
        nx       : num divisions along x
        ny       : num divisions along y
        nc       : num divisions along arc
        tag_hole : tag also points on the surface of hole
        uniform  : uniform weights?
        nit      : number of iterations for ps_mesh
                   does not run ps_mesh if <= 0
    RETURNS:
        a Mesh object
    """
    # hole function
    hole_fcn = lambda p: sqrt(npsum(p**2.,axis=1)) - r

    # generate (fixed) points on boundary
    if nx < 3: raise Exception('nx must be at least equal to 3')
    if ny < 3: raise Exception('ny must be at least equal to 3')
    th   = linspace(0., pi/2., nc)
    dxb  = Lx / float(nx-1)
    dyb  = Ly / float(ny-1)
    xbo  = linspace(0.0,     Lx,     nx  )
    xto  = linspace(0.0+dxb, Lx-dxb, nx-2)
    ylr  = linspace(0.0+dyb, Ly,     ny-1)
    xbo  = xbo[xbo > r]
    yle  = ylr[ylr > r]
    pfix = vstack([transpose(vstack([ r*cos(th), r*sin(th)   ])),
                   transpose(vstack([ xbo,   zeros(len(xbo)) ])),
                   transpose(vstack([ xto, Ly*ones(len(xto)) ])),
                   transpose(vstack([ Lx*ones(len(ylr)), ylr ])),
                   transpose(vstack([   zeros(len(yle)), yle ]))])

    # plot fixed points
    if 0:
        plot(pfix[:,0],pfix[:,1],'ro',clip_on=0,markersize=10)
        axis('equal'); show()
        check_overlap(pfix)

    # domain function
    h0      = 0.8 * sqrt(dxb**2.+dyb**2.)
    dd      = 0.4 * h0
    inside_ = lambda x,y: -1. if sqrt(x**2.+y**2.)>r+dd and \
                              x>dd and x<Lx-dd and y>dd and y<Ly-dd else +1.
    inside  = vectorize(inside_)

    # generate initial points on equilateral triangles
    p = vstack([pfix, equi_tri_points(Lx, Ly, h0, inside)])

    # plot all initial points
    if 0:
        Arc(0.,0.,r, 0.,pi/2.)
        plot(p[:,0],p[:,1],'yo',clip_on=0,markersize=10)
        check_overlap(p)
        #axis('equal'); show()

    # run ps_mesh
    if nit > 0:
        p = ps_mesh(pfix, p, inside, hole_fcn, nit, uniform)

    # mesh
    V, C = tri_get_vc(p, Lx, Ly, hole_fcn, tag_hole=tag_hole)
    return Mesh(V, C)

def GenQdisk(r, nr=5, nc=7, imethod=1, nit=20):
    """
    Generate one quarter of disk with hole
    ======================================
        y
        __
        | `-._           .
        |     `-.        .
        |        \       .
        |         `.     .
        +--- r --->| --> x
    INPUT:
        r       : radius
        nr      : num divisions along r
        nc      : num divisions along arc
        imethod : method for initial points
        nit     : number of iterations for ps_mesh
                  does not run ps_mesh if <= 0
    RETURNS:
        a Mesh object
    """
    # generate (fixed) points on boundary
    if nr < 3: raise Exception('nr must be at least equal to 3')
    if nc < 3: raise Exception('nc must be at least equal to 3')
    th   = linspace(0., pi/2., nc)
    dr   = r / float(nr-1)
    xbo  = linspace(0.0,    r-dr, nr-1)
    yle  = linspace(0.0+dr, r-dr, nr-2)
    pfix = vstack([transpose(vstack([ r*cos(th), r*sin(th)   ])),
                   transpose(vstack([ xbo,   zeros(len(xbo)) ])),
                   transpose(vstack([ zeros(len(yle)), yle   ]))])

    # plot fixed points
    if 0:
        plot(pfix[:,0],pfix[:,1],'ro',clip_on=0,markersize=10)
        axis('equal'); show()
        check_overlap(pfix)

    # domain function
    h0      = 0.8 * dr
    dd      = 0.4 * h0
    inside_ = lambda x,y: -1. if sqrt(x**2.+y**2.)<r-dd and x>dd and y>dd else +1.
    inside  = vectorize(inside_)

    # generate initial points on equilateral triangles
    if imethod == 1: p = vstack([pfix, equi_tri_points(r, r, h0, inside)])
    else:
        dth = pi/2. / float(nc-1)
        th = linspace(0.+dth, pi/2.-dth, nc-2)
        for i in range(1, nr-1):
            if i==1: p = transpose(vstack([i*dr*cos(th), i*dr*sin(th)]))
            else:    p = vstack([p, transpose(vstack([i*dr*cos(th), i*dr*sin(th)]))])
        p = vstack([pfix, p])

    # plot all initial points
    if 0:
        Arc(0.,0.,r, 0.,pi/2.)
        plot(p[:,0],p[:,1],'yo',clip_on=0,markersize=10)
        #axis('equal'); show()
        check_overlap(p)

    # run ps_mesh
    if nit > 0:
        p = ps_mesh(pfix, p, inside, hole_fcn=None, nit=nit, uniform=True)

    # mesh
    V, C = tri_get_vc(p, 2.*r, 2.*r, hole_fcn=None, tag_hole=False)
    return Mesh(V, C)

def ReadJson(fnkey):
    """
    Reads a .msh file in json format
    ================================
    INPUT:
        fnkey : filename key. ex: "bh-1.6"
    OUTPUT:
        a Mesh object
    --
    Example of file: <bh-1.6.msh>
        {
            "verts" : [
                {"id":0, "tag":-100, "c":[0.0, 0.0]},
                {"id":1, "tag":-100, "c":[0.0, 2.0]},
                {"id":2, "tag":   0, "c":[2.0, 0.0]},
                {"id":3, "tag":   0, "c":[2.0, 1.5]},
                {"id":4, "tag":   0, "c":[4.0, 0.0]},
                {"id":5, "tag":   0, "c":[4.0, 1.0]}
            ],
            "cells" : [
                {"id":0, "tag":-1, "geo":3, "part":0, "verts":[0,2,3], "ftags":[  0,0,0]},
                {"id":1, "tag":-1, "geo":3, "part":0, "verts":[3,1,0], "ftags":[-10,0,0]},
                {"id":2, "tag":-1, "geo":3, "part":0, "verts":[2,4,5], "ftags":[  0,0,0]},
                {"id":3, "tag":-1, "geo":3, "part":0, "verts":[5,3,2], "ftags":[-10,0,0]}
            ]
        }
    """
    f = open(fnkey+'.msh', 'r')
    r = json.load(f)
    V = []
    for v in r['verts']:
        if len(v['c']) != 2: raise Exception('ReadJson: this function works for 2D meshes only')
        V.append([v['id'], v['tag'], v['c'][0], v['c'][1]])
    C = []
    for c in r['cells']:
        if 'ftags' in c: ftags = {idx:tag for idx, tag in enumerate(c['ftags'])}
        else:            ftags = {}
        if 'part'  in c: part  = c['part']
        else:            part  = 0
        C.append([c['id'], c['tag'], c['verts'], ftags, part])
    f.close()
    m = Mesh(V, C)
    if 'patches' in r:
        m.p_vids, m.p_cids = {}, {}
        for p in r['patches']:
            key           = (p['vert'], p['ctag'], p['nvc'])
            m.p_vids[key] = set(p['vids'])
            m.p_cids[key] = set(p['cids'])
    return m

def Qua8SaveExtruded(m, fnk, thick, yToZ=False, with_patch=False):
    """
    Extrudes a 2D mesh
    ==================
      NOTE: this works with qua8 only, for now
    """
    n = len(m.V)
    # patches data
    if with_patch:
        m.find_patch()
        new_p_vids  = {}
        vid2patches = {}
        for key, val in m.p_vids.items():
            new_p_vids[key] = val
            vids = []
            for vid in val:
                vids.append(vid)
                vids.append(n+vid)
                if vid in vid2patches: vid2patches[vid].append(key)
                else:                  vid2patches[vid] = [key]
            new_p_vids[key] = vids

    # mesh data
    V = []
    for v in m.V:
        V.append([v[0], v[1], v[2], v[3], 0.0])
    for v in m.V:
        V.append([n+v[0], v[1], v[2], v[3], thick])
    o2n_back  = [0, 3, 7, 4, 11, 19, 15, 16] # old=>new
    o2n_front = [1, 2, 6, 5,  9, 18, 13, 17] # old=>new
    ed2face   = [4, 3, 5, 2]
    newed2nod = [8, 10, 14, 12]
    newedges  = {}
    C         = []
    minftag   = 0
    for c in m.C:
        vv = -ones(20, dtype=int)
        for k, nid in enumerate(c[2]):
            vv[o2n_back [k]] = V[nid  ][0]
            vv[o2n_front[k]] = V[nid+n][0]
        edges = [tuple(sorted([vv[0], vv[1]])),
                 tuple(sorted([vv[3], vv[2]])),
                 tuple(sorted([vv[7], vv[6]])),
                 tuple(sorted([vv[4], vv[5]]))]
        for k, ed in enumerate(edges):
            if ed in newedges:
                nid = newedges[ed]
            else:
                nid = len(V)
                n0  = ed[0]
                V.append([nid, 0, V[n0][2], V[n0][3], thick/2.0])
                newedges[ed] = nid
                # patches data
                if with_patch:
                    for pkey in vid2patches[n0]:
                        new_p_vids[pkey].append(nid)
            vv[newed2nod[k]] = nid
        ftags = {}
        if len(c) > 3:
            for fidx, ftag in c[3].items():
                ftags[ed2face[fidx]] = ftag
                if ftag < minftag:
                    minftag = ftag
        part = 0
        if len(c) > 4:
            part = c[4]
        C.append([c[0], c[1], vv, ftags, part])
    mtag_exp  = int(log10(abs(minftag)))
    tag_back  =     10 ** (1+mtag_exp)
    tag_front = 2 * 10 ** (1+mtag_exp)
    for k in range(len(C)):
        C[k][3][0] = -tag_back
        C[k][3][1] = -tag_front

    # verts
    l  = '{\n    "verts" : [\n'
    nv = len(V)
    for i, v in enumerate(V):
        y, z = v[3],v[4]
        if yToZ: y, z = v[4],v[3]
        l += '        { "id":%3d, "tag":%d, "c":[%23.15e, %23.15e, %23.15e] ' % (v[0],v[1],v[2],y,z)
        if i==nv-1: l += '}\n'
        else:       l += '},\n'
    # cells
    l += '    ],\n    "cells" : [\n'
    nc = len(C)
    for i, c in enumerate(C):
        nv = len(c[2])
        ge = 12
        pt = 0
        if len(c) > 4: pt = c[4] # part id
        l += '        { "id":%3d, "tag":%d, "part":%d, "geo":%d, "verts":[' % (i,c[1],pt,ge)
        for j in range(nv):
            l += '%3d' % c[2][j]
            if j==nv-1: l += '], '
            else:       l += ', '
        l += '"ftags":['
        nf = 6
        for j in range(nf):
            if len(c) > 3:
                if j in c[3]: l += '%3d' % c[3][j]
                else:         l += '%3d' % 0
            else:
                l += '%3d' % 0
            if j==nf-1: l += '] '
            else:       l += ', '
        if i==nc-1: l += '}\n'
        else:       l += '},\n'
    # patches
    if with_patch:
        l += '    ],\n    "patches" : [\n'
        np = len(new_p_vids)
        for i, pkey in enumerate(new_p_vids.keys()):  # for each patch key
            N   = pkey[0]                             # master node describ. the patch
            tag = pkey[1]                             # tag of cells in patch
            nvc = pkey[2]                             # number of vertices per cell
            nv  = len(new_p_vids[pkey])               # number of vertices in patch
            nc  = len(m.  p_cids[pkey])               # number of cells in patch
            l += '        { "vert":%3d, "ctag":%3d, "nvc":20, "vids":[' % (N,tag)
            for j, iv in enumerate(new_p_vids[pkey]):
                l += '%3d' % iv
                if j==nv-1: l += '], '
                else:       l += ', '
            l += '"cids":['
            for j, ic in enumerate(m.p_cids[pkey]):
                l += '%3d' % ic
                if j==nc-1: l += '] '
                else:       l += ', '
            if i==np-1: l += '}\n'
            else:       l += '},\n'
    # close
    l += '    ]\n}\n'

    # save file
    f = open('%s.msh' % fnk, 'w')
    f.write(l)
    f.close()

# --------------------------------- auxiliary functions ------------------------------------

def check_overlap(p, tol=0.001):
    """
    Check overlapping points
    ========================
    INPUT:
        p   : list of points. ex: [[0.0, 0.0], [1.0, 0.0]]
        tol : tolerance to consider two points equal to each other
    RETURNS:
        None
    """
    for i in range(len(p)):
        x0, y0 = p[i]
        for j in range(i+1,len(p)):
            x1, y1 = p[j]
            dist   = sqrt((x1-x0)**2.0+(y1-y0)**2.0)
            if dist < tol:
                raise Exception('found overlapping points: (%g,%g), (%g,%g)'%(x0,y0,x1,y1))

def tri_get_vc(p, Lx, Ly, hole_fcn=None, tol=0.0001, tag_hole=True):
    """
    Tri3: get V and C from a set of points
    ======================================
    INPUT:
        p        : list of points. ex: [[0.0, 0.0], [1.0, 0.0]]
        Lx       : domain length (for tagging boundaries)
        Ly       : domain height (for tagging boundaries)
        hole_fcn : a function representing a hole (to remove triangles)
                   ex: lambda p: sqrt(npsum(p**2.,axis=1)) - r
                   [negative => inside hole]
        tol      : tolerance to find points on boundaries
        tag_hole : tag also points on the surface of hole, if fcn is given
    RETURNS:
        a list of unique edges
    """
    res = Delaunay(p)                                         # triangulation
    t   = res.vertices                                        # list of triangles
    if hole_fcn!=None:                                        # with hole function
        pmid = (p[t[:,0],:] + p[t[:,1],:] + p[t[:,2],:]) / 3. # compute centroids
        t    = t[hole_fcn(pmid) > 0.0]                        # remove triangles in hole
    V = [[i, 0, p[i][0], p[i][1]] for i in range(len(p))]     # set vertices
    C = []                                                    # empty list of cells
    for i0, i1, i2 in t:                                      # for each triangle
        x0, y0 = p[i0][0], p[i0][1]                           # coords of vertex 0
        x1, y1 = p[i1][0], p[i1][1]                           # coords of vertex 1
        x2, y2 = p[i2][0], p[i2][1]                           # coords of vertex 2
        e01    = array([x1-x0, y1-y0, 0.0])                   # edge 0->1
        e02    = array([x2-x0, y2-y0, 0.0])                   # edge 0->2
        cro    = cross(e01, e02)                              # cross product
        if cro[2] < 0.0: i0, i2 = i2, i0                      # check out-of-plane vector
        btags = {}
        l2g   = [i0, i1, i2]
        for l in range(3):
            ia, ib = l2g[l], l2g[(l+1)%3]
            if abs(p[ia][1]   )<tol and abs(p[ib][1]   )<tol: btags[l] = -10
            if abs(p[ia][0]-Lx)<tol and abs(p[ib][0]-Lx)<tol: btags[l] = -11
            if abs(p[ia][1]-Ly)<tol and abs(p[ib][1]-Ly)<tol: btags[l] = -12
            if abs(p[ia][0]   )<tol and abs(p[ib][0]   )<tol: btags[l] = -13
            if hole_fcn!=None and tag_hole:
                if hole_fcn(array([p[ia]]))[0]<tol and hole_fcn(array([p[ib]]))[0]<tol:
                    btags[l] = -14
        C.append([len(C), -1, [i0,i1,i2], btags])
    return V, C

def equi_tri_points(Lx, Ly, h0, isinside):
    """
    Generate points on equilateral triangles
    ========================================
    INPUT:
        Lx       : x-length
        Ly       : y-length
        h0       : triangles edge length
        isinside : a function to check if points are inside the domain:
                   lambda p: sqrt(npsum(p**2.,axis=1)) - r
                   [negative => inside]
    RETURNS:
        a list of points
    """
    dx, dy = h0, h0*sqrt(3.)/2.
    x      = arange(0.0, Lx+dx, dx)
    y      = arange(0.0, Ly+dy, dy)
    xx, yy = meshgrid(x,y)
    odd    = arange(1, len(x), 2)
    if max(odd)>len(xx)-1: odd = odd[:-1]
    xx[odd,:] += dx/2.
    p = transpose(vstack([xx.flatten(), yy.flatten()]))
    return p[isinside(p[:,0], p[:,1]) < 0.0]

def get_edges(p, hole_fcn=None):
    """
    Get edges form a set of points
    ==============================
    INPUT:
        p        : list of points. ex: [[0.0, 0.0], [1.0, 0.0]]
        hole_fcn : a function representing a hole (to remove triangles)
                   ex: lambda p: sqrt(npsum(p**2.,axis=1)) - r
                   [negative => inside hole]
    RETURNS:
        a list of unique edges
    """
    # describe each edge by a unique pair of nodes
    res = Delaunay(p)                                         # triangulation
    t   = res.vertices                                        # list of triangles
    if hole_fcn!=None:                                        # with hole function
        pmid = (p[t[:,0],:] + p[t[:,1],:] + p[t[:,2],:]) / 3. # compute centroids
        t    = t[hole_fcn(pmid) > 0.0]                        # remove triangles in hole
    edges = vstack([t[:,[0,1]], t[:,[0,2]], t[:,[1,2]]])      # interior edges duplicated
    edges = sort(edges)                                       # edges as node pairs
    edges = vstack([array(u) for u in set([tuple(b) for b in edges])])
    return edges

def ps_mesh(pfix, p, isinside, hole_fcn=None, nit=20, uniform=True):
    """
    Persson-Strang mesh generator
    =============================
        Based on the paper:
           Per-Olof Persson and Gilbert Strang, "A Simple Mesh Generator in MATLAB,"
           SIAM Review Vol. 46 (2) 2004.
    INPUT:
        pfix     : list of fixed points
        p        : initial list of points
        isinside : a function to check if points are inside the domain
                   ex: lambda p: sqrt(npsum(p**2.,axis=1)) - R
                   [negative => inside]
        hole_fcn : a function representing a hole
                   ex: lambda p: sqrt(npsum(p**2.,axis=1)) - r
                   [negative => inside hole]
        nit      : number of iterations
        uniform  : use uniform weights from (0,0) and radially increasing?
    RETURNS:
        p : a list of points
    """
    Fscale, deltat, dptol = 1.1, 0.1, 0.01
    fh = (lambda p: ones(len(p))) if uniform else (lambda p: 0.05+0.3*hole_fcn(p))
    notfix = range(len(pfix), len(p))
    for it in range(nit):
        # move mesh points based on bar lengths L and forces F
        bars   = get_edges(p, hole_fcn)                   # retriangulation by the Delaunay algor.
        barvec = p[bars[:,0],:] - p[bars[:,1],:]          # list of bar vectors
        L      = sqrt(npsum(barvec**2.,axis=1))           # bar lengths
        cen    = (p[bars[:,0],:] + p[bars[:,1],:]) / 2.0  # centre of bars
        hbars  = fh(cen)                                  # length weights
        sf     = sqrt(npsum(L**2.) / npsum(hbars**2.0))   # scaling factor
        L0     = hbars * Fscale * sf                      # desired lengths

        # update
        F      = L0 - L # axial forces
        F[F<0] = 0.0    # remove negative values
        Ftot   = zeros((len(p),2))
        for k, b in enumerate(bars):
            Ftot[b[0]] += F[k] * barvec[k]/L[k]
            Ftot[b[1]] -= F[k] * barvec[k]/L[k]
            Ftot
        Ftot[0:len(pfix)] = 0.0 # fixed points have force = 0
        dp = deltat * Ftot      # delta-x-y
        p += dp                 # update node positions

        # remove outside points
        pnfix = p[notfix]
        allin = isinside(pnfix[:,0], pnfix[:,1]) < 0
        if False in allin:
            p = vstack([pfix, pnfix[allin]])
            notfix = range(len(pfix), len(p))
            #print('>>>>>>>>>>>>>> outsider')

        # plot updated point
        if 0: plot(p[:,0],p[:,1],'+',color='y')

        # stop if all interior nodes move less than dptol (scaled)
        #dpnorm = max(sqrt(npsum(dp**2.0,axis=1))/L0[0])
        #print('dpnorm = %13.4e' % dpnorm)
        #if dpnorm < dptol: break

    # plot final mesh and show
    if 0:
        edges = get_edges(p, hole_fcn)
        gca().add_collection(collections.LineCollection(p[edges], linewidth=1, color='b'))
        plot(p[:,0],p[:,1],'ko',clip_on=0)
        axis('equal'); show()
    return p
