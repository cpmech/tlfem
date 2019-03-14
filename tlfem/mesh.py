# Copyright 2012 Dorival de Moraes Pedroso. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

from numpy    import array, cross, zeros  # some functions from numpy
from numpy    import tan, pi, sqrt, where # more functions from numpy
from pylab    import axis, show           # used in 'draw' method
from drawmesh import DrawMesh             # mesh drawing routine
from fig      import column_nodes         # plotting routines

class Mesh:
    def __init__(self, V, C):
        """
        FEM: Mesh
        =========
        INPUT:
            V: list of vertices (nodes)
            C: list of cells (elements)

            1D example:
                         id  vtag   x
                    V = [[0, -100, 0.0],
                         [1, -100, 0.5],
                         [2, -100, 1.0]]
                         id  etag  verts
                    C = [[0,  -1,  [0,1]],
                         [1,  -2,  [1,2]]]

            2D example:
                         id  vtag   x    y
                    V = [[0, -100, 0.0, 0.0],
                         [1, -100, 1.0, 0.0],
                         [2, -200, 1.0, 1.0],
                         [3, -300, 0.0, 1.0]]
                         id  etag  verts        btags      part
                    C = [[0, -1,  [0,1,2], {0:-10, 1:-11},  0],
                         [1, -2,  [0,2,3], {1:-12, 2:-13},  0]]
            where:
                verts : indices of vertices (connectivity)
                vtag  : vertex tag
                ctag  : cell tag
                btag  : boundary tag
        STORED:
            V         : list of vertices
            C         : list of cells
          after initialise:
            nv        : number of vertices
            nc        : number of cells
            ndim      : number of space dimensions
            vtag2vids : maps vertex tag to vertices ids
            ctag2cids : maps cell tag to cells ids
            etag2cids : maps edge tag to (cells ids, edge idx)
            edg2vids  : maps (ndim,nvc) to a list of list with edges
                        indices and vertices on edges
        RETURNS:
            None
        """
        self.V = V
        self.C = C
        self.initialise()

    def extend(self, V, C):
        """
        Append vertices and cells
        =========================
        INPUT:
            V : list of vertices to be added
            C : list of cells to be added
        STORED:
            V and C are appended to V and C
        RETURNS:
            None
        """
        self.V.extend(V)
        self.C.extend(C)
        self.initialise()

    def join_along_edge(self, b, eta, etb, tol=0.001):
        """
        Join this mesh with b along a common edge
        =========================================
        INPUT:
            b   : second Mesh object
            eta : tag of edges (in this mesh) to be merged
            etb : edge tag of b along edge to be merged
            tol : tolerance to consider overlapping vertices
        """
        if self.ndim != 2: raise Exception('this method is for 2D meshes only')
        if b.ndim    != 2: raise Exception('this method is for 2D meshes only')

        # collect vertices along edge of this mesh (a)
        Va = set()
        for ic, ed in self.etag2cids[eta]:
            vids = Edg2Vids[(self.ndim, len(self.C[ic][2]))][ed]
            Va.update([self.C[ic][2][iv] for iv in vids])

        # collect vertices along edge of other mesh (b)
        Vb = set()
        for ic, ed in b.etag2cids[etb]:
            vids = Edg2Vids[(b.ndim, len(b.C[ic][2]))][ed]
            Vb.update([b.C[ic][2][iv] for iv in vids])

        # remap vertices of b
        v2v = zeros(b.nv, dtype=int)                          # map verts in b to new verts in a
        k   = self.nv                                         # new vertex index
        for vb in b.V:                                        # for each vertex in b
            found = -1                                        # found vb in Va (-1 => not found)
            if vb[0] in Vb:                                   # vertex of b is in edge to be merged
                for ia in Va:                                 # for each vertex in Va
                    va = self.V[ia]                           # vertex of a
                    dd = sqrt((vb[2]-va[2])**2.0+(vb[3]-va[3])**2.0) # distance between vertices
                    if dd < tol:                              # overlapping vertex
                        found = ia                            # found coincident vertex in a
                        break                                 # no need to check other vts in Va
            if found < 0:                                     # not found
                iv = k                                        # index of vertex to be added
                self.V.append([iv, vb[1], vb[2], vb[3]])      # add new vertex
                k += 1                                        # next new vertex index
            else:                                             # use vertex of a
                iv = found                                    # index of vertex to be added
            v2v[vb[0]] = iv                                   # set map

        # remap cells of b
        k = self.nc                                           # new cell index
        for c in b.C:                                         # for each cell in b
            if len(c) < 4:                                    # if there is no edge tags data
                etags = {}                                    # empty etags dictionary
            else:                                             # there is etags data
                etags = {k:v for k,v in c[3].items() if v!=etb} # filter etb out
            self.C.append([k, c[1], [v2v[i] for i in c[2]], etags]) # add new cell
            k += 1                                            # next cell number

        # (re)initialise this mesh data
        self.initialise()

    def check_overlap(self, tol=0.001):
        """
        Check overlapping vertices
        ==========================
        INPUT:
            tol : tolerance to consider two vertices equal to each other
        RETURNS:
            None
        """
        if self.ndim == 1: raise Exception('this method works with 2D meshes only')
        for i in range(self.nv):
            x0, y0 = self.V[i][2], self.V[i][3]
            for j in range(i+1,self.nv):
                x1, y1 = self.V[j][2], self.V[j][3]
                dist   = sqrt((x1-x0)**2.0+(y1-y0)**2.0)
                if dist < tol:
                    raise Exception('found overlapping vertices: (%g,%g), (%g,%g)'%(x0,y0,x1,y1))

    def tag_verts(self, vtag, vids):
        """
        Tag vertices
        ============
            Tag vertices in list vids with tag = vtag
            Example: tag_verts(-102, [0, 5, 10])
        INPUT:
            vtag : tag to be applied to vids
            vids : list with indices of vertices to be tagged
        STORED:
            None
        RETURNS:
            None
        """
        for vid in vids: self.V[vid][1] = vtag # set tag
        self.build_vtag2vids_map()             # rebuild auxiliary map

    def tag_vert(self, vtag, x, y=0.0, tol=1.0e-5):
        """
        Tag vertex
        ==========
            Tag vertex with coordinates (x,y)
            Note: works for 1D as well
        INPUT:
            vtag : tag to be applied to vids
            x, y : coordinates
        STORED:
            None
        RETURNS:
            None
        """
        for v in self.V:
            if abs(v[2]-x) < tol:
                if self.ndim == 1:
                    self.tag_verts(vtag, [v[0]])
                    return
                else:
                    if abs(v[3]-y) < tol:
                        self.tag_verts(vtag, [v[0]])
                        return

    def tag_verts_on_line(self, vtag, x0, y0, alp_deg, bbox=None, tol=1.0e-5):
        """
        Tag vertices along Line
        =======================
            Tag vertices along Line:
                y = y0 + tan(alp_rad) * x
        INPUT:
            vtag    : tag to be applied to vertices on line
            x0, y0  : initial point on line
            alp_deg : inclination of line
            bbox    : [xmin,xmax, ymin,ymax] to consider vertices
        STORED:
            tagged vertices
        RETURNS:
            None
        """
        if self.ndim != 2: raise Exception('this method works with 2D meshes only')
        if alp_deg > -90.0 and alp_deg < 90.0:
            m       = tan(alp_deg * pi / 180.0)
            err_fcn = lambda x,y: y - (y0 + m*(x-x0))
        else:
            if abs(abs(alp_deg)-90.0) < tol: # vertical line
                err_fcn = lambda x,y: x - x0
            else: raise Exception('this method cannot handle alp_deg == %g'%alp_deg)
        vids = []
        for v in self.V:
            err = err_fcn(v[2], v[3])
            if abs(err) < tol:
                if bbox == None: vids.append(v[0])
                else:
                    if v[2]>bbox[0]-tol and v[2]<bbox[1]+tol and \
                       v[3]>bbox[2]-tol and v[3]<bbox[3]+tol:
                        vids.append(v[0])
        self.tag_verts(vtag, vids)

    def tag_edges_on_line(self, etag, x0, y0, alp_deg, bbox=None, tol=1.0e-5):
        """
        Tag edges along a straight line
        ===============================
            The two nodes of edges must lie on
                y = y0 + tan(alp_rad) * x
            Note:
              1) o2 nodes are disregarded; therefore edges must be straight
              2) this method is not efficient, since it loops of every cell
                 and then compares each node of cell
        INPUT:
            etag    : tag to be applied to edges
            x0, y0  : initial point on line
            alp_deg : inclination of line
            bbox    : [xmin,xmax, ymin,ymax] to consider vertices
        STORED:
            edges will be tagged
        RETURNS:
            None
        """
        if self.ndim != 2: raise Exception('this method works with 2D meshes only')
        if alp_deg > -90.0 and alp_deg < 90.0:
            m       = tan(alp_deg * pi / 180.0)
            err_fcn = lambda x,y: y - (y0 + m*(x-x0))
        else:
            if abs(abs(alp_deg)-90.0) < tol: # vertical line
                err_fcn = lambda x,y: x - x0
            else: raise Exception('this method cannot handle alp_deg == %g'%alp_deg)
        for ic, c in enumerate(self.C):
            e2v = Edg2Vids[(self.ndim, len(c[2]))]
            for iedge, lvids in enumerate(e2v):
                xa, ya = self.V[c[2][lvids[0]]][2], self.V[c[2][lvids[0]]][3]
                xb, yb = self.V[c[2][lvids[1]]][2], self.V[c[2][lvids[1]]][3]
                erra   = err_fcn(xa, ya)
                errb   = err_fcn(xb, yb)
                if abs(erra) < tol and abs(errb) < tol:
                    if bbox == None:
                        if len(c) > 3: self.C[ic][3][iedge] = etag
                        else:          self.C[ic].append({iedge:etag})
                    else:
                        if xa>bbox[0]-tol and xa<bbox[1]+tol and \
                           ya>bbox[2]-tol and ya<bbox[3]+tol and \
                           xb>bbox[0]-tol and xb<bbox[1]+tol and \
                           yb>bbox[2]-tol and yb<bbox[3]+tol:
                            if len(c) > 3: self.C[ic][3][iedge] = etag
                            else:          self.C[ic].append({iedge:etag})
        self.build_etag2cids_map() # edge tag => [[cell id, edge idx]]

    def tag_cells(self, cids, ctag):
        """
        Tag cells
        =========
            Tag cells in list cids with tag = ctag
            Example: tag_cells([0, 1], -3)
        INPUT:
            cids: list with indices of cells to be tagged
        STORED:
            None
        RETURNS:
            None
        """
        for cid in cids: self.C[cid][1] = ctag # set tag
        self.build_ctag2cids_map()            # rebuild auxiliary map

    def get_verts(self, vtags, xsorted=False, ysorted=False):
        """
        Get vertices
        ============
            Find vertices with tag = vtag(s)
        INPUT:
            vtags   : vertex tag or many vtags in a list. ex: [-100,-102]
            xsorted : sorted with respect to x values ?
            ysorted : sorted with respect to y values ?
        STORED:
            None
        RETURNS:
            vids : list of vertices ids
        """
        if isinstance(vtags, int): vtags = [vtags]
        vids = []
        for vt in vtags: vids.extend(self.vtag2vids[vt])
        return self.sort_vids(vids, xsorted, ysorted)

    def get_verts_on_edges(self, etag, xsorted=False, ysorted=False):
        """
        Get vertices on tagged edges
        ============================
        INPUT:
            etag    : edge tag
            xsorted : sorted with respect to x values ?
            ysorted : sorted with respect to y values ?
        STORED:
            None
        RETURNS:
            vids : list of vertices ids
        """
        vids = []
        for pair in self.get_cells_with_etag(etag): # index of cell, index of edge
            vids.extend(self.get_edge_verts(pair[0], pair[1]))
        vids = set(vids) # unique items
        if not xsorted and not ysorted: return [i for i in vids]
        return self.sort_vids(vids, xsorted, ysorted)

    def get_cells(self, ctag):
        """
        Get cells
        =========
            Find cells with tag = ctag
        INPUT:
            ctag: cell tag
        STORED:
            None
        RETURNS:
            list of cells ids
        """
        return self.ctag2cids[ctag][:]

    def get_cells_with_etag(self, etag):
        """
        Get cells with edge tag
        =======================
            Find cells with tagged edges = etag
        INPUT:
            etag: edge tag
        STORED:
            None
        RETURNS:
            list of cells ids
        """
        return self.etag2cids[etag][:]

    def get_edge_verts(self, icell, iedge):
        """
        Get edge vertices
        =================
            Find vertices on edge iedge of cell icell
        INPUT:
            icell: index of cell
            iedge: local index of edge of cell icell
        STORED:
            None
        RETURNS:
            list of vertex ids
        """
        lverts = Edg2Vids[(self.ndim, len(self.C[icell][2]))][iedge]
        return [self.C[icell][2][iv] for iv in lverts]

    def get_lims(self):
        """
        Get x-y limits
        ==============
        RETURNS:
            allX : all X coordinates
            allY : all Y coordinates
            xlim : [xmin, xmax]
            ylim : [ymin, ymax]
        """
        allX = array([v[2] for v in self.V])
        xlim = [allX.min(), allX.max()]
        allY, ylim = None, None
        if self.ndim > 1:
            allY = array([v[3] for v in self.V])
            ylim = [allY.min(), allY.max()]
        return allX, allY, xlim, ylim

    def gen_mid_verts(self, centre=False):
        """
        Generate vertices at middle of edges
        ====================================
            Converts a linear mesh to a quadratic one
        INPUT:
            centre : add a node at the centre of the element? (ex.: qua9)
        """
        # check
        if self.ndim != 2: raise Exception('this method works only for 2D meshes')

        # number of vertices per cell
        nvc = len(self.C[0][2])
        for c in self.C:
            cnvc = len(c[2])
            if cnvc!=nvc or (cnvc<3 or cnvc>4):
                raise Exception('to generate middle vertices, all cells must have the '+
                                'same number of vertices (3 or 4). nvc=%d is invalid'%cnvc)
        e2v_old = Edg2Vids[(self.ndim, nvc  )] # indices of verts on edges of linear element
        e2v_new = Edg2Vids[(self.ndim, nvc*2)] # indices of verts on edges of quadratic element

        # generate neighbours map
        self.find_neighbours()

        # generate map of cells => edges with global nodes indices (sorted)
        c2e = {} # cell to edges: (left global id, right global id)
        for c in self.C:
            c2e[c[0]] = [tuple(sorted([c[2][v[0]], c[2][v[1]]])) for v in e2v_old]

        # create vertices and set cells
        for edg, cells in self.edgescells.items(): # loop over edges
            # add vertex
            x0, y0 = self.V[edg[0]][2], self.V[edg[0]][3]
            x1, y1 = self.V[edg[1]][2], self.V[edg[1]][3]
            xc, yc = (x0+x1)/2.0, (y0+y1)/2.0
            iv     = len(self.V)
            self.V.append([iv, 0, xc, yc])

            # set connectivity in cells
            for ic in cells: # loop over cells sharing this edge

                # allocate space for new vertices
                nv = len(self.C[ic][2])
                if nv==3 or nv==4:
                    self.C[ic][2].extend([-1 for _ in range(nv)])

                # set new vertex
                iedge = c2e[ic].index(edg)
                inewv = e2v_new[iedge][2]
                self.C[ic][2][inewv] = iv

        # centre nodes
        if centre:
            for ic, c in enumerate(self.C):
                xc, yc, nn = 0., 0., float(len(c[2]))
                for iv in c[2]:
                    xc += self.V[iv][2]
                    yc += self.V[iv][3]
                xc /= nn
                yc /= nn
                iv  = len(self.V)
                self.V.append([iv, 0, xc, yc])
                self.C[ic][2].extend([iv])

        # reset data
        self.nv = len(self.V) # number of vertices
        self.nc = len(self.C) # number of cells

        # re-create auxiliary maps
        self.build_vtag2vids_map() # vertex tag => vertices ids
        self.build_ctag2cids_map() # cell tag => cells ids
        self.build_etag2cids_map() # edge tag => [[cell id, edge idx]]

    def edges_to_lincells(self, etag, ctag):
        """
        Add linear cells on top of edges
        ================================
            Add linear elements (beams/rods) on top of edges
        INPUT:
            etag : edge tag
            ctag : cell tag of new lincell
        STORED:
            self.C is changed
        RETURNS:
            None
        """
        edges = set()
        for cid, leid in self.etag2cids[etag]: # cell ID, local edge ID
            key = (self.ndim, len(self.C[cid][2]))
            for seg in Edg2Segs[key][leid]:
                edges.add(tuple(sorted([self.V[self.C[cid][2][Seg2Vids[key][seg][0]]][0],
                                        self.V[self.C[cid][2][Seg2Vids[key][seg][1]]][0]])))
        for ed in edges:
            self.C.append([len(self.C), ctag, list(ed)])
        self.nc = len(self.C) # number of cells
        self.build_ctag2cids_map()

    def draw(self, ids=True, tags=True, ec=False, cc=False, vc=False, vv=False, pr=False,
                   ypct=0.0, vtags=True, etags=True, ctags=True, verts=False, pr_redo=True):
        """
        Draw mesh
        =========
        INPUT:
            ids     : show also vertex/cell ids
            tags    : show also vertex/cell tags
            ec      : neighbours info: show edges => cells map
            cc      : neighbours info: show cells => cells map
            vc      : neighbours info: show vertices => cells map
            vv      : neighbours info: show vertices => vertices map
            pr      : patch recovery data
            pr_redo : redo patch finding, regardless existent one
            ypct    : a little multiplier for y ids
            vtags   : with vertex tags? [works only when tags=True]
            etags   : with edge tags? [works only when tags=True]
            ctags   : with cell tags? [works only when tags=True]
            verts   : show verts?
        STORED:
            None
        RETURNS:
            None
        """
        dm = DrawMesh(self.V, self.C, yidpct=ypct)
        if ec or cc or vc or vv or pr:
            v_ec, v_cc, v_vc, v_vv, p_vi, p_ci = {},[],[],[],{},{}
            if pr:
                if hasattr(self, 'p_vids'):
                    if pr_redo: self.find_patch()
                else: self.find_patch()
            if not (pr and pr_redo):
                self.find_neighbours()
            if ec: v_ec = self.edgescells
            if cc: v_cc = self.cellscells
            if vc: v_vc = self.vertscells
            if vv: v_vv = self.vertsverts
            if pr: p_vi, p_ci = self.p_vids, self.p_cids
            dm.draw(ids, tags, v_ec, v_cc, v_vc, v_vv, p_vi, p_ci,
                    vert_tags=vtags, edge_tags=etags, cell_tags=ctags, show_verts=verts)
        else:
            dm.draw(ids, tags, vert_tags=vtags, edge_tags=etags, cell_tags=ctags, show_verts=verts)

    def show(self):
        """
        Show figure
        ===========
            Show figure with 'same-scale' axes
        INPUT:
            None
        STORED:
            None
        RETURNS:
            None
        """
        axis('equal')
        show()

    def write_json(self, fnkey, pr=False, pr_redo=True):
        """
        Write json file (.msh)
        ======================
        INPUT:
            fnkey   : filename key (.msh will be added)
            pr      : generate patches information (for patch recovery)
            pr_redo : redo patch finding, regardless existent one
        OUTPUT:
            None (a file will be written)
        """
        if self.ndim !=2: raise Exception('write_json: this method works for 2D only')
        # verts
        l  = '{\n    "verts" : [\n'
        nv = len(self.V)
        for i, v in enumerate(self.V):
            l += '        { "id":%3d, "tag":%d, "c":[%23.15e, %23.15e] ' % (v[0],v[1],v[2],v[3])
            if i==nv-1: l += '}\n'
            else:       l += '},\n'
        # cells
        l += '    ],\n    "cells" : [\n'
        nc = len(self.C)
        for i, c in enumerate(self.C):
            nv = len(c[2])
            gd = 1 if nv<3 else 2
            pt = 0
            if len(c) > 4: pt = c[4] # part id
            l += '        { "id":%3d, "tag":%d, "part":%d, "gdim":%d, "verts":[' % (i,c[1],pt,gd)
            for j in range(nv):
                l += '%3d' % c[2][j]
                if j==nv-1: l += '], '
                else:       l += ', '
            l += '"ftags":['
            e2v = Edg2Vids[(self.ndim, len(c[2]))]
            nf  = len(e2v)
            for j in range(nf):
                if len(c)>3:
                    if j in c[3]: l += '%3d' % c[3][j]
                    else:         l += '%3d' % 0
                else:
                    l += '%3d' % 0
                if j!=nf-1: l += ', '
            l += '] '
            if i==nc-1: l += '}\n'
            else:       l += '},\n'
        # patches
        if pr:
            if pr_redo: self.find_patch()
            l += '    ],\n    "patches" : [\n'
            np = len(self.p_vids)
            for i, pkey in enumerate(self.p_vids.keys()): # for each patch key
                N   = pkey[0]                             # master node describ. the patch
                tag = pkey[1]                             # tag of cells in patch
                nvc = pkey[2]                             # number of vertices per cell
                nv  = len(self.p_vids[pkey])              # number of vertices in patch
                nc  = len(self.p_cids[pkey])              # number of cells in patch
                l += '        { "vert":%3d, "ctag":%3d, "nvc":%3d, "vids":[' % (N,tag,nvc)
                for j, iv in enumerate(self.p_vids[pkey]):
                    l += '%3d' % iv
                    if j==nv-1: l += '], '
                    else:       l += ', '
                l += '"cids":['
                for j, ic in enumerate(self.p_cids[pkey]):
                    l += '%3d' % ic
                    if j==nc-1: l += '] '
                    else:       l += ', '
                if i==np-1: l += '}\n'
                else:       l += '},\n'
        l += '    ]\n}\n'
        # save file
        f = open(fnkey+'.msh', 'w')
        f.write(l)
        f.close()

    def part_rectangle(self, nx, ny, npx, npy):
        """
        Partition rectangle
        ===================
        """
        Nx,  Ny  = nx-1, ny-1
        nnx, nny = Nx/npx, Ny/npy
        #print nnx, nny
        #print '%4s'%'nc', '%3s'%'i', '%3s'%'j', '%3s'%'I', '%3s'%'J', '%6s'%'part'
        for k, c in enumerate(self.C):
            i, j = c[0]%Nx, c[0]/Nx
            I, J = i/nnx, j/nny
            part = (I+J*npx)
            #print '%4d'%c[0], '%3d'%i, '%3d'%j, '%3d'%I, '%3d'%J, '%6d'%part
            if len(c) == 5: self.C[k][4] = part
            else:           self.C[k].append(part)

    def patch_o2column(self, disjoint=True):
        """
        Generate patch information for column of o2 squares
        """
        self.p_vids   = {}
        self.p_cids   = {}
        ctag          = self.C[0][1]
        cnvc          = len(self.C[0][2])
        nc            = len(self.C)
        ids           = range(1,nc)
        l, c, r, L, R = column_nodes(nc, True)
        if cnvc != 8: raise Exception("patch_o2column: cnvc must be 8 (%d is invalid)" % cnvc)
        #print 'l =', l
        #print 'c =', c
        #print 'r =', r
        #print 'L =', L
        #print 'R =', R
        #print
        for i in range(len(self.C)):
            cnvc = len(self.C[i][2])
            if cnvc != 8: raise Exception("patch_o2column: cnvc must be 8 (%d is invalid)" % cnvc)
            con = [l[i],r[i],r[i+1],l[i+1],c[i],R[i*2+1],c[i+1],L[i*2+1]]
            #print con
            for j in range(8):
                if self.C[i][2][j] != con[j]:
                    raise Exception("patch_o2column: mesh does not correspond to a o2column: verts # %d != %d" % (self.C[i][2][j], con[j]))
        if disjoint: ids = range(1,nc,2)
        for i in ids:
            #print '%2d:  %2d %2d %2d %2d %2d %2d %2d %2d %2d %2d %2d %2d' % (c[i], l[i-1], c[i-1], r[i-1], R[i*2-1], r[i], R[i*2+1], r[i+1], c[i+1], l[i+1], L[i*2+1], l[i], L[i*2-1])
            key = (c[i],ctag,cnvc)
            self.p_vids[key] = [c[i], l[i-1], c[i-1], r[i-1], R[i*2-1], r[i], R[i*2+1], r[i+1], c[i+1], l[i+1], L[i*2+1], l[i], L[i*2-1]]
            self.p_cids[key] = [i-1,i]

    # ------------------------------------ internal methods ---------------------------------------

    def initialise(self):
        """
        Initialise mesh
        ===============
            Initialise all mesh data
        INPUT:
            None
        STORED:
            nv        : number of vertices
            nc        : number of cells
            ndim      : number of space dimensions
            vtag2vids : maps vertex tag to vertices ids
            ctag2cids : maps cell tag to cells ids
            etag2cids : maps edge tag to (cells ids, edge idx)
            edg2vids  : maps (ndim,nvc) to a list of list with edges
                        indices and vertices on edges
        RETURNS:
            None
        """
        # data
        self.nv   = len(self.V) # number of vertices
        self.nc   = len(self.C) # number of cells
        self.ndim = len(self.V[0]) - 2  # number of space dimensions

        # check vertices
        for k, v in enumerate(self.V):
            # check numbering of vertices
            if v[0] != k:
                raise Exception('numbering of vertices must start from zero and be sequential;' +
                                ' problem with vertex # %d or %d'%(k,v[0]))
            # check space dimension
            ndim = len(v) - 2
            if ndim != self.ndim:
                raise Exception('space dimension of all vertices must be the same')

        # check cells
        if self.ndim == 1: # 1D
            for k, c in enumerate(self.C):
                # check numbering of cells
                if c[0] != k:
                    raise Exception('numbering of cells must start from zero and be sequential;' +
                                    ' problem with cell # %d or %d'%(k,c[0]))
                # check order of vertices within cell
                if self.V[c[2][1]][2] < self.V[c[2][0]][2]:
                    raise Exception('the order of vertices of cell %d is incorrect;'%c[0] +
                                    ' it must be from left to right')
        else: # 2D
            for k, c in enumerate(self.C):
                # check numbering of cells
                if c[0] != k:
                    raise Exception('numbering of cells must start from zero and be sequential;' +
                                    ' problem with cell # %d or %d'%(k,c[0]))
                # check order of vertices within cell
                if len(c[2]) > 2:
                    v0, v1, vn = self.V[c[2][0]], self.V[c[2][1]], self.V[c[2][-1]]
                    e01 = array([v1[2]-v0[2], v1[3]-v0[3], 0.0])
                    e0n = array([vn[2]-v0[2], vn[3]-v0[3], 0.0])
                    cro = cross(e01, e0n)
                    if cro[2] < 0.0:
                        raise Exception('the order of vertices of cell %d is incorrect;'%c[0] +
                                        ' it must be counter-clockwise')
        # create auxiliary maps
        self.build_vtag2vids_map() # vertex tag => vertices ids
        self.build_ctag2cids_map() # cell tag => cells ids
        self.build_etag2cids_map() # edge tag => [[cell id, edge idx]]

    def find_neighbours(self):
        """
        Find Neighbours
        ===============
            Builds maps containing all information about neighbours
            vertices and cells, including what is shared with what
        INPUT:
            None:
        STORED:
            edgescells : edges (a pair) => cells sharing an edge
            cellscells : cells => neighbour cells
            vertscells : vertices => shared cell numbers
            vertsverts : vertices => connected vertices
            bryverts   : list of vertices on boundaries
        RETURNS:
            None
        """
        self.edgescells = {}
        if self.ndim == 1:
            self.cellscells     = [[ic-1,ic+1] for ic in range(self.nc)]
            self.cellscells[ 0] = [1]
            self.cellscells[-1] = [self.nc-2]
            self.vertsverts     = [[iv-1,iv+1] for iv in range(self.nv)]
            self.vertsverts[ 0] = [1]
            self.vertsverts[-1] = [self.nv-2]
            self.vertscells     = [[iv-1,iv] for iv in range(self.nv)]
            self.vertscells[ 0] = [0]
            self.vertscells[-1] = [self.nc-1]
            self.bryverts       = set([0, self.nv-1])
        else:
            self.cellscells = [set() for _ in range(self.nc)]
            self.vertsverts = [set() for _ in range(self.nv)]
            self.vertscells = [set() for _ in range(self.nv)]
            for c in self.C: # for each cell
                # map: verts => cells
                for iv in c[2]:                    # for each vertex in cell
                    self.vertscells[iv].add(c[0])  # set cell sharing this vertex iv
                # map: edges => cells
                ledges = Seg2Vids[(self.ndim, len(c[2]))]                      # local edges nums
                gedges = [sorted([c[2][ed[0]], c[2][ed[1]]]) for ed in ledges] # global edges nums
                for gedge in gedges:          # for each global edge in list of glob edges of cell
                    ed = (gedge[0], gedge[1]) # edge key (sorted)
                    if ed in self.edgescells: self.edgescells[ed].append(c[0])
                    else:                     self.edgescells[ed] =     [c[0]]
            for c in self.C:
                # map: cells => cells
                ledges = Seg2Vids[(self.ndim, len(c[2]))]                      # local edges nums
                gedges = [sorted([c[2][ed[0]], c[2][ed[1]]]) for ed in ledges] # global edges nums
                for gedge in gedges:                            # for each edge of cell
                    ed = (gedge[0], gedge[1])                   # edge key (sorted)
                    cs = list(self.edgescells[ed])              # cells sharing this edge
                    del cs[cs.index(c[0])]                      # remove this cell from list
                    self.cellscells[c[0]].update(cs)            # update this cell's neighbours
                    self.vertsverts[gedge[0]].add(gedge[1])     # update list of connected vertices
                    self.vertsverts[gedge[1]].add(gedge[0])     # update list of connected vertices
            # find vertices on boundary: those on edges shared by one single cell
            self.bryverts = set()                               # unique list of vertices on bry
            for ed, cs in self.edgescells.items():              # for each edge and cells sharing
                if len(cs) == 1:                                # edge is shared by only one cell
                    nvc = len(self.C[cs[0]][2])                 # number of vertices in cell
                    nedges = len(Edg2Vids[(self.ndim, nvc)])    # number of edges in cells
                    for ie in range(nedges):                    # for each edge index
                        vids = self.get_edge_verts(cs[0], ie)   # vertices on edge
                        if ed[0] in vids and ed[1] in vids:     # found a boundary edge
                            self.bryverts.update(vids)          # update set
                            break                               # found bry edge

    def find_patch(self):
        """
        Find patch around vertices
        ==========================
            Find patches around vertices for recovery of secondary variables
        INPUT:
            None
        STORED:
            p_vids : patch vertices: example: {(node,ctag,cnvc):set(n0,n1,n2,...)}
            p_cids : patch cells:    example: {(node,ctag,cnvc):set(c0,c1,c2,...)}
        RETURNS:
            None
        """
        # find neighbours data
        self.find_neighbours()

        # 1D mesh
        if self.ndim == 1:
            self.p_vids = {(iv,-1,2):set([iv])                for iv in range(1,self.nv-1)}
            self.p_cids = {(iv,-1,2):set(self.vertscells[iv]) for iv in range(1,self.nv-1)}
            self.p_vids[(1,        -1,2)].add(0)
            self.p_vids[(self.nv-2,-1,2)].add(self.nv-1)
            return

        # filter lincells out from verts-to-cells map
        v2cells = {}
        for n, cells in enumerate(self.vertscells):
            v2cells[n] = [ic for ic in cells if len(self.C[ic][2])>2]

        # find master and interface vertices
        M = set() # master vertices
        I = set() # interface vertices
        for n in range(self.nv):
            if n in self.bryverts: continue
            # if vertex is surrounded by at least 3 cells (excluding lincells)
            if len(v2cells[n]) > 2:
                mytypes = set()
                for ic in v2cells[n]:
                    ctag =     self.C[ic][1]
                    cnvc = len(self.C[ic][2])
                    mytypes.add((n,ctag,cnvc))
                if len(mytypes) > 1:
                    I.add(n)
                else:
                    M.add([t for t in mytypes][0])

        # counter: number of times a vertex was added to a patch
        p_cnt = zeros(self.nv, dtype=int)

        # find patches
        self.p_vids = {}
        self.p_cids = {}
        for key in M:
            n                = key[0]
            p_cnt[n]         = 1
            self.p_vids[key] = set([n])
            self.p_cids[key] = set()
            for ic in v2cells[n]:
                self.p_cids[key].add(ic)
                for m in self.C[ic][2]:
                    mnc = len(v2cells[m]) # m: num cells => indicates vert on edge
                    if (m in self.bryverts) or (mnc < 3) or (m in I):
                        p_cnt[m] += 1
                        self.p_vids[key].add(m)

        # handle hanging vertices
        hanging = where(p_cnt == 0)[0]
        if len(hanging) > 0:
            for m in hanging:
                # skip vertices connected to linear cells only (rods/beams)
                if max([len(self.C[ic][2]) for ic in self.vertscells[m]]) == 2:
                    p_cnt[m] = -1
                    continue
                # find master vertices connected to other cells connected to this vertex
                for i in v2cells[m]:
                    Found = False
                    for j in self.cellscells[i]:
                        ctag  =     self.C[j][1]
                        cnvc  = len(self.C[j][2])
                        found = False
                        for n in self.C[j][2]:
                            key = (n, ctag, cnvc)
                            if key in M:
                                p_cnt[m] += 1
                                self.p_vids[key].add(m)
                                self.p_cids[key].add(i)
                                found = True
                                break
                        if found:
                            Found = True
                            break
                    if Found: break

        # check hanging vertices
        hanging = where(p_cnt == 0)[0]
        if len(hanging) > 0:
            raise Exception('there are hanging vertices: %s' % hanging.__str__())

    def sort_vids(self, vids, xsorted=False, ysorted=False):
        """
        Sort a list of vertex ids
        =========================
        INPUT:
            vids    : a list with vertex ids
            xsorted : sorted with respect to x values ?
            ysorted : sorted with respect to y values ?
        STORED:
            None
        RETURNS:
            sorted list
        """
        if xsorted:
            ix   = [(i, self.V[i][2]) for i in vids]
            ix   = sorted(ix, key=lambda k: k[1])
            vids = [k[0] for k in ix]
        if ysorted:
            iy   = [(i, self.V[i][3]) for i in vids]
            iy   = sorted(iy, key=lambda k: k[1])
            vids = [k[0] for k in iy]
        return vids

    def build_vtag2vids_map(self):
        """
        Build vtag2vids
        ===============
            Builds a map that converts vertex tags to list of vertex ids
            Example:
                vids = self.vtag2vids[-101]
                ==> [0, 5, 10]
        INPUT:
            None
        STORED:
            vtag2vids: maps vertex tag to vertices ids
        RETURNS:
            None
        """
        self.vtag2vids = {} # converts vertex tag to vertices ids
        for v in self.V:
            vid, vtag = v[0], v[1]
            if vtag >= 0: continue
            if vtag in self.vtag2vids:
                self.vtag2vids[vtag].append(vid)
            else:
                self.vtag2vids[vtag] = [vid]

    def build_ctag2cids_map(self):
        """
        Builds ctag2cids
        ================
            Builds a map that converts cell tags to list of cell ids
            Example:
                cids = self.ctag2cids[-1]
                ==> [0,1,2,3,4]
        INPUT:
            None
        STORED:
            ctag2cids: maps cell tag to cells ids
        RETURNS:
            None
        """
        self.ctag2cids = {} # converts cell tag to cells ids
        for c in self.C:
            cid, ctag = c[0], c[1]
            if ctag >= 0: raise Exception('all cells must have a negative tag')
            if ctag in self.ctag2cids:
                self.ctag2cids[ctag].append(cid)
            else:
                self.ctag2cids[ctag] = [cid]

    def build_etag2cids_map(self):
        """
        Builds etag2cids
        ================
            Builds a map that converts edges tags to list of cell ids
            etag2cids is a map of edge-tags to a list with all
            cells sharing that tagged edge. Both the cell id and
            the edge through which that cell is shared are stored.
            Example:
                cids = self.etag2cids[-10]
                ==> [[cell#0,edge#0], [cell#1,edge#0]]
        INPUT:
            None
        STORED:
            etag2cids: maps edge tag to (cells ids, edge idx)
        RETURNS:
            None
        """
        self.etag2cids = {} # converts edge tag to (cells ids, edge idx)
        for c in self.C:
            nedgmax = len(Edg2Vids[(self.ndim, len(c[2]))]) # max number of edges
            cid = c[0]            # cell id
            if len(c)<4: continue # does not have btags dictionary
            for eidx, etag in c[3].items():
                if etag >= 0: continue
                if eidx >= nedgmax:
                    raise Exception('edge index = %d is invalid; '%eidx +
                                    'edge index must be smaller than %d'%nedgmax)
                if etag in self.etag2cids:
                    self.etag2cids[etag].append([cid,eidx])
                else:
                    self.etag2cids[etag] = [[cid,eidx]]

# ---------------------------------------- some constants -----------------------------------------

# map: (ndim,nvc) ==> edge index ==> local vertices indices 
Edg2Vids = {
    (1, 2) : [                                          ], # lin2
    (1, 3) : [                                          ], # lin3
    (2, 2) : [                                          ], # lin2
    (2, 3) : [(0, 1),    (1, 2),    (2, 0)              ], # tri3
    (2, 6) : [(0, 1, 3), (1, 2, 4), (2, 0, 5)           ], # tri6
    (2, 4) : [(0, 1),    (1, 2),    (2, 3),    (3, 0)   ], # qua4
    (2, 8) : [(0, 1, 4), (1, 2, 5), (2, 3, 6), (3, 0, 7)], # qua8
    (2, 9) : [(0, 1, 4), (1, 2, 5), (2, 3, 6), (3, 0, 7)]  # qua9
}

# map: (ndim,nvc) ==> segments (sub-edge) index ==> local vertices indices
Seg2Vids = {
    (1, 2) : [                                                      ], # lin2
    (1, 3) : [                                                      ], # lin3
    (2, 2) : [                                                      ], # lin2
    (2, 3) : [(0,1), (1,2), (2,0)                                   ], # tri3
    (2, 6) : [(0,3), (3,1), (1,4), (4,2), (2,5), (5,0)              ], # tri6
    (2, 4) : [(0,1), (1,2), (2,3), (3,0)                            ], # qua4
    (2, 8) : [(0,4), (4,1), (1,5), (5,2), (2,6), (6,3), (3,7), (7,0)], # qua8
    (2, 9) : [(0,4), (4,1), (1,5), (5,2), (2,6), (6,3), (3,7), (7,0)]  # qua9
}

# map: (ndim,nvc) ==> edge index ==> segments indices
Edg2Segs = {
    (1, 2) : [                          ], # lin2
    (1, 3) : [                          ], # lin3
    (2, 2) : [                          ], # lin2
    (2, 3) : [(0,),  (1,),  (2,)        ], # tri3
    (2, 6) : [(0,1), (2,3), (4,5)       ], # tri6
    (2, 4) : [(0,),  (1,),  (2,),  (3,) ], # qua4
    (2, 8) : [(0,1), (2,3), (4,5), (6,7)], # qua8
    (2, 9) : [(0,1), (2,3), (4,5), (6,7)]  # qua9
}
