# Copyright 2012 Dorival de Moraes Pedroso. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

from __future__ import print_function # for Python 3

import sys
from   numpy         import linspace
from   tlfem.genmesh import Gen1Dmesh, Gen2Dregion, GenQplateHole, GenQdisk, JoinAlongEdge, ReadJson
from   tlfem.mesh    import Mesh

def runtest(prob):
    if prob==1:
        m = Gen1Dmesh(1.0, 5)
        m.find_neighbours()
        print('edgescells =', m.edgescells)
        print('cellscells =', m.cellscells)
        print('vertscells =', m.vertscells)
        print('vertsverts =', m.vertsverts)
        m.draw(vc=1)
        m.show()

    if prob==2:
        V = [[0, -100, 0.0],
             [1, -100, 0.5],
             [2, -100, 1.0]]
        C = [[0, -1, [0,1]],
             [1, -2, [1,2]]]
        m = Mesh(V, C)
        m.find_neighbours()
        print('edgescells =', m.edgescells)
        print('cellscells =', m.cellscells)
        print('vertscells =', m.vertscells)
        print('vertsverts =', m.vertsverts)
        m.draw()
        m.show()

    if prob==3:
        V = [[0, -100, 0.0, 0.0],
             [1, -100, 1.0, 0.0],
             [2, -200, 1.0, 1.0],
             [3, -300, 0.0, 1.0]]
        C = [[0, -1, [0,1,2], {0:-10, 1:-11}],
             [1, -2, [0,2,3], {1:-12, 2:-13}]]
        m = Mesh(V, C)
        m.find_neighbours()
        print('edgescells =', m.edgescells)
        print('cellscells =', m.cellscells)
        print('vertscells =', m.vertscells)
        print('vertsverts =', m.vertsverts)
        m.draw(ec=1,vc=1,cc=1,vv=1)
        m.show()

    if prob==4:
        m = Gen2Dregion(linspace(0.,5.,5), linspace(0.,3.,3))
        #m.draw()
        m.draw(ec=1,vc=1,cc=1,vv=1)
        m.show()

    if prob==5:
        m = Gen2Dregion(linspace(0.,5.,5), linspace(0.,3.,3))
        m.tag_verts_on_line(-111, 0.,0.,0.)
        m.tag_verts_on_line(-222, 0.,0.,90.)
        m.tag_verts_on_line(-333, 5.,0.,-90.)
        m.tag_verts_on_line(-444, 0.,3.,0.)
        m.tag_verts_on_line(-555, 1.25,0.,90.)
        m.draw()
        m.show()

    if prob==6:
        m = Gen2Dregion(linspace(0.,5.,5), linspace(0.,5.,5))
        m.gen_mid_verts()
        m.tag_verts_on_line(-111, 0.,2.5,0.)
        m.tag_verts_on_line(-222, 0.,0.,45.)
        #m.draw()
        m.draw(ec=1,vc=1,cc=1,vv=1)
        m.show()
        vs1 = m.get_verts(-111, xsorted=True)
        vs2 = m.get_verts(-222, ysorted=True)
        print('-111 =>', vs1)
        print('-222 =>', vs2)

    if prob==7:
        m = Gen2Dregion(linspace(0.,3.,3), linspace(0.,2.,2), triangle=False)
        #m.gen_mid_verts()
        m.gen_mid_verts(centre=True)
        m.draw()
        m.show()

    if prob==8:
        V = [[0, -101, 0.0, 0.0],
             [1, -102, 1.5, 3.5],
             [2,    0, 0.0, 5.0],
             [3, -101, 5.0, 5.0]]
        C = [[0,  -1, [0,1]],
             [1,  -1, [1,3]],
             [2,  -2, [0,2]],
             [3,  -2, [2,3]],
             [4,  -3, [2,1]]]
        m = Mesh(V, C)
        m.draw()
        m.show()

    if prob==9:
        m = GenQplateHole(3., 2., 1.0, 10, 7, 7, nit=20)
        m.draw()
        m.show()

    if prob==10:
        m = GenQdisk(3.)
        m.draw(ec=1,vc=1,cc=1,vv=1)
        m.show()

    if prob==11:
        V = [[0, 0, 0.0, 0.0],
             [1, 0, 0.5, 0.0],
             [2, 0, 1.0, 0.0],
             [3, 0, 0.0, 0.5],
             [4, 0, 0.5, 0.5],
             [5, 0, 0.0, 1.0],
             [6, 0, 1.0, 1.0]]

        C = [[0, -1, [0,2,5,1,4,3]],
             [1, -2, [4,2,6]],
             [2, -3, [5,6]]]

        m = Mesh(V, C)
        #m.draw(ids=1, tags=0, ec=1, cc=1, vc=1, vv=1, pr=1); m.show()
        m.draw(ids=1, tags=0, ec=1, cc=1, vc=1, vv=1, pr=0); m.show()

    if prob==12:

        tri = 0

        x = linspace(0.,4.,5)
        y = linspace(0.,2.,3)
        a = Gen2Dregion(x, y, triangle=tri)
        a.gen_mid_verts()
        #a.draw(); a.show()

        x = linspace(0.,4.,9)
        y = linspace(2.,2.5,2)
        b = Gen2Dregion(x, y, triangle=tri, etags=[-10,-11,-14,-13])
        #b.draw(); b.show()

        m = JoinAlongEdge(a, b, -12, -10)
        m.check_overlap()
        m.draw(); m.show()

    if prob==13:

        x = linspace(0.,4.,2)
        y = linspace(0.,2.,2)
        m = Gen2Dregion(x, y, triangle=0, etags=[-10,-11,-12,-11])
        m.draw(); m.show()
        print('etag2cids =\n', m.etag2cids)

    if prob==14:
        m = ReadJson('../data/bh16')
        m.draw(); m.show()

# run tests
prob = int(sys.argv[1]) if len(sys.argv)>1 else -1
if prob < 0:
    for p in range(1,15):
        print()
        print('[1;33m####################################### %d #######################################[0m'%p)
        print()
        runtest(p)
else: runtest(prob)
