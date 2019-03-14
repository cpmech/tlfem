# Copyright 2012 Dorival de Moraes Pedroso. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

import numpy
from numpy import array, unique

class TrussGenerator:
    """
    TrussGenerator creates a 2D truss in a grid with origin at (0, 0).
    The horizontal length of the grid is L and the height of the grid is H.
    Members (rods) are defined by pairs of IDs corresponding to vertices
    in the grid.

    For example, with the grid below:

       24---25---26---27---28---29  4  ny=5
       |    |    |    |    |    |
       18---19---20---21---22---23  3
       |    |    |    |    |    |        n = i + j * nx
       12---13---14---15---16---17  2    i = n % nx
       |    |    |    |    |    |        j = n // nx
       6----7----8----9---10----11  1
       |    |    |    |    |    |
       0----1----2----3----4----5  j=0
     i=0    1    2    3    4    5  nx=6

    the following truss
       .    .    .    .    .
            o---------o         .
       .   / \   .   / \   .
          / . \     / . \       .
       . /  .  \ . /  .  \ .
        /   .   \ /   .   \     .
       o---------o---------o

    can be fully specified by means of:

     T = [[0,2], [2,4], [13,15], [0,13], [13,2], [2,15], [15,4]]
    """


    def __init__(self, nx, ny, L, H):
        """
        Constructor initialises object
        Input:
          nx -- number of columns in grid
          ny -- number of rows in grid
          L  -- horizontal length of grid
          H  -- vertical length (height) of grid
        """
        self.nx = nx
        self.dx = L / float(nx-1) # horz spacing
        self.dy = H / float(ny-1) # vert spacing


    def generate(self, T):
        """
        generate generates truss mesh as explained above
        Input:
          T -- pairs of IDs in grid defining rods
        Output:
          V -- list of vertices
          C -- list of cells
        Notes:
          1) vertices located @ the horizontal line with j=0 are tagged with -100
          2) a vertex located @ i=0 is tagged with -101
          3) a vertex located @ i=nx-1 is tagged with -102
        """

        # vertices
        if not isinstance(T, numpy.ndarray): T = array(T, dtype=int)
        nr  = len(T)                  # number of rods
        N   = unique(T.reshape(2*nr)) # all nodes
        g2l = {}                      # global => local indices
        V   = []
        for iv, n in enumerate(N):
            i = n % self.nx
            j = n // self.nx
            x = float(i) * self.dx
            y = float(j) * self.dy
            tag = 0
            if j==0: tag = -100
            if i==0: tag = -101
            if i==self.nx-1: tag = -102
            g2l[n] = iv
            V.append([iv, tag, x, y])

        # cells
        C = []
        for ic, p in enumerate(T):
            C.append([ic, -1, [g2l[p[0]], g2l[p[1]]]])

        # results
        return V, C


# test
if __name__ == "__main__":
    from FEMmesh import FEMmesh
    G = TrussGenerator(6, 5, 5.0, 4.0)
    T = [[0,2], [2,4], [13,15], [0,13], [13,2], [2,15], [15,4]]
    V, C = G.generate(T)
    m = FEMmesh(V, C)
    m.draw()
    m.show()
