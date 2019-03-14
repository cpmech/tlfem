# Copyright 2012 Dorival de Moraes Pedroso. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

from os.path import exists, normpath
from os      import mkdir, remove
from glob    import glob
from numpy   import sqrt

class Vtu:
    def __init__(self, pu_decoup=False):
        """
        Class for writing ParaView's VTU files
        ======================================
        INPUT:
            pu_decoup : pressure/displacements(or velocities) decoupling. This will generate
                        another file with the 'pw' scalar field on top of a reduced mesh:
                        tri6 => tri3 or qua8 => qua4
        STORED:
            pu_decoup : to decouple 'pw' from 'u'
        """
        # map (ndim,nvc) to VTK keycode
        # nvc is the number of vertices per cell
        self.geo2vtk = {
            (1, 2) : 3,   # (1D, 2  nodes) => lin2
            (1, 3) : 21,  # (1D, 3  nodes) => lin3
            (2, 2) : 3,   # (2D, 2  nodes) => lin2 (truss)
            (2, 3) : 5,   # (2D, 3  nodes) => tri3
            (2, 6) : 22,  # (2D, 6  nodes) => tri6
            (2, 4) : 9,   # (2D, 4  nodes) => qua4
            (2, 8) : 23,  # (2D, 8  nodes) => qua8
            (2, 9) : 23   # (2D, 9  nodes) => qua8 (cannot be quad9)
        }
        self.pu_decoup = pu_decoup

    def start(self, fnkey, directory='vtufiles', del_files=True):
        """
        Start
        =====
            Example: write('sim') produces 'sim_000...1.vtu'
        INPUT:
            fnkey     : filename-key used during sequential output
            directory : subdirectory to be created holding all vtu files
            del_files : delete previous files?
        """
        self.fnkey  = fnkey     # filename key
        self.dire   = directory # directory holding output files
        self.idxout = 0         # index of current output
        self.fnames = []        # list with all file names
        self.times  = []        # output times

        # create output directory
        if not exists(directory):
            mkdir(directory)

        # delete existent files
        if del_files:
            for fn in glob(normpath(directory+'/'+fnkey+'*.vtu')):
                remove(normpath(fn))

        # decoupled pw field
        if self.pu_decoup: self.fnames_pw = []

    def stop(self):
        """
        Stop
        ====
        """
        # generate collections file
        self.gen_collections_file(self.fnkey, self.times, self.fnames)
        if self.pu_decoup:
            self.gen_collections_file(self.fnkey+'_pw', self.times, self.fnames_pw)

    def write(self, time, msh, Uout={}, Eout={}, Lv=[], mdata=False, winvs=True):
        """
        Write VTU file for ParaView
        ===========================
        INPUT:
            time  : real time of output (stored as a comment in the vtu file)
            msh   : an instance of FEMmesh
            Uout  : U values at vertices
            Eout  : extrapolated values at vertices
            Lv    : eigenvectors
            mdata : mehsdata: output tags and ids as well
            winvs : with stress/strain invariants (if present)
        Note:
            Uout['ux'][3][tidx] => output at time 'tidx' of the 'ux' value of 'name' # 3
            Eout['sx'][3][tidx] => output at time 'tidx' of the 'sx' value of 'name' # 3
                ^    ^   ^
                |    |   |_ time index [optional]
                |    |_____ id
                |__________ key
        """
        # buffers
        a, b = '', ''
        if self.pu_decoup: bp = ''

        # header
        a += "<?xml version=\"1.0\"?>\n"
        a += "<!-- Time = %g -->\n" % time
        a += "<VTKFile type=\"UnstructuredGrid\" version=\"0.1\" byte_order=\"LittleEndian\">\n"
        a += "<UnstructuredGrid>\n"
        a += "<Piece NumberOfPoints=\"%d\" NumberOfCells=\"%d\">\n" % (msh.nv, msh.nc)

        # verts
        a += "<Points>\n"
        a += "<DataArray type=\"Float64\" NumberOfComponents=\"3\" format=\"ascii\">\n"
        for k in range(msh.nv):
            for j in range(msh.ndim):
                a += "%g " % (msh.V[k][2+j])
            if msh.ndim == 1: a += "0 0 "
            if msh.ndim == 2: a += "0 "
        a += "\n</DataArray>\n"
        a += "</Points>\n"

        # cells
        a += "<Cells>\n"
        a += "<DataArray type=\"Int32\" Name=\"connectivity\" format=\"ascii\">\n"
        for k in range(msh.nc):
            for vid in msh.C[k][2]:
                b += "%d " % vid
            if self.pu_decoup: # decouple pw
                nvc = len(msh.C[k][2])
                if nvc == 6 or nvc == 8: # tri6 or qua8
                    for iv in range(nvc/2):
                        bp += "%d " % msh.C[k][2][iv]
        b += "\n</DataArray>\n"
        b += "<DataArray type=\"Int32\" Name=\"offsets\" format=\"ascii\">\n"
        offset = 0
        if self.pu_decoup: # decouple pw
            bp += "\n</DataArray>\n"
            bp += "<DataArray type=\"Int32\" Name=\"offsets\" format=\"ascii\">\n"
            offset_p = 0
        for k in range(msh.nc):
            nvc     = len(msh.C[k][2])
            offset += nvc
            b      += "%d " % offset
            if self.pu_decoup: # decouple pw
                if nvc == 6 or nvc == 8: # tri6 or qua8
                    offset_p += nvc/2
                    bp       += "%d " % offset_p
        b += "\n</DataArray>\n"
        b += "<DataArray type=\"UInt8\" Name=\"types\" format=\"ascii\">\n"
        if self.pu_decoup: # decouple pw
            bp += "\n</DataArray>\n"
            bp += "<DataArray type=\"UInt8\" Name=\"types\" format=\"ascii\">\n"
        for k in range(msh.nc):
            nvc = len(msh.C[k][2])
            b += "%d " % self.geo2vtk[(msh.ndim, nvc)]
            if self.pu_decoup: # decouple pw
                if nvc == 6 or nvc == 8: # tri6 or qua8
                    bp += "%d " % self.geo2vtk[(msh.ndim, nvc/2)]
        b += "\n</DataArray>\n"
        b += "</Cells>\n"
        if self.pu_decoup: # decouple pw
            bp += "\n</DataArray>\n"
            bp += "</Cells>\n"

        # data: verts
        b += "<PointData Scalars=\"TheScalars\">\n"
        if self.pu_decoup: # decouple pw
            bp += "<PointData Scalars=\"TheScalars\">\n"
        if mdata:
            b += "<DataArray type=\"Int32\" Name=\"tag,id\" NumberOfComponents=\"2\" format=\"ascii\">\n"
            for k in range(msh.nv):
                b += "%d %d " % (msh.V[k][1], msh.V[k][0])
            b += "\n</DataArray>\n"

        # Uout: displacements
        if 'ux' in Uout and 'uy' in Uout:
            b += "<DataArray type=\"Float64\" Name=\"u\" NumberOfComponents=\"3\" format=\"ascii\">\n"
            for n in range(msh.nv):
                b += "%g %g 0  " % (Uout['ux'][n][-1], Uout['uy'][n][-1])
            b += "\n</DataArray>\n"

        # Uout: other variables
        for key in Uout.keys():
            if key in ['ux', 'uy', 'rz']: continue
            b += "<DataArray type=\"Float64\" Name=\"%s\" NumberOfComponents=\"1\" format=\"ascii\">\n" % key
            if self.pu_decoup: # decouple pw
                bp += "<DataArray type=\"Float64\" Name=\"%s\" NumberOfComponents=\"1\" format=\"ascii\">\n" % key
            for n in range(msh.nv):
                b += "%g " % Uout[key][n][-1]
                if self.pu_decoup: # decouple pw
                    bp += "%g " % Uout[key][n][-1]
            b += "\n</DataArray>\n"

        # Eout: diffusion velocity
        if 'wx' in Eout and 'wy' in Eout:
            b += "<DataArray type=\"Float64\" Name=\"w\" NumberOfComponents=\"3\" format=\"ascii\">\n"
            for n in range(msh.nv):
                b += "%g %g 0  " % (Eout['wx'][n][-1], Eout['wy'][n][-1])
            b += "\n</DataArray>\n"

        # Eout: other variables
        for key in Eout.keys():
            if key in ['wx', 'wy']: continue
            b += "<DataArray type=\"Float64\" Name=\"%s\" NumberOfComponents=\"1\" format=\"ascii\">\n" % key
            for n in range(msh.nv):
                b += "%g " % Eout[key][n][-1]
            b += "\n</DataArray>\n"

        # Eout: stress invariants
        if winvs and 'sx' in Eout and 'sy' in Eout and 'sz' in Eout and 'sxy' in Eout:
            str_p  = "<DataArray type=\"Float64\" Name=\"p\"  NumberOfComponents=\"1\" format=\"ascii\">\n"
            str_q  = "<DataArray type=\"Float64\" Name=\"q\"  NumberOfComponents=\"1\" format=\"ascii\">\n"
            str_ev = "<DataArray type=\"Float64\" Name=\"ev\" NumberOfComponents=\"1\" format=\"ascii\">\n"
            str_ed = "<DataArray type=\"Float64\" Name=\"ed\" NumberOfComponents=\"1\" format=\"ascii\">\n"
            sq2    = sqrt(2.0)
            for n in range(msh.nv):
                s  = [Eout['sx'][n][-1], Eout['sy'][n][-1], Eout['sz'][n][-1], Eout['sxy'][n][-1]]
                e  = [Eout['ex'][n][-1], Eout['ey'][n][-1], Eout['ez'][n][-1], Eout['exy'][n][-1]]
                p  = -(s[0]+s[1]+s[2])/3.0
                q  = sqrt((s[0]-s[1])**2.0+(s[1]-s[2])**2.0+(s[2]-s[0])**2.0+6.0*(s[3]**2.0))/sq2
                ev = e[0]+e[1]+e[2]
                ed = sqrt((e[0]-e[1])**2.0+(e[1]-e[2])**2.0+(e[2]-e[0])**2.0+6.0*(e[3]**2.0))*(sq2/3.0)
                str_p  += "%g " % p
                str_q  += "%g " % q
                str_ev += "%g " % ev
                str_ed += "%g " % ed
            b += str_p  + "\n</DataArray>\n"
            b += str_q  + "\n</DataArray>\n"
            b += str_ev + "\n</DataArray>\n"
            b += str_ed + "\n</DataArray>\n"

        # end of point data
        b += "</PointData>\n"

        # data: cells
        b += "<CellData Scalars=\"TheScalars\">\n"
        if mdata:
            b += "<DataArray type=\"Int32\" Name=\"tag,id\" NumberOfComponents=\"2\" format=\"ascii\">\n"
            for ic in range(msh.nc):
                b += "%d %d " % (msh.C[ic][1], msh.C[ic][0])
            b += "\n</DataArray>\n"

        # end of element data
        b += "</CellData>\n"

        # footer
        b += "</Piece>\n"
        b += "</UnstructuredGrid>\n"
        b += "</VTKFile>\n"

        # write file
        fn = "%s/%s_%06d.vtu" % (self.dire, self.fnkey, self.idxout)
        fi = open(normpath(fn), 'w')
        fi.write(a)
        fi.write(b)
        fi.close()
        self.idxout += 1
        self.fnames.append(normpath(fn))
        self.times .append(time)

        # decouple pw
        if self.pu_decoup:
            bp += "\n</DataArray>\n"
            bp += "</PointData>\n"
            bp += "</Piece>\n"
            bp += "</UnstructuredGrid>\n"
            bp += "</VTKFile>\n"
            fnp = "%s/%s_%06d.vtu" % (self.dire, self.fnkey+'_pw', self.idxout)
            fip = open(normpath(fnp), 'w')
            fip.write(a)
            fip.write(bp)
            fip.close()
            self.fnames_pw.append(normpath(fnp))

    def gen_collections_file(self, fnkey, times, fnames):
        """
        Generate collections file
        =========================
        INPUT:
            fnkey  : file name key => will generate 'fnkey.pvd'
            times  : list with output (real) times
            fnames : list with filenames corresponding to each output time
        STORED:
            None
        RETURNS:
            None
        """
        l  = "<?xml version=\"1.0\" ?>\n"
        l += "<VTKFile type=\"Collection\" version=\"0.1\" byte_order=\"LittleEndian\">\n"
        l += "<Collection>\n"
        for t, fn in zip(times, fnames):
            l += "<DataSet timestep=\"%g\" file=\"%s\" />\n" % (t, fn)
        l += "</Collection>\n"
        l += "</VTKFile>"
        f = open(normpath(fnkey+'.pvd'), 'w')
        f.write(l)
        f.close()
