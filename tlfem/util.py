# Copyright 2012 Dorival de Moraes Pedroso. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

from __future__ import print_function # for Python 3

def GetITout(all_output_times, time_stations_out, tol=1.0e-8):
    """
    Get indices and output times
    ============================
    INPUT:
        all_output_times  : array with all output times. ex: [0,0.1,0.2,0.22,0.3,0.4]
        time_stations_out : time stations for output: ex: [0,0.2,0.4]  # must be sorted ascending
        tol               : tolerance to compare times
    RETURNS:
        iout : indices of times in all_output_times
        tout : times corresponding to iout
    """
    I, T        = [], []                          # indices and times for output
    lower_index = 0                               # lower index in all_output_times
    len_aotimes = len(all_output_times)           # length of a_o_times
    for t in time_stations_out:                   # for all requested output times
        for k in range(lower_index, len_aotimes): # search within a_o_times
            if abs(t-all_output_times[k]) < tol:  # found near match
                lower_index += 1                  # update index
                I.append(k)                       # add index to iout
                T.append(all_output_times[k])     # add time to tout
                break                             # stop searching for this 't'
            if all_output_times[k] > t:           # failed to search for 't'
                lower_index = k                   # update idx to start from here on
                break                             # skip this 't' and try the next one
    return zip(I, T)                              # results

def PrintV(name, V, nf='%12.5f', Tol=1.0e-14):
    """
    Pretty-print a vector
    =====================
    """
    print(name)
    lin = ''
    for i in range(len(V)):
        if abs(V[i])<Tol: lin += nf % 0
        else:             lin += nf % V[i]
    lin += '\n'
    print(lin)

def PrintM(name, A, nf='%12.5f', Tol=1.0e-14):
    """
    Pretty-print a matrix
    =====================
    """
    print(name)
    m = A.shape[0] # number of rows
    n = A.shape[1] # number of columns
    lin = ''       # empty string
    for i in range(m):
        for j in range(n):
            if abs(A[i,j])<Tol: lin += nf % 0
            else:               lin += nf % A[i,j]
        lin += '\n'
    print(lin)

def PrintDiff(fmt, a, b, tol):
    """
    Print colored difference between a and b
    ========================================
    """
    if abs(a-b) < tol: print('[1;32m' + fmt % abs(a-b) + '[0m')
    else:              print('[1;31m' + fmt % abs(a-b) + '[0m')

def CompareArrays(a, b, tol=1.0e-12, table=False, namea='a', nameb='b', numformat='%17.10e', spaces=17, stride=0):
    """
    Compare two arrays
    ==================
    """
    if table:
        sfmt = '%%%ds' % spaces
        print('='*(spaces*3+4+5))
        print(sfmt % namea, sfmt % nameb, '[1;37m', sfmt % 'diff', '[0m')
        print('-'*(spaces*3+4+5))
    max_diff = 0.0
    for i in range(len(a)):
        diff = abs(a[i]-b[i])
        if diff > max_diff: max_diff = diff
        if table:
            clr = '[1;31m' if diff > tol else '[1;32m'
            print('%4d'%(i+stride), numformat % a[i], numformat % b[i], clr, numformat % diff, '[0m')
    if table: print('='*(spaces*3+4+5))
    if max_diff < tol: print('max difference = [1;32m%20.15e[0m' % max_diff)
    else:              print('max difference = [1;31m%20.15e[0m' % max_diff)

# test
if __name__=="__main__":
    prob = 1
    if prob==0:
        ITout = get_itout([0., 0.1, 0.15, 0.2, 0.23, 0.23, 0.23, 0.3, 0.8, 0.99],
                          [0., 0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9, 1.])
        print(ITout)
    if prob==1:
        ITout = get_itout([0.0, 0.1, 0.2, 0.30000000000000004, 0.4, 0.5, 0.6, 0.7, 0.7999999999999999, 0.8999999999999999, 0.9999999999999999],
                          [0.1, 0.1, 0.2, 0.5, 1.0])
        print(ITout)
