#__author__ = 'moser'
## coding=utf-8
#"""
#In SDC there are two different possibilities to
#compute the quadrature_matrix.
#
#.. math::
#    Su & \approx \int_{t_i}^{t_{i+1}} u(t) dt \mathbf{e}_i\\
#    Qu & \approx \int_{t_0}^{t_i} u(t) dt \mathbf{e}_i
#
#This test suite points out the difference in calculating
#.. math::
#    \int_{t_m}^{t_m+1} u(t) dt \mbox{ or } \int_{t_0}^{t_m} u(t) dt
#
#by using the S or the Q matrix.
#
#.. moduleauthor:: Torbjörn Klatt <t.klatt@fz-juelich.de>
#.. moduleauthor:: Dieter Moser <d.moser@fz-juelich.de>
#"""
#
##import unittest
##from nose.tools import *
#import numpy as np
##import pySDC.globals as Config
#
#from pySDC.integrate.gauss import Gauss
#
## define some usual functions for the integration
#
#def runge_glocke(x):
#    return 1/(1+x**2)
#def trig_polynom(x):
#    return np.sin(5*x)+np.cos(0.5*x)+np.sin(2*x)
#def generic_polynom(x):
#    return x**5+x*0.5+7*x**3+15
#
## Integrationsgrenzen
#
#a = -1
#b = 1
#
## Aufsetzen der Testmatrix
#testNumPoints = [3, 5, 7, 13, 31,111]
#testMethods = ["lobatto"]
#testFunctions = [runge_glocke, trig_polynom, generic_polynom]
#
#
##def compare_arrays(arr1, arr2):
##    assert_equal(len(arr1), len(arr2),
##                 "Length of the two arrays not equal: {:d} != {:d}"
##                 .format(len(arr1), len(arr2)))
##    for i in range(1, len(arr1)):
##        assert_almost_equals(arr1[i], arr2[i],
##                             msg="{:d}. element not equal:".format(i) +
##                                 " arr1[{:d}]={:f} != {:f}=arr2[{:d}]"
##                                 .format(i, arr1[i], arr2[i], i),
##                             places=None, delta=Config.PRECISION)
##
##def compare_ndarrays(arr1, arr2):
##    assert_equal(arr1.size, arr2.size,
##                 "Length of the two arrays not equal: {:d} != {:d}"
##                 .format(len(arr1), len(arr2)))
##    for i in range(1, arr1.size):
##        assert_almost_equals(arr1[i], arr2[i],
##                             msg="{:d}. element not equal:".format(i) +
##                                 " arr1[{:d}]={:f} != {:f}=arr2[{:d}]"
##                                 .format(i, arr1[i], arr2[i], i),
##                             places=None, delta=Config.PRECISION)
#
#def construct_s_matrix_matching_ends(nodes):
#    """
#    """
#    npoints=len(nodes)
#    smat = np.zeros((npoints,npoints), dtype=float)
#    smat[0,:]=np.zeros(npoints)
#    for i in range(1,npoints):
#        smat[i,:]= Gauss.compute_weights(nodes,nodes[i-1],nodes[i])
#
#    return smat
#
#def construct_q_matrix_matching_ends(nodes):
#    """
#    """
#    npoints=len(nodes)
#    qmat = np.zeros((npoints,npoints), dtype=float)
#    qmat[0,:]=np.zeros(npoints)
#    for i in range(1,npoints):
#        qmat[i,:]= Gauss.compute_weights(nodes,nodes[0],nodes[i])
#
#    return qmat
#
## Waiting for Weighting and Nodes Klass
#def construct_s_matrix_non_matching_ends(nodes):
#    """
#    """
#    npoints=len(nodes)
#    smat = np.zeros((npoints,npoints), dtype=float)
#    smat[0,:]=np.zeros(npoints)
#    for i in range(1,npoints):
#        smat[i,:]= Gauss.compute_weights(nodes,i-1,i)
#
#    return smat
#
## Waiting for Weighting and Nodes Klass
#def construct_q_matrix_non_matching_ends(nodes):
#    """
#    """
#    npoints=len(nodes)
#    qmat = np.zeros((npoints,npoints), dtype=float)
#    qmat[0,:]=np.zeros(npoints)
#    for i in range(1,npoints):
#        qmat[i,:]= Gauss.compute_weights(nodes,0,i)
#
#    return qmat
#
#
## integrating (t_m,t_{m+1})
## interval 1    :   (t_0,t_1)
##          2        (t_1,t_2)
##           ...
#
#def compare_s_vs_q_between_sub_steps(number_nodes,func):
#    nodes = Gauss.lobatto_nodes(number_nodes)
#    x = func(nodes)
#    q_mat = construct_q_matrix_matching_ends(nodes)
#    s_mat = construct_s_matrix_matching_ends(nodes)
#    s_result = np.dot(s_mat,x)
#    q_result = np.zeros(s_result.size)
#    q_pre_result = np.dot(q_mat,x)
#    q_result[0] = q_pre_result[0]
#    for i in range(1,q_result.size):
#        q_result[i] = q_pre_result[i] - q_pre_result[i-1]
#
#    #compare_ndarrays(q_result,s_result)
#    print("Between sub steps")
#    print("Number of nodes: ",number_nodes,"Used function: ", func.__name__)
#    print(s_result-q_result)
#
#def test_quadrature_between_sub_steps():
#    for nnodes in testNumPoints:
#        for funcs in testFunctions:
#            yield compare_s_vs_q_between_sub_steps, nnodes, funcs
#
#
## integrating (t_0,t_m)
#
#def compare_s_vs_q_until_sub_step(number_nodes,func):
#    nodes = Gauss.lobatto_nodes(number_nodes)
#    x = func(nodes)
#    q_mat = construct_q_matrix_matching_ends(nodes)
#    s_mat = construct_s_matrix_matching_ends(nodes)
#    q_result = np.dot(q_mat,x)
#    s_result = np.zeros(q_result.size)
#    s_pre_result = np.dot(s_mat,x)
#    s_result[0] = s_pre_result[0]
#    my_sum=0.0
#    for i in range(1,q_result.size):
#        my_sum=my_sum+s_pre_result[i]
#        s_result[i] = my_sum
#
#    #compare_ndarrays(q_result,s_result)
#    print("Until sub step")
#    print("Number of nodes: ",number_nodes,"Used function: ", func.__name__)
#    print(s_result-q_result)
#
#def test_quadrature_until_sub_step():
#    for nnodes in testNumPoints:
#        for funcs in testFunctions:
#            yield compare_s_vs_q_until_sub_step, nnodes, funcs
#
#if __name__ == "__main__":
#    #unittest.main()
#    for nnodes in testNumPoints:
#        for funcs in testFunctions:
#            compare_s_vs_q_between_sub_steps(nnodes, funcs  )
#            compare_s_vs_q_until_sub_step(nnodes, funcs  )
