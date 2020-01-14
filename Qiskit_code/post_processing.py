from qiskit import *
import matplotlib.pyplot as plt
import numpy as np
from qiskit.quantum_info.operators import Operator
from qiskit.tools.monitor import job_monitor
import csv

def theoretical_angle(theta):
    return np.arctan(np.tan(theta)**2)

def post_processing_angle():
    data = np.genfromtxt('data_sim.csv', delimiter="\t",skip_header=1)
    data_physical = np.genfromtxt('data_physical.csv',delimiter="\t",skip_header=1)
    x = np.linspace(0,np.pi/2,25)
    x_physical = np.linspace(0,np.pi/2,5)

    for i,k in zip([0,1,2,4],['.','^','x','*']):
        i_rounds = [data[j] for j in range(i,len(data),8)]
        arctan_xz = [np.arctan2(j[2],j[4])/2 for j in i_rounds[:-2]] #factor of two since every angle is multiplied by 2 on the bloch sphere

        if i == 0:#these conditions just shift the result to another solution which fits the curve
            arctan_xz += [(np.arctan(j[2]/j[4]) + np.pi)/2 for j in i_rounds[-2:]]
        if i ==3:
            arctan_xz += [(np.arctan(j[2]/j[4]) + np.pi)/2 for j in i_rounds[-2:]]
        if i == 1:
            arctan_xz += [(np.arctan2(j[2],j[4]))/2 for j in i_rounds[-2:-1]] + [(np.arctan(j[2]/j[4])+np.pi)/2 for j in i_rounds[-1:]]
        if i ==4:
            arctan_xz += [(np.arctan2(j[2],j[4]))/2 for j in i_rounds[-2:]]
        if i ==2:
            arctan_xz += [(np.arctan2(j[2],j[4]))/2 for j in i_rounds[-2:]]


        plt.plot(x,arctan_xz,label = '${} \ rounds$'.format(i+1),linewidth = 0,marker = k)

    for i in range(3):
        i_rounds = [data_physical[j] for j in range(i,len(data_physical),3)]
        arccos_z = [np.arccos(j[4])/2 for j in i_rounds]
        arctan_xz = [np.arctan2(j[2],j[4])/2 for j in i_rounds]
        #plt.plot(x_physical,arccos_z,linewidth = 0, marker = 'o',markersize = 10,fillstyle = 'none',label = '${} \ rounds \ QC$'.format(i+1))
        plt.plot(x_physical,arctan_xz,linewidth = 0, marker = 'o',markersize = 10,fillstyle = 'none',label = '${} \ rounds \ QC$'.format(i+1))

    plt.ylabel('$arctan(<X>/<Z>)$')
    plt.xlabel('$\\theta \ (radians)$',labelpad=-2)

    positions = (0,np.pi/8,np.pi/4, 3*np.pi/8,np.pi/2)
    labels = ("$0$", "$\\frac{\\pi}{8}$", "$\\frac{\\pi}{4}$","$\\frac{3\\pi}{8}$","$\\frac{\\pi}{2}$")
    plt.xticks(positions, labels,fontsize = 15)
    plt.plot(x,[theoretical_angle(theta) for theta in x],label = '$arctan(tan^2(\\theta))$',color = 'k')
    plt.legend()
    plt.savefig('Perceptron_angle.pdf')
    plt.show()


def expectation_value(theta,measure_type):
    q_theta = theoretical_angle(theta)
    state = np.array([np.cos(q_theta),np.sin(q_theta)]).T

    if measure_type == 'x':
        X = np.array([[0,1],[1,0]])
        return state @ X @ state

    if measure_type == 'y':
        Y = np.array([[0,-1j],[1j,0]])
        return state @ Y @ state

    if measure_type == 'z':
        Z = np.array([[1,0],[0,-1]])
        return state @ Z @ state

def computed_fidelity(theta,computed_list):
    expected_list = [expectation_value(theta,m) for m in ['x','y','z'] ]
    return (abs(np.sum([(expected_list[i]-computed_list[i])**2 for i in range(3)])))

def theoretical_fidelity(theta,n_rounds):
    p_theta = np.cos(theta)**4 + np.sin(theta)**4
    return 1 - (1-p_theta)**n_rounds


def post_processing_fidelity(): #do for angle pi/4
    data = np.genfromtxt('data_sim.csv', delimiter="\t",skip_header=1)
    fidelity_rounds = [1-computed_fidelity(np.pi/4,data[96+i][2:]) for i in range(8)] #row with angle pi/4
    print(fidelity_rounds)
    #plt.plot(range(1,9),fidelity_rounds)
    #plt.plot(range(1,9),[theoretical_fidelity(data[192][0],n) for n in range(1,9)])
    #plt.show()

theory = [theoretical_fidelity(np.pi/4,n) for n in range(1,9)]
print(theory)
post_processing_fidelity()
