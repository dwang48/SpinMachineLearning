import numpy 
from scipy.integrate import complex_ode,ode
import matplotlib.pyplot as plt 
from keras.layers import Lambda, Input, Dense, concatenate
from keras.models import Model
from keras.losses import mse, binary_crossentropy
from keras.utils import plot_model
from keras import backend as K
import argparse
import os

#Generate time series using the given Riccati form equation of motion from the paper
def eqm(t,z,args):
	b = args
	y = 1j*b
	dzdt = -1j*(1-1j*y)/2*(z**2-((1+y)/(1-1j*y)))
	return dzdt

#generate 400 sets of data for different initial value for zeta(0) and 
a = numpy.concatenate((numpy.arange(0,10,1),numpy.arrange(10,1000,100)),axis=0)
b = numpy.concatenate((numpy.arange(0,1,0.1),numpy.arrange(1,100,10)),axis=0)
os.mkdir('beta_larger_than_1')
os.mkdir('beta_smaller_than_1')
import itertools
for element in itertools.product(a,b):
	r = ode(eqm).set_integrator('zvode',method='bdf',with_jacobian=False)
	r.set_initial_value(element[0],0).set_f_params(element[1])
	t1 = 1000
	dt= 0.1
	results = numpy.zeros((t1,2))
	counter = 0
	while r.successful() and r.t<(t1*dt-0.0001):
		r.integrate(r.t+dt)
		print ('%10.5f %10.5f %10.5f' %(r.t,(r.y).real,(r.y).imag)) 
		results[counter,0] = r.t
		results[counter,1] = (r.y).imag/(r.y).real
		counter += 1
		
	if (element[1] < 1):
		numpy.savetxt('beta_smaller_than_1/alpha_%d_beta_%.2f.txt'%(element[0],element[1]),results)
	else:
		numpy.savetxt('beta_larger_than_1/alpha_%d_beta_%.2f.txt'%(element[0],element[1]),results)		

