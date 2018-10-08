import numpy 
from scipy.integrate import complex_ode,ode
import matplotlib.pyplot as plt 
from keras.layers import Lambda, Input, Dense
from keras.models import Model
from keras.datasets import mnist
from keras.losses import mse, binary_crossentropy
from keras.utils import plot_model
from keras import backend as K
import argparse
import os

#Generate time series using the given Riccati form equation of motion from the paper
def eqm(t,z,args):
	x = args[0]
	b = args[1]
	y = 1j*b
	dzdt = -1j*(x-1j*y)/2*(z**2-((x+y)/(x-1j*y)))
	return dzdt
	
r = ode(eqm).set_integrator('zvode',method='bdf',with_jacobian=False)
r.set_initial_value(0,0).set_f_params([1,0.5])
t1 = 100
dt=0.5
Time_array = numpy.zeros(t1)
results = numpy.zeros(t1,dtype=numpy.complex_)
Time_array[0] = 0
counter = 0
while r.successful() and r.t<t1:
	r.integrate(r.t+dt)
	counter+=1
	#Time_array[counter] = r.t
	#results[counter] = r.y
	print ('%10.5f %10.5f %10.5f' %(r.t,(r.y).real,(r.y).imag))



#Machine learning code starts from here, I refer to the script from 
#https://github.com/keras-team/keras/blob/master/examples/variational_autoencoder.py
'''batch_size = 
original_dim = 
latent_dim = 
intermediate_dim = 
epochs = 

x = Input(batch_shape=(batch_size, original_dim))
def sampling(args)
	z_mean, z_log_var = args
	batch = K.shape(z_mean)[0]
	dim = K.int_shape(z_mean)[1]


