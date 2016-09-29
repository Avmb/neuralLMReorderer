import sys
import cPickle
import numpy as np
import scipy.sparse as sp

import theano
import theano.tensor as tt
import theano.sparse as ts
import theano.ifelse as tif
import theano.printing as tp

from sklearn import base
from sklearn import utils

from adadelta import ADADeltaUpdates

# Setup

rng = utils.check_random_state(0)
w_in0 = rng.uniform(-0.1, 0.1, (10, 200))
w_rec0 = rng.uniform(-0.1, 0.1, (200, 200))
w_out0 = rng.uniform(-0.1, 0.1, (200,))

W_in = theano.shared(w_in0, name = 'w_in', borrow=False)
W_rec = theano.shared(w_rec0, name = 'w_rec', borrow=False)
W_out = theano.shared(w_out0, name = 'w_out', borrow=False)

def reset_weights():
	W_in.set_value(w_in0, borrow=False)
	W_rec.set_value(w_rec0, borrow=False)
	W_out.set_value(w_out0, borrow=False)

X = tt.matrix(name = 'X', dtype=theano.config.floatX)
Y = tt.vector(name = 'y')

x = rng.uniform(-1.0, 1.0, (6, 10))
y = rng.random_integers(0, 1, (6,)).astype(np.float)

S_0 = tt.zeros(200, dtype=theano.config.floatX)
def recurrence(X_t, S_tm1):
    S_t = tt.dot(X_t, W_in) + tt.dot(S_tm1, W_rec)
    Z_t = tt.dot(S_t, W_out)
    return Z_t, S_t

recurrence_seq, _ = theano.scan(recurrence, sequences=[X], outputs_info=[None, S_0])
Z, S = recurrence_seq

ZP = tt.nnet.softmax(Z)
Loss = tt.nnet.binary_crossentropy(ZP, Y)
Mean_loss = Loss.mean()
Mean_loss_proc = theano.function([X, Y], Mean_loss)

# Standard training

gr_Mean_loss_wrt_W_in = theano.grad(Mean_loss, W_in)
gr_Mean_loss_wrt_W_rec = theano.grad(Mean_loss, W_rec)
gr_Mean_loss_wrt_W_out = theano.grad(Mean_loss, W_out)

alpha = 0.1
updates = [
	(W_in, W_in - alpha * gr_Mean_loss_wrt_W_in),
	(W_rec, W_rec - alpha * gr_Mean_loss_wrt_W_rec),
	(W_out, W_out - alpha * gr_Mean_loss_wrt_W_out)
]

Train_proc = theano.function([X, Y], Mean_loss, updates=updates)

def train(max_epochs = 100, min_loss = 0.01):
	for i in xrange(max_epochs):
		train_loss = Train_proc(x, y)
		print i, train_loss
		if train_loss <= min_loss:
			break

# Custom BTTP training

#WX = tt.dot(X, W_in)

def forward_pass(X_t, Y_t, S_tm1):
	S_t = tt.dot(X_t, W_in) + tt.dot(S_tm1, W_rec)
	Z_t = tt.dot(S_t, W_out)
	ZP_t = tt.nnet.sigmoid(Z_t)
	Loss_t = tt.nnet.binary_crossentropy(ZP_t, Y_t)
	
	gr_Loss_t_wrt_W_out_t = theano.grad(Loss_t, W_out)
	gr_Loss_t_wrt_S_t = theano.grad(Loss_t, S_t)
	
	return Loss_t, gr_Loss_t_wrt_W_out_t, gr_Loss_t_wrt_S_t, S_t
	
Forward_pass_seq, _ = theano.scan(forward_pass, sequences=[X, Y], outputs_info=[None, None, None, S_0])
Forward_pass_seq_proc = theano.function([X, Y], Forward_pass_seq)

Train_losses = Forward_pass_seq[0]
Train_gr_Loss_t_wrt_W_out = Forward_pass_seq[1]
Train_gr_Loss_t_wrt_S = Forward_pass_seq[2]
Train_states = Forward_pass_seq[3]

gr_bp_wrt_S_Fin = tt.zeros(200, dtype=theano.config.floatX)
def backwards_pass(X_t, S_tm1, gr_Loss_t_wrt_S_t, gr_bp_wrt_S_t):	# Buggy?
	S_t = tt.dot(X_t, W_in) + tt.dot(S_tm1, W_rec)
	
	# Local loss gradient components
	#l_gr_wrt_W_rec_t = theano.grad(None, W_rec, known_grads = {S_t : gr_Loss_t_wrt_S_t})
	#l_gr_wrt_W_in_t = theano.grad(None, W_in, known_grads = {S_t : gr_Loss_t_wrt_S_t})
	#l_gr_wrt_S_tm1_t = theano.grad(None, S_tm1, known_grads = {S_t : gr_Loss_t_wrt_S_t})
	
	# Backpropagated gradient components
	#bp_gr_wrt_W_rec_t = theano.grad(None, W_rec, known_grads = {S_t : gr_bp_wrt_S_t})
	#bp_gr_wrt_W_in_t = theano.grad(None, W_in, known_grads = {S_t : gr_bp_wrt_S_t})
	#bp_gr_wrt_S_tm1_t = theano.grad(None, S_tm1, known_grads = {S_t : gr_bp_wrt_S_t})
	
	# Cumulative gradient
	#gr_wrt_W_rec_t = l_gr_wrt_W_rec_t + bp_gr_wrt_W_rec_t
	#gr_wrt_W_in_t = l_gr_wrt_W_in_t + bp_gr_wrt_W_in_t
	#gr_wrt_S_tm1 = l_gr_wrt_S_tm1_t + bp_gr_wrt_S_tm1_t
	
	gr_wrt_W_rec_t = theano.grad(None, W_rec, known_grads = {S_t : gr_Loss_t_wrt_S_t + gr_bp_wrt_S_t})
	gr_wrt_W_in_t = theano.grad(None, W_in, known_grads = {S_t : gr_Loss_t_wrt_S_t + gr_bp_wrt_S_t})
	
	# Backpropagate
	gr_wrt_S_tm1 = theano.grad(None, S_tm1, known_grads = {S_t : gr_Loss_t_wrt_S_t + gr_bp_wrt_S_t})
	
	return gr_wrt_W_rec_t, gr_wrt_W_in_t, gr_wrt_S_tm1

Backwards_pass_seq, _ = theano.scan(backwards_pass, sequences=[X, Train_states, Train_gr_Loss_t_wrt_S], outputs_info=[None, None, gr_bp_wrt_S_Fin])
Backwards_pass_seq_proc = theano.function([X, Y], Backwards_pass_seq)

Train_acc_gr_wrt_W_rec = Backwards_pass_seq[0]
Train_gr_wrt_W_in = Backwards_pass_seq[1]

Train_mean_gr_wrt_W_out = Train_gr_Loss_t_wrt_W_out.mean(axis = 0)
Train_mean_gr_wrt_W_in = Train_gr_wrt_W_in.mean(axis = 0)
Train_mean_gr_wrt_W_rec = Train_acc_gr_wrt_W_rec[0] / X.shape[0]

cbptt_updates = [
	#(W_in, W_in - alpha * gr_Mean_loss_wrt_W_in),
	(W_in, W_in - alpha * Train_mean_gr_wrt_W_in),
	
	#(W_rec, W_rec - alpha * gr_Mean_loss_wrt_W_rec),
	(W_rec, W_rec - alpha * Train_mean_gr_wrt_W_rec),
	
	#(W_out, W_out - alpha * gr_Mean_loss_wrt_W_out)
	(W_out, W_out - alpha * Train_mean_gr_wrt_W_out)
]

Train_cbptt_proc = theano.function([X, Y], Mean_loss, updates=cbptt_updates)

def train_cbptt(max_epochs = 100, min_loss = 0.01):
	for i in xrange(max_epochs):
		train_loss = Train_cbptt_proc(x, y)
		print i, train_loss
		if train_loss <= min_loss:
			break
