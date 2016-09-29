import sys
from collections import OrderedDict
import numpy as np
import scipy.sparse as sp
import theano
import theano.tensor as tt
import theano.sparse as ts
import theano.ifelse as tif
import theano.printing as tp

from sklearn import base
from sklearn import utils

import spnonzeroop as tspnz

class ADADelta(base.BaseEstimator):
	def __init__(self, additive_epsilon = 1e-06, inv_decay_rate = 0.95, validation_frequency = 60, validation_patience = 10, dtype = None, verbose = 0):
		self.additive_epsilon = additive_epsilon
		self.inv_decay_rate = inv_decay_rate
		self.validation_frequency = validation_frequency
		self.validation_patience = validation_patience
		self.dtype = None
		self.verbose = verbose
	
	def fit_setup(self, loss = None, inputs = [], params_list = [], validation_set = None):
		self.loss_ = loss
		self.inputs_ = inputs
		self.params_list_ = params_list
		self.validation_set_ = validation_set
		#self.input_features_constructor_ = input_features_constructor if (input_features_constructor != None) else tt.matrix
		self.dtype_ = self.dtype if (self.dtype != None) else theano.config.floatX
		self.n_epochs_ = 0
		self.n_minibatches_ = 0
		self.validation_loss_value_ = np.inf
		self.validation_count_ = 0
		self.cur_patience_ = self.validation_patience
		self.done_ = False
		self.make_model()
		self.best_value_ = np.inf
		self.best_params_ = self.get_model_params()
	
	def fit_new_epoch(self):
		self.n_epochs_ += 1
		if self.verbose >= 1:
			print "Epoch:", self.n_epochs_
	
	def fit_minibatch(self, X):
		self.n_minibatches_ += 1
		minibatch_train_loss = self.train_minibatch_proc_(*X)
		if self.verbose >= 3:
			print "Minibatch: %s, loss: %s" % (self.n_minibatches_, minibatch_train_loss)
		if (self.n_minibatches_ % self.validation_frequency) == 0:
			self.validate_and_check_termination()
	
	def fit_end(self):
		self.set_model_params(self.best_params_)
	
	def fit_stream(self, stream_generator, **fit_params):
		self.fit_setup(**fit_params)
		for stream in stream_generator:
			self.fit_new_epoch()
			for X in stream:
				if self.done_:
					if self.verbose >= 1:
						print "Done."
					self.fit_end()
					return self
				self.fit_minibatch(X)
				#raw_input("Press Enter to continue...")
		if self.verbose >= 1:
			print "No more data, finishing."
		self.fit_end()
	
	def fit(self, X, **fit_params):
		def default_stream_generator(X):
			while True:
				yield X
		return self.fit_stream(default_stream_generator(X), **fit_params)
	
	def make_model(self):
		self.n_params_ = len(self.params_list_)
		self.params_ = []
		updates_dict = OrderedDict()
		for param_dict in self.params_list_:
			w = ADADeltaParam(param_dict)
			w.nabla = theano.grad(self.loss_, w.var) if w.nabla == None else w.nabla
			self.params_.append(w)
			if w.sparse_updates:
				if isinstance(w.nabla.type, ts.SparseType):
					#nz_indices_rows = ts.csm_indices(w.nabla)
					#nabla_indptr = ts.csm_indptr(w.nabla)
					#col_fn = lambda I, O_tm1: tif.ifelse(tp.Print('I')(I) < nabla_indptr[O_tm1+1], O_tm1, O_tm1 + 1)
					#nz_indices_cols, _ = theano.scan(col_fn, sequences=[tt.arange(tp.Print('w.nabla.size')(w.nabla.size))], outputs_info=[dict(initial=tt.constant(0))])
					nz_indices_rows, nz_indices_cols = tspnz.sp_nonzero(w.nabla)
					nz_indices = nz_indices_rows, nz_indices_cols
					nabla_nonzeros = ts.csm_data(w.nabla)
					param_to_update = w.var[nz_indices]
					nabla_to_update = w.nabla_sqr_acc[nz_indices]
					delta_to_update = w.delta_sqr_acc[nz_indices]
					
				else:
					nz_indices = tt.nonzero(w.nabla)
					nabla_nonzeros = w.nabla[nz_indices].flatten()
					param_to_update = w.var[nz_indices]
					nabla_to_update = w.nabla_sqr_acc[nz_indices]
					delta_to_update = w.delta_sqr_acc[nz_indices]
				
				rms_delta = tt.sqrt(delta_to_update.flatten() + self.additive_epsilon)
				nabla_sqr_acc_tp1 = self.inv_decay_rate * nabla_to_update.flatten() + (1.0 - self.inv_decay_rate) * tt.sqr(nabla_nonzeros)
				rms_nabla_tp1 = tt.sqrt(nabla_sqr_acc_tp1 + self.additive_epsilon)
				neg_delta = nabla_nonzeros * rms_delta / rms_nabla_tp1
				delta_sqr_acc_tp1 = self.inv_decay_rate * delta_to_update.flatten() + (1.0 - self.inv_decay_rate) * tt.sqr(neg_delta)
				
				updates_dict[w.var] = tt.inc_subtensor(param_to_update, -(neg_delta.reshape(param_to_update.shape)))
				updates_dict[w.nabla_sqr_acc] = tt.set_subtensor(nabla_to_update, nabla_sqr_acc_tp1.reshape(nabla_to_update.shape))
				updates_dict[w.delta_sqr_acc] = tt.set_subtensor(delta_to_update, delta_sqr_acc_tp1.reshape(delta_to_update.shape))
			else:
				rms_delta = tt.sqrt(w.delta_sqr_acc + self.additive_epsilon)
				nabla_sqr_acc_tp1 = self.inv_decay_rate * w.nabla_sqr_acc + (1.0 - self.inv_decay_rate) * tt.sqr(w.nabla)
				rms_nabla_tp1 = tt.sqrt(nabla_sqr_acc_tp1 + self.additive_epsilon)
				neg_delta = w.nabla * rms_delta / rms_nabla_tp1
				delta_sqr_acc_tp1 = self.inv_decay_rate * w.delta_sqr_acc + (1.0 - self.inv_decay_rate) * tt.sqr(neg_delta)
				updates_dict[w.var] = w.var - neg_delta
				updates_dict[w.nabla_sqr_acc] = nabla_sqr_acc_tp1
				updates_dict[w.delta_sqr_acc] = delta_sqr_acc_tp1
		self.train_minibatch_proc_ = theano.function(self.inputs_, self.loss_, updates=updates_dict)
		self.loss_proc_ = theano.function(self.inputs_, self.loss_)
	
	def validate_and_check_termination(self):
		self.validation_count_ += 1
		self.validation_loss_value_ = self.loss_proc_(*self.validation_set_)
		if self.validation_loss_value_ < self.best_value_:
			new_best = True
			self.done_ = False
			self.best_value_ = self.validation_loss_value_
			self.best_params_ = self.get_model_params()
			self.cur_patience_ = self.validation_patience
		else:
			new_best = False
			self.cur_patience_ -= 1
			self.done_ = (self.cur_patience_ <= 0)
		
		if self.verbose >= 2:
			print "Validation loss:", self.validation_loss_value_, "patience:", self.cur_patience_, "new best:", new_best
	
	def get_model_params(self, borrow=False):
		return [w.var.get_value(borrow=borrow) for w in self.params_]
	
	def set_model_params(self, params_val_list, borrow=False):
		for i, w in enumerate(self.params_):
			w.var.set_value(params_val_list[i], borrow=borrow)
		
class ADADeltaParam(object):
	def __init__(self, param_dict):
		self.var = param_dict['var']
		self.sparse_updates = param_dict.get('sparse_updates', False)
		self.nabla = param_dict.get('nabla', None)
		self.l2_regularization = param_dict.get('l2_regularization', 0.0)
		
		val = self.var.get_value(borrow=True)
		self.nabla_sqr_acc = theano.shared(np.zeros_like(val), name=self.var.name + '_nabla_sqr_acc', borrow=True)
		self.delta_sqr_acc = theano.shared(np.zeros_like(val), name=self.var.name + '_delta_sqr_acc', borrow=True)

def makeADADeltaParamDict(params_list):
	rv = {}
	for param in params_list:
		rv[param['var']] = param
	return rv


class ADADeltaUpdates(base.BaseEstimator):
	def __init__(self, additive_epsilon = 1e-06, inv_decay_rate = 0.95, clip_nabla = None, dtype = None, verbose = 0):
		self.additive_epsilon = additive_epsilon
		self.inv_decay_rate = inv_decay_rate
		self.clip_nabla = clip_nabla
		self.dtype = None
		self.verbose = verbose
	
	def make_updates(self, loss, params_list = []):
		def clip(nabla):
			if self.clip_nabla == None:
				return nabla
			return tt.minimum(tt.maximum(nabla, self.clip_nabla[0]), self.clip_nabla[1])
		
		self.loss_ = loss
		self.params_list_ = params_list
		self.dtype_ = self.dtype if (self.dtype != None) else theano.config.floatX
		self.n_params_ = len(self.params_list_)
		self.params_ = []
		updates_dict = OrderedDict()
		for param_dict in self.params_list_:
			w = ADADeltaParam(param_dict)
			if self.verbose > 2:
				print >> sys.stderr, 'Making updates for:', w.var
			w.nabla = theano.grad(self.loss_, w.var) if w.nabla == None else w.nabla
			self.params_.append(w)
			if w.sparse_updates:
				if isinstance(w.nabla.type, ts.SparseType):
					print >> sys.stderr, "(sparse with sparse updates)"
					nz_indices_rows, nz_indices_cols = tspnz.sp_nonzero(w.nabla)
					nz_indices = nz_indices_rows, nz_indices_cols
					nabla_nonzeros = clip(ts.csm_data(w.nabla))
					param_to_update = w.var[nz_indices]
					nabla_to_update = w.nabla_sqr_acc[nz_indices]
					delta_to_update = w.delta_sqr_acc[nz_indices]
					
				else:
					nz_indices = tt.nonzero(w.nabla)
					nabla_nonzeros = clip(w.nabla[nz_indices].flatten())
					param_to_update = w.var[nz_indices]
					nabla_to_update = w.nabla_sqr_acc[nz_indices]
					delta_to_update = w.delta_sqr_acc[nz_indices]
				
				rcwnabla_nonzeros = nabla_nonzeros if (w.l2_regularization == 0.0) else nabla_nonzeros + w.l2_regularization * param_to_update
				rms_delta = tt.sqrt(delta_to_update.flatten() + self.additive_epsilon)
				nabla_sqr_acc_tp1 = self.inv_decay_rate * nabla_to_update.flatten() + (1.0 - self.inv_decay_rate) * tt.sqr(rcwnabla_nonzeros)
				rms_nabla_tp1 = tt.sqrt(nabla_sqr_acc_tp1 + self.additive_epsilon)
				neg_delta = rcwnabla_nonzeros * rms_delta / rms_nabla_tp1
				delta_sqr_acc_tp1 = self.inv_decay_rate * delta_to_update.flatten() + (1.0 - self.inv_decay_rate) * tt.sqr(neg_delta)
				
				updates_dict[w.var] = tt.inc_subtensor(param_to_update, -(neg_delta.reshape(param_to_update.shape)))
				updates_dict[w.nabla_sqr_acc] = tt.set_subtensor(nabla_to_update, nabla_sqr_acc_tp1.reshape(nabla_to_update.shape))
				updates_dict[w.delta_sqr_acc] = tt.set_subtensor(delta_to_update, delta_sqr_acc_tp1.reshape(delta_to_update.shape))
			else:
				cwnabla = clip(w.nabla)
				rcwnabla = cwnabla if (w.l2_regularization == 0.0) else cwnabla + w.l2_regularization * w.var
				rms_delta = tt.sqrt(w.delta_sqr_acc + self.additive_epsilon)
				nabla_sqr_acc_tp1 = self.inv_decay_rate * w.nabla_sqr_acc + (1.0 - self.inv_decay_rate) * tt.sqr(rcwnabla)
				rms_nabla_tp1 = tt.sqrt(nabla_sqr_acc_tp1 + self.additive_epsilon)
				neg_delta = rcwnabla * rms_delta / rms_nabla_tp1
				delta_sqr_acc_tp1 = self.inv_decay_rate * w.delta_sqr_acc + (1.0 - self.inv_decay_rate) * tt.sqr(neg_delta)
				updates_dict[w.var] = w.var - neg_delta
				updates_dict[w.nabla_sqr_acc] = nabla_sqr_acc_tp1
				updates_dict[w.delta_sqr_acc] = delta_sqr_acc_tp1
		self.updates_ = updates_dict
		if self.verbose > 2:
			print >> sys.stderr, '(No more parameters)'

	def get_model_params(self, borrow=False):
		return [w.var.get_value(borrow=borrow) for w in self.params_]
	
	def set_model_params(self, params_val_list, borrow=False):
		for i, w in enumerate(self.params_):
			w.var.set_value(params_val_list[i], borrow=borrow)
