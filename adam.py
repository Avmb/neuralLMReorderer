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

def scalarAsType(x, t):
	return np.asarray(x, t)

class AdamParam(object):
	def __init__(self, param_dict):
		self.var = param_dict['var']
		self.sparse_updates = param_dict.get('sparse_updates', False)
		self.nabla = param_dict.get('nabla', None)
		self.l2_regularization = param_dict.get('l2_regularization', 0.0)
		
		val = self.var.get_value(borrow=True)
		self.m = theano.shared(np.zeros_like(val), name=self.var.name + '_m', borrow=True)
		self.v = theano.shared(np.zeros_like(val), name=self.var.name + '_v', borrow=True)

def makeAdamParamDict(params_list):
	rv = {}
	for param in params_list:
		rv[param['var']] = param
	return rv

class AdamUpdates(base.BaseEstimator):
	def __init__(self, alpha=0.0002, beta1 = 0.1, beta2 = 0.001, epsilon = 1e-08, decay_lambda=1e-08, clip_nabla = None, dtype = None, verbose = 0):
		self.alpha = alpha
		self.beta1 = beta1
		self.beta2 = beta2
		self.epsilon = epsilon
		self.decay_lambda = decay_lambda
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
		
		ombeta1 = 1.0 - self.beta1
		ombeta2 = 1.0 - self.beta2
		
		ombeta1_t = theano.shared(scalarAsType(ombeta1, self.dtype_), name="ombeta1_t")
		ombeta2_t = theano.shared(scalarAsType(ombeta2, self.dtype_), name="ombeta2_t")
		lambda_t  = theano.shared(scalarAsType(1.0, self.dtype_), name="lambda_t")
		
		
		beta1_t = 1.0 - ombeta1 * lambda_t
		#beta1_t = theano.shared(scalarAsType(self.beta1, self.dtype_), name="ombeta1_t")
		#updates_dict[beta1_t] = 1.0 - (1.0 - self.decay_lambda) * (1.0 - beta1_t)
		
		updates_dict[ombeta1_t] = ombeta1_t * ombeta1
		updates_dict[ombeta2_t] = ombeta2_t * ombeta2
		updates_dict[lambda_t]  = lambda_t * self.decay_lambda	
		
		for param_dict in self.params_list_:
			w = AdamParam(param_dict)
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
					m_to_update = w.m[nz_indices]
					v_to_update = w.v[nz_indices]
					
				else:
					nz_indices = tt.nonzero(w.nabla)
					nabla_nonzeros = clip(w.nabla[nz_indices].flatten())
					param_to_update = w.var[nz_indices]
					m_to_update = w.m[nz_indices]
					v_to_update = w.v[nz_indices]
				
				rcwnabla_nonzeros = nabla_nonzeros if (w.l2_regularization == 0.0) else nabla_nonzeros + w.l2_regularization * param_to_update
				m_tp1 = beta1_t * rcwnabla_nonzeros + (1.0 - beta1_t) * m_to_update.flatten()
				v_tp1 = self.beta2   * tt.sqr(rcwnabla_nonzeros) + ombeta2 * v_to_update.flatten()
				neg_delta = self.alpha * (tt.sqrt(1.0 - ombeta2_t) / (1.0 - ombeta1_t)) * m_tp1 / (tt.sqrt(v_tp1) + self.epsilon)
				updates_dict[w.var] = tt.inc_subtensor(param_to_update, -(neg_delta.reshape(param_to_update.shape)))
				updates_dict[w.m] = tt.set_subtensor(m_to_update, m_tp1.reshape(m_to_update.shape))
				updates_dict[w.v] = tt.set_subtensor(v_to_update, v_tp1.reshape(v_to_update.shape))
			else:
				cwnabla = clip(w.nabla)
				rcwnabla = cwnabla if (w.l2_regularization == 0.0) else cwnabla + w.l2_regularization * w.var
				m_tp1 = beta1_t * rcwnabla + (1.0 - beta1_t) * w.m
				v_tp1 = self.beta2   * tt.sqr(rcwnabla) + ombeta2 * w.v
				neg_delta = self.alpha * (tt.sqrt(1.0 - ombeta2_t) / (1.0 - ombeta1_t)) * m_tp1 / (tt.sqrt(v_tp1) + self.epsilon)
				#neg_delta = self.alpha * m_tp1 / (tt.sqrt(v_tp1) + self.epsilon)
				updates_dict[w.var] = w.var - neg_delta
				updates_dict[w.m] = m_tp1
				updates_dict[w.v] = v_tp1
		self.updates_ = updates_dict
		if self.verbose > 2:
			print >> sys.stderr, '(No more parameters)'

	def get_model_params(self, borrow=False):
		return [w.var.get_value(borrow=borrow) for w in self.params_]
	
	def set_model_params(self, params_val_list, borrow=False):
		for i, w in enumerate(self.params_):
			w.var.set_value(params_val_list[i], borrow=borrow)
