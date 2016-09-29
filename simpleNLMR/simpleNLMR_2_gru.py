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

from adadelta import ADADeltaUpdates, makeADADeltaParamDict
from adam import AdamUpdates
default_optimizer = ADADeltaUpdates

from Node import *

class SimpleNLMR_2(base.BaseEstimator):
	def __init__(self, state_dim = 100, initial_feature_scaling_factor = 0.1, initial_contraction_coefficient = 0.99, is_esn = False, bttp_truncation = -1, optimizer = None, optimizer_opts = {}, tr_l2_reg = 0.0, ta_l2_reg = 0.0, rf_l2_reg = 0.0, rec_l2_reg = 0.0, initial_state_l2_reg = 0.0, dtype = None, random_state = 0, verbose = 0):
		self.state_dim = state_dim
		self.initial_feature_scaling_factor = initial_feature_scaling_factor
		self.initial_contraction_coefficient = initial_contraction_coefficient
		self.is_esn = is_esn
		self.bttp_truncation = bttp_truncation
		self.optimizer = optimizer
		self.optimizer_opts = optimizer_opts
		self.tr_l2_reg = tr_l2_reg
		self.ta_l2_reg = ta_l2_reg
		self.rf_l2_reg = rf_l2_reg
		self.rec_l2_reg = rec_l2_reg
		self.initial_state_l2_reg = initial_state_l2_reg
		self.dtype = dtype
		self.random_state = random_state
		self.verbose = verbose
	
	def fit(self, X = None, y = None, transition_features_map = None, target_features_map = None, readout_features_map = None, enable_training = False):
		"""Initialize model, does not use any data."""
		self.transition_features_map_ = transition_features_map
		self.target_features_map_ = target_features_map
		self.readout_features_map_ = readout_features_map
		self.enable_training_ = enable_training
		self.make_model()
		self.initializeParams()
	
	def make_model(self):
		if self.verbose > 2:
			print >> sys.stderr, 'make_model begin'
		
		spgr = theano.sparse_grad
		#spgr = lambda x : x
		self.dtype_ = self.dtype if self.dtype else theano.config.floatX
		self.optimizer_ = self.optimizer if (self.optimizer != None) else default_optimizer
		self.rng_ = utils.check_random_state(self.random_state)
		
		# Parameters
		
		self.train_params_list_ = []
		
		b_to_hidden = np.zeros(3 * self.state_dim, dtype = self.dtype_)
		w_rec = np.zeros((3, self.state_dim, self.state_dim), dtype = self.dtype_)
		w_rf_to_out = np.zeros((self.readout_features_map_.get_sparse_dim(), 1), dtype = self.dtype_)
		b_to_out = np.zeros(1, dtype = self.dtype_)
		initial_state = np.zeros(self.state_dim, dtype = self.dtype_)
		
		self.B_to_hidden_ = theano.shared(b_to_hidden, name = 'b_to_hidden', borrow = True)
		self.W_rec_ = theano.shared(w_rec, name = 'w_rec', borrow = True)
		self.W_rf_to_out_ = theano.shared(w_rf_to_out, name = 'w_rf_to_out', borrow = True)
		self.B_to_out_ = theano.shared(b_to_out, name = 'b_to_out', borrow = True)
		self.Initial_state_ = theano.shared(initial_state, name = 'initial_state', borrow = True)
		
		self.train_params_list_.append({'var': self.W_rf_to_out_, 'sparse_updates': True, 'l2_regularization' : self.rf_l2_reg})
		self.train_params_list_.append({'var': self.B_to_out_, 'sparse_updates': False})

		if not self.is_esn:
			self.train_params_list_.append({'var': self.B_to_hidden_, 'sparse_updates': False})
			self.train_params_list_.append({'var': self.W_rec_, 'sparse_updates': False, 'l2_regularization' : self.rec_l2_reg})
			self.train_params_list_.append({'var': self.Initial_state_, 'sparse_updates': False, 'l2_regularization' : self.initial_state_l2_reg})

		self.tr_params_rid_ = []
		c = 0
		for k in xrange(len(self.transition_features_map_.inv_templates_)):
			nc = c + self.transition_features_map_.get_indices_dim()[k]
			self.tr_params_rid_.append(nc)
			c = nc
		w_tr_to_hidden = np.zeros((self.tr_params_rid_[-1], 3 * self.state_dim), dtype = self.dtype_)
		self.W_tr_to_hidden_ = theano.shared(w_tr_to_hidden, name = 'w_tr_to_hidden', borrow = True)
		if not self.is_esn:
			self.train_params_list_.append({'var': self.W_tr_to_hidden_, 'sparse_updates': True, 'l2_regularization' : self.tr_l2_reg})
		
		self.ta_params_rid_ = []
		c = 0
		for k in xrange(len(self.target_features_map_.inv_templates_)):
			nc = c + self.target_features_map_.get_indices_dim()[k]
			self.ta_params_rid_.append(nc)
			c = nc
		w_ta_to_hidden = np.zeros((self.ta_params_rid_[-1], self.state_dim), dtype = self.dtype_)
		self.W_ta_to_hidden_ = theano.shared(w_ta_to_hidden, name = 'w_ta_to_hidden', borrow = True)
		self.train_params_list_.append({'var': self.W_ta_to_hidden_, 'sparse_updates': True, 'l2_regularization' : self.ta_l2_reg})

				
		# Inputs
		self.X_state_ = tt.matrix(name = 'x_state', dtype=self.dtype_)
		self.X_in_tr_ = tt.imatrix(name = 'x_in_tr')
		self.X_in_ta_ = tt.imatrix(name = 'x_in_ta')
		self.X_in_re_ = ts.csr_matrix(name = 'x_in_re', dtype=self.dtype_)
		
		# State transition
		def get_transition_embeddings_graph(X_in_tr, pad_one = False):
			Xp = X_in_tr if not pad_one else tt.concatenate([X_in_tr, tt.zeros((1, X_in_tr.shape[1]), dtype=np.int64)])
			N = Xp.shape[0]
			Acc_initial_row = self.B_to_hidden_.reshape((1, -1))
			Acc = tt.repeat(Acc_initial_row, N, axis=0)
			c = 0
			for j, l in enumerate(self.tr_params_rid_):
				Embeddings = spgr(self.W_tr_to_hidden_[Xp[:, j].flatten() + c]).reshape((-1, 3*self.state_dim))
				Acc += Embeddings
				c = l
			return Acc
		
		self.Transition_embeddings_ = get_transition_embeddings_graph(self.X_in_tr_)
		self.Transition_embeddings_proc_ = theano.function([self.X_in_tr_], self.Transition_embeddings_)
		
		def get_state_transition_graph(X_state, Transition_embeddings, matrix_mode=False):
			Emb_r = Transition_embeddings[:, :self.state_dim] if matrix_mode else Transition_embeddings[:self.state_dim]
			Emb_u = Transition_embeddings[:, self.state_dim:2*self.state_dim] if matrix_mode else Transition_embeddings[self.state_dim:2*self.state_dim]
			Emb_m = Transition_embeddings[:, 2*self.state_dim:] if matrix_mode else Transition_embeddings[2*self.state_dim:]
			reset_activation = tt.nnet.sigmoid(Emb_r + tt.dot(X_state, self.W_rec_[0]))
			update_activation = tt.nnet.sigmoid(Emb_u + tt.dot(X_state, self.W_rec_[1]))
			main_activation = tt.tanh(Emb_m + tt.dot(X_state * reset_activation, self.W_rec_[2]))
			X_next_state = (update_activation * X_state) + ((1.0 - update_activation) * main_activation)
			return X_next_state
		
		self.X_state_tp1_ = get_state_transition_graph(self.X_state_, self.Transition_embeddings_, matrix_mode=True)
		self.X_state_tp1_proc_ = theano.function([self.X_state_, self.X_in_tr_], self.X_state_tp1_)
		
		# Target embeddings
		def get_target_embeddings_graph(X_in_ta):
			N = X_in_ta.shape[0]
			Acc = tt.zeros((N, self.state_dim), dtype=self.dtype_)
			c = 0
			for j, l in enumerate(self.ta_params_rid_):
				Embeddings = spgr(self.W_ta_to_hidden_[X_in_ta[:, j].flatten() + c]).reshape((-1, self.state_dim))
				Acc += Embeddings
				c = l
			return Acc
		
		self.Target_embeddings_ = get_target_embeddings_graph(self.X_in_ta_)
		self.Target_embeddings_proc_ = theano.function([self.X_in_ta_], self.Target_embeddings_)
		
		# Readout features partial score
		def get_readout_features_partial_score(X_in_re):
			#return self.B_to_out_[0] + ts.dot(X_in_re, self.W_rf_to_out_).flatten()
			#return self.B_to_out_[0] + ts.structured_dot(X_in_re, self.W_rf_to_out_).flatten()
			return self.B_to_out_[0] + ts.true_dot(X_in_re, self.W_rf_to_out_, grad_preserves_dense=False).toarray().flatten()
		
		self.Readout_features_partial_score_ = get_readout_features_partial_score(self.X_in_re_)
		self.Readout_features_partial_score_proc_ = theano.function([self.X_in_re_], self.Readout_features_partial_score_)
		
		# Raw target score
		def get_raw_target_score_graph(X_state, Target_embeddings, Readout_features_partial_score):
			return (X_state * Target_embeddings).sum(axis = 1) + Readout_features_partial_score
		
			
		self.Raw_target_score_ = get_raw_target_score_graph(self.X_state_, self.Target_embeddings_, self.Readout_features_partial_score_)
		self.Raw_target_score_proc_ = theano.function([self.X_state_, self.X_in_ta_, self.X_in_re_], self.Raw_target_score_)
		
		# Auxiliary procedures to compute values of a batch from a single initial state
		self.X_state_to_repeat_ = tt.matrix(name = 'x_state_to_repeat', dtype=self.dtype_)
		self.X_repeats_num_ = tt.iscalar(name = 'repeats_num')
		X_state_repeated = tt.extra_ops.repeat(self.X_state_to_repeat_, self.X_repeats_num_, axis=0)
		
		self.X_state_tp1_with_repeated_state_ = get_state_transition_graph(X_state_repeated, self.Transition_embeddings_, matrix_mode=True)
		self.X_state_tp1_with_repeated_state_proc_ = theano.function([self.X_state_to_repeat_, self.X_repeats_num_, self.X_in_tr_], self.X_state_tp1_with_repeated_state_)
		
		self.Raw_target_score_with_repeated_state_ = get_raw_target_score_graph(X_state_repeated, self.Target_embeddings_, self.Readout_features_partial_score_)
		self.Raw_target_score_with_repeated_state_proc_ = theano.function([self.X_state_to_repeat_, self.X_repeats_num_, self.X_in_ta_, self.X_in_re_], self.Raw_target_score_with_repeated_state_)
		
		# Probabilities
		
		self.Probabilities_with_repeated_state_ = tt.nnet.softmax(self.Raw_target_score_with_repeated_state_).flatten()
		self.Probabilities_with_repeated_state_proc_ = theano.function([self.X_state_to_repeat_, self.X_repeats_num_, self.X_in_ta_, self.X_in_re_], self.Probabilities_with_repeated_state_)
				
		# Loss
		
		self.X_in_tr_seq_ = tt.imatrix(name = 'x_in_tr_seq')
		self.X_in_tas_seq_ = tt.imatrix(name = 'x_in_tas_seq')
		self.X_in_res_seq_ = ts.csr_matrix(name = 'x_in_monos_seq', dtype=self.dtype_)
		self.X_in_ids_seq_ = tt.ivector(name = 'x_in_ids_seq')
		
		def get_loss_graph(X_in_tr_seq, X_in_tas_seq, X_in_res_seq, X_in_ids_seq, X_state_0 = None):
			if  X_state_0 == None:
				X_state_0 = self.Initial_state_
			Transition_embeddings_seq_ext = get_transition_embeddings_graph(X_in_tr_seq, pad_one = True)
			#Transition_embeddings_seq_ext = tt.vertical_stack(Transition_embeddings_seq, tt.zeros((1, self.state_dim), dtype=self.dtype_))
			Seq_len = X_in_tr_seq.shape[0] + 1
			Nte_seq = tt.arange(Seq_len + 1, 1, -1)
			
			Target_embeddings_seq = get_target_embeddings_graph(X_in_tas_seq)
			Readout_features_partial_score_seq = get_readout_features_partial_score(X_in_res_seq)
			
			def seq_loss_seq_scan_proc(K, X_in_id, Nte, X_state_tm1, I_tm1):
				I = I_tm1 + Nte
				Readout_features_partial_score = Readout_features_partial_score_seq[I_tm1:I]
				
				Prereadout = X_state_tm1 * Target_embeddings_seq[I_tm1:I]
				Raw_scores = Prereadout.sum(axis = 1) + Readout_features_partial_score
				
				Probs = tt.nnet.softmax(Raw_scores).flatten()
				Loss = -tt.log(Probs[X_in_id])
				
				X_state = get_state_transition_graph(X_state_tm1, Transition_embeddings_seq_ext[K])
				return X_state, I, Loss
				
			(_, _, Loss_seq), _ = theano.scan(seq_loss_seq_scan_proc, sequences=[tt.arange(Seq_len), X_in_ids_seq, Nte_seq], outputs_info = [X_state_0, np.int64(0), None], truncate_gradient = self.bttp_truncation, name = 'seq_loss_seq_scan')
			return Loss_seq
		
		self.Loss_seq_ = get_loss_graph(self.X_in_tr_seq_, self.X_in_tas_seq_, self.X_in_res_seq_, self.X_in_ids_seq_)
		self.Loss_seq_proc_ = theano.function([self.X_in_tr_seq_, self.X_in_tas_seq_, self.X_in_res_seq_, self.X_in_ids_seq_], self.Loss_seq_, on_unused_input='warn')
		
		self.Mean_loss_ = self.Loss_seq_.mean()
		self.Mean_loss_proc_ = theano.function([self.X_in_tr_seq_, self.X_in_tas_seq_, self.X_in_res_seq_, self.X_in_ids_seq_], self.Mean_loss_, on_unused_input='warn')
		
		
		if self.verbose > 2:
			print >> sys.stderr, 'make_model end'
			
		if not self.enable_training_:
			return
		
		if self.verbose > 2:
			print >> sys.stderr, 'make_training begin'
		
		def get_train_loss_graph(X_in_tr_seq, X_in_tas_seq, X_in_res_seq, X_in_ids_seq, X_state_0 = None):
			if  X_state_0 == None:
				X_state_0 = self.Initial_state_
			Transition_embeddings_seq_ext = get_transition_embeddings_graph(X_in_tr_seq, pad_one = True)
			#Transition_embeddings_seq_ext = tt.vertical_stack(Transition_embeddings_seq, tt.zeros((1, self.state_dim), dtype=self.dtype_))
			Seq_len = X_in_tr_seq.shape[0] + 1
			Nte_seq = tt.arange(Seq_len + 1, 1, -1)
			
			Target_embeddings_seq = get_target_embeddings_graph(X_in_tas_seq)
			Readout_features_partial_score_seq = get_readout_features_partial_score(X_in_res_seq)
			
			def seq_tloss_seq_scan_proc(K, X_in_id, Nte, X_state_tm1, I_tm1, Readout_features_partial_score_seq, Target_embeddings_seq, Transition_embeddings_seq_ext):
				I = I_tm1 + Nte
				Readout_features_partial_score = Readout_features_partial_score_seq[I_tm1:I]
				
				Prereadout = X_state_tm1 * Target_embeddings_seq[I_tm1:I]
				Raw_scores = Prereadout.sum(axis = 1) + Readout_features_partial_score
				
				Probs = tt.nnet.softmax(Raw_scores).flatten()
				Loss = -tt.log(Probs[X_in_id])
				
				X_state = get_state_transition_graph(X_state_tm1, Transition_embeddings_seq_ext[K])
				return X_state, I, Loss
				
			(_, _, Loss_seq), _ = theano.scan(seq_tloss_seq_scan_proc, sequences=[tt.arange(Seq_len), X_in_ids_seq, Nte_seq], outputs_info = [X_state_0, np.int64(0), None], non_sequences=[Readout_features_partial_score_seq, Target_embeddings_seq, Transition_embeddings_seq_ext], truncate_gradient = self.bttp_truncation, name = 'seq_tloss_seq_scan')
			
			Mean_Loss = Loss_seq.mean()
			gr_MTL_wrt_Readout_features_partial_score_seq = theano.grad(Mean_Loss, Readout_features_partial_score_seq)
			gr_MTL_wrt_W_rf_to_out = theano.grad(None, self.W_rf_to_out_, known_grads = {Readout_features_partial_score_seq : gr_MTL_wrt_Readout_features_partial_score_seq})
			gr_MTL_wrt_B_to_out = theano.grad(None, self.B_to_out_, known_grads = {Readout_features_partial_score_seq : gr_MTL_wrt_Readout_features_partial_score_seq})
			gr_MTL_wrt_Target_embeddings_seq = theano.grad(Mean_Loss, Target_embeddings_seq)
			gr_MTL_wrt_W_ta_to_hidden_ = theano.grad(None, self.W_ta_to_hidden_, known_grads = {Target_embeddings_seq : gr_MTL_wrt_Target_embeddings_seq})
			gr_MTL_wrt_Transition_embeddings_seq_ext = theano.grad(Mean_Loss, Transition_embeddings_seq_ext)
			gr_MTL_wrt_W_tr_to_hidden_ = theano.grad(None, self.W_tr_to_hidden_, known_grads = {Transition_embeddings_seq_ext : gr_MTL_wrt_Transition_embeddings_seq_ext})
			gr_MTL_wrt_B_to_hidden_ = theano.grad(None, self.B_to_hidden_, known_grads = {Transition_embeddings_seq_ext : gr_MTL_wrt_Transition_embeddings_seq_ext})
			
			return Mean_Loss, gr_MTL_wrt_W_rf_to_out, gr_MTL_wrt_B_to_out, gr_MTL_wrt_W_ta_to_hidden_, gr_MTL_wrt_W_tr_to_hidden_, gr_MTL_wrt_B_to_hidden_
			
		
		self.Train_Loss_, gr_MTL_wrt_W_rf_to_out, gr_MTL_wrt_B_to_out, gr_MTL_wrt_W_ta_to_hidden_, gr_MTL_wrt_W_tr_to_hidden_, gr_MTL_wrt_B_to_hidden_ = get_train_loss_graph(self.X_in_tr_seq_, self.X_in_tas_seq_, self.X_in_res_seq_, self.X_in_ids_seq_)
		
		train_params_dict = makeADADeltaParamDict(self.train_params_list_)
		train_params_dict[self.W_rf_to_out_]['nabla'] = gr_MTL_wrt_W_rf_to_out
		train_params_dict[self.B_to_out_]['nabla'] = gr_MTL_wrt_B_to_out
		train_params_dict[self.W_ta_to_hidden_]['nabla'] = gr_MTL_wrt_W_ta_to_hidden_
		train_params_dict[self.W_tr_to_hidden_]['nabla'] = gr_MTL_wrt_W_tr_to_hidden_
		train_params_dict[self.B_to_hidden_]['nabla'] = gr_MTL_wrt_B_to_hidden_

		self.optimizer_ = self.optimizer_(**self.optimizer_opts)
		self.optimizer_.make_updates(self.Train_Loss_, self.train_params_list_)
		self.Train_proc_ = theano.function([self.X_in_tr_seq_, self.X_in_tas_seq_, self.X_in_res_seq_, self.X_in_ids_seq_], self.Train_Loss_, updates=self.optimizer_.updates_, on_unused_input='warn')
		
		if self.verbose > 2:
			print >> sys.stderr, 'make_training end'
	
	def initializeParams(self):
		if self.verbose > 2:
			print >> sys.stderr, 'initialize_params begin'
			
		shape_proc = lambda shared_var: shared_var.get_value(borrow=True).shape
		rand_proc = lambda magnitude, shape: (2 * magnitude) * self.rng_.random_sample(shape) - magnitude
		
		for param in self.train_params_list_:
			Param = param['var']
			Param.set_value(rand_proc(self.initial_feature_scaling_factor, shape_proc(Param)))
			
		w_rec_value = self.W_rec_.get_value(borrow = False)
		w_rec_value *= self.initial_contraction_coefficient / np.linalg.norm(w_rec_value)
		self.W_rec_.set_value(w_rec_value, borrow=True)
		
		if self.verbose > 2:
			print >> sys.stderr, 'initialize_params end'
	
	def getInitialState(self, borrow=False):
		return self.Initial_state_.get_value(borrow=borrow)
	
	def saveModelParams(self, fs):
		"""Note: In ESN mode, does not save untrained ESN parameters. Rely on the random state for reinitialization."""
		model_params = self.optimizer_.get_model_params()
		cPickle.dump(model_params, fs, protocol=2)
		fs.flush()
	
	def loadModelParams(self, fs):
		model_params = cPickle.load(fs)
		#self.optimizer_.set_model_params(model_params)
		for k, param in enumerate(self.train_params_list_):
			val = model_params[k]
			param['var'].set_value(val)
	
	
