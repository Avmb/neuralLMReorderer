import numpy as np
import scipy.sparse as sp
import theano
import theano.tensor as tt
import theano.sparse as ts

class SpNonzeroOp(theano.Op):
	"""
	Wrapper for the nonzero() method of scipy sparse matrices
	"""
	
	def __eq__(self, other):
		return type(self) == type(other)

	def __hash__(self):
		return hash(type(self))

	def make_node(self, x):
		x_ = ts.as_sparse_variable(x)
		indices = ts.csm_indices(x)
		return theano.Apply(self, inputs = [x_], outputs = [indices.type(), indices.type()])
	
	def perform(self, node, inputs, output_storage):
		x, = inputs
		ind_rows, ind_cols = x.nonzero()
		output_storage[0][0] = ind_rows
		output_storage[1][0] = ind_cols

sp_nonzero = SpNonzeroOp()


