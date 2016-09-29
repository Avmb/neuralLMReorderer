import sys
from collections import defaultdict
import numpy as np
import scipy.sparse as sp

from sklearn import base
from sklearn import utils

from Node import *

def nodeOffset(n0, n1):
	return n1.id - n0.id

def binnedNodeOffset(n0, n1):
	offset = nodeOffset(n0, n1)
	os, om = np.sign(offset), np.abs(offset)
	if om < 5:
		bom = om
	elif om < 10:
		bom = 5
	else:
		bom = 10
	return bom * os

def withBinnedNodeOffsetFeatures(feats, binnedOffset):
	bF = 'o%s_' % binnedOffset
	rv = [(bF, 1)]
	
	for fk, fv in feats:
		rv.append((fk, fv))
		rv.append((bF + fk, fv))
	return rv

class FeatureMap(base.BaseEstimator):
	def __init__(self):
		pass
	
	def fit(self, X = None, y = None):
		self.templates_ = {}
		self.inv_templates_ = {}
		self.map_ = {}
		self.inv_map_ = {}
		self.counts_ = defaultdict(int)
		self.tf_map_ = {}
		self.inv_tf_map_ = {}
	
	def fit_feature(self, template_str, value):
		value_str = value if type(value) is str else '<NUM>'
		if template_str not in self.templates_:
			new_template_id = len(self.templates_)
			self.templates_[template_str] = new_template_id
			self.inv_templates_[new_template_id] = template_str
			self.map_[new_template_id] = {'<UNK>' : 0}
			self.inv_map_[new_template_id] = {0 : '<UNK>'}
			new_tf_id = len(self.tf_map_)
			self.tf_map_[(template_str, '<UNK>')] = new_tf_id
			self.inv_tf_map_[new_tf_id] = (template_str, '<UNK>')
			
		template_id = self.templates_[template_str]
		fm = self.map_[template_id]
		if value_str not in fm:
			new_id = len(fm)
			fm[value_str] = new_id
			self.inv_map_[template_id][new_id] = value_str
		if (template_str, value_str) not in self.tf_map_:
			new_tf_id = len(self.tf_map_)
			self.tf_map_[(template_str, value_str)] = new_tf_id
			self.inv_tf_map_[new_tf_id] = (template_str, value_str)
		self.counts_[(template_str, value_str)] += 1
	
	def get_indices_dim(self):
		return [len(self.map_[template_id]) for template_id in sorted(self.inv_templates_.keys())]
	
	def get_sparse_dim(self):
		return len(self.tf_map_)
	
	def transform_to_indices(self, feats):
		n = len(self.templates_)
		rv = np.zeros(n, dtype=np.int32)
		for template_str, value_str in feats:
			template_id = self.templates_[template_str]
			fm = self.map_[template_id]
			val_id = fm.get(value_str, 0)
			rv[template_id] = val_id
		return rv

	def transform_seq_to_indices(self, feats_seq):
		n = len(self.templates_)
		m = len(feats_seq)
		rv = np.zeros((m, n), dtype=np.int32)
		for j in xrange(m):
			for template_str, value_str in feats_seq[j]:
				template_id = self.templates_[template_str]
				fm = self.map_[template_id]
				val_id = fm.get(value_str, 0)
				rv[j, template_id] = val_id
		return rv

	
	def transform_to_sparse(self, feats, weights = None):
		n = len(self.tf_map_)
		rv = sp.dok_matrix((1, n), dtype=np.float64)
		for i, feat in enumerate(feats):
			template_str, value = feat
			value_str = value if type(value) is str else '<NUM>'
			tf_id = self.tf_map_.get((template_str, value_str), -1)
			if tf_id == -1:
				tf_id = self.tf_map_[(template_str, '<UNK>')]
			w = 1.0 if weights == None else weights[i]
			if type(value) is not str:
				w *= value
			rv[0, tf_id] += w
		return rv.tocsr()
	
	def transform_seq_to_sparse(self, feats_seq, weights_seq = None):
		n = len(self.tf_map_)
		m = len(feats_seq)
		rv = sp.dok_matrix((m, n), dtype=np.float64)
		for j in xrange(m):
			for i, feat in enumerate(feats_seq[j]):
				template_str, value = feat
				value_str = value if type(value) is str else '<NUM>'
				tf_id = self.tf_map_.get((template_str, value_str), -1)
				if tf_id == -1:
					tf_id = self.tf_map_[(template_str, '<UNK>')]
				w = 1.0 if weights_seq == None else weights_seq[j][i]
				if type(value) is not str:
					w *= value
				rv[j, tf_id] += w
		return rv.tocsr()
	
	def get_filtered_feature_map(self, count_threshold = 5):
		rv = FeatureMap()
		rv.fit()
		for feat, count in self.counts_.iteritems():
			template_str, value_str = feat
			if count >= count_threshold:
				rv.fit_feature(template_str, value_str)
				rv.counts_[(template_str, value_str)] += count - 1
			else:
				rv.fit_feature(template_str, '<UNK>')
				rv.counts_[(template_str, '<UNK>')] += count - 1
		return rv
	
# Path info

leftPaddingNode.descendents = {}

def prepareDescendents(node):
	node.descendents = {}
	for child in node.children:
		prepareDescendents(child)
		for subDescendent in child.descendents.iterkeys():
			node.descendents[subDescendent] = child
		node.descendents[child] = child

def findPath(root, startNode, endNode):
	rv = []
	
	if (startNode.id + 1 == endNode.id) or (startNode == leftPaddingNode and endNode.id == 1):
		rv.append(('R', startNode, endNode))
		return rv
	
	curNode = startNode
	while curNode != endNode:
		nextDownNode = curNode.descendents.get(endNode, None)
		if nextDownNode != None:
			rv.append(('D', curNode, nextDownNode))
			curNode = nextDownNode
		else:
			parentNode = root if (curNode == leftPaddingNode) else curNode.parent
			rv.append(('U', curNode, parentNode))
			curNode = parentNode
	return rv

