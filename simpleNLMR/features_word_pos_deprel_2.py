import sys
import numpy as np
import scipy.sparse as sp

from sklearn import base
from sklearn import utils

from Node import *

from featuresBase import *

def getStateFeatureData(state):
	return True

def getTargetFeatures(state, node):
	return getTransitionFeatures(state, node)

def getTransitionFeatures(state, node):
	t = node
	tm1 = nodeAtPadded(state.nodes, node.id - 2)
	tp1 = nodeAtPadded(state.nodes, node.id)
	parent = nodeAtPadded(state.nodes, node.parentId - 1)
	feats = [
		# Target node unigram features
		('t_form', t.form),
		('t_pos', t.pos),
		('t_deprel', t.deprel),
		('t_pos*t_deprel', '%s*%s' % (t.pos, t.deprel)),
		
		# Target left, right and parent features
		('tm1_pos', tm1.pos),
		('tm1_deprel', tm1.deprel),
		('tm1_pos*tm1_deprel', '%s*%s' % (tm1.pos, tm1.deprel)),
		('tp1_pos', tp1.pos),
		('tp1_deprel', tp1.deprel),
		('tp1_pos*tp1_deprel', '%s*%s' % (tp1.pos, tp1.deprel)),
		('parent_pos', parent.pos),
		('parent_deprel', parent.deprel),
		('parent_pos*parent_deprel', '%s*%s' % (parent.pos, parent.deprel)),
		
		# Target pair features
		('t_pos*tm1_pos', '%s*%s' % (t.pos, tm1.pos)),
		('t_pos*tp1_pos', '%s*%s' % (t.pos, tp1.pos)),
		('t_pos*parent_pos', '%s*%s' % (t.pos, parent.pos)),
		('t_deprel*tm1_deprel', '%s*%s' % (t.deprel, tm1.deprel)),
		('t_deprel*tp1_deprel', '%s*%s' % (t.deprel, tp1.deprel)),
		('t_deprel*parent_deprel', '%s*%s' % (t.deprel, parent.deprel)),
		('t_deprel*parent_pos', '%s*%s' % (t.deprel, parent.pos)),
		('t_pos*parent_deprel', '%s*%s' % (t.pos, parent.deprel)),
		
		# Target triple features
		('t_pos*tm1_pos*tp1_pos', '%s*%s*%s' % (t.pos, tm1.pos, tp1.pos)),
		('t_pos*tm1_pos*parent_pos', '%s*%s*%s' % (t.pos, tm1.pos, parent.pos)),
		('t_pos*tp1_pos*parent_pos', '%s*%s*%s' % (t.pos, tp1.pos, parent.pos))	
	]
	return feats


def getReadoutFeatures(state, node):
	t = node
	parent = nodeAtPadded(state.nodes, node.parentId - 1)
	e1 = state.emittedNodesL[-1] if len(state.emittedNodesL) > 0 else nodeAtPadded(state.nodes, -1)
	e1_parent = nodeAtPadded(state.nodes, node.parentId - 1)
	e2 = state.emittedNodesL[-2] if len(state.emittedNodesL) > 1 else nodeAtPadded(state.nodes, -1)
	
	feats = [
		# Bigram features
		('t_pos*e1_pos', '%s*%s' % (t.pos, e1.pos)),
		('t_pos*e1_deprel', '%s*%s' % (t.pos, e1.deprel)),
		('t_deprel*e1_pos', '%s*%s' % (t.deprel, e1.pos)),
		('t_deprel*e1_deprel', '%s*%s' % (t.deprel, e1.deprel)),
		
		('parent_pos*e1_pos', '%s*%s' % (parent.pos, e1.pos)),
		('parent_pos*e1_deprel', '%s*%s' % (parent.pos, e1.deprel)),
		('parent_deprel*e1_pos', '%s*%s' % (parent.deprel, e1.pos)),
		('parent_deprel*e1_deprel', '%s*%s' % (parent.deprel, e1.deprel)),
		
		('t_pos*e1_parent_pos', '%s*%s' % (t.pos, e1_parent.pos)),
		('t_pos*e1_parent_deprel', '%s*%s' % (t.pos, e1_parent.deprel)),
		('t_deprel*e1_parent_pos', '%s*%s' % (t.deprel, e1_parent.pos)),
		('t_deprel*e1_parent_deprel', '%s*%s' % (t.deprel, e1_parent.deprel)),
				
		# Trigram features
		('t_pos*e1_pos*e2_pos', '%s*%s*%s' % (t.pos, e1.pos, e2.pos)),
		('t_deprel*e1_deprel*e2_deprel', '%s*%s*%s' % (t.deprel, e1.deprel, e2.deprel))
	]
	
	# Bigram dependency topology features
	if t.parent == e1:
		feats.append(('t->e1', '1'))
		feats.append(('t->e1*t_pos', t.pos))
		feats.append(('t->e1*t_deprel', t.deprel))
		feats.append(('t->e1*t_pos*e1_pos', '%s*%s' % (t.pos, e1.pos)))
		feats.append(('t->e1*t_deprel*e1_pos', '%s*%s' % (t.deprel, e1.pos)))
		feats.append(('t->e1*t_pos*e1_deprel', '%s*%s' % (t.pos, e1.deprel)))
		feats.append(('t->e1*t_deprel*e1_deprel', '%s*%s' % (t.deprel, e1.deprel)))
	elif e1.parent == t:
		feats.append(('t<-e1', '1'))
		feats.append(('t<-e1*t_pos', t.pos))
		feats.append(('t<-e1*t_deprel', t.deprel))
		feats.append(('t<-e1*t_pos*e1_pos', '%s*%s' % (t.pos, e1.pos)))
		feats.append(('t<-e1*t_deprel*e1_pos', '%s*%s' % (t.deprel, e1.pos)))
		feats.append(('t<-e1*t_pos*e1_deprel', '%s*%s' % (t.pos, e1.deprel)))
		feats.append(('t<-e1*t_deprel*e1_deprel', '%s*%s' % (t.deprel, e1.deprel)))
	elif t.parent == e1.parent:
		feats.append(('t-><-e1', '1'))
		feats.append(('t-><-e1*t_pos', t.pos))
		feats.append(('t-><-e1*t_deprel', t.deprel))
		feats.append(('t-><-e1*t_pos*e1_pos', '%s*%s' % (t.pos, e1.pos)))
		feats.append(('t-><-e1*t_deprel*e1_pos', '%s*%s' % (t.deprel, e1.pos)))
		feats.append(('t-><-e1*t_pos*e1_deprel', '%s*%s' % (t.pos, e1.deprel)))
		feats.append(('t-><-e1*t_deprel*e1_deprel', '%s*%s' % (t.deprel, e1.deprel)))
	
	if t.parent == e2:
		feats.append(('t->e2', '1'))
	elif e2.parent == t:
		feats.append(('t<-e2', '1'))
	elif t.parent == e2.parent:
		feats.append(('t-><-e2', '1'))
	
	offset =  nodeOffset(e1, node)
	featsWithOffset = withBinnedNodeOffsetFeatures(feats, binnedNodeOffset(e1, node))
	
	# Numeric features
	nte = len(state.nodes) - len(state.emittedNodesL)
	featsWithOffset.append(('log_nte', np.log(nte)))
	featsWithOffset.append(('po_div30', max(offset, 0.0) / 30.0))
	featsWithOffset.append(('no_div30', min(offset, 0.0) / 30.0))
	
	return featsWithOffset

