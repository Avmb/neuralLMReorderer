import sys
import numpy as np
import scipy.sparse as sp

import theano
import theano.tensor as tt
import theano.sparse as ts

from sklearn import base
from sklearn import utils

from adadelta import ADADelta

from Node import *

#import features_pos_deprel_0 as features
#import features_pos_deprel_1 as features
#import features_pos_deprel_2 as features
#import features_pos_deprel_3 as features
import features_pos_deprel_4 as features
#import features_word_pos_deprel_1 as features
#import features_word_pos_deprel_2 as features
#import features_word_1 as features
#import features_pos_deprel_path_0 as features

class ReordererState(object):
	def __init__(self, sentNR, modelState = None):
		self.nodes, self.root = sentNR
		self.modelState = modelState
		self.prevState = None
		self.emittedNodesL = []
		self.nodeZIdsToEmit = set(xrange(len(self.nodes)))
		self.stateFeatureData = None
	
	def isFinalState(self):
		return len(self.emittedNodesL) == len(self.nodes)
	
	def lastEmittedId(self):
		return 0 if len(self.emittedNodesL) == 0 else self.emittedNodesL[-1].id
	
	def reorderNodes(self):
		for i, node in enumerate(self.emittedNodesL):
			node.hint = i + 1

	def successorToNode(self, node, newModelState = None):
		if node.id - 1 not in self.nodeZIdsToEmit:
			raise "Node already emitted"
		
		nextState = ReordererState((self.nodes, self.root), newModelState)
		nextState.prevState = self
		nextState.emittedNodesL = list(self.emittedNodesL) + [node]
		nextState.nodeZIdsToEmit = set(self.nodeZIdsToEmit)
		nextState.nodeZIdsToEmit.remove(node.id - 1)
		return nextState

	def transitionFeaturesToNode(self, node):
		if self.stateFeatureData == None:
			self.stateFeatureData = features.getStateFeatureData(self)
		transitionFeatures = features.getTransitionFeatures(self, node)
		return transitionFeatures
	
	def targetFeaturesToNode(self, node):
		if self.stateFeatureData == None:
			self.stateFeatureData = features.getStateFeatureData(self)
		targetFeatures = features.getTargetFeatures(self, node)
		return targetFeatures
	
	def readoutFeaturesToNode(self, node):
		if self.stateFeatureData == None:
			self.stateFeatureData = features.getStateFeatureData(self)
		readoutFeatures = features.getReadoutFeatures(self, node)
		return readoutFeatures
	
	def pathFeaturesToNode(self, node):
		if self.stateFeatureData == None:
			self.stateFeatureData = features.getStateFeatureData(self)
		pathFeatures = features.getPathFeatures(self, node)
		return pathFeatures

		
