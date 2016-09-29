#! /usr/bin/python

import sys
import cPickle
import numpy as np
from collections import defaultdict
from sortedcontainers.sortedset import SortedSet
import sklearn.feature_extraction as skfe

from Node import *

from reordererState import ReordererState
from simpleNLMR_3 import SimpleNLMR_3

def findPunctNodeZIds(nodes):
	return [n.id - 1 for n in nodes if n.pos.startswith('$')]

def processDepSentence(sid, sent, snlmr, trFM, taFM, reFM, beam_size):
	nodes, root = createNodes(sent, sid)
	punctNodeZIds = findPunctNodeZIds(nodes)
	state0 = ReordererState((nodes, root), snlmr.getInitialState().reshape((1, -1)))
	beam = SortedSet([(0.0, 0, state0)])
	c = 1
	punctL = punctNodeZIds[0] if len(punctNodeZIds) > 0 else -1
	punctR = punctNodeZIds[1] if len(punctNodeZIds) > 1 else len(nodes)
	for j in xrange(len(nodes) - 1):
		nextBeam = SortedSet()
	
		for totalNegScore, _, state in beam:
			#print >> sys.stderr, totalNegScore
			taL = []
			reL = []
			nodeZIdsToEmitL = list(state.nodeZIdsToEmit)
			for candidateNodeZId in nodeZIdsToEmitL:
				candidateNode = nodes[candidateNodeZId]
				taL.append(state.targetFeaturesToNode(candidateNode))
				reL.append(state.readoutFeaturesToNode(candidateNode))
			
			ta = taFM.transform_seq_to_indices(taL)
			re = reFM.transform_seq_to_sparse(reL)
			probs = snlmr.Probabilities_with_repeated_state_proc_(state.modelState, len(nodeZIdsToEmitL), ta, re)
			scores = np.log(probs)
			
			for i, candidateNodeZId in enumerate(nodeZIdsToEmitL):
				candidateNode = nodes[candidateNodeZId]
				nextState = state.successorToNode(candidateNode)
				nextBeam.add((totalNegScore - scores[i], c, nextState))
				c += 1
				if len(nextBeam) > beam_size:
					nextBeam.pop()

		if j < len(nodes) - 2:
			trL = []
			modelStateL = []
			for _, _, state in nextBeam:
				node = state.emittedNodesL[-1]
				trL.append(state.prevState.transitionFeaturesToNode(node))
				modelStateL.append(state.prevState.modelState)
			#print >> sys.stderr, trDictL
			tr = trFM.transform_seq_to_indices(trL)
			modelStateMat = np.vstack(modelStateL)
			nextModelStateMat = snlmr.X_state_tp1_proc_(modelStateMat, tr)
			i = 0
			for _, _, state in nextBeam:
				state.modelState = nextModelStateMat[i].reshape((1, -1))
				i += 1
		
		beam = nextBeam	
	bestNegScore, _, bestState = beam.pop(0)
	
	# Emit last node
	lastNodeZId = next(bestState.nodeZIdsToEmit.__iter__())
	lastState = bestState.successorToNode(nodes[lastNodeZId])
	
	# Update hints and output
	lastState.reorderNodes()
	for node in nodes:
		print node.toConllStr()
	print ''
	

def decode(sentF, paramsF, trFM, taFM, reFM, state_dim = 100, beam_size = 2, verbose = 0):
	snlmr = SimpleNLMR_3(state_dim = state_dim, random_state = 0, verbose = 3)
	snlmr.fit(transition_features_map = trFM, target_features_map = taFM, readout_features_map = reFM, enable_training = False)
	snlmr.loadModelParamsTxt(paramsF)

	for i, sent in enumerate(readDepSentences(sentF)):
		processDepSentence(i+1, sent, snlmr, trFM, taFM, reFM, beam_size)
		if verbose > 0:
			print >> sys.stderr, i + 1

def main():
	argn = len(sys.argv)
	if (argn != 3):
		usage()
		sys.exit(-1)
	
	dvF = open(sys.argv[1])
	paramsF = open(sys.argv[2])
	tmp = cPickle.load(dvF)
	trFM, taFM, reFM = tmp['transitionFeatureMap'], tmp['targetFeatureMap'], tmp['readoutFeatureMap']

	decode(sys.stdin, paramsF, trFM, taFM, reFM, state_dim = 100, beam_size = 4, verbose = 1)

def usage():
	print >> sys.stderr, 'Usage:'
	print >> sys.stderr, sys.argv[0], 'feature-maps-file parameters-file'

if __name__ == '__main__':
	main()

