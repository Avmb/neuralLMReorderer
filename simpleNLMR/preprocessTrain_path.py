#! /usr/bin/python

import sys
import cPickle
from collections import defaultdict
import sklearn.feature_extraction as skfe

from Node import *
from featuresBase import FeatureMap, prepareDescendents

from reordererState import ReordererState

cutoffCount = 100

def processDepSentence(sid, sent, acc):
	# Note: Accurate for Bigram features only
	print >> sys.stderr, sid

	trFM, taFM, reFM, pfFM = acc['transitionFeatureMap'], acc['targetFeatureMap'], acc['readoutFeatureMap'], acc['pathFeatureMap']
	nodes, root = createNodes(sent, sid)
	prepareDescendents(root)
	reorderedNodes = sorted(nodes, key = lambda node: node.hint)
	m = len(reorderedNodes)
	state = ReordererState((nodes, root))
	prevNode = None
	nextNode = None
	for i in xrange(m):
		for candidateNode in nodes:
			if candidateNode == prevNode:
				continue
			if candidateNode == reorderedNodes[i]:
				nextNode = candidateNode
				nextState = state.successorToNode(candidateNode)
			trF = state.transitionFeaturesToNode(candidateNode)
			for k, v in trF:
				trFM.fit_feature(k, v)
			taF = state.targetFeaturesToNode(candidateNode)
			for k, v in taF:
				taFM.fit_feature(k, v)
			reF = state.readoutFeaturesToNode(candidateNode)
			for k, v in reF:
				reFM.fit_feature(k, v)
			pfFL = state.pathFeaturesToNode(candidateNode)
			for pfF in pfFL:
				for k, v in pfF:
					pfFM.fit_feature(k, v)
		prevNode = nextNode
		state = nextState	

def main():
	argn = len(sys.argv)
	if argn != 2:
		usage()
		sys.exit(-1)
	outF = open(sys.argv[1], 'wb')

	acc = {'transitionFeatureMap' : FeatureMap(), 'targetFeatureMap' : FeatureMap(), 'readoutFeatureMap' : FeatureMap(), 'pathFeatureMap' : FeatureMap()}
	for fm in acc.itervalues():
		fm.fit()
	
	for i, sent in enumerate(readDepSentences(sys.stdin)):
		processDepSentence(i+1, sent, acc)
	
	rv = {}
	for k, fm in acc.iteritems():
		ffm = fm.get_filtered_feature_map(cutoffCount)
		rv[k] = ffm
	cPickle.dump(rv, outF, protocol=2)
	
def usage():
	print >> sys.stderr, 'Usage:'
	print >> sys.stderr, sys.argv[0], 'out-feature-maps-file'
if __name__ == '__main__':
	main()

