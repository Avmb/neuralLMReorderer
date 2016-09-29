#! /usr/bin/python

import sys
import cPickle
from collections import defaultdict
import sklearn.feature_extraction as skfe

from Node import *

from reordererState import ReordererState

cutoffCount = 5

g_n = 0
g_next = 0
g_skipForward = 0
g_skipBackwards = 0

g_transitionFeatsD = defaultdict(int)
g_targetFeatsD = defaultdict(int)
g_featuresN = 0
g_transitionFeatsL = []
g_targetFeatsL = []

def updateNoiseProbabilities(sid, nodes, root):
	global g_n, g_next, g_skipForward, g_skipBackwards
	
	reorderedNodes = sorted(nodes, key = lambda node: node.hint)
	lastId = 0
	for node in reorderedNodes[:-1]:		# Ignore last node
		if node.id == lastId + 1:
			g_next += 1
		elif node.id > lastId + 1:
			g_skipForward += 1
		else:
			g_skipBackwards += 1
		g_n += 1
		lastId = node.id

def updateFeatsDict(sid, nodes, root):
	# Bigram features only
	global g_transitionFeatsD, g_targetFeatD, g_featuresN, g_transitionFeatsL, g_targetFeatsL
	
	print >> sys.stderr, sid
	state0 = ReordererState((nodes, root))
	for e1 in nodes:
		state1 = state0.successorToNode(e1)
		for t in nodes:
			if t == e1:
				continue
			transitionFeats = state1.transitionFeaturesToNode(t)
			g_transitionFeatsL.append(dict(transitionFeats))
			for f in transitionFeats:
				g_transitionFeatsD[f] += 1
			targetFeats = state1.targetFeaturesToNode(t)
			g_targetFeatsL.append(dict(targetFeats))
			for f in targetFeats:
				g_targetFeatsD[f] += 1
			g_featuresN += 1

def processDepSentence(sid, sent):
	nodes, root = createNodes(sent, sid)
	updateNoiseProbabilities(sid, nodes, root)
	updateFeatsDict(sid, nodes, root)

def makeDV(featsD, featsL):
	accS = set([])
	accL = []
	for featsRow in featsL:
		accRow = {}
		for f, v in featsRow.iteritems():
			if (f, v) in accS:
				continue
			accS.add((f, v))
			if featsD[(f, v)] >= cutoffCount:
				accRow[f] = v
			else:
				accRow[f] = '<unk>'
		accL.append(accRow)
	rv = skfe.DictVectorizer()
	rv.fit(accL)
	return rv

def finalizeFeatDict(outF):
	global g_transitionFeatsD, g_targetFeatD, g_featuresN, g_transitionFeatsL, g_targetFeatsL
	
	trFeatsL = g_transitionFeatsD.items()
	trFeatsL.sort(key = lambda x: -x[1])
	taFeatsL = g_targetFeatsD.items()
	taFeatsL.sort(key = lambda x: -x[1])
	
	if outF != None:
		trDV = makeDV(g_transitionFeatsD, g_transitionFeatsL)
		taDV = makeDV(g_targetFeatsD, g_targetFeatsL)
		#trDV = skfe.DictVectorizer()
		#trDV.fit(g_transitionFeatsL)
		#taDV = skfe.DictVectorizer()
		#taDV.fit(g_targetFeatsL)
		d = {'transitionFeatureDictVectorizer' : trDV, 'targetFeatureDictVectorizer' : taDV}
		cPickle.dump(d, outF, protocol=2)
		outF.flush()
	
	print 'Transition features:'
	for (fk, fv), c in trFeatsL:
		print "%s=%s:\t%s" % (fk, fv, c)
	print '\nTarget features'
	for (fk, fv), c in taFeatsL:
		print "%s=%s:\t%s" % (fk, fv, c)

def main():
	global g_n, g_next, g_skipForward, g_skipBackwards
	
	argn = len(sys.argv)
	if (argn != 1) and (argn != 2):
		usage()
		sys.exit(-1)
	if argn == 2:
		outF = open(sys.argv[1], 'wb')
	else:
		outF = None

	for i, sent in enumerate(readDepSentences(sys.stdin)):
		processDepSentence(i+1, sent)
	
	finalizeFeatDict(outF)
	
	n = float(g_n)
	d = {'nextP' : g_next / n, 'skipForwardP' : g_skipForward / n, 'skipBackwardsP' : g_skipBackwards / n}
	print d

def usage():
	print >> sys.stderr, 'Usage:'
	print >> sys.stderr, sys.argv[0], '[out-dictvectorizer-file]'
if __name__ == '__main__':
	main()

