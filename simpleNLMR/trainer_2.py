#! /usr/bin/python

import sys
import cPickle
import numpy as np
from collections import defaultdict
import sklearn.feature_extraction as skfe

from Node import *

from reordererState import ReordererState
from simpleNLMR_1 import SimpleNLMR

def getNoiseSample(prevId, toEmitZId, rng, monotonic_transition_prob):
	if prevId in toEmitZId:
		if (len(toEmitZId) == 1) or (rng.rand() < monotonic_transition_prob):
			return prevId
	return rng.choice(toEmitZId)

def getDepSentenceFeatures(sid, sent, dtype, trDV, taDV):
	nodes, root = createNodes(sent, sid)
	reorderedNodes = sorted(nodes, key = lambda n: n.hint)
	toEmitZId = range(len(reorderedNodes))
	
	trDictL = []
	tasDictL = []
	monosL = []
	vidL = []
	prevState = None
	prevId = 0
	state = ReordererState((nodes, root))
	for j in xrange(len(reorderedNodes) - 1):
		positiveNode = reorderedNodes[j]
		for vid, k in enumerate(toEmitZId):
			candidateNode = nodes[k]
			tasDictL.append(dict(state.targetFeaturesToNode(candidateNode)))
			monosL.append(candidateNode.id == prevId + 1)
			if candidateNode == positiveNode:
				vidL.append(vid)
		
		toEmitZId.remove(positiveNode.id - 1)
		if j < len(reorderedNodes) - 2:
			trDictL.append(dict(state.transitionFeaturesToNode(positiveNode)))
		prevState = state
		state = state.successorToNode(positiveNode)
		prevId = positiveNode.id
	trSeq = trDV.transform(trDictL)
	tasSeq = taDV.transform(tasDictL)
	monosSeq = np.array(monosL, dtype=dtype)
	vidSeq = np.array(vidL, dtype=np.int32)
	return (trSeq, tasSeq, monosSeq, vidSeq)

def trainDepSentence(sid, sent, snlmr, trDV, taDV):
	return snlmr.FN_train_proc_(*getDepSentenceFeatures(sid, sent, snlmr.dtype_, trDV, taDV))

def validate(snlmr, validationSet, verbose = 0):
	loss_acc = 0.0
	for i, feats in enumerate(validationSet):
		loss = snlmr.FN_mean_loss_proc_(*feats)
		if verbose > 2:
			print >> sys.stderr, "\tValidation sentence", i+1, "loss:", loss
		loss_acc += loss
	return loss_acc / len(validationSet)

def prepareValidationSet(validSentF, dtype, trDV, taDV, verbose = 0):
	rv = []
	for i, sent in enumerate(readDepSentences(validSentF)):
		rv.append(getDepSentenceFeatures(i+1, sent, dtype, trDV, taDV))
		if verbose > 2:
			print >> sys.stderr, "\tValidation sentence", i+1, "prepared."
	return rv

def saveParams(params, outParamsF):
	outParamsF.seek(0)
	cPickle.dump(params, outParamsF, protocol = 2)
	outParamsF.flush()

def train(trainSentF, validSentF, outParamsF, trDV, taDV, monotonic_transition_prob = 0.7, patience = 5, loss_averaging_factor = 0.1, valid_freq = 300, cutoff_loss = 0.0, verbose = 0):
	snlmr = SimpleNLMR(make_full_normalized_model = True, random_state = 0, verbose = 3)
	snlmr.fit(transition_features_dim = len(trDV.feature_names_), target_features_dim = len(taDV.feature_names_), monotonic_transition_prob = monotonic_transition_prob)
	
	if verbose > 0:
		print >> sys.stderr, "Preparing validation set"
	validationSet = prepareValidationSet(validSentF, snlmr.dtype_, trDV, taDV, verbose)

	best_validation_loss = np.inf
	validation_loss = np.inf
	avg_training_loss = np.inf
	remaining_patience = patience
	epoch = 0
	sentences_before_validation = 0
	total_sentences = 0
	best_params = snlmr.optimizer_.get_model_params()
	live = True
	while live:
		if verbose > 0:
			print >> sys.stderr, "Epoch:", epoch
		epoch += 1
		trainSentF.seek(0)
		for i, sent in enumerate(readDepSentences(trainSentF)):
			if sentences_before_validation <= 0:
				if verbose > 1:
					print >> sys.stderr, "Validation after", total_sentences
				validation_loss = validate(snlmr, validationSet, verbose)
				if validation_loss < best_validation_loss:
					if verbose > 1:
						print >> sys.stderr, "Validation loss improvement. Loss:", validation_loss
						print >> sys.stderr, "Saving parameters..."
					best_params = snlmr.fn_optimizer_.get_model_params()
					best_validation_loss = validation_loss
					remaining_patience = patience
					saveParams(best_params, outParamsF)
					if best_validation_loss <= cutoff_loss:
						if verbose > 1:
							print >> sys.stderr, "Cutoff loss reached."
						live = False
						break
				else:
					remaining_patience -= 1
					if verbose > 1:
						print >> sys.stderr, "No improvement. Loss:", validation_loss, ", remaining patience:", remaining_patience
					if remaining_patience <= 0:
						live = False
						break
				sentences_before_validation = valid_freq
			else:
				sentences_before_validation -= 1
			
			training_loss = trainDepSentence(i+1, sent, snlmr, trDV, taDV)
			total_sentences += 1
			avg_training_loss = training_loss if avg_training_loss == np.inf else loss_averaging_factor * training_loss + (1.0 - loss_averaging_factor) * avg_training_loss
			if verbose > 2:
				print >> sys.stderr, "Total sentences:", total_sentences, "training loss:", training_loss, "Avg training loss:", avg_training_loss
	if verbose > 0:
		print >> sys.stderr, "Training completed. Sentences:", total_sentences, "."

def main():
	argn = len(sys.argv)
	if (argn != 5):
		usage()
		sys.exit(-1)
	
	trainSentF = open(sys.argv[1])
	validSentF = open(sys.argv[2])
	dvF = open(sys.argv[3])
	outParamsF = open(sys.argv[4], 'wb')
	tmp = cPickle.load(dvF)
	trDV, taDV = tmp['transitionFeatureDictVectorizer'], tmp['targetFeatureDictVectorizer']

	train(trainSentF, validSentF, outParamsF, trDV, taDV, valid_freq = 300, patience = 2, cutoff_loss = 0.0, verbose = 3)

def usage():
	print >> sys.stderr, 'Usage:'
	print >> sys.stderr, sys.argv[0], 'training-depsentences-file validation-depsentences-file dictionary-vectorizers-file out-parameters-file'

if __name__ == '__main__':
	main()

