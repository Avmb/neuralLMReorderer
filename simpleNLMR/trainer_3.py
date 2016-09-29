#! /usr/bin/python

import sys
import cPickle
import numpy as np
from collections import defaultdict
import sklearn.feature_extraction as skfe

from Node import *

from reordererState import ReordererState
#from simpleNLMR_2 import SimpleNLMR_2
from simpleNLMR_2_gru import SimpleNLMR_2, AdamUpdates

def getDepSentenceFeatures(sid, sent, dtype, trFM, taFM, reFM):
	nodes, root = createNodes(sent, sid)
	reorderedNodes = sorted(nodes, key = lambda n: n.hint)
	toEmitZId = range(len(reorderedNodes))
	
	trL = []
	tasL = []
	resL = []
	vidL = []
	prevState = None
	prevId = 0
	state = ReordererState((nodes, root))
	for j in xrange(len(reorderedNodes) - 1):
		positiveNode = reorderedNodes[j]
		for vid, k in enumerate(toEmitZId):
			candidateNode = nodes[k]
			tasL.append(state.targetFeaturesToNode(candidateNode))
			resL.append(state.readoutFeaturesToNode(candidateNode))
			if candidateNode == positiveNode:
				vidL.append(vid)
		
		toEmitZId.remove(positiveNode.id - 1)
		if j < len(reorderedNodes) - 2:
			trL.append(state.transitionFeaturesToNode(positiveNode))
		prevState = state
		state = state.successorToNode(positiveNode)
		prevId = positiveNode.id
	trSeq = trFM.transform_seq_to_indices(trL)
	tasSeq = taFM.transform_seq_to_indices(tasL)
	resSeq = reFM.transform_seq_to_sparse(resL)
	vidSeq = np.array(vidL, dtype=np.int32)
	return (trSeq, tasSeq, resSeq, vidSeq)

def trainDepSentence(sid, sent, snlmr, trFM, taFM, reFM):
	return snlmr.Train_proc_(*getDepSentenceFeatures(sid, sent, snlmr.dtype_, trFM, taFM, reFM))

def validate(snlmr, validationSet, verbose = 0):
	loss_acc = 0.0
	for i, feats in enumerate(validationSet):
		loss = snlmr.Mean_loss_proc_(*feats)
		if verbose > 2:
			print >> sys.stderr, "\tValidation sentence", i+1, "loss:", loss
		loss_acc += loss
	return loss_acc / len(validationSet)

def prepareValidationSet(validSentF, dtype, trFM, taFM, reFM, verbose = 0):
	rv = []
	for i, sent in enumerate(readDepSentences(validSentF)):
		if len(sent) < 3:
			if verbose > 2:
				print >> sys.stderr, "\t Skipping short validation sentence", i+1
			continue
		rv.append(getDepSentenceFeatures(i+1, sent, dtype, trFM, taFM, reFM))
		if verbose > 2:
			print >> sys.stderr, "\tValidation sentence", i+1, "prepared."
	return rv

def saveParams(params, outParamsF):
	outParamsF.seek(0)
	cPickle.dump(params, outParamsF, protocol = 2)
	outParamsF.flush()

def train(trainSentF, validSentF, outParamsF, trFM, taFM, reFM, state_dim = 100, patience = 5, loss_averaging_factor = 0.01, valid_freq = 300, cutoff_loss = 0.0, verbose = 0):
	#snlmr = SimpleNLMR_2(state_dim = state_dim, optimizer_opts={'clip_nabla' : (-5.0, 5.0), 'verbose' : 3}, tr_l2_reg = 0.0001, ta_l2_reg = 0.0001, rf_l2_reg = 0.001, rec_l2_reg = 0.001, initial_state_l2_reg = 0.0001, random_state = 0, verbose = 3)
	#snlmr = SimpleNLMR_2(state_dim = state_dim, optimizer=AdamUpdates, optimizer_opts={'clip_nabla' : (-5.0, 5.0), 'verbose' : 3, 'alpha' : 0.0002, 'decay_lambda' : 1-1e-6}, tr_l2_reg = 0.0001, ta_l2_reg = 0.0001, rf_l2_reg = 0.001, rec_l2_reg = 0.001, initial_state_l2_reg = 0.0001, random_state = 0, verbose = 3)
	#snlmr = SimpleNLMR_2(state_dim = state_dim, optimizer=AdamUpdates, optimizer_opts={'clip_nabla' : (-5.0, 5.0), 'verbose' : 3, 'alpha' : 0.0002, 'decay_lambda' : 1-1e-6}, tr_l2_reg = 0.0, ta_l2_reg = 1e-4, rf_l2_reg = 1e-4, rec_l2_reg = 1e-5, initial_state_l2_reg = 0.0, random_state = 0, verbose = 3)
	snlmr = SimpleNLMR_2(state_dim = state_dim, optimizer=AdamUpdates, optimizer_opts={'clip_nabla' : (-5.0, 5.0), 'verbose' : 3, 'alpha' : 0.0002, 'decay_lambda' : 1-1e-6}, tr_l2_reg = 0.0, ta_l2_reg = 0.0, rf_l2_reg = 1e-4, rec_l2_reg = 1e-6, initial_state_l2_reg = 0.0, random_state = 0, verbose = 3)
	#snlmr = SimpleNLMR_2(state_dim = state_dim, optimizer=AdamUpdates, optimizer_opts={'clip_nabla' : (-5.0, 5.0), 'verbose' : 3, 'alpha' : 0.0002, 'decay_lambda' : 1-1e-6}, tr_l2_reg = 0.0, ta_l2_reg = 0.0, rf_l2_reg = 0.0, rec_l2_reg = 0.0, initial_state_l2_reg = 0.0, random_state = 0, verbose = 3)
	snlmr.fit(transition_features_map = trFM, target_features_map = taFM, readout_features_map = reFM, enable_training = True)
	
	if verbose > 0:
		print >> sys.stderr, "Preparing validation set"
	validationSet = prepareValidationSet(validSentF, snlmr.dtype_, trFM, taFM, reFM, verbose)

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
					best_params = snlmr.optimizer_.get_model_params()
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
			
			if len(sent) < 3:
				if verbose > 2:
					print >> sys.stderr, " Skipping short training sentence", i+1
				total_sentences += 1
				continue
			training_loss = trainDepSentence(i+1, sent, snlmr, trFM, taFM, reFM)
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
	trFM, taFM, reFM = tmp['transitionFeatureMap'], tmp['targetFeatureMap'], tmp['readoutFeatureMap']

	train(trainSentF, validSentF, outParamsF, trFM, taFM, reFM, state_dim = 30, valid_freq = 2000, patience = 50, cutoff_loss = 0.0, verbose = 3)

def usage():
	print >> sys.stderr, 'Usage:'
	print >> sys.stderr, sys.argv[0], 'training-depsentences-file validation-depsentences-file feature-maps-file out-parameters-file'

if __name__ == '__main__':
	main()

