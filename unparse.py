#! /usr/bin/python

import sys

from Node import *

def processDepParseTree(tokens):
	global sentenceSeqId

	for i in xrange(len(tokens)):
		tokens[i][0] = i + 1
	
	removeClitics(tokens)
	
	nodesAndRoot = createNodes(tokens, sentenceSeqId)
	if (nodesAndRoot == None):
		return
	nodes, root = nodesAndRoot
	if (root == None):
		print >> sys.stderr, "Error: no root in sentence:", sentenceSeqId
		return
	
	for curNode in nodes:
		print curNode.form,
	
	print

sentenceSeqId = 1

def processSentences():
	global sentenceSeqId

	curSentenceNodes = []

	for line in sys.stdin:
		line = line.strip()
		if (line == ""):
			if (len(curSentenceNodes) > 0):
				processDepParseTree(curSentenceNodes)
				sentenceSeqId += 1
			curSentenceNodes = []
		else:
			tokens = line.split()
			curSentenceNodes.append(tokens)

	if (len(curSentenceNodes) > 0):
		processDepParseTree(curSentenceNodes)
		sentenceSeqId += 1

def main():
	processSentences()

if (__name__ == '__main__'):
	main()

