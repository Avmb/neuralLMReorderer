#! /usr/bin/python

""" Generate reordering hints from symmetrized Giza++ alignment using heuristic from Al-Onaizan and Papineni (2006)
Antonio Valerio Miceli Barone 2012 """

import sys
import getopt

def removeClitics(tokens):
	i = 0
	end = len(tokens)-1
	while i < end:
		tok = tokens[i]
		if (tok[1][-1] != '-') or (len(tok[1]) == 1):
			i += 1
			continue
		next = tokens[i+1]
		tok[1] = tok[1][0:-1] + next[1]
		if (tok[6] == str(next[0])):
			tok[6] = next[6]
		tokens.remove(next)
		# renumber tokens
		for tok in tokens:
			if tok[0] > i + 1:
				tok[0] = tok[0] - 1
			if int(tok[6]) > i + 1:
				tok[6] = str(int(tok[6]) - 1)
		end -= 1

def usage():
	print >> sys.stderr, "Usage:"
	print >> sys.stderr, sys.argv[0], "alignmentFile"
	sys.exit(-1)

reorder = False

def main():
	global reorder
	global sentenceSeqId

	opts, argv = getopt.getopt(sys.argv[1:], "r", "reorder")
	for opt, a in opts:
		if (opt == '-r'):
			reorder = True
		else:
			usage()
	
	if (len(argv) != 1):
		usage()
	alFileName = argv[0]
	
	alFs = open(alFileName)
	
	sentenceSeqId = 1
	curSentenceNodes = []
	for line in sys.stdin:
		line = line.strip()
		if (line == ""):
			processSentence(curSentenceNodes, alFs)
			curSentenceNodes = []
			sentenceSeqId += 1
		else:
			tokens = line.split("\t")
			curSentenceNodes.append(tokens)
		if (tokens[1] == "<BRK>"):
			processSentence(curSentenceNodes, alFs)
			curSentenceNodes = []
			sentenceSeqId += 1
	if (len(curSentenceNodes) > 0):
		processSentence(curSentenceNodes, alFs)
		sentenceSeqId += 1
			
def processSentence(tokens, alFs):
	global reorder
	global sentenceSeqId

	for i in xrange(len(tokens)):
		tokens[i][0] = i + 1
		if (len(tokens[i]) < 11):
                        tokens[i].insert(8, "0")

	#removeClitics(tokens)
	
	tokPos = [[i, None] for i in xrange(len(tokens))]
	alTokens = alFs.readline().split()

	for alToken in alTokens:
		p = alToken.split('-')
		i = int(p[0])
		j = int(p[1])
		if i >= len(tokens):
			print >> sys.stderr, "Token number does not match alignment in sentence id:", sentenceSeqId, "i:", i, "len:", len(tokens)
			return
		if (tokPos[i][1] == None):
			tokPos[i][1] = j
		else:
			tokPos[i][1] = min(tokPos[i][1], j)

	if (tokPos[0][1] == None):
		tokPos[0][1] = 0
	for i in xrange(1, len(tokens)):
		if (tokPos[i][1] == None):
			tokPos[i][1] = tokPos[i - 1][1]

	tokPos.sort(key = lambda (x, y): (y, x))
	i = 1
	for x, y in tokPos:
		tokens[x][8] = str(i)
		i += 1
	
	if (reorder):
		tokens.sort(key = lambda token: int(token[8]))
	
	for token in tokens:
		print str(token[0]) + "\t" + "\t".join(token[1:])
	print

if (__name__ == '__main__'):
	main()

