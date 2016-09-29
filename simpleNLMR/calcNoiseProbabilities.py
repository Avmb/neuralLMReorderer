#! /usr/bin/python

import sys

from Node import *

g_n = 0
g_next = 0
g_skipForward = 0
g_skipBackwards = 0

def processDepSentence(sid, sent):
	global g_n, g_next, g_skipForward, g_skipBackwards
	
	nodes, root = createNodes(sent, sid)
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

def main():
	global g_n, g_next, g_skipForward, g_skipBackwards

	for i, sent in enumerate(readDepSentences(sys.stdin)):
		processDepSentence(i+1, sent)
	
	n = float(g_n)
	d = {'nextP' : g_next / n, 'skipForwardP' : g_skipForward / n, 'skipBackwardsP' : g_skipBackwards / n}
	print d

if __name__ == '__main__':
	main()

