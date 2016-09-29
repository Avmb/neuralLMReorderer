import sys
import re

def removeClitics(tokens):
	# German doesn't need this
	return
#	i = 0
#	end = len(tokens)-1
#	while i < end:
#		tok = tokens[i]
#		if (tok[1][-1] != '-') or (len(tok[1]) == 1):
#			i += 1
#			continue
#		next = tokens[i+1]
#		tok[1] = tok[1][0:-1] + next[1]
#		if (tok[6] == str(next[0])):
#			tok[6] = next[6]
#		tokens.remove(next)
#		# renumber tokens
#		for tok in tokens:
#			if tok[0] > i + 1:
#				tok[0] = tok[0] - 1
#			if int(tok[6]) > i + 1:
#				tok[6] = str(int(tok[6]) - 1)
#		end -= 1

class Node:
	def __init__(self, token):
		if (len(token) < 9) or (token[8] == '_'):
			token.insert(8, '-1')
		self.token = token
		self.children = set([])
		self.id = int(token[0])
		self.form = token[1]
		self.lemma = token[2]
		self.cpos = token[3]
		self.pos = token[4]
		self.morphStr = token[5]
		self.morph = morphDict(self.morphStr)
		self.parentId = int(token[6])
		self.deprel = token[7]
		self.hint = int(token[8])
		self.parent = None
		self.isEmitted = False
		self.emitCount = 0
		self.leftChildrenCount = 0
		self.rightChildrenCount = 0
		self.leftEmittedCount = 0
		self.rightEmittedCount = 0
		self.visitCount = 0
		self.unemittedChildrenIds = set()
	
	def toConllStr(self):
		return '\t'.join((str(self.id), self.form, self.lemma, self.cpos, self.pos, self.morphStr, str(self.parentId), self.deprel, str(self.hint), '_', '_'))
	
	def getMorphStr(self):
		morphList = ['='.join(morphPair) for morphPair in self.morph.iteritems()]
		morphList.sort()
		return '|'.join(morphList)
			
	
	def getClosestLeftChild(self):
		if (self.leftChildrenCount == 0):
			return leftPaddingNode
		rv = None
		rvId = -1
		for child in self.children:
			if (child.id < self.id) and (child.id > rvId):
				rv = child
				rvId = child.id
		return rv
	
	def getClosestRightChild(self):
		if (self.rightChildrenCount == 0):
			return rightPaddingNode
		rv = None
		rvId = float('inf')
		for child in self.children:
			if (child.id > self.id) and (child.id < rvId):
				rv = child
				rvId = child.id
		return rv
	
	
	def isLeaf(self):
		return len(self.children) == 0
	
	def isSubTreeEmitted(self):
		return self.emitCount == 1 + len(self.children)
	
	def hasAllLeftEmitted(self):
		return self.leftEmittedCount == self.leftChildrenCount

	def hasAllRightEmitted(self):
		return self.rightEmittedCount == self.rightChildrenCount
	
	def hasAllChildrenEmitted(self, exceptChildId = None):
		unemittedChildrenNum = len(self.unemittedChildrenIds)
		if not exceptChildId:
			return unemittedChildrenNum == 0
		else:
			return (unemittedChildrenNum == 0) or ((unemittedChildrenNum == 1) and (exceptChildId in self.unemittedChildrenIds))
		
	def incEmitted(self):
		self.emitCount += 1
		if (self.parent != None) and (self.isSubTreeEmitted()):
			self.parent.incEmitted()
			if (self.id < self.parentId):
				self.parent.leftEmittedCount += 1
			else:
				self.parent.rightEmittedCount += 1
			self.parent.unemittedChildrenIds.remove(self.id)
	
def morphDict(morphStr):
		rv = dict()
		morphStr = morphStr.strip()
		if (morphStr == '_'):
			return rv
		
		morphToks = morphStr.split('|')
		mi = 0
		for morphTok in morphToks:
			kv = morphTok.split('=')
			#print >> sys.stderr, kv
			if len(kv) >= 2:
				rv[kv[0].strip()] = kv[1].strip()
			else:
				mk = 'm'+str(mi)
				mi += 1
				rv[mk] = kv[0].strip()
		return rv
		
def createNodes(tokens, sentenceSeqId, escapeSpecialChars = False):
	root = None
	nodes = [Node(token) for token in tokens]
	for node in nodes:
		if (escapeSpecialChars):
			node.form = re.sub(r',', '<comma>', node.form)
			node.form = re.sub(r'"', '<quote>', node.form)
			node.form = re.sub(r'%', '<percent>', node.form)
			
		if (node.parentId != 0):
			node.parent = nodes[node.parentId - 1]
			node.parent.children.add(node)
			if (node.id < node.parentId):
				node.parent.leftChildrenCount += 1
			else:
				node.parent.rightChildrenCount += 1
			node.parent.unemittedChildrenIds.add(node.id)
		else:
			if (root == None):
				root = node
			else:
				print >> sys.stderr, "Error: multiple roots in sentence:", sentenceSeqId
				#for node in nodes:
				#	print >> sys.stderr, node.toConllStr()
				#print >> sys.stderr
				return None
	return (nodes, root)

leftPaddingNode = Node(["-1", "<s>", "<s>", "<s>", "<s>", "<s>=<s>", "-1", "<s>", "-1", "_"])
rightPaddingNode = Node(["-2", "</s>", "</s>", "</s>", "</s>", "</s>=</s>", "-2", "</s>", "-2", "_"])

def nodeAtPadded(nodes, idx):
	global leftPaddingNode, rightPaddingNode
	
	if (idx < 0):
		return leftPaddingNode
	elif (idx >= len(nodes)):
		return rightPaddingNode
	return nodes[idx]
	

def readDepSentences(fs):
	sent = []
	for line in fs:
		tokens = line.split()
		if len(tokens) == 0:
			yield sent
			sent = []
		else:
			sent.append(tokens)
	if len(sent) > 0:
		yield sent

