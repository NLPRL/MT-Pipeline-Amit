import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR += '/'

adj=BASE_DIR + "adj-crossLinks.idx"
adv=BASE_DIR + "adv-crossLinks.idx"
fw=BASE_DIR + "fw-crossLinks.idx"
noun=BASE_DIR + "noun-crossLinks.idx"
verb=BASE_DIR + "verb-crossLinks.idx"
# htb="Unk"
def convertBhoj(a,pos_tag):
	if (pos_tag=="adj"):
		f1=open(adj,'r')
	elif (pos_tag=='n'):
		f1=open(noun,'r')
	elif (pos_tag=='adv'):
		f1=open(adv,'r')
	elif (pos_tag=="v"):
		f1=open(verb,'r')
	else:
		f1=open(fw,'r')
	# global htb
	# global minidex
	htb = "Unk"
	line=f1.readline()
	while(line):
		line=line.rstrip('\n')
		rooth,rootb=line.split('\t')[:2]
		# roothid=0
		roothw=""
		for i in range(len(rooth)):
			if (rooth[i].isdigit()):
				roothw=rooth[:i]
				# roothid=int(rooth[i:])
				break
		rootb=rootb.split("-")[1]
		if (rootb==a):
			htb=roothw
			# minidex=roothid
		line=f1.readline()
	if (htb != 'Unk'):
		return htb
	else:
		return a

# print (convertBhoj("nevawA","avy"))
# print (convertBhoj("xeKa","v"))
# print (convertBhoj("mana","punc"))
# print (convertBhoj("cataka","n"))
# print (convertBhoj("gaila","v"))