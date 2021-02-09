import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR += '/'

adj=BASE_DIR + "adj-crossLinks.idx"
adv=BASE_DIR + "adv-crossLinks.idx"
fw=BASE_DIR + "fw-crossLinks.idx"
noun=BASE_DIR + "noun-crossLinks.idx"
verb=BASE_DIR + "verb-crossLinks.idx"
htb="Unk"
minidex=100000000000
def convert(a,pos_tag):
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
	global htb
	global minidex
	line=f1.readline()
	while(line):
		line=line.rstrip('\n')
		try:
			rooth,rootb=line.split('\t')[:2]
		except:
			print (line)
			exit(0)
		roothid=0
		roothw=""
		for i in range(len(rooth)):
			if (rooth[i].isdigit()):
				roothw=rooth[:i]
				roothid=int(rooth[i:])
				break
		rootb=rootb.split("-")[1]
		if (roothw==a and roothid<minidex):
			htb=rootb
			minidex=roothid
		line=f1.readline()
	if (htb != 'Unk'):
		return htb
	else:
		return a
	# return htb
print (convert("pawwala","noun"))


