import os
import string
file_in = './data.conllu'
error= './error.txt'
temp_file = './input_file2.txt'
input_file=open(file_in,'r')
output_file=open(temp_file,'w')
lines=input_file.readlines()
final_list=[]
prev=''
sentence=[]

for i in lines:
    i=i.rstrip('\n')
    if (i==''):
        output_file.write('\n')
        final_list.append(sentence)
        sentence=[]
        prev=''
        continue
    if (i[0]=='#'):
        continue
    split_line=i.split('\t')
    word=[]
    word.append(split_line[1]) #word
    output_file.write(split_line[1]+'\t')
    d={}
    word.append(split_line[3]) #pos_tag
    output_file.write(split_line[3]+'\t')
    if i[5]!='_':
        features1=split_line[5].split('|')
        for j in features1:
            try:
                key,value=j.split('=')
                d[key]=value
            except:
                continue
    if i[9]!='_':
        features2=split_line[9].split('|')
        for j in features2:
            try:
                key,value=j.split('=')
                d[key]=value
            except:
                continue
    word.append(d.get('Case','<unk>'))
    output_file.write(d.get('Case','<unk>')+'\t')
    word.append(d.get('Gender','<unk>'))
    output_file.write(d.get('Gender','<unk>')+'\t')
    word.append(d.get('Number','<unk>'))
    output_file.write(d.get('Number','<unk>')+'\t')
    word.append(d.get('Person','<unk>'))
    output_file.write(d.get('Person','<unk>')+'\t')
    word.append(d.get('Tam','<unk>'))
    output_file.write(d.get('Tam','<unk>')+'\t')
    word.append(d.get('ChunkId','<unk>'))
    r1=word[0]
    r2=d.get('ChunkId','<unk>')
    r3=r2.rstrip(string.digits)
    print (r3)
    if (r1=='।' or r1=='.'):
        output_file.write('O'+'\n')
        prev=''
    else:
        if (r2==prev):
            output_file.write('I-'+r3+'\n')
        else:
            output_file.write('B-'+r3+'\n')
            prev=r2
    sentence.append(word)
print (final_list[0])




