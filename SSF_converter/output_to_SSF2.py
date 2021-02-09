import os
import queue
import SSF_converter.output_to_SSF as ssf_converter

    
def file_writer(list_list):
    try:
        ssf_converter.out_temp_file.write("\t".join(list_list)+'\n')
    except:
        ssf_converter.error1.write("\t".join(list_list)+'\n')


def print_close_brackets():
    try:
        ssf_converter.out_temp_file.write('\t'+'))'+'\n')
    except:
        ssf_converter.error1.write('Unable to write in file'+'\n')


def print_open_brackets(pos,tag):
    try:
        ssf_converter.out_temp_file.write(str(pos[0])+'\t'+'(('+'\t'+tag+'\n')
    except:
        ssf_converter.error1.write('Unable to write in file at'+pos+'\n')


def main_call(pos,word):
    output = []
    output.append(str(pos[0])+'.'+str(pos[1]))
    try:
        output.append(word[3])
        output.append(word[4])
        tmp=','.join(word[5:12])
        tmp='<fs af='+tmp+'>'
        output.append(tmp)
    except:
        pass
    file_writer(output)


def sentence_builder(sentence):
    open_till=0
    pos = [0,1]
    for word in sentence:
        if word[1]=='open_bracket_here':
            continue
        else:
            if word[12][0] != 'I':
                pos[0]+=1
                pos[1]=1
                for num in range(open_till):
                    print_close_brackets()
                open_till=1
                if word[12][0]=='O':
                    print_open_brackets(pos,word[12])
                else:
                    print_open_brackets(pos,word[12][2:]) 
            else:
                pos[1]+=1
            main_call(pos,word)
    for num in range(open_till):
        print_close_brackets()
    ssf_converter.out_temp_file.write('\n')
    ssf_converter.out_temp_file.flush()



def func():
    sentence_ = []
    with open(ssf_converter.file_in, 'r', encoding='utf-8') as f1:
        for line in f1:
            if line != '\n':
                pair = line.strip().split('\t')
                sentence_.append(pair)
            else:
                sentence_builder(sentence_)
                sentence_.clear()
    sentence_builder(sentence_)