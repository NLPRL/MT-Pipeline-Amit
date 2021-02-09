import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR+='/'
file_in = BASE_DIR+"temp_SSF.txt"
error= BASE_DIR+"error_conv1.txt"
temp_file = BASE_DIR+"out.txt"

error1=open(error,'w',encoding='utf-8')
out_temp_file = open(temp_file, 'w', encoding='utf-8')
    
def file_writer(list_list):
    size = len(list_list)
    for num in range(14-size):
        list_list.append("")
    try:
        out_temp_file.write("\t".join(list_list)+'\n')
    except:
        error1.write("\t".join(list_list)+'\n')

def open_bracket(inp):
    try:
        out_temp_file.write(inp+"\topen_bracket_here"+'\n')
    except:
        error1.write(inp+'\n')    

def attribute_pair_extractor(raw_data):
    attribute_pair = []
    if len(raw_data)==0:
        return attribute_pair
    start=0
    while raw_data[start]!='=':
        start+=1
    start+=1
    end=start
    while raw_data[end]!='>':
        end+=1
    attribute_pair = raw_data[start:end].split(",")
    return attribute_pair 

def sentence_cleaner(sentence):
    group = []
    count_open=0
    count_close=0
    for word in sentence:
        if word[0]=='))':
            count_close+=1
            try:
                group.pop()
            except:
                pass
        elif word[1]=='((':
            count_open+=1
            open_bracket(word[0])
            try:
                group.append(word[2])
            except:
                pass
        else:
            atom = []
            atom.append(word[0])
            atom.append(str(count_close))
            atom.append(str(count_open))
            atom.append(word[1])
            try:
                atom.append(word[2])
                attribute_pair = attribute_pair_extractor(word[3])
                for each in attribute_pair:
                    atom.append(each)
                for each in group:
                    atom.append(each)
            except:
                pass
            file_writer(atom)
    out_temp_file.write('\n')

def func():
    sentence_ = []
    with open(file_in, 'r', encoding='utf-8') as f1:
        for line in f1:
            if line != '\n':
                pair = line.strip().split('\t')
                sentence_.append(pair)
            else:
                sentence_builder(sentence_)
                sentence_.clear()
    sentence_cleaner(sentence_)
