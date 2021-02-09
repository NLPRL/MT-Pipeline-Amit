import os
import SSF_converter.SSF_to_Input as input_converter
import SSF_converter.output_to_SSF as ssf_converter
import SSF_converter.output_to_SSF2 as ssf_converter2
import morph_analyser.make_prediction as morph_analyser
import Pos_Tagger.final_predict_model as pos_tagger
import chunking.predict as chunker
import lexical.dictionaryAmit1 as lexical
# import morph_generation.morph_inflection as morph_generator
import torch

from wxconv import WXC

con = WXC(order='utf2wx')
con1 = WXC(order='wx2utf', lang='hin')

BASE_DIR = os.path.dirname(os.path.abspath('__file__'))
BASE_DIR += '/SSF_converter/'
main_file = BASE_DIR + "main_format.txt"
local_add = os.path.dirname(os.path.abspath('__file__'))
pos_tagger_input_file = local_add + '/Pos_Tagger/sentinput.txt'
chunker_input_file = local_add + '/chunking/input.txt'

match_list=[]

checklist = []

def main_format_writer(data):
    # This file writes in main_format.txt.
    out_main_file = open(main_file, 'w', encoding='utf-8')
    for each in main_format_data:
        out_main_file.write('\t'.join(each) + '\n')
    out_main_file.write('\n')
    out_main_file.flush()
    out_main_file.close()

def block_maker():
    # This function print a line in the output file SSF.txt
    for i in range(80):
        ssf_converter.out_temp_file.write('-')
    ssf_converter.out_temp_file.write('\n\n')
    ssf_converter.out_temp_file.flush()

while 1:
    block_maker()
    inp = input("Enter the sentence in Bhojpuri: ").split()
    print(inp)
    ssf_converter.out_temp_file.write('New Sentence = ' + ' '.join(inp) + '\n\n')
    # main format data store the output of different modules.
    main_format_data = []

    for i in range(1, len(inp) + 1):
        temp = []
        temp.append(str(i))
        temp.append('open_bracket_here')
        main_format_data.append(temp)
        temp = []
        temp.append(str(i) + '.1')
        temp.append(str(i - 1))
        temp.append(str(i))
        temp.append(inp[i - 1])
        main_format_data.append(temp)

    # Morph analyser is run here and output is store in
    # "output" variable.
    output = morph_analyser.main(inp)
    outputM = output
    # print(outputM)
    print(output)
    # storelist = []
    for c in range(len(output)):
        # match_list.append(output[c][0][0])
         match_list.append(output[c][0][2])
         # storelist.append(output[c][0][2])
         # storelist.append(output[c][0][3])
         # storelist.append(output[c][0][4])
         # storelist.append(output[c][0][5])
         # storelist.append(output[c][0][7])
    print(match_list)
    # print(storelist)

    # output is stored in "main_format_data" from "output"
    # variable
    j = 0
    for i in range(len(output)):
        while main_format_data[j][1] == 'open_bracket_here':
            j += 1
        main_format_data[j].append(output[i][0][2])
        main_format_data[j].append(output[i][0][1])
        main_format_data[j].append('')
        main_format_data[j].append(output[i][0][3])
        main_format_data[j].append(output[i][0][4])
        main_format_data[j].append(output[i][0][5])
        main_format_data[j].append(output[i][0][6])
        main_format_data[j].append(output[i][0][7])
        main_format_data[j].append('')
        main_format_data[j].append('')
        j += 1

    # output from morph analyser is stored in file
    # main_format.txt
    print(main_format_data)
    main_format_writer(main_format_data)

    ssf_converter.out_temp_file.write('\t\t***Output after Morph Analyser***\n\n')

    # this function converts the data from main_format.txt
    # to SSF and stored in SSF.txt
    ssf_converter.func()

    # Input is written in sentinput.txt file in POS_Tagger
    # directory in the order word , pos , gender , number,
    # person, case ,tam
    pos_tagger_input = open(pos_tagger_input_file, 'w', encoding='utf-8')

    for j in range(len(main_format_data)):
        if main_format_data[j][1] == 'open_bracket_here':
            continue
        temp = main_format_data[j][3]
        for k in range(7, 12):
            temp += '\t' + main_format_data[j][k]
        temp += '\n'
        pos_tagger_input.write(temp)
    pos_tagger_input.flush()
    pos_tagger_input.close()

    # POS tagger module is run here and output is taken in
    # in "output" variable
    ssf_converter.out_temp_file.write('\t\t***Output after POS Tagger***\n\n')
    output = pos_tagger.pos_main()
    # print(output)

    # output is stored in "main_format_data" from "output"
    # variable
    i = 0
    for j in range(len(main_format_data)):
        if main_format_data[j][1] == 'open_bracket_here':
            continue
        main_format_data[j][4] = output[0][i]
        i += 1

    # POS tagger output is written in main_format.txt
    main_format_writer(main_format_data)

    # output is converted in SSF and stored in SSF.txt
    ssf_converter.func()

    # Input is written in input.txt file in chunking
    # directory in the order word , pos , gender , number,
    # person, case ,tam
    chunker_input = open(chunker_input_file, 'w', encoding='utf-8')

    for j in range(len(main_format_data)):
        if main_format_data[j][1] == 'open_bracket_here':
            continue
        temp = main_format_data[j][3]
        temp += '\t' + main_format_data[j][4]
        for k in range(7, 12):
            temp += '\t' + main_format_data[j][k]
        temp += '\n'
        chunker_input.write(temp)
    chunker_input.flush()
    chunker_input.close()

    # chunker module is run here and output is taken in
    # in "output" variable
    ssf_converter.out_temp_file.write('\t\t***Output after Chunker***\n\n')
    output = chunker.main_chunker()
    # print(output)

    i = 0
    for j in range(len(main_format_data)):
        if main_format_data[j][1] == 'open_bracket_here':
            continue
        main_format_data[j][12] = output[0][i]
        i += 1

    # Chunker output is written in main_format.txt

    main_format_writer(main_format_data)

    # output is converted in SSF with the help of second
    # type of converter and stored in SSF.txt
    ssf_converter2.func()
    print(main_format_data)

    i = 0
    # print(len(main_format_data))

    for j in range(len(main_format_data)):

        if main_format_data[j][1] == 'open_bracket_here':
            # main_format_data[j+1][5] = con1.convert(lexical.convertBhoj(con.convert(main_format_data[j+1][5]), match_list[j//2]))
            continue

        main_format_data[j][5] = con1.convert(
            lexical.convertBhoj(con.convert(main_format_data[j][5]), match_list[(j//2)]))
        print(main_format_data[j][5])

        i += 1






    # print(main_format_data)
    ssf_converter.out_temp_file.write('\t\t***Output after Lexical Generator***\n\n')

    main_format_writer(main_format_data)

    ssf_converter2.func()
    print(main_format_data)

    # for j in range(len(main_format_data)):
    #     if main_format_data[j][1] == 'open_bracket_here':
    #         continue
    #     if outputM[j//2][0][2] == 'v':
    #         main_format_data[j][3] = morph_generator.main(main_format_data[j][5], outputM[j//2][0][2] +';'+ outputM[j//2][0][3] +';'+ outputM[j//2][0][4] +';'+ outputM[j//2][0][5] +';'+ outputM[j//2][0][7])
    # print(main_format_data)
    g = ''
    for j in range(len(main_format_data)):
        if (j % 2) != 0:
            print(main_format_data[j][5])
            g = g + main_format_data[j][5] +" "
    # print(main_format_data[1][5])
    # ssf_converter.out_temp_file.write('Final Output = ' + ' '.join(output) + '\n\n')
    print(g)
    ssf_converter.out_temp_file.write('\t\t***Target Sentence in Hindi***\n\n')
    ssf_converter.out_temp_file.write('Final Output = '+' '+ g)
