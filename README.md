# pipeline-mt
Source code is for Bhojpuri to Hindi baseline Machine Translation System.

  
Repository contains MorphAnalyzer, pos tagger, chunker, lexical, Transduction, morph generator modules.

We have to implement all this module sequentially the same as in the pipeline, so we can get the end-to-end translation. For example, If we give input  "स्नान-पूजा का बाद कुन्ती अनमनाइल मन से
कुछ सोचत रहली।" as Bhojpuri sentence then we get output "स्नान-पूजा  के बाद कुन्ती बेचैन मन से कुछ सोच रही थीं।" in Hindi. We have to assemble all this module.

Run the mainFinal.py file to execute the model.

Our model follow the below mention pipeline order:

Morph-analyzer: Morph-analyzer implemented using data in conll format. It takes input sentence and process it and gives output in conll format. you have to convert this output conll format into SSF format.

Pos-tagger: This module takes input in SSF format and process the input and generates tags in SSF format.

Chunker: This module takes pos tagger output and process it and perform chunking and gives output in SSF format.

Lexical: This module search the lemma of source language in Bhojpuri-Hindi bilingual dictionary and replace it with lemma of the target language.

Morph-generator: This module generates inflection of word. You have to apply it on lemma of output we got in Lexical module.

Finally, we got the target language output in SSF format.

then convert the output in SSF format to a Sentence.
