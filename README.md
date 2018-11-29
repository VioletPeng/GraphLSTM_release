# Cross-Sentence N-ary Relation Extraction with Graph LSTMs 

This is the data and source code of the papers:

**Cross-Sentence N-ary Relation Extraction with Graph LSTMs**
Nanyun Peng, Hoifung Poon, Chris Quirk, Kristina Toutanova and Wen-tau Yih  
*Transactions of the Association for Computational Linguistics*, Vol 5, 2017

If you use the code, please kindly cite the following bibtex:

@article{peng2017cross,  
  title={Cross-Sentence N-ary Relation Extraction with Graph LSTMs},  
  author={Peng, Nanyun and Poon, Hoifung and Quirk, Chris and Toutanova, Kristina and Yih, Wen-tau},  
  journal={Transactions of the Association for Computational Linguistics},  
  volume={5},  
  pages={101--115},  
  year={2017}  
}  

## Data
File system hierarchy:
data/
    drug_gene_var/
        0/
            data_graph
            sentences_2nd
            graph_arcs
        1/
        2/
        3/
        4/
    drug_var/
        the same structure as in drug_gene_var
    drug_gene/
        the same structure as in drug_gene_var
### Source attribution: 
The full information of the instances are contained in the file "data_graph", it's a json format file containing information such as PubMed articleID, paragraph number, sentence number, and the information about the tokens including part-of-speech tags, dependencies, etc. produced by Stanford coreNLP tool. 

### Preprocessing
We processed the source data into the format that is easier for our code to consume, which includes two files: "sentences_2nd" and "graph_arcs". The "sentences_2nd" file contains the information of the raw input, and the format is:

the-original-sentences<TAB>indices-to-the-first-entity(drug)<TAB>indices-to-the-second-entity(gene/variant)[<TAB>indices-to-the-third-entity(variant)]<TAB>relation-label   

The "graph_arcs" file contains the information of the dependencies between the words, including time sequence adjacency, syntactic dependency, and discourse dependency. The format is:

dependencies-for-node-0<WHITESPACE>dependencies-for-node-1...
dependencies-for-node-n = dependency-0,,,dependency-1...
dependency-n = dependency-type::dependent-node

## Experiments
To reproduce the results in our paper, the script ./scripts/batch_run_lstm.sh contains the command for running all the cross-validation folds for both drug-gene-variant triple and drug-variant binary relations.

The script ./scripts/batch_run_multitask.sh contains the command for running all the multi-task learning experiments.
