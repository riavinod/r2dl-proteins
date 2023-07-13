# Source: https://pypi.org/project/bert-embedding/
# pip install bert-embedding
from bert_embedding import BertEmbedding
import pandas as pd

def get_bert_embeddings(path_to_dataset, file_type):
    '''TODO specify dataset which is path to a csv file of protein sequences or english sentences'''
    df = pd.read_csv(path_to_dataset) # read in enligh or protein sequence vocabularies here
 
    if file_type == 'csv':
        myFile = open(path_to_dataset) # file should be a csv
        text = myFile.readline()
        while text != "":
            print(text)
            text = myFile.readline()
        myFile.close()
    
        bert_embedding = BertEmbedding()
        result = bert_embedding(text.split('\n'))
        
        return result
    
    if file_type == 'txt':
        with open(path_to_dataset) as f:
            lines = f.readlines()
        bert_embedding = BertEmbedding()
        result = bert_embedding(lines.split('\n'))
        
        return result
    
    return None


def get_protein_embeddings(path_to_dataset, file_type):
    '''function to get per presidue protein embeddings.'''
    

# get the embeddings for english data
V_S = get_bert_embeddings('data/english/imdb_data', txt)
V_T = get_bert_embeddings('data/protein/secondary_structure.csv', txt) # change data set as per protein task

np.save('embeddings/english/bert_sentiment.npy') # TODO: change the language model task name here so it goes to the correct folder
np.save('embeddings/protein/secondary_structure.npy') # TODO: change the protein task name here so it goes to the correct folder

