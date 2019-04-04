

import Dataset_load
from flair.embeddings import CharLMEmbeddings
from flair.data import Sentence

from flair.embeddings import WordEmbeddings, CharLMEmbeddings

import Corpus
import Evaluation
import NER
import os
import gensim
import numpy as np
from gensim.test.utils import datapath, get_tmpfile
from gensim.models import KeyedVectors
from flair.data import Sentence
from flair.embeddings import StackedEmbeddings

import cv2

SEED = 86
np.random.seed(SEED)



os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from gensim.models import word2vec


custom_embedding = WordEmbeddings('pathtotoEmbeddings.vec')




# now create the StackedEmbedding object that combines all embeddings
stacked_embeddings =StackedEmbeddings(embeddings=[custom_embedding])# , charlm_embedding_forward,charlm_embedding_backward])




dataset_dict = Dataset_load.load()



corp = Corpus.Corpus(dataset_dict, embeddings_file_path=None,stacked_embeddings = stacked_embeddings)


model_params = {"filter_width": 3,
                "embeddings_dropout": True,
                "n_filters": [
                    256
                ],
                "dense_dropout" : True,
                "token_embeddings_dim": 300,
                "char_embeddings_dim": 50,
                "cell_type":'lstm',
                "use_batch_norm": True,
                "concat_embeddings":True,
                "use_crf": True,
                "use_char_embeddins":True,
                "net_type": 'rnn',
                "use_capitalization":False ,
               }
 

net = NER.NER(corp,stacked_embeddings, **model_params)




learning_params = {'dropout_rate': 0.5,
                     'epochs':200,
                     'learning_rate': 0.001,# 0.0003
                     'batch_size':20,
                     'learning_rate_decay': 0.94}

  
results = net.fit(**learning_params)

