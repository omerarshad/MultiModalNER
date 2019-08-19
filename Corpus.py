"""
Copyright 2017 Neural Networks and Deep Learning lab, MIPT
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

from collections import Counter
from collections import defaultdict
import random
import numpy as np

import tensorflow as tf

from tensorflow.contrib.slim import nets

from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
from flair.embeddings import CharLMEmbeddings
from flair.data import Sentence

from flair.embeddings import WordEmbeddings, CharLMEmbeddings

from gensim.models import word2vec



DATA_PATH = '/tmp/ner'
DOC_START_STRING = '-DOCSTART-'
SEED = 86
SPECIAL_TOKENS = ['<PAD>', '<UNK>']
SPECIAL_TAGS = ['<PAD>']

np.random.seed(SEED)
random.seed(SEED)


# Dictionary class. Each instance holds tags or tokens or characters and provides
# dictionary like functionality like indices to tokens and tokens to indices.
class Vocabulary:
    def __init__(self, tokens=None, default_token='<UNK>', is_tags=False):
        if is_tags:
            special_tokens = SPECIAL_TAGS
            self._t2i = dict()
        else:
            special_tokens = SPECIAL_TOKENS
            if default_token not in special_tokens:
                raise Exception('SPECIAL_TOKENS must contain <UNK> token!')
            # We set default ind to position of <UNK> in SPECIAL_TOKENS
            # because the tokens will be added to dict in the same order as
            # in SPECIAL_TOKENS
            default_ind = special_tokens.index('<UNK>')
            self._t2i = defaultdict(lambda: default_ind)
        self._i2t = list()
        self.frequencies = Counter()

        self.counter = 0
        for token in special_tokens:
            self._t2i[token] = self.counter
            self.frequencies[token] += 0
            self._i2t.append(token)
            self.counter += 1
        if tokens is not None:
            self.update_dict(tokens)

    def update_dict(self, tokens):
        for token in tokens:
            if token not in self._t2i:
                self._t2i[token] = self.counter
                self._i2t.append(token)
                self.counter += 1
            self.frequencies[token] += 1

    def idx2tok(self, idx):
        return self._i2t[idx]

    def idxs2toks(self, idxs, filter_paddings=False):
        toks = []
        for idx in idxs:
            if not filter_paddings or idx != self.tok2idx('<PAD>'):
                toks.append(self._i2t[idx])
        return toks

    def tok2idx(self, tok):
        return self._t2i[tok]


    def toks2idxs(self, toks):
        return [self._t2i[tok] for tok in toks]

    def batch_toks2batch_idxs(self, b_toks):
        max_len = max(len(toks) for toks in b_toks)
        # Create array filled with paddings
        batch = np.ones([len(b_toks), max_len]) * self.tok2idx('<PAD>')
        for n, tokens in enumerate(b_toks):
            idxs = self.toks2idxs(tokens)
            batch[n, :len(idxs)] = idxs
        return batch

    def batch_idxs2batch_toks(self, b_idxs, filter_paddings=False):
        return [self.idxs2toks(idxs, filter_paddings) for idxs in b_idxs]

    def is_pad(self, x_t):
        assert type(x_t) == np.ndarray
        return x_t == self.tok2idx('<PAD>')

    def __getitem__(self, key):
        return self._t2i[key]

    def __len__(self):
        return self.counter

    def __contains__(self, item):
        return item in self._t2i


class Corpus:
    def __init__(self, dataset=None, embeddings_file_path=None, dicts_filepath=None,stacked_embeddings = None):
        if dataset is not None:
            self.dataset = dataset
            self.token_dict = Vocabulary(self.get_tokens())
            self.tag_dict = Vocabulary(self.get_tags(), is_tags=True)
            self.char_dict = Vocabulary(self.get_characters())
            self.stacked_embeddings = stacked_embeddings
        elif dicts_filepath is not None:
            self.dataset = None
            self.load_corpus_dicts(dicts_filepath)
        if embeddings_file_path is not None:
            self.embeddings = self.load_embeddings(embeddings_file_path)
        else:
            self.embeddings = None

    # All tokens for dictionary building
    def get_tokens(self, data_type='train'):
        for tokens, _,_ in self.dataset[data_type]:
            for token in tokens:
                yield token

    # All tags for dictionary building
    def get_tags(self, data_type=None):
        if data_type is None:
            data_types = self.dataset.keys()
        else:
            data_types = [data_type]
        for data_type in data_types:
            for _, tags,_ in self.dataset[data_type]:
                for tag in tags:
                    yield tag

    # All characters for dictionary building
    def get_characters(self, data_type='train'):
        for tokens, _,_ in self.dataset[data_type]:
            for token in tokens:
                for character in token:
                    yield character

    def load_embeddings(self, file_path):
        # Embeddins must be in fastText format either bin or
        print('Loading embeddins...')
        if file_path.endswith('.bin'):
            from gensim.models.wrappers import FastText
            embeddings = FastText.load_fasttext_format(file_path)
        else:
            from gensim.models import KeyedVectors
            embeddings = KeyedVectors.load_word2vec_format(file_path)
        return embeddings

    def tokens_to_x_and_xc(self, tokens):
        n_tokens = len(tokens)
        tok_idxs = self.token_dict.toks2idxs(tokens)
        char_idxs = []
        max_char_len = 0
        for token in tokens:
            char_idxs.append(self.char_dict.toks2idxs(token))
            max_char_len = max(max_char_len, len(token))
        toks = np.zeros([1, n_tokens], dtype=np.int32)
        chars = np.zeros([1, n_tokens, max_char_len], dtype=np.int32)
        toks[0, :] = tok_idxs
        for n, char_line in enumerate(char_idxs):
            chars[0, n, :len(char_line)] = char_line
        return toks, chars

    def batch_generator(self,
                        batch_size,
                        dataset_type='train',
                        shuffle=False,
                        allow_smaller_last_batch=True,
                        start = 0):
        tokens_tags_pairs = self.dataset[dataset_type][start:start+1]
        
        
        
#         tokens_tags_pairs_sound = self.dataset[dataset_type+"_phonetics"]
        
        
        
        n_samples = len(tokens_tags_pairs)
        print("n_samples : ",n_samples)
        if shuffle:
            order = np.random.permutation(n_samples)
        else:
            order = np.arange(n_samples)
        n_batches = n_samples // batch_size
        if allow_smaller_last_batch and n_samples % batch_size:
            n_batches += 1
        for k in range(n_batches):
            batch_start = k * batch_size
            batch_end = min((k + 1) * batch_size, n_samples)
            x_batch = [tokens_tags_pairs[ind][0] for ind in order[batch_start: batch_end]]
            y_batch = [tokens_tags_pairs[ind][1] for ind in order[batch_start: batch_end]]
            img_batch = [tokens_tags_pairs[ind][2] for ind in order[batch_start: batch_end]]

            
            

#             s_batch = [tokens_tags_pairs_sound[ind][0] for ind in order[batch_start: batch_end]]

            
            
            
            x, y,img = self.tokens_batch_to_numpy_batch(x_batch, y_batch,img_batch)

            yield x, y, img

    def tokens_batch_to_numpy_batch(self, batch_x, batch_y=None,img_batch = None,scope = "one"):
        
        x = dict()
        # Determine dimensions
        batch_size = len(batch_x)
        max_utt_len = max([len(utt) for utt in batch_x])
        # Fix batch with len 1 issue (https://github.com/deepmipt/ner/issues/4) 
        max_utt_len = max(max_utt_len, 2)
        max_token_len = max([len(token) for utt in batch_x for token in utt])

        # Check whether bin file is used (if so then embeddings will be prepared on the go using gensim)
        prepare_embeddings_onthego = True#self.embeddings is not None
        # Prepare numpy arrays
        if prepare_embeddings_onthego:  # If the embeddings is a fastText model
            x['emb'] = np.zeros([batch_size, max_utt_len,self.stacked_embeddings.embedding_length], dtype=np.float32)
#             x['emb_sound'] = np.zeros([batch_size, max_utt_len,stacked_embeddings_sound.embedding_length], dtype=np.float32)
#             x['emb_backward'] = np.zeros([batch_size, max_utt_len,charlm_embedding_backward.embedding_length], dtype=np.float32)

        x['token'] = np.ones([batch_size, max_utt_len], dtype=np.int32) * self.token_dict['<PAD>']
        x['wordindex'] = np.ones([batch_size, max_utt_len], dtype=np.int32) * self.token_dict['<PAD>']
        
        x['char'] = np.ones([batch_size, max_utt_len, max_token_len], dtype=np.int32) * self.char_dict['<PAD>']

        # Capitalization
        x['capitalization'] = np.zeros([batch_size, max_utt_len], dtype=np.float32)
        for n, utt in enumerate(batch_x):
            for k, tok in enumerate(utt):
                if len(tok) > 0 and tok[0].isupper():
                    x['capitalization'][n, k] = 1

        # Prepare x batch
        for n, utterance in enumerate(batch_x):
            if prepare_embeddings_onthego:
                utterance_vectors = np.zeros([len(utterance), self.stacked_embeddings.embedding_length])#np.zeros([len(utterance), self.embeddings.vector_size])
#                 utterance_vectors_sound = np.zeros([len(utterance), stacked_embeddings_sound.embedding_length])
#                 utterence_vectors_forward = np.zeros([len(utterance), charlm_embedding_forward.embedding_length])#np.zeros([len(utterance), self.embeddings.vector_size])
#                 utterence_vectors_backward = np.zeros([len(utterance), charlm_embedding_backward.embedding_length])
#                 sounds = []
#                 for w in utterance:
                  

                  
#                   sounds.append(''.join(g2p(w)))
        
                sentence = Sentence(' '.join(utterance))
#                 sentence_2 = Sentence(' '.join(s))
#                 sentence_3 = Sentence(' '.join(utterance))

                # just embed a sentence using the StackedEmbedding as you would with any single embedding.
                self.stacked_embeddings.embed(sentence)
#                 stacked_embeddings_sound.embed(sentence_2)
#                 stacked_embeddings.embed(sentence_2)
#                 charlm_embedding_backward.embed(sentence_3)

#                 a=[]
#                 for token in sentence:
#                   a.append(np.asarray(token.embedding))
                
               
                for q,token in enumerate(sentence):
                    
                    try:
                        utterance_vectors[q] = token.embedding #self.embeddings[token.lower()]
#                         utterance_vectors_sound[q] = stoken.embedding
#                         utterence_vectors_forward[q] = token2.embedding
#                         utterence_vectors_backward[q] = token3.embedding
                    except KeyError:
                        pass
                x['emb'][n, :len(utterance), :] = utterance_vectors
#                 x['emb_sound'][n, :len(utterance), :] = utterance_vectors_sound
                
#                 x['emb_forward'][n, :len(utterence_vectors_forward), :] = utterence_vectors_forward
#                 x['emb_backward'][n, :len(utterence_vectors_backward), :] = utterence_vectors_backward
                
            x['token'][n, :len(utterance)] = self.token_dict.toks2idxs(utterance)# using flair embeddings ::
#             x['token'][n, :len(utterance)] = self.token_dict.toks2idxs(utterance)
          #  utterance.pop()
           # utterance.append(".")
            x['wordindex'][n, :len(utterance)] = self.token_dict.toks2idxs(utterance)
#             utterance.remove(".")
            
#             print("shape:: ", x[wordindex].shape)
#             print("tokenn :: ", x['token'])
#             print("utterance__ ", np.asarray(x['emb']).shape)
            for k, token in enumerate(utterance):
                x['char'][n, k, :len(token)] = self.char_dict.toks2idxs(token)

        # Mask for paddings
        x['mask'] = np.zeros([batch_size, max_utt_len], dtype=np.float32)
        for n in range(batch_size):
            x['mask'][n, :len(batch_x[n])] = 1

        # Prepare y batch
        if batch_y is not None:
            y = np.ones([batch_size, max_utt_len], dtype=np.int32) * self.tag_dict['<PAD>']
        else:
            y = None

        if batch_y is not None:
            for n, tags in enumerate(batch_y):
                y[n, :len(tags)] = self.tag_dict.toks2idxs(tags)



        # img_path ='../../datasets/multi_model_twitter_ACL/Twitter_datasets/ner_img/'
        img_path ='../../datasets/multi_model_twitter_AAAI/ner_img/'
        All_Images = []
        x2 = []

        for img in img_batch  : 

            
            img2 = image.load_img(str(img_path+img[0]+".jpg"), target_size=(224, 224))
            # img2 = image.load_img(str(img_path+img[0]), target_size=(224, 224))

            img_Arr = image.img_to_array(img2)
            img_Arr = np.expand_dims(img_Arr, axis=0)
            img_Arr = preprocess_input(img_Arr)
            img_Arr = img_Arr.reshape(224,224,3)

           
            x2.append(img_Arr)
            



        x2 = np.asarray(x2)



     

        
        # model.summary()

        # img_path = 'train/dogs/1.jpg'
        # img = image.load_img(img_path, target_size=(224, 224))
        # img_data = image.img_to_array(img)
        # img_data = np.expand_dims(img_data, axis=0)
        # img_data = preprocess_input(img_data)

        # vgg16_feature = model.predict(x2)
            
     
       



        



        

        # last_layer_logits, end_points = nets.vgg.vgg_16(x2, num_classes=0)

          


            
        #     # examples
        # pool5_features = end_points['vgg_16/pool5']

      





        



        # All_Images.append(pool5_features)

        # All_Images = np.asarray(All_Images)

        return x, y,x2

    def save_corpus_dicts(self, filename='dict.txt'):
        # Token dict
        token_dict = self.token_dict._i2t
        with open(filename, 'w', encoding="utf8") as f:
            f.write('-TOKEN-DICT-\n')
            for ind in range(len(token_dict)):
                f.write(token_dict[ind] + '\n')
            f.write('\n')

        # Tag dict
        token_dict = self.tag_dict._i2t
        with open(filename, 'a', encoding="utf8") as f:
            f.write('-TAG-DICT-\n')
            for ind in range(len(token_dict)):
                f.write(token_dict[ind] + '\n')
            f.write('\n')

        # Character dict
        token_dict = self.char_dict._i2t
        with open(filename, 'a', encoding="utf8") as f:
            f.write('-CHAR-DICT-\n')
            for ind in range(len(token_dict)):
                f.write(token_dict[ind] + '\n')
            f.write('\n')

    def load_corpus_dicts(self, filename='dict.txt'):
        with open(filename, encoding="utf8") as f:
            # Token dict
            tokens = list()
            line = f.readline()
            assert line.strip() == '-TOKEN-DICT-'
            while len(line) > 0:
                line = f.readline().strip()
                if len(line) > 0:
                    tokens.append(line)
            self.token_dict = Vocabulary(tokens)

            # Tag dictappend
            line = f.readline()
            tags = list()
            assert line.strip() == '-TAG-DICT-'
            while len(line) > 0:
                line = f.readline().strip()
                if len(line) > 0:
                    tags.append(line)
            self.tag_dict = Vocabulary(tags, is_tags=True)

            # Char dict
            line = f.readline()
            chars = list()
            assert line.strip() == '-CHAR-DICT-'
            while len(line) > 0:
                line = f.readline().strip()
                if len(line) > 0:
                    chars.append(line)
            self.char_dict = Vocabulary(chars)
