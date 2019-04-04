
import Evaluation
import tensorflow as tf
import os



from flair.embeddings import CharLMEmbeddings
from flair.data import Sentence

from flair.embeddings import WordEmbeddings, CharLMEmbeddings
import layers
import numpy as np
# import Attention_multihead
import Attention
from tensorflow.contrib.layers import xavier_initializer
from collections import defaultdict


from tensorflow.contrib.slim import nets

from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input

from keras.preprocessing import image
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input


from keras.applications.resnet50 import ResNet50



import numpy as np






class NER:
    def __init__(self,
                 corpus,
                 stacked_embeddings = None,
                 emb_matrix=None,
                 n_filters=(128, 256),
                 filter_width=3,
                 token_embeddings_dim=128,
                 char_embeddings_dim=50,
                 use_char_embeddins=True,
                 pretrained_model_filepath=None,
                 embeddings_dropout=False,
                 dense_dropout=False,
                 use_batch_norm=False,
                 logging=False,
                 use_crf=False,
                 net_type='cnn',
                 char_filter_width=5,
                 verbouse=True,
                 use_capitalization=False,
                 concat_embeddings=False,
                 cell_type=None):
        tf.reset_default_graph()

        n_tags = len(corpus.tag_dict)
        n_tokens = len(corpus.token_dict)
        n_chars = len(corpus.char_dict)
        embeddings_onethego = False#not concat_embeddings and \
                              #corpus.embeddings is not None and \
                              #not isinstance(corpus.embeddings, dict)

        # Create placeholders
#         if embeddings_onethego:
#             x_word = tf.placeholder(dtype=tf.float32, shape=[None, None, stacked_embeddings.embedding_length], name='x_word')
#         else:
        x_word = tf.placeholder(dtype=tf.int32, shape=[None, None], name='x_word')
#         if concat_embeddings:
            
        x_emb = tf.placeholder(dtype=tf.float32, shape=[None, None, stacked_embeddings.embedding_length], name='x_emb')
        images = tf.placeholder(dtype=tf.float32, shape=[None, 7,7,512], name='images')
       
            
#         x_emb_forward = tf.placeholder(dtype=tf.float32, shape=[None, None, charlm_embedding_forward.embedding_length], name='x_forward')
#         x_emb_backward = tf.placeholder(dtype=tf.float32, shape=[None, None, charlm_embedding_backward.embedding_length], name='x_backward')
         
        x_char = tf.placeholder(dtype=tf.int32, shape=[None, None, None], name='x_char')
        word_index = tf.placeholder(dtype=tf.int32, shape=[None, None], name='wordidnex')
        y_true = tf.placeholder(dtype=tf.int32, shape=[None, None], name='y_tag')
        mask = tf.placeholder(dtype=tf.float32, shape=[None, None], name='mask')
        x_capi = tf.placeholder(dtype=tf.float32, shape=[None, None], name='x_capi')

        # Auxiliary placeholders
        learning_rate_ph = tf.placeholder(dtype=tf.float32, shape=[], name='learning_rate')
        dropout_ph = tf.placeholder_with_default(1.0, shape=[])
        training_ph = tf.placeholder_with_default(False, shape=[])
        learning_rate_decay_ph = tf.placeholder(dtype=tf.float32, shape=[], name='learning_rate_decay')
        
        # print("x_wordx_wordx_word : ",x_word.shape)
        
        # w_emb = layers.embedding_layer(x_word, n_tokens=n_tokens ,token_embedding_matrix = emb_matrix,token_embedding_dim=token_embeddings_dim,trainable=True)
        
        # print("w_embw_embw_emb :: ",w_emb.shape)

        # Embeddings
        if not embeddings_onethego:
            print("not embeddings_onethego" , x_word.shape)
            with tf.variable_scope('Embeddings'):
                emb = x_word
#                 w_emb = embedding_layer(x_word, n_tokens=n_tokens, token_embedding_dim=token_embeddings_dim,token_embedding_matrix=corpus.embeddings.vectors,trainable=False)
                if use_char_embeddins:
                    print("use char embeddings")
                    
#                     Char_stacked_rnn
                    
#                     c_emb = character_embedding_network(x_char,
#                                                         n_characters=n_chars,
#                                                         char_embedding_dim=char_embeddings_dim,
#                                                         filter_width=char_filter_width,
#                                                         training_ph=training_ph
#                                                        )
#                     emb = tf.concat([w_emb, c_emb], axis=-1)
                else:
                    emb = w_emb
        else:
            emb = x_word




        c_emb = layers.character_embedding_network(x_char,
                                            n_characters=n_chars,
                                            char_embedding_dim=50,
                                            filter_width=char_filter_width,
                                            
                                           )
            
            
        emb = tf.concat([x_emb,c_emb], -1)
            
        

#         if concat_embeddings:
#             c_emb = character_embedding_network(x_char,
#                                                         n_characters=n_chars,
#                                                         char_embedding_dim=char_embeddings_dim,
#                                                         filter_width=char_filter_width,
#                                                         training_ph=training_ph
#                                                        )
      
      
#             x_emb = add_timing_signal(x_emb)
      
      
      
#             emb = multi_head_attention(x_emb, c_emb, 5, None,drop_rate=0.5, is_train=training_ph,reuse=False,scope="embedding")
#             print("w_emb.shape :: ",w_emb.shape)
#             emb = x_emb #tf.concat([c_emb, x_emb], axis=2)

        if use_capitalization:
            cap = tf.expand_dims(x_capi, 2)
            emb = tf.concat([emb, cap], axis=2)

        # Dropout for embeddings
        if embeddings_dropout:
            emb = tf.layers.dropout(emb, dropout_ph, training=training_ph)

        if 'cnn' in net_type.lower():
            # Convolutional network
            with tf.variable_scope('ConvNet'):
                units = stacked_convolutions(emb,
                                             n_filters=n_filters,
                                             filter_width=filter_width,
                                             use_batch_norm=use_batch_norm,
                                             training_ph=training_ph)
                
                
        elif 'rnn' in net_type.lower():
            if cell_type is None or cell_type not in {'lstm', 'gru'}:
                raise RuntimeError('You must specify the type of the cell! It could be either "lstm" or "gru"')

            print("units before attention ",emb.shape)
            with tf.variable_scope("self_attention"):
              
                  token_mask = tf.cast(x_word, tf.bool)

                  shape = tf.shape(images)
                  images_s1 = tf.reshape(images,[shape[0],49,512])


  
                  fw_res,new_logits,img_tensor_ = Attention.directional_attention_Image_dense( emb, token_mask, images_s1,'forward',
                   'dir_attn_fw',0.8,training_ph,1e-4, 'relu',tensor_dict={}, name='fw_fw_attn')

                  # bw_res = Attention.directional_attention_Image_dense( emb, token_mask, images_s1,'backward',
                  #  'dir_attn_bw',0.9,training_ph,1e-4, 'relu',tensor_dict={}, name='bwattn')
              #    print("fw_res.shape :: ", fw_res.shape)

                  #fw_res = Attention.directional_attention_with_dense( emb, token_mask,'forward',
                 # 'dir_attn_fw',0.8,training_ph,1e-4, 'relu',tensor_dict={}, name='fw_fw_attn')
                  # print("fw_res.shape :: ", fw_res.shape) 

                  bw_res = Attention.directional_attention_with_dense( emb, token_mask,'backward',
                   'dir_attn_bw',0.8 ,training_ph,1e-4, 'relu',tensor_dict={}, name='bw_bw_attn')                                
                  units = tf.concat([fw_res,bw_res],axis=2)

                  output = units
                  print("units :: " ,units.shape)           

        else:
            raise KeyError('There is no such type of network: {}'.format(net_type))

        # Classifier
        with tf.variable_scope('Classifier'):
            pre_logits = tf.nn.relu(layers.linear([output],600, True, scope='pre_logits_linear',
                                          wd = 1e-4, input_keep_prob=0.8,
                                          is_train=training_ph))  # bs, hn
            pre_logits = tf.layers.dropout(pre_logits, 0.5, training=training_ph)
            # pre_logits2 = tf.nn.relu(layers.linear(pre_logits,100, True, scope='pre_logits_linear2',
            #                               wd = 1e-4, input_keep_prob=1,
            #                               is_train=training_ph))
            logits = tf.layers.dense(pre_logits, n_tags, kernel_initializer=xavier_initializer())
#             logits = tf.layers.dropout(logits, rate=0.2, training=training_ph)
            
            
#             logits = multi_head_attention(logits, logits, 5, None,drop_rate=0.5, is_train=training_ph,reuse=False,scope="first")
# # #                 attn_outs = attn_outs_rnn + FFN
#             logits = layer_normalize(logits) 

            
            
            

        if use_crf:
            sequence_lengths = tf.reduce_sum(mask, axis=1)
            log_likelihood, trainsition_params = tf.contrib.crf.crf_log_likelihood(logits,
                                                                                   y_true,
                                                                                   sequence_lengths)
            loss_tensor = -log_likelihood
            predictions = None
        else:
            ground_truth_labels = tf.one_hot(y_true, n_tags)
            loss_tensor = tf.nn.softmax_cross_entropy_with_logits(labels=ground_truth_labels, logits=logits)
            loss_tensor = loss_tensor * mask
            predictions = tf.argmax(logits, axis=-1)
            
            
            
        def output_embedding(current_output):
          return tf.matmul(current_output, tf.transpose(self.output_embedding_mat))
            
#             return tf.add(
#                 tf.matmul(current_output, tf.transpose(self.output_embedding_mat)), self.output_embedding_bias)
            
            
#         self.output_embedding_mat = tf.get_variable("output_embedding_mat",[22684, 512],  dtype=tf.float32)
# # #         self.output_embedding_bias = tf.get_variable("output_embedding_bias", [0],   dtype=tf.float32)
        
#         self.word_index = word_index
            
#         non_zero_weights = tf.sign(self.word_index)

#         logits2 = tf.map_fn(output_embedding, units)
#         output_embedding = logits2
#         logits2 = tf.reshape(output_embedding, [-1, 22684])
        
        
#         self.LM_output = logits2
        
        
#         loss2 = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf.reshape(self.word_index, [-1]),logits=logits2)* tf.cast(tf.reshape(non_zero_weights, [-1]), tf.float32)
#         aux_loss = tf.reduce_sum(loss2)  

            
        

        loss = tf.reduce_mean(loss_tensor)  

        self.vgg  = VGG16(weights='imagenet', include_top=False)
        self.image_atten = new_logits
        self.img_tensor_ = img_tensor_
        for l in self.vgg.layers:
          l.trainable=False 
#         
#         final_loss = aux_loss * 0.2 + loss * 0.8

        # Initialize session
        sess = tf.Session()
        if verbouse:
            self.print_number_of_parameters()
        if logging:
            self.train_writer = tf.summary.FileWriter('summary', sess.graph)

        self._use_crf = use_crf
        self.summary = tf.summary.merge_all()

        self._learning_rate_decay_ph = learning_rate_decay_ph
        self._x_w = x_word
        self._emb = x_emb
#         self._x_s = x_emb_sound
        self._x_c = x_char
        
        self._y_true = y_true
        self._y_pred = predictions
        
        self._img = images
        if concat_embeddings:
            self._x_emb = x_emb
#             self._x_emb_sound = x_emb_sound
#             self._x_emb_backward = x_emb_backward
        if use_crf:
            self._logits = logits
            self._trainsition_params = trainsition_params
            self._sequence_lengths = sequence_lengths
        self._learning_rate_ph = learning_rate_ph
        self._dropout = dropout_ph
        
       

        self._loss = loss#loss * 0.7 + aux_loss * 0.98
        
        self.corpus = corpus

        self._loss_tensor = loss_tensor 
#         self._loss_tensor_LM = loss2
        self._use_dropout = True if embeddings_dropout or dense_dropout else None

        self._training_ph = training_ph
        self._logging = logging

        # Get training op
        self._train_op = self.get_train_op(loss, learning_rate_ph, lr_decay_rate=learning_rate_decay_ph)
        self._embeddings_onethego = embeddings_onethego
        self.verbouse = verbouse
        sess.run(tf.global_variables_initializer())

        self._sess = sess
        
        self._mask = mask
        if use_capitalization:
            self._x_capi = x_capi
        self._use_capitalization = use_capitalization
        self._concat_embeddings = concat_embeddings
        if pretrained_model_filepath is not None:
            self.load(pretrained_model_filepath)

    def save(self, model_file_path=None,filename = None):
        print("saving")
        if model_file_path is None:
#             if not os.path.exists(MODEL_PATH):
#                 os.mkdir(MODEL_PATH)
            model_file_path = './'+str(filename)+'/ner_model.ckpt' + str(filename)#os.path.join(MODEL_PATH, MODEL_FILE_NAME)
            print("model_file_path :: ", model_file_path)
        saver = tf.train.Saver()
        saver.save(self._sess, model_file_path)
        self.corpus.save_corpus_dicts('./dict.txt')

    def load(self, model_file_path):
        saver = tf.train.Saver()
        # saver = tf.train.Saver(tf.global_variables())
        # saver = tf.train.import_meta_graph(model_file_path+".meta")

        print("model loaded")
        saver.restore(self._sess, model_file_path)

    def train_on_batch(self, x_word, x_char, y_tag):
        feed_dict = {self._x_w: x_word, self._x_c: x_char, self._y_true: y_tag}
        self._sess.run(self._train_op, feed_dict=feed_dict)

    @staticmethod
    def print_number_of_parameters():
        print('Number of parameters: ')
        vars = tf.trainable_variables()
        blocks = defaultdict(int)
        for var in vars:
            # Get the top level scope name of variable
            block_name = var.name.split('/')[0]
            number_of_parameters = np.prod(var.get_shape().as_list())
            blocks[block_name] += number_of_parameters
        for block_name in blocks:
            print(block_name, blocks[block_name])
        total_num_parameters = np.sum(list(blocks.values()))
        print('Total number of parameters equal {}'.format(total_num_parameters))

    def fit(self,batch_gen=None, batch_size=32, learning_rate=1e-3, epochs=1, dropout_rate=0.5, learning_rate_decay=1):
        


        Best_F1_valid = 0
        Best_F1_test = 0
        for epoch in range(epochs):
            print("learning_rate :: ",learning_rate)
            
            if self.verbouse:
                print('Epoch {}'.format(epoch))


            
            if batch_gen is None:
                batch_generator = self.corpus.batch_generator(batch_size, dataset_type='train', shuffle=True)
            for x, y, img in batch_generator:
                feed_dict = self._fill_feed_dict(x,
                                                img,
                                                 y,
                                                 
                                                 learning_rate,
                                                 dropout_rate=dropout_rate,
                                                 training=True,
                                                 learning_rate_decay=learning_rate_decay)
                if self._logging:
                    summary, _ = self._sess.run([self.summary, self._train_op], feed_dict=feed_dict)
                    
                    self.train_writer.add_summary(summary)

                self._sess.run(self._train_op, feed_dict=feed_dict)
                
                
            if self.verbouse:
                f = self.eval_conll('valid',print_results=True)['__total__']['f1']
                if f  > Best_F1_valid:
                  self.save(filename="valid")
                  Best_F1_valid = f
                  print("new best valid model saved with F1 : ", Best_F1_valid)
                  
                  
                f2 = self.eval_conll('test',print_results=True)
                if f2['__total__']['f1'] > Best_F1_test:
                  self.save(filename="test")
                  Best_F1_test = f2['__total__']['f1']
                  print("new best test model saved with F1 : ", Best_F1_test)
                  with open ("best_score.txt","a") as f:
                    f.write(str(f2 ))
                    f.write("\n")


            

        if self.verbouse:
            self.eval_conll(dataset_type='train', short_report=False)
            self.eval_conll(dataset_type='valid', short_report=False)
            results = self.eval_conll(dataset_type='test', short_report=False)
        else:
            results = self.eval_conll(dataset_type='test',short_report=True)
        return results

    def predict(self, x,img):
        feed_dict = self._fill_feed_dict(x,img,None,training=False)
        if self._use_crf:
            y_pred = []
            logits, trans_params, sequence_lengths = self._sess.run([self._logits,
                                                                     self._trainsition_params,
                                                                     self._sequence_lengths
                                                                     ],
                                                                    feed_dict=feed_dict)

            # iterate over the sentences because no batching in viterbi_decode
            for logit, sequence_length in zip(logits, sequence_lengths):
                logit = logit[:int(sequence_length)]  # keep only the valid steps
                viterbi_seq, viterbi_score = tf.contrib.crf.viterbi_decode(logit, trans_params)
                y_pred += [viterbi_seq]
        else:
            y_pred = self._sess.run(self._y_pred, feed_dict=feed_dict)
        return self.corpus.tag_dict.batch_idxs2batch_toks(y_pred, filter_paddings=True)

    def eval_conll(self, dataset_type='test', print_results=True, short_report=True):
        y_true_list = list()
        y_pred_list = list()
        loss = 0
        batch = 0
        print('Eval on {}:'.format(dataset_type))

        # print("tag_vectors in eval :: " , tag_vectors.shape)

        for x, y_gt,img in self.corpus.batch_generator(batch_size=16, dataset_type=dataset_type):
            y_pred = self.predict(x,img)
            y_gt = self.corpus.tag_dict.batch_idxs2batch_toks(y_gt, filter_paddings=True)
            for tags_pred, tags_gt in zip(y_pred, y_gt):
                for tag_predicted, tag_ground_truth in zip(tags_pred, tags_gt):
                    y_true_list.append(tag_ground_truth)
                    y_pred_list.append(tag_predicted)
                y_true_list.append('O')
                y_pred_list.append('O')
                
                
             


        return Evaluation.precision_recall_f1(y_true_list,
                                   y_pred_list,
                                   print_results,
                                   short_report)

    def _fill_feed_dict(self,
                        x,
                        img,                       
                        y_t=None,
                        
                        learning_rate=None,
                        training=False,
                        dropout_rate=1,
                        learning_rate_decay=1):

        feed_dict = dict()


        # sess = tf.sess.run(tf.global_variables_initializer())

        # last_layer_logits, end_points = nets.vgg.vgg_16(img, num_classes=0)



          


            
        #     # examples
        # pool5_features = self.vgg['vgg_16/pool5']



        try:
          feed_dict[self._img] = self.vgg.predict(img)
        except:
          pass

        # print("tag_vectorstag_vectorstag_vectors ",tag_vectors.shape)
        

        if self._embeddings_onethego:

            feed_dict[self._x_w] = x['token'] #x['emb']
            feed_dict[self._x_emb] = x['emb']
#             feed_dict[self._x_s] = x['emb_sound']
            
            
        else:
            # print("x['token'] ", x['token'])
            feed_dict[self._x_w] = x['token']
            
            
#         feed_dict[self.word_index]= x['wordindex']
            
        feed_dict[self._x_c] = x['char']
        feed_dict[self._mask] = x['mask']
        feed_dict[self._training_ph] = training
        if y_t is not None:
            feed_dict[self._y_true] = y_t

        # Optional arguments
        if self._use_capitalization:
            feed_dict[self._x_capi] = x['capitalization']

        if self._concat_embeddings:
            feed_dict[self._x_emb] = x['emb']
#             feed_dict[self._x_emb_sound] = x['emb_sound']
#             feed_dict[self._x_emb_backward] = x['emb_backward']
            

        # Learning rate
        if learning_rate is not None:
            feed_dict[self._learning_rate_ph] = learning_rate
            feed_dict[self._learning_rate_decay_ph] = learning_rate_decay

        # Dropout
        if self._use_dropout is not None and training:
            feed_dict[self._dropout] = dropout_rate
        else:
            feed_dict[self._dropout] = 1.0
        return feed_dict

    def eval_loss(self, data_type='test', batch_size=32):
        # TODO: fixup
        num_tokens = 0
        loss = 0
        for x, y_t in self.corpus.batch_generator(batch_size=batch_size, dataset_type=data_type):
            feed_dict = self._fill_feed_dict(x, y_t, training=False)
            loss += np.sum(self._sess.run(self._loss_tensor, feed_dict=feed_dict))
            num_tokens += np.sum(self.corpus.token_dict.is_pad(x_w))
        return loss / num_tokens
      
 

    @staticmethod
    def get_trainable_variables(trainable_scope_names=None):
        vars = tf.trainable_variables()
        if trainable_scope_names is not None:
            vars_to_train = list()
            for scope_name in trainable_scope_names:
                for var in vars:
                    if var.name.startswith(scope_name):
                        vars_to_train.append(var)
            return vars_to_train
        else:
            return vars

    def get_train_op(self, loss, learning_rate, learnable_scopes=None, lr_decay_rate=None):
        # global_step = tf.Variable(0, trainable=False)
        # try:
        #     n_training_samples = len(self.corpus.dataset['train'])
        # except TypeError:
        #     n_training_samples = 1024
#         batch_size = tf.shape(self._x_w)[0]
#         decay_steps = tf.cast(n_training_samples / batch_size, tf.int32)
#         if lr_decay_rate is not None:
#             learning_rate = tf.train.exponential_decay(learning_rate,
#                                                        global_step,
#                                                        decay_steps=decay_steps,
#                                                        decay_rate=lr_decay_rate,
#                                                        staircase=True)
#             self._learning_rate_decayed = learning_rate
# #             print("learning_rate :: ",learning_rate)
        variables = self.get_trainable_variables(learnable_scopes)

        # For batch norm it is necessary to update running averages
        extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(extra_update_ops):
#               train_op = tf.train.MomentumOptimizer(learning_rate,momentum=0.9,use_nesterov=True).minimize(loss, var_list=variables)
#             train_op = tf.train.AdamOptimizer(learning_rate).minimize(loss, global_step=global_step, var_list=variables)
          train_op = tf.train.AdamOptimizer(learning_rate).minimize(loss, var_list=variables)
#         print("learning rate :: ",  learning_rate)
        print("loss :: ", loss)
        return train_op

    def predict_for_token_batch(self, tokens_batch):
        batch_x, _ = self.corpus.tokens_batch_to_numpy_batch(tokens_batch)
        # Prediction indices
        predictions_batch = self.predict(batch_x)
        predictions_batch_no_pad = list()
        for n, predicted_tags in enumerate(predictions_batch):
            predictions_batch_no_pad.append(predicted_tags[: len(tokens_batch[n])])
        return predictions_batch_no_pad
