 
import tensorflow as tf
import os


import layers

def multi_head_attention(queries, keys, num_heads, attention_size, drop_rate=0.0, is_train=True, reuse=None,
                         scope=None,expand_dims = False):
    # borrowed from: https://github.com/Kyubyong/transformer/blob/master/modules.py
    with tf.variable_scope(scope or "multi_head_attention", reuse=reuse):
        if attention_size is None:
            attention_size = queries.get_shape().as_list()[-1]
            print("attention size ::: ",attention_size)
        # linear projections, shape=(batch_size, max_time, attention_size)
        
        print ( " queries.shape : ", queries.shape )
        print ( " keys.shape : ", keys.shape )
        
        if expand_dims:
        
          queries = tf.expand_dims(queries, 1)
        
        
        query = tf.layers.dense(queries, attention_size, activation=tf.nn.relu, name="query_project")
        
#         key = tf.layers.dense(keys, 1024, activation=tf.nn.relu, name="key_project", reuse = False)
#         key = tf.layers.dropout(key, rate=drop_rate, training=is_train)
        key = tf.layers.dense(keys, attention_size, activation=tf.nn.relu, name="key_project2" , reuse = False)
#         key = tf.layers.dropout(key, rate=drop_rate, training=is_train)
        
#         value = tf.layers.dense(keys, 1024, activation=tf.nn.relu, name="value_project", reuse = False)
#         value = tf.layers.dropout(value, rate=drop_rate, training=is_train)
        value = tf.layers.dense(keys, attention_size, activation=tf.nn.relu, name="value_project2", reuse = False)
#         value = tf.layers.dropout(value, rate=drop_rate, training=is_train)
#         value = key
        
        
        # split and concatenation, shape=(batch_size * num_heads, max_time, attention_size / num_heads)
        query_ = tf.concat(tf.split(query, num_heads, axis=2), axis=0)
        key_ = tf.concat(tf.split(key, num_heads, axis=2), axis=0)
        value_ = tf.concat(tf.split(value, num_heads, axis=2), axis=0)
        # multiplication
        attn_outs = tf.matmul(query_, tf.transpose(key_, [0, 2, 1]))
        # scale
        attn_outs = attn_outs / (key_.get_shape().as_list()[-1] ** 0.5)
        # key masking
        key_masks = tf.sign(tf.abs(tf.reduce_sum(keys, axis=-1)))  # shape=(batch_size, max_time)
        key_masks = tf.tile(key_masks, [num_heads, 1])  # shape=(batch_size * num_heads, max_time)
        # shape=(batch_size * num_heads, max_time, max_time)
        key_masks = tf.tile(tf.expand_dims(key_masks, 1), [1, tf.shape(queries)[1], 1])
        paddings = tf.ones_like(attn_outs) * (-2 ** 32 + 1)
        # shape=(batch_size, max_time, attention_size)
        attn_outs = tf.where(tf.equal(key_masks, 0), paddings, attn_outs)
        # activation
        attn_outs = tf.nn.softmax(attn_outs)
        # query masking
        query_masks = tf.sign(tf.abs(tf.reduce_sum(queries, axis=-1)))
        query_masks = tf.tile(query_masks, [num_heads, 1])
        query_masks = tf.tile(tf.expand_dims(query_masks, -1), [1, 1, tf.shape(keys)[1]])
        attn_outs *= query_masks
        # dropout
        attn_outs = tf.layers.dropout(attn_outs, rate=drop_rate, training=is_train)
        # weighted sum
        outputs = tf.matmul(attn_outs, value_)
        # restore shape
        outputs = tf.concat(tf.split(outputs, num_heads, axis=0), axis=2)
        
        
        
        outputs += queries  # residual connection
        
        print ( " outputs.shape before  :: ",outputs.shape)
        
        if expand_dims:  
        
          outputs = tf.contrib.layers.flatten(outputs)
    
          outputs =  tf.contrib.layers.layer_norm(outputs)
#         outputs = layer_normalize(outputs)
        

        else:
            outputs = layers.layer_normalize(outputs)


        print ( " outputs.shape after  :: ",outputs.shape)
        
        
        
    return outputs



def MultiDm_image_word_with_dense(rep_tensor, rep_mask,img_tensor, direction=None, scope=None,
                                     keep_prob=1., is_train=None, wd=0., activation='elu',
                                     tensor_dict=None, name=None):
    def scaled_tanh(x, scale=5.):
        return scale * tf.nn.tanh(1./scale * x)

    bs, sl, vec = tf.shape(img_tensor)[0], tf.shape(img_tensor)[1], tf.shape(img_tensor)[2]
    ivec = 600 #rep_tensor.get_shape()[2]
    with tf.variable_scope(scope or 'directional_attention_%s' % direction or 'diag'):
        # mask generation
        sl_indices = tf.range(sl, dtype=tf.int32)
        sl_col, sl_row = tf.meshgrid(sl_indices, sl_indices)
        # if direction is None:
        direct_mask = tf.cast(tf.diag(- tf.ones([sl], tf.int32)) + 1, tf.bool)
        # else:
        #     if direction == 'forward':
        #         direct_mask = tf.greater(sl_row, sl_col)
        #     else:
        #         direct_mask = tf.greater(sl_col, sl_row)
        direct_mask_tile = tf.tile(tf.expand_dims(direct_mask, 0), [bs, 1, 1])  # bs,sl,sl
        rep_mask_tile = tf.tile(tf.expand_dims(rep_mask, 1), [1, sl, 1])  # bs,sl,sl
        attn_mask = tf.logical_and(direct_mask_tile, rep_mask_tile)  # bs,sl,sl

        # non-linear
        rep_map = layers.bn_dense_layer(rep_tensor, ivec, True, 0., 'bn_dense_map', activation,
                                 False, wd, keep_prob, is_train)


        img_tensor_ = layers.bn_dense_layer(img_tensor, ivec, True, 0., 'img_tensor_', activation,
                                 False, wd, keep_prob, is_train)


        print("rep_map shape :: ",rep_map.shape)
        print("img_tensor_.shape :: ",img_tensor_.shape)


        rep_map_tile = tf.tile(tf.expand_dims(rep_map, 1), [1, sl, 1, 1])  # bs,sl,sl,vec




        img_tensor_tile = tf.tile(tf.expand_dims(img_tensor_, 1), [1, sl, 1, 1])

        rep_map_dp = layers.dropout(rep_map, keep_prob, is_train)

        # attention
        with tf.variable_scope('attention'):  # bs,sl,sl,vec
            f_bias = tf.get_variable('f_bias',[ivec], tf.float32, tf.constant_initializer(0.))
            dependent = layers.linear(img_tensor_, ivec, False, scope='linear_dependent')  # bs,sl,vec
            dependent_etd = tf.expand_dims(dependent, 1)  # bs,1,sl,vec

            head = layers.linear(rep_map_dp    , ivec, False, scope='linear_head') # bs,sl,vec
            head_etd = tf.expand_dims(head, 2)  # bs,sl,1,vec


            print("head_etd :: ",head_etd.shape)
            print("dependent_etd :: ",dependent_etd.shape)

            logits = scaled_tanh(dependent_etd + head_etd + f_bias, 5.0)  # bs,sl,sl,vec

            # logits_masked = layers.exp_mask_for_high_rank(logits, attn_mask)
            attn_score = tf.nn.softmax(logits, 2)  # bs,sl,sl,vec
            # attn_score = layers.mask_for_high_rank(attn_score, attn_mask)

            print("attn_score ", attn_score.shape)

            print("img_tensor_tile ", img_tensor_tile.shape)


            # concat_images = multi_dimensional_attention(attn_score * img_tensor_tile)

            # print("concat_images.shape : ",concat_images.shape)

            attn_result = tf.reduce_sum(attn_score * img_tensor_tile, 2)  # bs,sl,vec

        with tf.variable_scope('output'):
            o_bias = tf.get_variable('o_bias',[ivec], tf.float32, tf.constant_initializer(0.))
            # input gate
            fusion_gate = tf.nn.sigmoid(
                layers.linear(rep_map, ivec, True, 0., 'linear_fusion_i', False, wd, keep_prob, is_train) +
                layers.linear(attn_result, ivec, True, 0., 'linear_fusion_a', False, wd, keep_prob, is_train) +
                o_bias)
            output = fusion_gate * rep_map + (1-fusion_gate) * attn_result
            # output = layers.mask_for_high_rank(output, rep_mask)

        # save attn
        if tensor_dict is not None and name is not None:
            tensor_dict[name + '_dependent'] = dependent
            tensor_dict[name + '_head'] = head
            tensor_dict[name] = attn_score
            tensor_dict[name + '_gate'] = fusion_gate
        return output


def Visual_Attention(rep_tensor,img_tensor):

    def scaled_tanh(x, scale=5.):
        return scale * tf.nn.tanh(1./scale * x)

    

def MultiDm_attention_toktag(rep_tensor, rep_mask,img_tensor, direction=None, scope=None,
                                     keep_prob=1., is_train=None, wd=0., activation='elu',
                                     tensor_dict=None, name=None):
    def scaled_tanh(x, scale=5.):
        return scale * tf.nn.tanh(1./scale * x)

    bs, sl, vec = tf.shape(rep_tensor)[0], tf.shape(rep_tensor)[1], tf.shape(rep_tensor)[2]
    ivec = rep_tensor.get_shape()[2]
    with tf.variable_scope(scope or 'directional_attention_%s' % direction or 'diag'):
        # mask generation
        sl_indices = tf.range(sl, dtype=tf.int32)
        sl_col, sl_row = tf.meshgrid(sl_indices, sl_indices)
        # if direction is None:
        direct_mask = tf.cast(tf.diag(- tf.ones([sl], tf.int32)) + 1, tf.bool)
        # else:
        #     if direction == 'forward':
        #         direct_mask = tf.greater(sl_row, sl_col)
        #     else:
        #         direct_mask = tf.greater(sl_col, sl_row)
        direct_mask_tile = tf.tile(tf.expand_dims(direct_mask, 0), [bs, 1, 1])  # bs,sl,sl
        rep_mask_tile = tf.tile(tf.expand_dims(rep_mask, 1), [1, sl, 1])  # bs,sl,sl
        attn_mask = tf.logical_and(direct_mask_tile, rep_mask_tile)  # bs,sl,sl

        # non-linear
        rep_map = layers.bn_dense_layer(rep_tensor, ivec, True, 0., 'bn_dense_map', activation,
                                 False, wd, keep_prob, is_train)


        img_tensor_ = layers.bn_dense_layer(img_tensor, ivec, True, 0., 'img_tensor_', activation,
                                 False, wd, keep_prob, is_train)


        print("rep_map shape :: ",rep_map.shape)
        print("img_tensor_.shape :: ",img_tensor_.shape)


        rep_map_tile = tf.tile(tf.expand_dims(rep_map, 1), [1, sl, 1, 1])  # bs,sl,sl,vec



        img_tensor_ = tf.expand_dims(img_tensor_,0)
        img_tensor_tile = tf.tile(tf.expand_dims(img_tensor_, 1), [1, sl, 1,1])

        rep_map_dp = layers.dropout(rep_map, keep_prob, is_train)

        img_tensor_dp = layers.dropout(img_tensor_, keep_prob, is_train)

        # attention
        with tf.variable_scope('attention'):  # bs,sl,sl,vec
            f_bias = tf.get_variable('f_bias',[ivec], tf.float32, tf.constant_initializer(0.))
            dependent = layers.linear(img_tensor_tile, ivec, False, scope='linear_dependent')  # bs,sl,vec
            # dependent_etd = tf.expand_dims(dependent, 1)  # bs,1,sl,vec

            head = layers.linear(rep_map_dp, ivec, False, scope='linear_head') # bs,sl,vec
            head_etd = tf.expand_dims(head, 2)  # bs,sl,1,vec


            print("head_etd :: ",head_etd.shape)
            print("dependent :: ",dependent.shape)

            logits = scaled_tanh(dependent + head_etd + f_bias, 5.0)  # bs,sl,sl,vec


            print("logits.sape :: ", logits.shape)

            # logits_masked = layers.exp_mask_for_high_rank(logits, attn_mask)
            attn_score = tf.nn.softmax(logits, 2)  # bs,sl,sl,vec
            # attn_score = layers.mask_for_high_rank(attn_score, attn_mask)

            print("attn_score ", attn_score.shape)

            print("img_tensor_tile ", img_tensor_tile.shape)


            # concat_images = multi_dimensional_attention(attn_score * img_tensor_tile)

            # print("concat_images.shape : ",concat_images.shape).

            apply_attention = attn_score * img_tensor_tile

            simple = True

            if simple :

                


                attn_result = tf.reduce_sum(apply_attention, 2)
                # attn_result = tf.reduce_sum(attn_result, 2)
                
                attn_result = tf.layers.dense(attn_result, ivec)

                  # bs,sl,vec
                #attn_result = fc1
                print(" attn_result :: ",attn_result.shape)

            # return attn_result

            
                # return attn_result

        with tf.variable_scope('output'):
            o_bias = tf.get_variable('o_bias',[ivec], tf.float32, tf.constant_initializer(0.))
            # input gate
            fusion_gate = tf.nn.sigmoid(
                layers.linear(rep_map, ivec, True, 0., 'linear_fusion_i', False, wd, keep_prob, is_train) +
                layers.linear(attn_result, ivec, True, 0., 'linear_fusion_a', False, wd, keep_prob, is_train) +
                o_bias)
            output = fusion_gate * rep_map + (1-fusion_gate) * attn_result
            # output = layers.mask_for_high_rank(output, rep_mask)

        # save attn
        if tensor_dict is not None and name is not None:
            tensor_dict[name + '_dependent'] = dependent
            tensor_dict[name + '_head'] = head
            tensor_dict[name] = attn_score
            tensor_dict[name + '_gate'] = fusion_gate
        return output




def MultiDm_attention_with_dense(rep_tensor, rep_mask,img_tensor, direction=None, scope=None,
                                     keep_prob=1., is_train=None, wd=0., activation='elu',
                                     tensor_dict=None, name=None):
    def scaled_tanh(x, scale=5.):
        return scale * tf.nn.tanh(1./scale * x)

    bs, sl, vec = tf.shape(rep_tensor)[0], tf.shape(rep_tensor)[1], tf.shape(rep_tensor)[2]
    ivec = rep_tensor.get_shape()[2]
    with tf.variable_scope(scope or 'directional_attention_%s' % direction or 'diag'):
        # mask generation
        sl_indices = tf.range(sl, dtype=tf.int32)
        sl_col, sl_row = tf.meshgrid(sl_indices, sl_indices)
        # if direction is None:
        direct_mask = tf.cast(tf.diag(- tf.ones([sl], tf.int32)) + 1, tf.bool)
        # else:
        #     if direction == 'forward':
        #         direct_mask = tf.greater(sl_row, sl_col)
        #     else:
        #         direct_mask = tf.greater(sl_col, sl_row)
        direct_mask_tile = tf.tile(tf.expand_dims(direct_mask, 0), [bs, 1, 1])  # bs,sl,sl
        rep_mask_tile = tf.tile(tf.expand_dims(rep_mask, 1), [1, sl, 1])  # bs,sl,sl
        attn_mask = tf.logical_and(direct_mask_tile, rep_mask_tile)  # bs,sl,sl

        # non-linear
        rep_map = layers.bn_dense_layer(rep_tensor, ivec, True, 0., 'bn_dense_map', activation,
                                 False, wd, keep_prob, is_train)


        img_tensor_ = layers.bn_dense_layer(img_tensor, ivec, True, 0., 'img_tensor_', activation,
                                 False, wd, keep_prob, is_train)


        print("rep_map shape :: ",rep_map.shape)
        print("img_tensor_.shape :: ",img_tensor_.shape)


        rep_map_tile = tf.tile(tf.expand_dims(rep_map, 1), [1, sl, 1, 1])  # bs,sl,sl,vec




        img_tensor_tile = tf.tile(tf.expand_dims(img_tensor_, 1), [1, sl, 1, 1])

        rep_map_dp = layers.dropout(rep_map, keep_prob, is_train)


        

        img_tensor_dp = layers.dropout(img_tensor_, keep_prob, is_train)

        # attention
        with tf.variable_scope('attention'):  # bs,sl,sl,vec
            f_bias = tf.get_variable('f_bias',[ivec], tf.float32, tf.constant_initializer(0.))
            dependent = layers.linear(img_tensor_dp, ivec, False, scope='linear_dependent')  # bs,sl,vec
            dependent_etd = tf.expand_dims(dependent, 1)  # bs,1,sl,vec

            head = layers.linear(rep_map_dp, ivec, False, scope='linear_head') # bs,sl,vec
            head_etd = tf.expand_dims(head, 2)  # bs,sl,1,vec


            print("head_etd :: ",head_etd.shape)
            print("dependent_etd :: ",dependent_etd.shape)

            logits = scaled_tanh(dependent_etd + head_etd + f_bias, 5.0)  # bs,sl,sl,vec


            print("logits.sape :: ", logits.shape)

            # logits_masked = layers.exp_mask_for_high_rank(logits, attn_mask)
            attn_score = tf.nn.softmax(logits, 2)  # bs,sl,sl,vec
            # attn_score = layers.mask_for_high_rank(attn_score, attn_mask)

            print("attn_score ", attn_score.shape)

            print("img_tensor_tile ", img_tensor_tile.shape)


            # concat_images = multi_dimensional_attention(attn_score * img_tensor_tile)

            # print("concat_images.shape : ",concat_images.shape).

            apply_attention = attn_score * img_tensor_tile

            simple = True

            if simple :

                


                attn_result = tf.reduce_sum(apply_attention, 2)
                # attn_result = tf.reduce_sum(attn_result, 2)
                
                attn_result = tf.layers.dense(attn_result, ivec)

                  # bs,sl,vec
                #attn_result = fc1
                print(" attn_result :: ",attn_result.shape)

            # return attn_result

            

            else :

                # print("apply_attention ",apply_attention.shape)

                i = tf.constant(0)
                

                matrix_rows = tf.shape(apply_attention)[0]

                while_condition = lambda i,data: i < matrix_rows#tf.less(i, 32)

                data = tf.TensorArray(dtype='float32', size= matrix_rows )

                # init_state = (i, data)

                def body(i,data):

                    inp = tf.expand_dims(apply_attention[i], 3)
                    print("input size :: ",inp.shape)
                    conv1 = tf.layers.conv2d(inp, 32, 5)
                # Max Pooling (down-sampling) with strides of 2 and kernel size of 2
                    conv1 = tf.layers.max_pooling2d(conv1, 2, 2)

                    print("conv1 " ,conv1.shape)
                    
                    data = data.write(i,conv1)



                    i = i + 1
                    return [i,data]


                all_data,ta_final= tf.while_loop(while_condition, body, [i,data])#,
                    # s//////////////hape_invariants=[i.get_shape(), [None,128]])

                # fc1 = tf.contrib.layers.flatten(conv1)

                ta_final_result = ta_final.stack()

                print("all_data ",ta_final_result.shape)

                # print("conv1.shape :: ",conv1.shape)

            # Fully connected layer (in tf contrib folder for now)

                attn_result = tf.reduce_sum(ta_final_result, 2)
                attn_result = tf.reduce_sum(attn_result, 2)
                
                attn_result = tf.layers.dense(attn_result, 600)

                  # bs,sl,vec
                #attn_result = fc1
                print(" attn_result :: ",attn_result.shape)

                # return attn_result

        with tf.variable_scope('output'):
            o_bias = tf.get_variable('o_bias',[ivec], tf.float32, tf.constant_initializer(0.))
            # input gate
            fusion_gate = tf.nn.sigmoid(
                layers.linear(rep_map, ivec, True, 0., 'linear_fusion_i', False, wd, keep_prob, is_train) +
                layers.linear(attn_result, ivec, True, 0., 'linear_fusion_a', False, wd, keep_prob, is_train) +
                o_bias)
            output = fusion_gate * rep_map + (1-fusion_gate) * attn_result
            # output = layers.mask_for_high_rank(output, rep_mask)

        # save attn
        if tensor_dict is not None and name is not None:
            tensor_dict[name + '_dependent'] = dependent
            tensor_dict[name + '_head'] = head
            tensor_dict[name] = attn_score
            tensor_dict[name + '_gate'] = fusion_gate
        return output



def directional_attention_with_dense(rep_tensor, rep_mask,direction=None, scope=None,
                                     keep_prob=1., is_train=None, wd=0., activation='elu',
                                     tensor_dict=None, name=None):
    def scaled_tanh(x, scale=5.):
        return scale * tf.nn.tanh(1./scale * x)

    bs, sl, vec = tf.shape(rep_tensor)[0], tf.shape(rep_tensor)[1], tf.shape(rep_tensor)[2]
    ivec = rep_tensor.get_shape()[2]
    with tf.variable_scope(scope or 'directional_attention_%s' % direction or 'diag'):
        # mask generation
        sl_indices = tf.range(sl, dtype=tf.int32)
        sl_col, sl_row = tf.meshgrid(sl_indices, sl_indices)
        if direction is None:
            direct_mask = tf.cast(tf.diag(- tf.ones([sl], tf.int32)) + 1, tf.bool)
        else:
            if direction == 'forward':
                direct_mask = tf.greater(sl_row, sl_col)
            else:
                direct_mask = tf.greater(sl_col, sl_row)
        direct_mask_tile = tf.tile(tf.expand_dims(direct_mask, 0), [bs, 1, 1])  # bs,sl,sl
        rep_mask_tile = tf.tile(tf.expand_dims(rep_mask, 1), [1, sl, 1])  # bs,sl,sl
        attn_mask = tf.logical_and(direct_mask_tile, rep_mask_tile)  # bs,sl,sl

        # non-linear
        rep_map = layers.bn_dense_layer(rep_tensor, ivec, True, 0., 'bn_dense_map', activation,
                                 False, wd, keep_prob, is_train)
        rep_map_tile = tf.tile(tf.expand_dims(rep_map, 1), [1, sl, 1, 1])  # bs,sl,sl,vec
        
        print("rep_map_tile : ", rep_map_tile.shape)


        
        rep_map_dp = layers.dropout(rep_map, keep_prob, is_train)

      
       



       

        # attention
        with tf.variable_scope('attention'):  # bs,sl,sl,vec
            f_bias = tf.get_variable('f_bias',[ivec], tf.float32, tf.constant_initializer(0.))
            dependent = layers.linear(rep_map_dp, ivec, False, scope='linear_dependent')  # bs,sl,vec
            dependent_etd = tf.expand_dims(dependent, 1)  # bs,1,sl,vec




            head = layers.linear(rep_map_dp, ivec, False, scope='linear_head') # bs,sl,vec
            head_etd = tf.expand_dims(head, 2)  # bs,sl,1,vec



            
            logits = scaled_tanh(dependent_etd + head_etd    + f_bias, 5.0)  # bs,sl,sl,vec
 


            logits_masked = layers.exp_mask_for_high_rank(logits, attn_mask)

            print("logits_masked : ", logits_masked.shape)




            attn_score = tf.nn.softmax(logits_masked, 2)  # bs,sl,sl,vec


            # attn_score = tf.clip_by_value(
            #                             attn_score,
            #                             0.,
            #                             1.0,
            #                             name=None
            #                         )


            attn_score = layers.mask_for_high_rank(attn_score, attn_mask)

            print("attn_score : ", attn_score.shape)

            attn_result = tf.reduce_sum(attn_score * rep_map_tile, 2)  # bs,sl,vec

            print("attn_result : ", attn_result.shape)





        # img_atten = MultiDm_attention_with_dense( attn_result, rep_mask,img_tensor,None, 'img_atten',
        #                0.80, is_train,1e-4, 'relu',tensor_dict={}, name='fw_fw_attn2')





        with tf.variable_scope('output'):
            o_bias = tf.get_variable('o_bias',[ivec], tf.float32, tf.constant_initializer(0.))
            # input gate
            fusion_gate = tf.nn.sigmoid(
                layers.linear(rep_map, ivec, True, 0., 'linear_fusion_i', False, wd, keep_prob, is_train) +
                layers.linear(attn_result, ivec, True, 0., 'linear_fusion_a', False, wd, keep_prob, is_train) +
                o_bias)
            output = fusion_gate * rep_map + (1-fusion_gate) * attn_result
            output = layers.mask_for_high_rank(output, rep_mask)



        
        # with tf.variable_scope('output2'):
        #     o_bias = tf.get_variable('o_bias2',[ivec], tf.float32, tf.constant_initializer(0.))
        #     # input gate
        #     fusion_gate = tf.nn.sigmoid(
        #         layers.linear(output, ivec, True, 0., 'linear_fusion_i2', False, wd, keep_prob, is_train) +
        #         layers.linear(img_atten, ivec, True, 0., 'linear_fusion_a2', False, wd, keep_prob, is_train) +
        #         o_bias)
        #     output = fusion_gate * output + (1-fusion_gate) * img_atten
        #     output = layers.mask_for_high_rank(output, rep_mask)

        # save attn
        if tensor_dict is not None and name is not None:
            tensor_dict[name + '_dependent'] = dependent
            tensor_dict[name + '_head'] = head
            tensor_dict[name] = attn_score
            tensor_dict[name + '_gate'] = fusion_gate



        # output = layers.layer_normalize(output)
        return output
      
      


def directional_attention_Image_dense(rep_tensor, rep_mask, img_tensor,direction=None, scope=None,
                                     keep_prob=1., is_train=None, wd=0., activation='elu',
                                     tensor_dict=None, name=None):
    def scaled_tanh(x, scale=5.):
        return scale * tf.nn.tanh(1./scale * x)

    bs, sl, vec = tf.shape(rep_tensor)[0], tf.shape(rep_tensor)[1], tf.shape(rep_tensor)[2]
    ivec = rep_tensor.get_shape()[2]
    with tf.variable_scope(scope or 'directional_attention_%s' % direction or 'diag'):
        # mask generation
        sl_indices = tf.range(sl, dtype=tf.int32)
        sl_col, sl_row = tf.meshgrid(sl_indices, sl_indices)
        if direction is None:
            direct_mask = tf.cast(tf.diag(- tf.ones([sl], tf.int32)) + 1, tf.bool)
        else:
            if direction == 'forward':
                direct_mask = tf.greater(sl_row, sl_col)
            else:
                direct_mask = tf.greater(sl_col, sl_row)
        direct_mask_tile = tf.tile(tf.expand_dims(direct_mask, 0), [bs, 1, 1])  # bs,sl,sl
        rep_mask_tile = tf.tile(tf.expand_dims(rep_mask, 1), [1, sl, 1])  # bs,sl,sl
        attn_mask = tf.logical_and(direct_mask_tile, rep_mask_tile)  # bs,sl,sl

        # non-linear
        rep_map = layers.bn_dense_layer(rep_tensor, ivec, True, 0., 'bn_dense_map', activation,
                                 False, wd, keep_prob, is_train)
        rep_map_tile = tf.tile(tf.expand_dims(rep_map, 1), [1, sl, 1, 1])  # bs,sl,sl,vec
        


        img_tensor_ = layers.bn_dense_layer(img_tensor, ivec, True, 0., 'img_tensor_', activation,
                                 False, wd, keep_prob, is_train)


        img_tensor_dp = layers.dropout(img_tensor_, keep_prob, is_train)



        img_tensor_tile = tf.tile(tf.expand_dims(img_tensor_, 1), [1, sl, 1, 1])

        # img_tensor_tile = tf.tile(tf.expand_dims(img_tensor_tile, 1), [1, sl, 1, 1,1])

        # print("img_tensor_tile.shape ", img_tensor_tile.shape)

        
        rep_map_dp = layers.dropout(rep_map, keep_prob, is_train)

        shape = tf.shape(img_tensor_tile)
       



        T1 = False

        

        # attention
        with tf.variable_scope('attention'):  # bs,sl,sl,vec
            f_bias = tf.get_variable('f_bias',[ivec], tf.float32, tf.constant_initializer(0.))
            dependent = layers.linear(rep_map_dp, ivec, False, scope='linear_dependent')  # bs,sl,vec
            dependent_etd = tf.expand_dims(dependent, 1)  # bs,1,sl,vec

            # dependent_etd = tf.concat([dependent_etd,img_tensor_tile],2)




            # print("dependent_etd.shape :: ",dependent_etd.shape)

            dependent2 = layers.linear(img_tensor_dp, ivec, False, scope='linear_dependent2')  # bs,sl,vec

            if T1:
                dependent_etd2 = tf.expand_dims(dependent2, 2)
            else:

                dependent_etd2 = tf.expand_dims(dependent2, 1)  # bs,1,sl,vec
                dependent_etd2 = tf.expand_dims(dependent_etd2, 2)  # bs,1,sl,vec


         

            head = layers.linear(rep_map_dp, ivec, False, scope='linear_head') # bs,sl,vec
            head_etd = tf.expand_dims(head, 2)  # bs,sl,1,vec

            

            # img_tensor_tile = layers.linear(img_tensor_tile, ivec, False, scope='img_tensor_tile2')
            img_tensor_tile= tf.expand_dims(img_tensor_tile,2)
            print("new img_tensor_tile.shape ", img_tensor_tile.shape)

            logits = scaled_tanh(dependent_etd + head_etd    + f_bias, 5.0)  # bs,sl,sl,vec




            logits_ = layers.bn_dense_layer(logits, ivec, True, 0., 'logits_', activation,
                                 False, wd, keep_prob, is_train)


            logits_ = layers.dropout(logits_, keep_prob, is_train)


            logits_ = layers.linear(logits_, ivec, False, scope='logits_2')

 
            logits_etd = tf.expand_dims(logits_,3)

            new_logits =scaled_tanh(logits_etd + dependent_etd2+f_bias,5.0) 

            print("dependent_etd2.shape ", dependent_etd2.shape)

            print("dir new_logits : ", new_logits.shape)
            attn  = tf.nn.softmax(new_logits,3)

            apply_Atten = img_tensor_tile*attn
            # print("apply_Atten ", apply_Atten.shape)
            # new_logits = tf.reduce_sum(apply_Atten,3)
            new_logits = tf.reduce_sum(apply_Atten,3)


            print("new_logits.shape :: ",new_logits.shape)

            # logits = new_logits + logits 


            o_bias = tf.get_variable('o_bias1',[ivec], tf.float32, tf.constant_initializer(1.))





            fusion_gate = tf.nn.sigmoid(
            layers.linear(new_logits, ivec, True, 0., 'linear_fusion_1i', False, wd, keep_prob, is_train) +
            layers.linear(logits, ivec, True, 0., 'linear_fusion_1a', False, wd, keep_prob, is_train) +
            o_bias)
            logits = fusion_gate * logits  + (1-fusion_gate) *  new_logits


            print("logits.shape ",logits.shape)

 


            logits_masked = layers.exp_mask_for_high_rank(logits, attn_mask)

            print("logits_masked : ", logits_masked.shape)




            attn_score = tf.nn.softmax(logits_masked, 2)  # bs,sl,sl,vec


            # attn_score = tf.clip_by_value(
            #                             attn_score,
            #                             0.,
            #                             1.0,
            #                             name=None
            #                         )


            attn_score = layers.mask_for_high_rank(attn_score, attn_mask)

            print("attn_score : ", attn_score.shape)

            attn_result = tf.reduce_sum(attn_score * rep_map_tile, 2)  # bs,sl,vec

            print("attn_result : ", attn_result.shape)





        # img_atten = MultiDm_attention_with_dense( attn_result, rep_mask,img_tensor,None, 'img_atten',
        #                0.80, is_train,1e-4, 'relu',tensor_dict={}, name='fw_fw_attn2')





        with tf.variable_scope('output'):
            o_bias = tf.get_variable('o_bias',[ivec], tf.float32, tf.constant_initializer(0.))
            # input gate
            fusion_gate = tf.nn.sigmoid(
                layers.linear(rep_map, ivec, True, 0., 'linear_fusion_i', False, wd, keep_prob, is_train) +
                layers.linear(attn_result, ivec, True, 0., 'linear_fusion_a', False, wd, keep_prob, is_train) +
                o_bias)
            output = fusion_gate * rep_map + (1-fusion_gate) * attn_result
            output = layers.mask_for_high_rank(output, rep_mask)


        #output = tf.add(new_logits,output)



        # with tf.variable_scope('output2'):
        #     o_bias = tf.get_variable('o_bias2',[ivec], tf.float32, tf.constant_initializer(0.))
        #     # input gate
        #     fusion_gate = tf.nn.sigmoid(
        #         layers.linear(output, ivec, True, 0., 'linear_fusion_2i', False, wd, keep_prob, is_train) +
        #         layers.linear(new_logits, ivec, True, 0., 'linear_fusion_2a', False, wd, keep_prob, is_train) +
        #         o_bias)
        #     output = fusion_gate * output + (1-fusion_gate) * new_logits
        #     output = layers.mask_for_high_rank(output, rep_mask)



        
        # with tf.variable_scope('output2'):
        #     o_bias = tf.get_variable('o_bias2',[ivec], tf.float32, tf.constant_initializer(0.))
        #     # input gate
        #     fusion_gate = tf.nn.sigmoid(
        #         layers.linear(output, ivec, True, 0., 'linear_fusion_i2', False, wd, keep_prob, is_train) +
        #         layers.linear(img_atten, ivec, True, 0., 'linear_fusion_a2', False, wd, keep_prob, is_train) +
        #         o_bias)
        #     output = fusion_gate * output + (1-fusion_gate) * img_atten
        #     output = layers.mask_for_high_rank(output, rep_mask)

        # save attn
        if tensor_dict is not None and name is not None:
            tensor_dict[name + '_dependent'] = dependent
            tensor_dict[name + '_head'] = head
            tensor_dict[name] = attn_score
            tensor_dict[name + '_gate'] = fusion_gate



        # output = layers.layer_normalize(output)



        return output,apply_Atten,attn
      
      
      
def multi_dimensional_attention(rep_tensor, scope=None,
                                keep_prob=1., is_train=None, wd=0., activation='elu',
                                tensor_dict=None, name=None):
    bs, sl, vec = tf.shape(rep_tensor)[0], tf.shape(rep_tensor)[1], tf.shape(rep_tensor)[2]
    ivec = rep_tensor.get_shape()[3]
    with tf.variable_scope(scope or 'multi_dimensional_attention'):
        map1 = layers.bn_dense_layer(rep_tensor, ivec, True, 0., 'bn_dense_map1', activation,
                              False, wd, keep_prob, is_train)
        map2 = layers.bn_dense_layer(map1, ivec, True, 0., 'bn_dense_map2', 'linear',
                              False, wd, keep_prob, is_train)
        # map2_masked = layers.exp_mask_for_high_rank(map2, rep_mask)

        soft = tf.nn.softmax(map2, 1)  # bs,sl,vec
        print("soft.shape :: ", soft.shape)
        return soft * rep_tensor
        attn_output = tf.reduce_sum(soft * rep_tensor, 1)  # bs, vec

        # save attn
        if tensor_dict is not None and name is not None:
            tensor_dict[name] = soft

        return 0