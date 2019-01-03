from keras.layers import Input, Dense, Embedding, Conv2D, MaxPool2D, Conv1D, MaxPooling1D, concatenate
from keras.layers import Reshape, Flatten, Dropout, Concatenate
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam
from keras.models import Model
from keras import backend as K
import keras.layers as L
import keras

from sklearn.model_selection import train_test_split

def text_cnn(sequence_length, num_classes, vocab_size, embedding_size, 
             filter_sizes, num_filters, embedding_matrix, drop_out=0.5 ,l2_reg_lambda=0.0):
    input_x = L.Input(shape=(sequence_length,), name='input_x')
    
    # embedding layer
    if embedding_matrix is None:
        embedding = L.Embedding(vocab_size, embedding_size, name='embedding')(input_x)
    else:
        embedding = L.Embedding(vocab_size, embedding_size, weights=[embedding_matrix], name='embedding')(input_x)
    expend_shape = [embedding.get_shape().as_list()[1], embedding.get_shape().as_list()[2], 1]
    # embedding_chars = K.expand_dims(embedding, -1)    
    # 4D tensor [batch_size, seq_len, embeding_size, 1] seems like a gray picture
    embedding_chars = L.Reshape(expend_shape)(embedding)
    
    # conv->max pool
    pooled_outputs = []
    for i, filter_size in enumerate(filter_sizes):
        conv = L.Conv2D(filters=num_filters, 
                        kernel_size=[filter_size, embedding_size],
                        strides=1,
                        padding='valid',
                        activation='relu',
                        kernel_initializer=keras.initializers.TruncatedNormal(mean=0.0, stddev=0.1),
                        bias_initializer=keras.initializers.constant(value=0.1),
                        name=('conv_%d' % filter_size))(embedding_chars)
        # print("conv-%d: " % i, conv)
        max_pool = L.MaxPool2D(pool_size=[sequence_length - filter_size + 1, 1],
                               strides=(1, 1),
                               padding='valid',
                               name=('max_pool_%d' % filter_size))(conv)
        pooled_outputs.append(max_pool)
        # print("max_pool-%d: " % i, max_pool)
    
    # combine all the pooled features
    num_filters_total = num_filters * len(filter_sizes)
    h_pool = L.Concatenate(axis=3)(pooled_outputs)
    h_pool_flat = L.Reshape([num_filters_total])(h_pool)
    # add dropout
    dropout = L.Dropout(drop_out)(h_pool_flat)
    
    # output layer
    output = L.Dense(num_classes,
                     kernel_initializer='glorot_normal',
                     bias_initializer=keras.initializers.constant(0.1),
                     activation='softmax',
                     name='output')(dropout)
    
    model = keras.models.Model(inputs=input_x, outputs=output)
    print('Use TextCNN model')
    return model