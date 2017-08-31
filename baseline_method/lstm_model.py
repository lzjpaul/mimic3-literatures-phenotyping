import keras
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers import Embedding
from keras.layers import LSTM
from keras.layers import Conv1D, MaxPooling1D
from keras.utils.vis_utils import plot_model

from data_loader import *

# embedding
embedding_size = 128        # word_embedding size
maxlen = 34                 # used to pad input tweet sequence
max_features = 61380        # vocabulary size

# cnn
kernel_size = 5
filters = 64
pool_size = 4

# lstm
lstm_output_size = 70

# dense
dense_size = 256            # optional, depends on performance

# training
batch_size = 32
epochs = 2


# loading training data
print 'loading data...'
x_train, y_train, x_test, y_test = load_corpus()


# building model
print 'building model...'
model = Sequential()
model.add(Embedding(max_features, embedding_size, input_length=maxlen))
model.add(Dropout(0.25))
model.add(Conv1D(filters=filters,
                 kernel_size=kernel_size,
                 padding='valid',
                 activation='relu',
                 strides=1))
model.add(MaxPooling1D(pool_size=pool_size))
model.add(LSTM(lstm_output_size))
model.add(Dense(1))
model.add(Activation('sigmoid'))

print 'compiling model...'
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'],)

# creating some callbacks
tensorboard_callback = keras.callbacks.TensorBoard(log_dir='logs/lstm_cnn')
checkpoint_callback = keras.callbacks.ModelCheckpoint(filepath='checkpoint/lstm_cnn.{epoch:02d}.hdf5', period=1)

print 'training model...'
model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          callbacks=[tensorboard_callback, checkpoint_callback],   # wait for specification
          validation_data=(x_test, y_test))

print 'saving model...'
model.save('model/lstm_cnn.final')


'''
1.  origin
maxlen = 53                 # used to pad input tweet sequence
max_features = 33366        # vocabulary size
2.  no stem
1, 5, 10, 20, 50, 80, 90, 95, 99 percentile:	[  2.   3.   4.   6.  12.  21.  26.  29.  34.]
input minimum word count:           6
vocabulary size:	                61380
input maximum sentence length:      34
input minimum sentence length:      4
Train on 1006333 samples, validate on 431287 samples
1006333/1006333 [==============================] - 9937s - loss: 0.4790 - acc: 0.7736 - val_loss: 0.4595 - val_acc: 0.7885
3.  stem
input minimum word count:           10
vocabulary size:	                32399
input maximum sentence length:      34
input minimum sentence length:      4
'''