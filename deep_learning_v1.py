#!/usr/bin/env python3

import os
import keras
import itertools
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from keras.utils.np_utils import to_categorical
from keras.layers import Conv2D, Conv3D, MaxPooling2D, MaxPooling3D, LeakyReLU, concatenate, Input
from keras.layers.normalization import BatchNormalization
from keras.models import load_model, Sequential, model_from_json, Model
from keras.layers.core import Dense, Dropout, Activation, Flatten

from data import load_data
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

class Deep_Learning:
    def __init__(self):
        data_net       = 'ModelNet10'
        # data_net       = 'ModelNet40'
        self.model_net = 'voxnet'
        # self.model_net = 'V_CNN_1'
        # self.model_net = 'V_CNN_2'
        # self.model_net = 'MV_CNN_1'

        self.batch_size = 64
        self.epochs     = 50

        self.data_dir   = 'data/{}'.format(data_net)
        self.dl_model   = '{}/{}_{}.json'.format(self.data_dir, self.model_net, data_net)
        self.dl_weight  = '{}/{}_{}.hdf5'.format(self.data_dir, self.model_net, data_net)
        
        self.growth_memory() # limit tensorflow to use all of GPU resources

    def growth_memory(self):
        physical_devices = tf.config.experimental.list_physical_devices('GPU')
        if len(physical_devices) > 0:
            for k in range(len(physical_devices)):
                tf.config.experimental.set_memory_growth(physical_devices[k], True)
                print('memory growth:', tf.config.experimental.get_memory_growth(physical_devices[k]))
        else:
            print("Not enough GPU hardware devices available")

    def load_data(self, path):
        (X_train, y_train), (X_test, y_test), target_names = load_data(path)

        if self.model_net == 'V_CNN_1' or self.model_net == 'V_CNN_2':
            X_train = X_train.reshape(X_train.shape[:-1])
            X_test  = X_test.reshape(X_test.shape[:-1])

        y_train = to_categorical(y_train)
        y_test  = to_categorical(y_test)

        return (X_train, y_train), (X_test, y_test), target_names

    def voxnet(self, train_data, test_data, batch_size=128, epochs=50):
        model = Sequential()
        model.add(Conv3D(32, input_shape=(30, 30, 30, 1), kernel_size=(5, 5, 5), strides=(2, 2, 2), data_format='channels_last'))
        model.add(BatchNormalization())
        model.add(LeakyReLU(alpha=0.1))

        model.add(Conv3D(32, kernel_size=(3, 3, 3), strides=(1, 1, 1), data_format='channels_last'))
        model.add(BatchNormalization())
        model.add(LeakyReLU(alpha=0.1))
        model.add(MaxPooling3D(pool_size=(2, 2, 2), data_format='channels_last',))
        
        model.add(Flatten())
        model.add(Dense(128, activation='linear'))
        model.add(BatchNormalization())
        model.add(Dense(units=self.number_class, activation='softmax'))
        model.summary()
        model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.0001), metrics=['accuracy'])

        json_config = model.to_json()
        with open(self.dl_model, 'w') as json_file:
            json_file.write(json_config)

        checkpoint       = ModelCheckpoint(self.dl_weight, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
        x_train, y_train = train_data
        x_test,  y_test  = test_data

        self.history = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1, validation_data=(x_test, y_test), callbacks=[checkpoint])
        return model

    def MV_CNN_1(self, train_data, test_data, batch_size=128, epochs=50):
        model = Sequential()
        model.add(Conv3D(32, input_shape=(30, 30, 30, 1), kernel_size=(3, 3, 3), strides=(1, 1, 1), data_format='channels_last'))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2), data_format='channels_last',))
        
        model.add(Conv3D(32, kernel_size=(3, 3, 3), strides=(1, 1, 1), data_format='channels_last'))
        model.add(BatchNormalization())
        model.add(Activation('relu'))

        model.add(Conv3D(32, kernel_size=(3, 3, 3), strides=(1, 1, 1), data_format='channels_last'))
        model.add(BatchNormalization())
        model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2), data_format='channels_last',))
        model.add(Dropout(0.5))

        model.add(Flatten())
        model.add(Dense(2048, activation='linear'))
        model.add(BatchNormalization())
        model.add(Dense(units=self.number_class, activation='softmax'))
        model.summary()

        model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.0001), metrics=['accuracy'])
        # print(model.metrics_names)

        json_config = model.to_json()
        with open(self.dl_model, 'w') as json_file:
            json_file.write(json_config)

        checkpoint       = ModelCheckpoint(self.dl_weight, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
        x_train, y_train = train_data
        x_test,  y_test  = test_data

        self.history = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1, validation_data=(x_test, y_test), callbacks=[checkpoint])
        return model

    def V_CNN_1(self, train_data, test_data, batch_size=128, epochs=50):
        model = Sequential()
        model.add(Conv2D(64, input_shape=(30, 30, 30), kernel_size=(3, 3), strides=(1, 1), data_format='channels_last'))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), data_format='channels_last',))
        
        model.add(Conv2D(64, kernel_size=(3, 3), strides=(1, 1), data_format='channels_last'))
        model.add(BatchNormalization())
        model.add(Activation('relu'))

        model.add(Conv2D(32, kernel_size=(3, 3), strides=(1, 1), data_format='channels_last'))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), data_format='channels_last',))
        # model.add(Dropout(0.5))

        model.add(Flatten())
        model.add(Dense(2048, activation='linear'))
        model.add(BatchNormalization())
        model.add(Dense(units=self.number_class, activation='softmax'))
        model.summary()

        model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.0001), metrics=['accuracy'])
        # print(model.metrics_names)

        json_config = model.to_json()
        with open(self.dl_model, 'w') as json_file:
            json_file.write(json_config)

        checkpoint       = ModelCheckpoint(self.dl_weight, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
        x_train, y_train = train_data
        x_test,  y_test  = test_data

        self.history = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1, validation_data=(x_test, y_test), callbacks=[checkpoint])
        return model

    def V_CNN_2(self, train_data, test_data, batch_size=128, epochs=50):
        input_img = Input(shape=(30, 30, 30))

        layer_1  = Conv2D(20, kernel_size=(1, 1), strides=(1, 1), padding='same')(input_img)
        layer_2  = Conv2D(20, kernel_size=(3, 3), strides=(1, 1), padding='same')(input_img)
        layer_3  = Conv2D(20, kernel_size=(5, 5), strides=(1, 1), padding='same')(input_img)
        mid_1    = concatenate([layer_1, layer_2, layer_3], axis = 3)
        layer_4  = Activation('relu')(mid_1)
        layer_4  = Dropout(0.2)(layer_4)

        layer_5  = Conv2D(30, kernel_size=(1, 1), strides=(1, 1), padding='same')(layer_4)
        layer_6  = Conv2D(30, kernel_size=(3, 3), strides=(1, 1), padding='same')(layer_4)
        mid_2    = concatenate([layer_5, layer_6], axis = 3)
        layer_7  = Activation('relu')(mid_2)
        layer_7  = Dropout(0.3)(layer_7)

        layer_8  = Conv2D(30, kernel_size=(3, 3), strides=(1, 1), padding='same')(layer_7)
        layer_8  = Activation('relu')(layer_8)
        # layer_8  = Dropout(0.5)(layer_8)
        layer_8  = BatchNormalization()(layer_8)
        layer_8  = Dropout(0.5)(layer_8)
        
        flat_3   = Flatten()(layer_8)
        dense_1  = Dense(2048, activation='relu')(flat_3)
        dense_1  = Dense(128, activation='relu')(dense_1)
        dense_1  = BatchNormalization()(dense_1)
        output   = Dense(self.number_class, activation='softmax')(dense_1)

        model = Model([input_img], output)
        model.summary()
        model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.0001), metrics=['accuracy'])

        # print(model.metrics_names)

        json_config = model.to_json()
        with open(self.dl_model, 'w') as json_file:
            json_file.write(json_config)

        checkpoint       = ModelCheckpoint(self.dl_weight, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
        x_train, y_train = train_data
        x_test,  y_test  = test_data

        self.history = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1, validation_data=(x_test, y_test), callbacks=[checkpoint])
        return model

    def plot_history(self, history):
        loss_list     = [s for s in history.history.keys() if 'loss' in s and 'val' not in s]
        val_loss_list = [s for s in history.history.keys() if 'loss' in s and 'val' in s]
        acc_list      = [s for s in history.history.keys() if 'acc'  in s and 'val' not in s]
        val_acc_list  = [s for s in history.history.keys() if 'acc'  in s and 'val' in s]
        
        if len(loss_list) == 0:
            print('Loss is missing in history')
            return 
        
        ## As loss always exists
        epochs = range(1,len(history.history[loss_list[0]]) + 1)
        
        ## Loss
        plt.figure(1)
        for l in loss_list:
            plt.plot(epochs, history.history[l], 'b', label='Training loss (' + str(str(format(history.history[l][-1],'.5f'))+')'))
        for l in val_loss_list:
            plt.plot(epochs, history.history[l], 'g', label='Validation loss (' + str(str(format(history.history[l][-1],'.5f'))+')'))
        
        plt.title('Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        
        ## Accuracy
        plt.figure(2)
        for l in acc_list:
            plt.plot(epochs, history.history[l], 'b', label='Training accuracy (' + str(format(history.history[l][-1],'.5f'))+')')
        for l in val_acc_list:    
            plt.plot(epochs, history.history[l], 'g', label='Validation accuracy (' + str(format(history.history[l][-1],'.5f'))+')')

        plt.title('Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()

    def plot_confusion_matrix(self, cm, classes, normalize=False, cmap=plt.cm.Blues):
        """
        This function prints and plots the confusion matrix.
        Normalization can be applied by setting `normalize=True`.
        """
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            title='Normalized confusion matrix'
        else:
            title='Confusion matrix'

        plt.figure(3)
        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)

        fmt = '.2f' if normalize else 'd'
        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, format(cm[i, j], fmt), horizontalalignment="center", color="white" if cm[i, j] > thresh else "black")

        # plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')

    # multiclass or binary report, If binary (sigmoid output), set binary parameter to True
    def full_multiclass_report(self, model, x, y_true, classes, batch_size=128, binary=False):
        # 1. Transform one-hot encoded y_true into their class number
        if not binary:
            y_true = np.argmax(y_true, axis=1)

        # 2. Predict classes and stores in y_pred
        if self.model_net == 'V_CNN_2':
            y_pred = model.predict(x, batch_size=batch_size)
            y_pred = np.argmax(y_pred, axis=1)
        else:
            y_pred = model.predict_classes(x, batch_size=batch_size)
        
        # 3. Print accuracy score
        # accuracy = np.mean(y_pred==y_true)
        print("Accuracy : "+ str(accuracy_score(y_true, y_pred)))
        print("")

        # 4. Print classification report
        print("Classification Report")
        print(classification_report(y_true, y_pred, digits=3))

        # 5. Plot confusion matrix
        cnf_matrix = confusion_matrix(y_true, y_pred)
        print(cnf_matrix.shape)
        self.plot_confusion_matrix(cnf_matrix, classes=classes)

    def run(self):
        # Load data
        (X_train, y_train), (X_test, y_test), target_names = self.load_data(self.data_dir)

        # preprocess
        self.number_class   = np.unique(target_names).shape[0]
        print(self.number_class)
        self.le             = LabelEncoder()
        self.encoded_labels = self.le.fit_transform(target_names)

        # training
        if self.model_net == 'voxnet':
            model = self.voxnet((X_train, y_train),  (X_test, y_test), batch_size=self.batch_size, epochs=self.epochs)
        elif self.model_net == 'V_CNN_1':
            model = self.V_CNN_1((X_train, y_train), (X_test, y_test), batch_size=self.batch_size, epochs=self.epochs) 
        elif self.model_net == 'V_CNN_2':
            model = self.V_CNN_2((X_train, y_train), (X_test, y_test), batch_size=self.batch_size, epochs=self.epochs) 
        elif self.model_net == 'MV_CNN_1':
            model = self.MV_CNN_1((X_train, y_train), (X_test, y_test), batch_size=self.batch_size, epochs=self.epochs) 

        self.plot_history(self.history)
        del model

        # load best weight
        with open(self.dl_model) as json_file:
            json_config = json_file.read()
            model       = model_from_json(json_config)
        model.load_weights(self.dl_weight)
        # model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

        # evaluate
        self.full_multiclass_report(model, X_test, y_test, self.le.inverse_transform(np.arange(len(target_names))), batch_size=self.batch_size)
        
        # show graph
        plt.show(block=False)
        input('Close: ')
        plt.close('all')

if __name__ == '__main__':
    dl = Deep_Learning()
    dl.run()