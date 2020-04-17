#!/usr/bin/env python3

import os
import sys
import yaml
import keras
import pickle
import itertools
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from time import sleep

from keras.optimizers import Adam
from keras.models import Sequential, Model
from keras.utils.np_utils import to_categorical
from keras.layers.normalization import BatchNormalization
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, Conv3D, MaxPooling2D, MaxPooling3D, LeakyReLU

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

class Deep_Learning:
    def __init__(self):
        # self.growth_memory() # limit tensorflow to use all of GPU resources
        
        self.pcl_dataset  = "wolf_3D_object_dataset.npz"
        self.pcl_dataset1 = "wolf_dataset_1.npz"
        self.pcl_dataset2 = "wolf_dataset_2.npz"

        # directory
        self.init_folder()

        # hyperparameter
        self.hyperparameter()

    def hyperparameter(self):
        self.data_mode = 'combine' # 'general'
        self.batch_sz  = 2 #10
        self.epochs    = 100
        self.lr_rate   = 0.0001
        self.drop_out  = 0 # 0.2
        self.loss_func = 'categorical_crossentropy' # 'kullback_leibler_divergence'

        # K-fold parameter
        self.n_splits  = 5

    def init_folder(self):
        result_path      = 'result/'
        self.n_folder    = len(os.walk(result_path).__next__()[1])

        if self.n_folder >= 1:
            prev_folder = "{}/{}".format(result_path, self.n_folder-1)
            if len(os.walk(prev_folder).__next__()[2]) == 0:
                self.n_folder -= 1

        self.res_folder  = "{}/{}".format(result_path, self.n_folder)           
        if not os.path.exists(self.res_folder):
            os.mkdir(self.res_folder)

    def growth_memory(self):
        physical_devices = tf.config.experimental.list_physical_devices('GPU')
        if len(physical_devices) > 0:
            for k in range(len(physical_devices)):
                tf.config.experimental.set_memory_growth(physical_devices[k], True)
                print('memory growth:', tf.config.experimental.get_memory_growth(physical_devices[k]))
        else:
            print("Not enough GPU hardware devices available")

    def load_data(self, path):
        datasets = np.load(path)
        data     = datasets['data']
        labels   = datasets['labels']
        self.number_class = np.unique(labels).shape[0]

        # if self.data_mode == 'general':
        #     self.number_class = np.unique(labels).shape[0]
        # else:
            # self.number_class = 5

        data   = data.reshape(data.shape[0], 32, 32, 32, 1) # reshaping data
        labels = to_categorical(labels)   
        return data, labels

    def preprocess_data(self, data, labels):
        # data   = data.reshape(data.shape[0], 32, 32, 32, 1) # reshaping data
        # labels = to_categorical(labels)                     # one hot encoded
        # print('[DL] Total data : {}, {}'.format(data.shape,   type(data)))
        # print('[DL] Total label: {}, {}'.format(labels.shape, type(labels)))

        X = data
        y = labels
        x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
        print('[DL] Train Data : {}, {}'.format(x_train.shape, y_train.shape))
        print('[DL] Test  Data : {}, {}'.format(x_test.shape,  y_test.shape))

        return (x_train, y_train), (x_test, y_test)

    def V_CNN_1(self, train_data, test_data):
        model = Sequential()
        model.add(Conv2D(64, input_shape=(32, 32, 32), kernel_size=(3, 3), strides=(1, 1), data_format='channels_last'))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), data_format='channels_last',))
        
        model.add(Conv2D(64, kernel_size=(3, 3), strides=(1, 1), data_format='channels_last'))
        model.add(BatchNormalization())
        model.add(Activation('relu'))

        model.add(Conv2D(64, kernel_size=(3, 3), strides=(1, 1), data_format='channels_last'))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), data_format='channels_last',))
        model.add(Dropout(self.drop_out))

        model.add(Flatten())
        model.add(Dense(2048, activation='linear'))
        model.add(BatchNormalization())
        model.add(Dense(units=self.number_class, activation='softmax'))
        model.summary()

        model.compile(loss=self.loss_func, optimizer=Adam(lr=self.lr_rate), metrics=['accuracy'])
        # print(model.metrics_names)

        x_train, y_train = train_data
        x_test,  y_test  = test_data

        history = model.fit(x_train, y_train, batch_size=self.batch_sz, epochs=self.epochs, verbose=1, validation_data=(x_test, y_test))
        return model, history

    def VoxNet(self, train_data, test_data):
        model = Sequential()
        model.add(Conv3D(32, input_shape=(32, 32, 32, 1), kernel_size=(5, 5, 5), strides=(2, 2, 2), data_format='channels_last'))
        model.add(BatchNormalization())
        model.add(LeakyReLU(alpha=0.1))

        model.add(Conv3D(32, kernel_size=(3, 3, 3), strides=(1, 1, 1), data_format='channels_last'))
        model.add(BatchNormalization())
        model.add(LeakyReLU(alpha=0.1))
        model.add(MaxPooling3D(pool_size=(2, 2, 2), data_format='channels_last',))
        model.add(Dropout(self.drop_out))
        
        model.add(Flatten())
        model.add(Dense(128, activation='linear'))
        model.add(BatchNormalization())
        model.add(Dense(units=self.number_class, activation='softmax'))
        model.summary()
        model.compile(loss=self.loss_func, optimizer=Adam(lr=self.lr_rate), metrics=['accuracy'])
        # print(model.metrics_names)

        x_train, y_train = train_data
        x_test,  y_test  = test_data

        history = model.fit(x_train, y_train, batch_size=self.batch_sz, epochs=self.epochs, verbose=1, validation_data=(x_test, y_test))
        return model, history

    def MV_CNN_1(self, train_data, test_data):
        model = Sequential()
        model.add(Conv3D(32, input_shape=(32, 32, 32, 1), kernel_size=(3, 3, 3), strides=(1, 1, 1), data_format='channels_last'))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2), data_format='channels_last',))
        
        model.add(Conv3D(32, kernel_size=(3, 3, 3), strides=(1, 1, 1), data_format='channels_last'))
        model.add(BatchNormalization())
        model.add(Activation('relu'))

        model.add(Conv3D(32, kernel_size=(3, 3, 3), strides=(1, 1, 1), data_format='channels_last'))
        model.add(BatchNormalization())
        model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2), data_format='channels_last',))
        model.add(Dropout(self.drop_out))

        model.add(Flatten())
        model.add(Dense(2048, activation='linear'))
        model.add(BatchNormalization())
        model.add(Dense(units=self.number_class, activation='softmax'))
        model.summary()

        model.compile(loss=self.loss_func, optimizer=Adam(lr=self.lr_rate), metrics=['accuracy'])
        # print(model.metrics_names)

        x_train, y_train = train_data
        x_test,  y_test  = test_data

        history = model.fit(x_train, y_train, batch_size=self.batch_sz, epochs=self.epochs, verbose=1, validation_data=(x_test, y_test))
        return model, history

    def plot_single_graph(self, mode, vcnn1, voxnet, mvcnn):
        plt.style.use('seaborn-deep')
        plt.rcParams.update({'font.size': 22})

        fig, ax = plt.subplots(1, 1, figsize=(12,8))
        
        # mode = 'val_loss' or 'val_accuracy' or 'accuracy' or 'loss'])
        if 'loss' in mode:
            val_vcnn1  = np.min( np.array(vcnn1.history[mode]) )
            val_voxnet = np.min( np.array(voxnet.history[mode]) )
            val_mvcnn1 = np.min( np.array(mvcnn.history[mode]) )
        else:
            val_vcnn1  = np.max( np.array(vcnn1.history[mode]) )
            val_voxnet = np.max( np.array(voxnet.history[mode]) )
            val_mvcnn1 = np.max( np.array(mvcnn.history[mode]) )

        epochs = range(1,self.epochs + 1)
        ax.plot(epochs, vcnn1.history [mode], 'r', label='VCNN1 - {0:.2f}' .format(val_vcnn1))
        ax.plot(epochs, voxnet.history[mode], 'b', label='VoxNet - {0:.2f}'.format(val_voxnet))
        ax.plot(epochs, mvcnn.history[mode],  'g', label='MVCNN - {0:.2f}'.format(val_mvcnn1))

        ax.legend()
        ax.grid()
        ax.set_xlabel('Epochs')
        ax.set_ylabel(mode)
        return fig

    def plot_double_graph(self, mode, vcnn1, voxnet, mvcnn):
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12,8))

        # mode = 'val_loss' or 'val_accuracy' or 'accuracy' or 'loss'])

        if 'loss' in mode:
            train_vcnn1  = np.min( np.array(vcnn1. history['loss']) )
            train_voxnet = np.min( np.array(voxnet.history['loss']) )
            train_mvcnn1 = np.min( np.array(mvcnn.history['loss']) )
            val_vcnn1    = np.min( np.array(vcnn1. history['val_loss']) )
            val_voxnet   = np.min( np.array(voxnet.history['val_loss']) )
            val_mvcnn1   = np.min( np.array(mvcnn.history['val_loss']) )
            ax1.set_ylabel('Training Loss')
            ax2.set_ylabel('Validation Loss')
        else:
            train_vcnn1  = np.max( np.array(vcnn1. history['accuracy']) )
            train_voxnet = np.max( np.array(voxnet.history['accuracy']) )
            train_mvcnn1 = np.max( np.array(mvcnn.history['accuracy']) )
            val_vcnn1    = np.max( np.array(vcnn1. history['val_accuracy']) )
            val_voxnet   = np.max( np.array(voxnet.history['val_accuracy']) )
            val_mvcnn1   = np.max( np.array(mvcnn.history['val_accuracy']) )
            ax1.set_ylabel('Training Accuracy')
            ax2.set_ylabel('Validation Accuracy')

        epochs = range(1,self.epochs + 1)
        ax1.plot(epochs, vcnn1.history [mode], 'r', label='VCNN1 - {0:.2f}' .format(train_vcnn1))
        ax1.plot(epochs, voxnet.history[mode], 'b', label='VoxNet - {0:.2f}'.format(train_voxnet))
        ax1.plot(epochs, mvcnn.history[mode],  'g', label='MVCNN - {0:.2f}'.format(train_mvcnn1))

        ax2.plot(epochs, vcnn1.history ['val_'+mode], 'r', label='VCNN1 - {0:.2f}' .format(val_vcnn1))
        ax2.plot(epochs, voxnet.history['val_'+mode], 'b', label='VoxNet - {0:.2f}'.format(val_voxnet))
        ax2.plot(epochs, mvcnn.history['val_'+mode],  'g', label='MVCNN - {0:.2f}'.format(val_mvcnn1))

        ax1.legend()
        ax2.legend()
        ax1.grid()
        ax2.grid()
        ax1.set_xlabel('Epochs')
        ax2.set_xlabel('Epochs')
        fig.tight_layout()

    def plot_confusion_matrix(self, name, cm, classes, normalize=False, cmap=plt.cm.Blues):
        """
        This function prints and plots the confusion matrix.
        Normalization can be applied by setting `normalize=True`.
        """

        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            title='Normalized confusion matrix'
        else:
            title='Confusion matrix'

        # plt.figure(self.plt_num, figsize=(7.5, 6))
        # plt.figure(plt_num, figsize=(12, 8))
        fig, ax = plt.subplots(1, 1, figsize=(12,8))

        # plt.imshow(cm, interpolation='nearest', cmap=cmap)
        # plt.title(title)
        # plt.colorbar()

        im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
        fig.colorbar(im, ax=ax)
        
        tick_marks = np.arange(len(classes))

        ax.set_xticks(tick_marks)
        ax.set_xticklabels(classes)
        plt.xticks(rotation=45)

        ax.set_yticks(tick_marks)
        ax.set_yticklabels(classes)

        fmt = '.2f' if normalize else 'd'
        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            ax.text(j, i, format(cm[i, j], fmt), horizontalalignment="center", color="white" if cm[i, j] > thresh else "black")

        fig.tight_layout()
        ax.set_ylabel('True label')
        ax.set_xlabel('Predicted label')
        fig_name = '{}/{}_{}.png'.format(self.res_folder, self.n_folder, name)
        fig.savefig(fig_name, dpi=fig.dpi)

    def full_multiclass_report(self, model, name, x, y_true, classes, batch_size=1, binary=False):
        # 1. Transform one-hot encoded y_true into their class number
        if not binary:
            y_true = np.argmax(y_true,axis=1)

        # 2. Predict classes and stores in y_pred
        y_pred = model.predict_classes(x, batch_size=batch_size)

        # 3. Print accuracy score
        print("Accuracy : "+ str(accuracy_score(y_true,y_pred)))
        print("")

        # 4. Print classification report
        print("Classification Report")
        print(classification_report(y_true,y_pred,digits=5))    

        # 5. Plot confusion matrix
        cnf_matrix = confusion_matrix(y_true,y_pred)
        self.plot_confusion_matrix(name, cnf_matrix, classes=classes)

        return cnf_matrix

    def run(self):
        if self.data_mode == 'general':
            data, labels = self.load_data(self.pcl_dataset)      

        elif self.data_mode == 'combine':
            X1, Y1    = self.load_data(self.pcl_dataset1)
            X2, Y2    = self.load_data(self.pcl_dataset2)
            # Y2        = np.zeros((tY2.shape[0], tY2.shape[1]+1))
            # Y2[:,:-1] = tY2
            data      = np.concatenate((X1, X2), axis=0)
            labels    = np.concatenate((Y1, Y2), axis=0)

        (x_train, y_train), (x_test, y_test) = self.preprocess_data(data, labels)
        x_train_2D = np.squeeze(x_train)
        x_test_2D  = np.squeeze(x_test)

        # training
        model_vcnn1,  vcnn1  = self.V_CNN_1 ((x_train_2D, y_train), (x_test_2D, y_test))
        model_voxnet, voxnet = self.VoxNet  ((x_train, y_train),    (x_test, y_test))
        model_mvcnn,  mvcnn  = self.MV_CNN_1((x_train, y_train),    (x_test, y_test))

        # plot data
        fig1 = self.plot_single_graph('loss',         vcnn1, voxnet, mvcnn) # training loss        
        fig2 = self.plot_single_graph('val_loss',     vcnn1, voxnet, mvcnn) # validation loss
        fig3 = self.plot_single_graph('accuracy',     vcnn1, voxnet, mvcnn) # training accuracy
        fig4 = self.plot_single_graph('val_accuracy', vcnn1, voxnet, mvcnn) # validation accuracy

        # plot multi graph
        # self.plot_double_graph('loss',     vcnn1, voxnet, mvcnn) # loss        
        # self.plot_double_graph('accuracy', vcnn1, voxnet, mvcnn) # accuracy        

        le = LabelEncoder()
        
        # if self.data_mode == 'general':
        #     classes  = [ 'small_suitcase', 'big_suitcase', 'black_chair', 'blue_chair']
        # else:
        #     classes  = [ 'small_suitcase', 'big_suitcase', 'black_chair', 'blue_chair', 'table' ]
        classes  = [ 'big_suitcase', 'black_chair', 'blue_chair', 'small_suitcase']
        encoded_labels = le.fit_transform(classes)

        cnf_vccn1  = self.full_multiclass_report(model_vcnn1,  'vcnn1',  x_test_2D, y_test, le.inverse_transform(np.arange(len(classes))))
        cnf_voxnet = self.full_multiclass_report(model_voxnet, 'voxnet', x_test,    y_test, le.inverse_transform(np.arange(len(classes))))
        cnf_mvcnn  = self.full_multiclass_report(model_mvcnn,  'mvcnn',  x_test,    y_test, le.inverse_transform(np.arange(len(classes))))

        # dump pickle
        self.memory = { 'vcnn1' : vcnn1,  'cnf_vcnn1':  cnf_vccn1,
                        'voxnet': voxnet, 'cnf_voxnet': cnf_voxnet,
                        'mvcnn' : mvcnn,  'cnf_mvcnn' : cnf_mvcnn }

        self.pickle_file = "{}/{}_history.p".format(self.res_folder, self.n_folder)
        with open(self.pickle_file, 'wb') as filehandle:
            pickle.dump(self.memory, filehandle)

        # save figure
        fig1_name = "{}/{}_train_loss.png".    format(self.res_folder, self.n_folder)
        fig2_name = "{}/{}_val_loss.png".      format(self.res_folder, self.n_folder)
        fig3_name = "{}/{}_train_accuracy.png".format(self.res_folder, self.n_folder)
        fig4_name = "{}/{}_val_accuracy.png".  format(self.res_folder, self.n_folder)
        fig1.savefig(fig1_name, dpi=fig1.dpi)
        fig2.savefig(fig2_name, dpi=fig2.dpi)
        fig3.savefig(fig3_name, dpi=fig3.dpi)
        fig4.savefig(fig4_name, dpi=fig4.dpi)

        # save model
        model_vcnn1.save ("{}/{}_VCNN1.h5" .format(self.res_folder, self.n_folder))
        model_voxnet.save("{}/{}_VoxNet.h5".format(self.res_folder, self.n_folder))
        model_mvcnn.save ("{}/{}_MVCNN.h5" .format(self.res_folder, self.n_folder))

        # save hyperparameters
        params = {
            'data_mode' : self.data_mode,
            'batch_size': self.batch_sz,
            'epochs'    : self.epochs,
            'lr_rate'   : self.lr_rate,
            'drop_out'  : self.drop_out,
            'loss_func' : self.loss_func,
        }
        yaml_file = "{}/{}_params.yaml".format(self.res_folder, self.n_folder)
        with open(yaml_file, 'w') as f:
            yaml.dump(params, f, default_flow_style=False)

        # plt.show(block=False)
        # input('Close: ')
        # plt.close('all')
        del model_vcnn1
        del model_voxnet
        del model_mvcnn
        sleep(3)

    ################################################
    ############### Cross Validation ###############

    def eval_model(self, name, model, x_test, y_test):
        print('\nModel evaluation: {}'.format(name))
        loss, acc = model.evaluate(x_test, y_test)
        print('Loss: {}'.format(loss))
        print('Acc:  {}'.format(acc))

        return loss, acc

    def plot_crossval(self, mode, voxnet, vcnn1, mvcnn):
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12,8))
        n_split = range(1, self.n_splits+1)

        mean_voxnet = np.mean(voxnet)
        mean_vcnn1  = np.mean(vcnn1)
        mean_mvcnn  = np.mean(mvcnn)
        
        ax1.axhline(y=mean_voxnet, color='r', ls='--', label='Mean score = {:.2f}'.format(mean_voxnet))
        ax2.axhline(y=mean_vcnn1,  color='b', ls='--', label='Mean score = {:.2f}'.format(mean_vcnn1))
        ax3.axhline(y=mean_mvcnn,  color='g', ls='--', label='Mean score = {:.2f}'.format(mean_mvcnn))

        ax1.bar(n_split, voxnet, color='r', label='VoxNet')
        ax2.bar(n_split, vcnn1,  color='b', label='VCNN1')
        ax3.bar(n_split, mvcnn,  color='g', label='MVCNN')

        ax1.legend()
        ax2.legend()
        ax3.legend()

        ax1.grid()
        ax2.grid()
        ax3.grid()

        ax1.set_xlabel('Training Instances')
        ax1.set_ylabel(mode)

        ax2.set_xlabel('Training Instances')
        ax2.set_ylabel(mode)

        ax3.set_xlabel('Training Instances')
        ax3.set_ylabel(mode)
        fig.tight_layout()

        return fig

    def plot_mean_cross(self, loss, acc):
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12,8))
        xtick_label = ['VoxNet', 'VCNN1', 'MVCNN']
        n_model     = range(3)

        ax1.bar(n_model, loss, color='r')
        ax2.bar(n_model, acc,  color='g')

        ax1.grid()
        ax2.grid()

        ax1.set_xticks(n_model)
        ax1.set_xticklabels(xtick_label)
        ax1.set_ylabel('Loss')

        ax2.set_xticks(n_model)
        ax2.set_xticklabels(xtick_label)
        ax2.set_ylabel('Accuracy')

        fig.tight_layout()
        return fig
    
    def run_cross_val(self):
        fold_mvcnn  = []
        fold_voxnet = []
        fold_vcnn1  = []

        # data
        X, Y = self.load_data(self.pcl_dataset)
        X_2D = np.squeeze(X)

        # X1, Y1 = self.load_data(self.pcl_dataset1)
        # X2, temp_Y2 = self.load_data(self.pcl_dataset2)

        # # add one column
        # Y2 = np.zeros((temp_Y2.shape[0], temp_Y2.shape[1]+1))
        # Y2[:,:-1] = temp_Y2

        # X = np.concatenate((X1, X2), axis=0)
        # Y = np.concatenate((Y1, Y2), axis=0)
        # X_2D = np.squeeze(X)

        for train_index, test_index in KFold(n_splits=self.n_splits, shuffle=True).split(X):
            # 3D data
            x_train, x_test = X[train_index], X[test_index]
            y_train, y_test = Y[train_index], Y[test_index]

            model_mvcnn, mvcnn   = self.MV_CNN_1((x_train, y_train), (x_test, y_test))
            loss, acc = self.eval_model('MVCNN', model_mvcnn, x_test, y_test)
            fold_mvcnn.append((loss, acc))

            model_voxnet, voxnet = self.voxnet  ((x_train, y_train), (x_test, y_test))
            loss, acc = self.eval_model('VoxNet', model_voxnet, x_test, y_test)
            fold_voxnet.append((loss, acc))

            # 2D data
            x_train_2D, x_test_2D = X_2D[train_index], X_2D[test_index]

            model_vcnn1, vcnn1   = self.V_CNN_1((x_train_2D, y_train), (x_test_2D, y_test))
            loss, acc = self.eval_model('VCNN1', model_vcnn1, x_test_2D, y_test)
            fold_vcnn1.append((loss, acc))


        # cross validation
        fold_mvcnn  = np.array(fold_mvcnn)
        fold_voxnet = np.array(fold_voxnet)
        fold_vcnn1  = np.array(fold_vcnn1)

        # mean loss
        loss_mvcnn  = np.mean(fold_mvcnn[:,0])
        loss_voxnet = np.mean(fold_voxnet[:,0])
        loss_vcnn1  = np.mean(fold_vcnn1[:,0])

        # mean accuracy
        acc_mvcnn  = np.mean(fold_mvcnn[:,1])
        acc_voxnet = np.mean(fold_voxnet[:,1])
        acc_vcnn1  = np.mean(fold_vcnn1[:,1])
        
        print('\nCross Validation Results')
        print('MVCNN')
        print('Loss    : {}, {}'.format(fold_mvcnn[:,0], acc_mvcnn  ))
        print('Accuracy: {}, {}'.format(fold_mvcnn[:,1], loss_mvcnn ))
                
        print('\nVoxNet')
        print('Loss    : {}, {}'.format(fold_voxnet[:,0], acc_voxnet  ))
        print('Accuracy: {}, {}'.format(fold_voxnet[:,1], loss_voxnet ))
              
        print('\nVCNN1')
        print('Loss    : {}, {}'.format(fold_vcnn1[:,0], acc_vcnn1  ))
        print('Accuracy: {}, {}'.format(fold_vcnn1[:,1], loss_vcnn1 ))

        # plot data
        fig_acc = self.plot_crossval('Loss',     fold_voxnet[:,0], fold_vcnn1[:,0], fold_mvcnn[:,0])
        fig_err = self.plot_crossval('Accuracy', fold_voxnet[:,1], fold_vcnn1[:,1], fold_mvcnn[:,1])
        
        loss     = [loss_voxnet, loss_vcnn1, loss_mvcnn]
        acc      = [acc_voxnet,  acc_vcnn1,  acc_mvcnn]
        fig_mean = self.plot_mean_cross(loss, acc)

        # dump pickle
        self.memory = {'fold_vcnn1': fold_vcnn1, 'fold_voxnet': fold_voxnet, 'fold_mvcnn': fold_mvcnn} #, 'cm': self.cnf_matrix }
        with open(self.pickle_file, 'wb') as filehandle:
            pickle.dump(self.memory, filehandle)

        # save figure
        fig1_name = "{}/{}_loss_fold.png".format(self.res_folder, self.n_folder)
        fig_err.savefig(fig1_name, dpi=fig_err.dpi)

        fig2_name = "{}/{}_acc_fold.png".format(self.res_folder, self.n_folder)
        fig_acc.savefig(fig2_name, dpi=fig_acc.dpi)

        fig3_name = "{}/{}_mean_fold.png".format(self.res_folder, self.n_folder)
        fig_mean.savefig(fig3_name, dpi=fig_mean.dpi)

        plt.show(block=False)
        input('Close: ')
        plt.close('all')

if __name__ == '__main__':
    dl = Deep_Learning()
    # 19
    dl.run()

    # 20
    dl.loss_func = 'kullback_leibler_divergence'
    dl.init_folder()
    dl.run()