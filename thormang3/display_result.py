#!/usr/bin/env python3

import os
import sys
import yaml
import pickle
import itertools
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

class Results:
    def __init__(self):
        self.n_folder   = 9
        self.res_folder = "result/{}".format(self.n_folder)

        plt.style.use('seaborn-deep')
        plt.rcParams.update({'font.size': 22})         

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
        # fig_name = '{}/{}_{}.png'.format(self.res_folder, self.n_folder, name)
        # fig.savefig(fig_name, dpi=fig.dpi)

    def run(self):
        # load pickle
        self.pickle_file = "{}/{}_history.p".format(self.res_folder, self.n_folder)
        with open(self.pickle_file, 'rb') as filehandle:
            data = pickle.load(filehandle)

        cm      = data['cm']
        # classes  = [ 'big_suitcase', 'black_chair', 'blue_chair', 'small_suitcase', 'table']
        classes  = [ 'big_suitcase', 'black_chair', 'blue_chair', 'small_suitcase']
        self.plot_confusion_matrix('MVCNN', cm, classes=classes)

        plt.show(block=False)
        input('Close: ')
        plt.close('all')

if __name__ == '__main__':
    res = Results()
    res.run()