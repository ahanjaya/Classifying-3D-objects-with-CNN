#!/usr/bin/env python3

import sys
import keras
import itertools
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from keras.optimizers import Adam
from keras.models import load_model, Sequential
from keras.utils.np_utils import to_categorical
from keras.layers.normalization import BatchNormalization
from keras.layers import Conv3D, MaxPooling3D, LeakyReLU
from keras.layers.core import Dense, Dropout, Activation, Flatten

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

class Deep_Learning:
    def __init__(self):
        augmented_data  = False
        # self.model_type = 'voxnet'
        self.model_type = 'm-vcnn1'

        if not augmented_data:
            self.pcl_dataset  = "wolf_3D_object_dataset.npz"
            self.pcl_model    = "wolf_model_{}.h5".format(self.model_type)
        else:
            self.pcl_dataset  = "aug_wolf_3D_object_dataset.npz"
            self.pcl_model    = "aug_wolf_model_{}.h5".format(self.model_type)

        self.load_model   = False
        self.debug        = True
        self.plot_data    = True
        self.save_data    = False
        self.batch_size   = 10
        self.epochs       = 50

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
        datasets = np.load(path)
        data     = datasets['data']
        labels   = datasets['labels']
        self.number_class = np.unique(labels).shape[0]
        return data, labels

    def preprocess_data(self, data, labels):
        data   = data.reshape(data.shape[0], 32, 32, 32, 1) # reshaping data
        labels = to_categorical(labels)                     # one hot encoded
        print('[DL] Total data : {}, {}'.format(data.shape,   type(data)))
        print('[DL] Total label: {}, {}'.format(labels.shape, type(labels)))

        X = data
        y = labels
        x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
        print('[DL] Train Data : {}, {}'.format(x_train.shape, y_train.shape))
        print('[DL] Test  Data : {}, {}'.format(x_test.shape,  y_test.shape))

        return (x_train, y_train), (x_test, y_test)

    def voxnet(self, train_data, test_data):
        model = Sequential()
        model.add(Conv3D(32, input_shape=(32, 32, 32, 1), kernel_size=(5, 5, 5), strides=(2, 2, 2), data_format='channels_last'))
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
        # print(model.metrics_names)

        x_train, y_train = train_data
        x_test,  y_test  = test_data

        self.history = model.fit(x_train, y_train, batch_size=self.batch_size, epochs=self.epochs, verbose=1, validation_data=(x_test, y_test))
        return model

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
        model.add(Dropout(0.5))

        model.add(Flatten())
        model.add(Dense(2048, activation='linear'))
        model.add(BatchNormalization())
        model.add(Dense(units=self.number_class, activation='softmax'))
        model.summary()

        model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.0001), metrics=['accuracy'])
        # print(model.metrics_names)

        x_train, y_train = train_data
        x_test,  y_test  = test_data

        self.history = model.fit(x_train, y_train, batch_size=self.batch_size, epochs=self.epochs, verbose=1, validation_data=(x_test, y_test))
        return model

    def evaluate(self, model, test_data):
        x_test, y_test = test_data
        score = model.evaluate(x_test, y_test, verbose=0)
        print('[DL] Test loss : {}, {}'.format(score[0], score[1]))

        predictions = model.predict_classes(x_test)
        y_test      = np.argmax(y_test, axis=1)
        report      = classification_report(y_test, predictions)
        print(report)

    def save(self, model):
        model.save(self.pcl_model)
        print('[DL] Saved model: {}'.format(self.pcl_model))

    def prediction(self, model, x):
        return model.predict_classes(x)

    def plot_history(self, history):
        loss_list     = [s for s in history.history.keys() if 'loss' in s and 'val' not in s]
        val_loss_list = [s for s in history.history.keys() if 'loss' in s and 'val' in s]
        acc_list      = [s for s in history.history.keys() if 'acc' in s and 'val' not in s]
        val_acc_list  = [s for s in history.history.keys() if 'acc' in s and 'val' in s]
        
        if len(loss_list) == 0:
            rospy.logerr('[DL] Loss is missing in history')
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
        # plt.show()

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

        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')

    ## multiclass or binary report
    ## If binary (sigmoid output), set binary parameter to True
    def full_multiclass_report(self, model, x, y_true, classes, batch_size=1, binary=False):
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
        print(cnf_matrix)
        self.plot_confusion_matrix(cnf_matrix,classes=classes)

    def run(self):
        print('[DL] Load model: {}'.format(self.load_model))

        data, labels = self.load_data(self.pcl_dataset)
        (x_train, y_train), (x_test, y_test) = self.preprocess_data(data, labels)

        if not self.load_model:
            if self.model_type == 'voxnet':
                model = self.voxnet((x_train, y_train), (x_test, y_test))
            elif self.model_type == 'm-vcnn1':
                model = self.MV_CNN_1((x_train, y_train), (x_test, y_test))
            else:
                print('[DL] Wrong model')
                sys.exit()

            if self.plot_data:
                self.plot_history(self.history)
                le       = LabelEncoder()
                classes  = [ 'small_suitcase', 'big_suitcase', 'black_chair', 'blue_chair', 'table' ]
                encoded_labels = le.fit_transform(classes)
                self.full_multiclass_report(model, x_test, y_test, le.inverse_transform(np.arange(len(classes))))

            if self.save_data:
                self.save(model)
            
            self.evaluate(model, (x_test, y_test))
        else:
            model = load_model(self.pcl_model)
            self.evaluate(model, (x_test, y_test))

        # pred = self.prediction(model, x_train)
        # print(pred)
        # print(np.argmax(y_train, axis=1))

        # test_data = x_test[0].reshape(1, 32, 32, 32, 1)
        # print(self.prediction(model, test_data))

        plt.show(block=False)
        input('Close: ')
        plt.close('all')

if __name__ == '__main__':
    dl = Deep_Learning()
    dl.run()