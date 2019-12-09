#!/usr/bin/env python3

import os
import pptk
import numpy as np
from data import load_data
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class Display:
    def __init__(self):
        model_net       = 'ModelNet10'
        # model_net       = 'ModelNet40'
        self.data_dir   = 'data/{}'.format(model_net)
        self.visual_ptk = []

    def plot_point_cloud(self, label, pcl_data, big_point=False, color=False):
        print("[PPTK] {} - length pcl : {}".format(label, pcl_data.shape))
        visual_ptk = pptk.viewer(pcl_data[:,:3])

        if color:
            visual_ptk.attributes(pcl_data[:,-1])
        if big_point:
            # visual_ptk.set(point_size=0.0025)
            visual_ptk.set(point_size=0.1)

        return visual_ptk

    def plot_pcl(self, ax, x, y, z, title=""):
        ax.scatter(x, y, z, c='green')
        # ax.scatter(x, y, z)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        ax.set_xlim(0, 32)
        ax.set_ylim(0, 32)
        ax.set_zlim(0, 32)
        ax.set_title(title)
        ax.view_init(azim=225)        

    def plot_voxel(self, ax, voxel, title=""):
        ax.voxels(voxel, facecolors='green', edgecolor='k')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        ax.set_xlim(0, 32)
        ax.set_ylim(0, 32)
        ax.set_zlim(0, 32)
        ax.set_title(title) 
        ax.view_init(azim=225)

    def close_all(self):
        for v in self.visual_ptk:
            if v is not None:
                v.close()

        plt.close('all')

    def voxel_to_pcl(self, data):
        x, y, z = data.nonzero()
        return np.stack((x, y, z), axis=1)

    def run(self):
        (X_train, y_train), (X_test, y_test), target_names = load_data(self.data_dir)
        num_classes  = len(target_names)
        # display_mode = 'train_data'
        display_mode = 'test_data'

        if display_mode == 'train_data':
            X, y = X_train, y_train
        elif display_mode == 'test_data':
            X, y = X_test, y_test
        
        classes_indices  = [np.where(y == i)[0] for i in range(num_classes)]

        for idx, val in enumerate(classes_indices):
            rand_idx   = np.random.choice(val, 1)
            voxel_size = X[rand_idx].shape[1:-1]
            temp_data  = X[rand_idx].reshape(voxel_size)

            # visualize pptk
            pcl = self.voxel_to_pcl(temp_data)
            v   = self.plot_point_cloud(target_names[idx], pcl, big_point=True)
            self.visual_ptk.append(v)

            # visualize matplotplib
            fig = plt.figure()
            ax  = fig.gca(projection='3d')
            self.plot_voxel(ax, temp_data)

        plt.show(block=False)
        input('Enter to close all')
        self.close_all()

if __name__ == '__main__':
    display = Display()
    display.run()