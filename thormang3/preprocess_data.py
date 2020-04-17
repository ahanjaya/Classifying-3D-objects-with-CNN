#!/usr/bin/env python3

import os
import pcl
import pptk
import natsort
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from time import sleep
from pyntcloud import PyntCloud
from sklearn.cluster import KMeans
from mpl_toolkits.mplot3d import Axes3D

class Preprocess_Data:
    def __init__(self):
        print("[Pre.] Pioneer Wolf Preprocess Data - Running")

        self.pcl_dataset  = "wolf_3D_object_dataset.npz"
        self.aug_dataset  = "aug_wolf_3D_object_dataset.npz"
        self.pcl_raw_path = "raw_pcl/"
        self.point_clouds = None
        self.data         = []
        self.labels       = []
        self.visual_ptk   = []

        self.show_plt     = False
        self.augmentating = False
        self.aug_theta    = np.arange(-45, 50, 20)
        # self.aug_theta    = np.arange(-90, 100, 10)
        self.aug_theta    = np.delete(self.aug_theta, np.argwhere(self.aug_theta==0))
        self.aug_dist     = np.arange(-1, 2, 1)
        # self.aug_dist     = np.delete(self.aug_dist, np.argwhere(self.aug_dist==0))

    def plot_point_cloud(self, label, pcl_data, big_point=False, color=True):
        print("[Pre.] {} - length pcl : {}".format(label, pcl_data.shape))
        visual_ptk = pptk.viewer(pcl_data[:,:3])

        if color:
            visual_ptk.attributes(pcl_data[:,-1])
        if big_point:
            visual_ptk.set(point_size=0.0025)

        visual_ptk.set(r=2.88)
        visual_ptk.set(phi=3.24)
        visual_ptk.set(theta=0.64)
        visual_ptk.set(lookat=(0.78, -0.01, -0.05))
        visual_ptk.set(show_axis=False)

        return visual_ptk

    def plot_2d(self, axes_, x_, y_, legend_=None, xlabel_="", ylabel_="", title_=""):
        axes_.plot(x_, y_, 'o-', label=legend_)
        axes_.set_xlabel(xlabel_)
        axes_.set_ylabel(ylabel_)
        axes_.set_title(title_)
        axes_.grid()

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

    def plot_voxel(self, ax, voxel, title=""):
        ax.voxels(voxel, facecolors='blue', edgecolor='k')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        ax.set_xlim(0, 32)
        ax.set_ylim(0, 32)
        ax.set_zlim(0, 32)
        ax.set_title(title) 

    def filter_raw_data(self, data_pcl):
        # 1st filter (knee joint)
        data_x   = data_pcl[:,0]
        x_min = np.round( np.min(data_x), 1) + 0.2
        x_max = np.round( np.max(data_x), 1) - 0.2
        data_pcl = data_pcl[np.where( (data_x >= x_min) & (data_x <= x_max))]

        # 2nd filter (wall)
        data_y   = data_pcl[:,1]
        y_min = np.round( np.min(data_y), 1) + 0.1
        y_max = np.round( np.max(data_y), 1) - 0.1
        data_pcl = data_pcl[np.where( (data_y >= y_min) & (data_y <= y_max))]

        # 3nd filter (ground)
        data_z   = data_pcl[:,2]
        z_min = np.round( np.min(data_z), 1) + 0.1
        data_pcl = data_pcl[np.where( (data_z >= z_min) )]

        cloud = pcl.PointCloud(np.array(data_pcl[:,:3], dtype=np.float32))
        sor   = cloud.make_voxel_grid_filter()
        # sor.set_leaf_size(0.01, 0.01, 0.01)
        sor.set_leaf_size(0.02, 0.02, 0.02)
        cloud_filtered = sor.filter()

        return np.array(cloud_filtered)
        # return np.array(data_pcl)

    def clustering(self, cloud):
        est = KMeans(n_clusters=2)
        fig = plt.figure(1, figsize=(10, 9))
        ax  = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)
        est.fit(cloud)
        labels = est.labels_
        ax.scatter(cloud[:, 0], cloud[:, 1], cloud[:, 2], c=labels.astype(np.float), edgecolor='k')

        ax.w_xaxis.set_ticklabels([])
        ax.w_yaxis.set_ticklabels([])
        ax.w_zaxis.set_ticklabels([])
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title('3 clusters')

    def voxelization_raw_data(self, human_body):
        dataset      = pd.DataFrame({'x': human_body[:,0], 'y': human_body[:,1], 'z': human_body[:,2]})
        cloud        = PyntCloud(dataset)
        voxelgrid_id = cloud.add_structure("voxelgrid", n_x=32, n_y=32, n_z=32)
        voxelgrid    = cloud.structures[voxelgrid_id]
        x_cords      = voxelgrid.voxel_x
        y_cords      = voxelgrid.voxel_y
        z_cords      = voxelgrid.voxel_z
        voxel        = np.zeros((32, 32, 32)).astype(np.bool)

        for x, y, z in zip(x_cords, y_cords, z_cords):
            voxel[x][y][z] = True
        return voxel

    def close_all(self):
        for v in self.visual_ptk:
            if v is not None:
                v.close()
        
        plt.close('all')

    def load_data(self, path):
        datasets = np.load(path)
        data     = datasets['data']
        labels   = datasets['labels']
        return data, labels

    def save_data(self, path, data, labels):
        data   = np.array(data)
        labels = np.array(labels)
        print('[Pre.] Saving data: {}'.format(path))
        print('[Pre.] Total data : {}'.format(data.shape))
        print('[Pre.] Total label: {}'.format(labels.shape))
        np.savez(path, data=data, labels=labels)

    def rotate_pcl(self, pcl, theta):
        x      = pcl[:,0]
        y      = pcl[:,1]
        z      = pcl[:,2]
        theta  = np.radians(theta)
        ox, oy = np.mean(x), np.mean(y)

        qx = ((x - ox) * np.cos(theta)) - ((y - oy) * np.sin(theta)) + ox
        qy = ((x - ox) * np.sin(theta)) + ((y - oy) * np.cos(theta)) + oy
        qx = qx.astype(int)
        qy = qy.astype(int)
        return np.stack((qx, qy, z), axis=1)

    def translate_pcl(self, pcl, diff):
        diff      = np.array(diff)
        pcl[:,:2] = pcl[:,:2] + diff 
        return pcl

    def voxelization(self, pcl):
        x_cords = pcl[:,0]
        y_cords = pcl[:,1]
        z_cords = pcl[:,2]
        voxel   = np.zeros((32, 32, 32)).astype(np.bool)
        for x, y, z in zip(x_cords, y_cords, z_cords):
            voxel[x][y][z] = True
        return voxel

    def augmentation_data(self, data, labels):
        # get voxel point
        augmented_data   = []
        augmented_labels = []

        for idx, voxel in enumerate(data):
            print('[Pre.] Augmenting: {}'.format(idx))
            # original data
            x, y, z = voxel.nonzero()
            pcl     = np.stack((x, y, z), axis=1)

            if self.show_plt:
                # original data
                fig = plt.figure()
                ax  = fig.add_subplot(1,2,1, projection='3d')
                self.plot_voxel(ax, data[0], title='Original Voxel')
                ax = fig.add_subplot(1,2,2,  projection='3d')
                self.plot_pcl(ax, x, y, z,  title='Original PCL')

            # augmentation rotation
            for theta in self.aug_theta:
                try:
                    rotated_pcl   = self.rotate_pcl(pcl, theta)
                    rotated_voxel = self.voxelization(rotated_pcl)

                    augmented_data.append(rotated_voxel)
                    augmented_labels.append(labels[idx])

                    if self.show_plt:
                        fig = plt.figure()
                        ax  = fig.add_subplot(1,2,1, projection='3d')
                        x, y, z = rotated_pcl[:,0], rotated_pcl[:,1], rotated_pcl[:,2]
                        title = 'Rotated PCL ({}deg)'. format(theta)
                        self.plot_pcl(ax, x, y, z, title=title)
                        ax = fig.add_subplot(1,2,2, projection='3d')
                        title = 'Rotated Voxel ({}deg)'. format(theta)
                        self.plot_voxel(ax, rotated_voxel, title=title)
                except:
                    pass

            # augmentation translation
            for x in self.aug_dist:
                for y in self.aug_dist:
                    if x != 0 or y != 0:
                        diff = (x, y)
                        translated_pcl   = self.translate_pcl(pcl, diff)
                        translated_voxel = self.voxelization(translated_pcl)

                        augmented_data.append(translated_voxel)
                        augmented_labels.append(labels[idx])

                        if self.show_plt:
                            fig = plt.figure()
                            ax  = fig.add_subplot(1,2,1, projection='3d')
                            tx, ty, tz = translated_pcl[:,0], translated_pcl[:,1], translated_pcl[:,2]
                            title = 'Translated PCL (X:{}, Y:{})'. format(diff[0], diff[1])
                            self.plot_pcl(ax, tx, ty, tz, title=title)

                            ax = fig.add_subplot(1,2,2, projection='3d')
                            title = 'Translated Voxel (X:{}, Y:{})'. format(diff[0], diff[1])
                            self.plot_voxel(ax, translated_voxel, title=title)

        if self.show_plt:
            # holding plot
            plt.show(block=False)
            plt.pause(0.1)
            input('[Close]')

        # append original data with augmented data
        data   = list(data)
        labels = list(labels)
        data   = data + augmented_data
        labels = labels + augmented_labels

        return data, labels

    def run(self):
        raw_files = os.listdir(self.pcl_raw_path)
        raw_files = natsort.natsorted(raw_files)
        print('[Pre.] Raw data : {}'.format(raw_files))
        print()

        for f in raw_files:
            file_name         = self.pcl_raw_path + f
            pcl_file          = np.load(file_name)
            self.point_clouds = pcl_file['pcl']
            self.label        = pcl_file['label']
            print('[Pre.] {} : {}'.format(f, self.point_clouds.shape))

            if self.show_plt:
                self.visual_ptk.append(self.plot_point_cloud(f, self.point_clouds)) # <-- plot

            # filter object
            cloud_filtered = self.filter_raw_data(self.point_clouds)
            if self.show_plt:
                self.visual_ptk.append(self.plot_point_cloud(f+'-filtered', cloud_filtered, big_point=True, color=True))

            # voxelization_raw_data
            voxel  = self.voxelization_raw_data(cloud_filtered)

            # acquiring data
            self.data.append(voxel)
            self.labels.append(self.label)

            if self.show_plt:
                fig = plt.figure()
                ax = fig.gca(projection='3d')
                self.plot_voxel(ax, voxel)
                plt.show(block=False)
                plt.pause(1)
            #     input('Enter to close all')
                
            if self.show_plt:
                state = input("\nContinue : ")
                if state == 'q':
                    print('[Pre.] Exit..')
                    self.close_all()
                    break
                else:
                    self.close_all()
            else:
                self.close_all()

        # # self.data & self.labels is list type data
        self.save_data(self.pcl_dataset, self.data, self.labels)
        
        if self.augmentating:
            print('[Pre.] Loaded data')
            self.data, self.labels = self.load_data(self.pcl_dataset)

            data, labels = self.augmentation_data(self.data, self.labels)
            self.save_data(self.aug_dataset, data, labels)

        print('[Pre.] Exit code')

if __name__ == '__main__':
    pre = Preprocess_Data()
    pre.run()