#!/usr/bin/env python3

import os
import pcl
import pptk
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pyntcloud import PyntCloud
from sklearn.cluster import KMeans
from mpl_toolkits.mplot3d import Axes3D

class Display:
    def __init__(self):
        self.data_dir   = 'display'
        self.visual_ptk = []
        self.objects    = [ 'small_suitcase', 'big_suitcase', 'black_chair', 'blue_chair', 'table' ]

    def filter_raw_data(self, data_pcl):
        # 1st filter (knee joint)
        data_x   = data_pcl[:,0]
        x_min = np.round( np.min(data_x), 1) + 0.5
        x_max = np.round( np.max(data_x), 1) - 0.3
        data_pcl = data_pcl[np.where( (data_x >= x_min) & (data_x <= x_max))]

        # 2nd filter (wall)
        data_y   = data_pcl[:,1]
        y_min = np.round( np.min(data_y), 1) + 0.2
        y_max = np.round( np.max(data_y), 1) - 1.2
        data_pcl = data_pcl[np.where( (data_y >= y_min) & (data_y <= y_max))]

        # 3nd filter (ground)
        data_z   = data_pcl[:,2]
        z_min = np.round( np.min(data_z), 1) + 0.08
        data_pcl = data_pcl[np.where( (data_z >= z_min) )]

        cloud = pcl.PointCloud(np.array(data_pcl[:,:3], dtype=np.float32))
        sor   = cloud.make_voxel_grid_filter()
        # sor.set_leaf_size(0.01, 0.01, 0.01)
        sor.set_leaf_size(0.02, 0.02, 0.02)
        cloud_filtered = sor.filter()

        return np.array(cloud_filtered)

    def plot_point_cloud(self, label, pcl_data, big_point=False, color=False, view_dist=3):
        print("[PPTK] {} - length pcl : {}".format(label, pcl_data.shape))
        visual_ptk = pptk.viewer(pcl_data[:,:3])

        if color:
            visual_ptk.attributes(pcl_data[:,-1])
        if big_point:
            # visual_ptk.set(point_size=0.0025)
            visual_ptk.set(point_size=0.1)

        visual_ptk.set(phi=3.5)
        visual_ptk.set(r=view_dist)

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

    def clustering(self, cloud):
        est = KMeans(n_clusters=3)
        fig = plt.figure(1, figsize=(6, 5))
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
        ax.view_init(azim=225)

    def combo(self):
        temp_data = '{}/combo.npz'.format(self.data_dir)
        print(temp_data)

        pcl_file = np.load(temp_data)
        pcl      = pcl_file['pcl']
        v = self.plot_point_cloud('combo', pcl, big_point=False, color=True, view_dist=5)
        v.capture('{}/combo.png'.format(self.data_dir))
        self.visual_ptk.append(v)

        # filter raw data
        pcl = self.filter_raw_data(pcl)
        v   = self.plot_point_cloud('combo-filter', pcl, big_point=False, color=True, view_dist=2.5)
        v.capture('{}/combo-filter.png'.format(self.data_dir))
        self.visual_ptk.append(v)
        self.clustering(pcl)
        plt.show()


        input('Enter to close all')
        self.close_all()

    def run(self):

        for i in self.objects :
            temp_data = '{}/{}.npz'.format(self.data_dir, i)
            print(temp_data)

            pcl_file = np.load(temp_data)
            pcl      = pcl_file['pcl']
            v = self.plot_point_cloud(i, pcl, big_point=False, color=True, view_dist=5)
            v.capture('{}/{}.png'.format(self.data_dir, i))
            self.visual_ptk.append(v)

            # filter raw data
            pcl = self.filter_raw_data(pcl)
            v   = self.plot_point_cloud(i, pcl, big_point=False, color=True, view_dist=2.5)
            v.capture('{}/{}-filter.png'.format(self.data_dir, i))
            self.visual_ptk.append(v)

            # voxelization
            voxel = self.voxelization_raw_data(pcl)
            fig = plt.figure()
            ax  = fig.gca(projection='3d')
            self.plot_voxel(ax, voxel)

        plt.show(block =False)
        input('Enter to close all')
        self.close_all()

if __name__ == '__main__':
    display = Display()
    # display.run()
    display.combo()