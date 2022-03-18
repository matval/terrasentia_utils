import os
import bisect
import random

import cv2
import numpy as np
import pandas as pd

import torch.utils.data as DataLoader

class TraversabilityDataset(DataLoader.Dataset):
    def __init__(self, params, transform):
        super().__init__()
        self.transform  = transform
        self.preproc    = params.preproc
        self.input_size = params.input_size
        self.output_size= params.output_size
        self.path       = params.data_path
        self.verbose    = params.verbose
        self.use_depth  = params.use_depth
        self.N          = params.horizon
        self.dt         = params.sample_time

        # Read lines in csv file
        bags_list = pd.read_csv(params.csv_path)

        self.rosbags = []
        self.bag_sizes = [0]
        for bag in bags_list.values:
            data = pd.read_csv(os.path.join(self.path, bag[0]))
            formatted_data =  self.read_data(data)
            self.rosbags.append(formatted_data)
            self.bag_sizes.append(self.bag_sizes[-1] + len(data))

        self.weights, self.bins = self.prepare_weights()

        print('weights:', self.weights)
        print('bins:', self.bins)

    def __getitem__(self,index):
        bag_idx = bisect.bisect_right(self.bag_sizes, index)-1
        rosbag_df = self.rosbags[bag_idx]
        seq_index = index - self.bag_sizes[bag_idx]
        data_df = rosbag_df.iloc[seq_index]

        hor_flip_data = random.random() < 0.5

        rgb_name, depth_name, linear_vel, angular_vel, \
            position_x, position_y, traversability = data_df.iloc
            
        # Prepare RGB image        
        rgb_img = cv2.imread(os.path.join(self.path, rgb_name),-1)
        rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB)

        if (self.input_size[0] != rgb_img.shape[1]) and (self.input_size[1] != rgb_img.shape[0]):
            rgb_img = cv2.resize(rgb_img, self.input_size, interpolation = cv2.INTER_AREA)

        if self.use_depth:
            # Prepare Depth image 
            depth_img = cv2.imread(os.path.join(self.path, depth_name),-1)
            depth_img = cv2.resize(depth_img, self.input_size, interpolation = cv2.INTER_NEAREST)
            # Convert depth to meters
            depth_img = np.uint16(depth_img)
            depth_img = depth_img*10**-3
        else:
            depth_img = np.zeros((self.input_size[1], self.input_size[0]))

        if self.preproc and hor_flip_data:
            rgb_img, depth_img, angular_vel, position_y = self.horizontal_flip(rgb_img, depth_img, angular_vel, position_y)

        if self.transform is not None:
            rgb_img = self.transform(rgb_img)
        else:
            rgb_img = np.uint8(rgb_img)
            rgb_img = rgb_img/255.0
            rgb_img = (rgb_img-0.5)/0.5
            rgb_img = np.transpose(rgb_img, (2, 0, 1))

        if self.use_depth:
            # Normalize depth image
            depth_img = (depth_img-5.0)/5.0
            depth_img = np.expand_dims(depth_img, axis=2)
            depth_img = np.transpose(depth_img, (2, 0, 1))

        '''
        # Add random noise to depth image
        if self.augment:
            mean = 0.0
            std = 0.005
            gauss = np.random.normal(mean,std,newsize)
            depth_img += gauss.reshape(newsize[1],newsize[0])
        '''

        weight_idxs = np.digitize(traversability, self.bins[:-1]) - 1
        weight = self.weights[weight_idxs]
        
        seq_tensor = [rgb_img, depth_img, linear_vel, angular_vel, position_x, position_y, traversability, weight]
        return seq_tensor

    def __len__(self):
        total_length = 0
        for data in self.rosbags:
            total_length += len(data.index)

        return total_length

    def read_data(self, data):
        color_fname_list = []
        depth_fname_list = []
        lin_ctrl_list = []
        ang_ctrl_list = []
        pos_x_list = []
        pos_y_list = []
        traversability_list = []

        map_float = lambda x: np.array(list(map(float, x)))

        for color_fname, depth_fname, linear_vel, angular_vel, pos_x, pos_y, traversability, timestamps in data.iloc:
            # Read timestamps for interpolation
            timestamps_seq = timestamps[1:-1].split()
            timestamps_seq = map_float(timestamps_seq)
            init_timestamp_idx = int(len(timestamps_seq)/2)
            timestamps_seq -= timestamps_seq[init_timestamp_idx]

            # Read linear control action
            lin_ctrl_seq = linear_vel[1:-1].split()
            lin_ctrl_seq = map_float(lin_ctrl_seq)
            lin_ctrl_seq = np.interp(np.arange(0, 2*self.N) * self.dt - self.N*self.dt, timestamps_seq, lin_ctrl_seq)

            # Read angular control action
            ang_ctrl_seq = angular_vel[1:-1].split()
            ang_ctrl_seq = map_float(ang_ctrl_seq)
            ang_ctrl_seq = np.interp(np.arange(0, 2*self.N) * self.dt - self.N*self.dt, timestamps_seq, ang_ctrl_seq)

            # Read x position sequence
            pos_x_seq = pos_x[1:-1].split()
            pos_x_seq = map_float(pos_x_seq)
            pos_x_seq = np.interp(np.arange(0, 2*self.N) * self.dt - self.N*self.dt, timestamps_seq, pos_x_seq)

            # Read y position sequence
            pos_y_seq = pos_y[1:-1].split()
            pos_y_seq = map_float(pos_y_seq)
            pos_y_seq = np.interp(np.arange(0, 2*self.N) * self.dt - self.N*self.dt, timestamps_seq, pos_y_seq)

            # Read traversability values
            traversability_seq = traversability[1:-1].split()
            traversability_seq = map_float(traversability_seq)
            traversability_seq = np.interp(np.arange(0, 2*self.N) * self.dt - self.N*self.dt, timestamps_seq, traversability_seq)

            # Append values to lists
            color_fname_list.append(color_fname)
            depth_fname_list.append(depth_fname)
            lin_ctrl_list.append(lin_ctrl_seq)
            ang_ctrl_list.append(ang_ctrl_seq)
            pos_x_list.append(pos_x_seq)
            pos_y_list.append(pos_y_seq)
            traversability_list.append(traversability_seq)

        if self.verbose:
            print("All data have been loaded from bag! Total dataset size: {:d}".format(len(color_fname_list)))

        dataframe = pd.DataFrame({
            'rgb_image': color_fname_list,
            'depth_image': depth_fname_list,
            'lin_vel': lin_ctrl_list,
            'ang_vel': ang_ctrl_list,
            'pos_x': pos_x_list,
            'pos_y': pos_y_list,
            'traversability': traversability_list})

        return dataframe

    def horizontal_flip(self, rgb_img, depth_img, angular_vel, position_y):
        # Augment data with a random horizontal image flip
        rgb_img_lr = np.fliplr(rgb_img).copy()
        depth_img_lr = np.fliplr(depth_img).copy()
        angular_vel_lr = -angular_vel.copy()
        position_y_lr = -position_y.copy()
        return rgb_img_lr, depth_img_lr, angular_vel_lr, position_y_lr

    def prepare_weights(self):
        bin_width = 0.2

        label = []
        for data in self.rosbags:
            traversability = data.loc[:,'traversability']
            label.extend(traversability)

        print('len(label):', len(label))

        # Flatten the entire list of numpy arrays
        flat_label = np.concatenate(label)

        print('flat_label.shape:', flat_label.shape)

        # Draw the plot
        values, bins = np.histogram(flat_label, bins=int(1/bin_width), range=(0,1), density=True)

        return 0.1/values, bins