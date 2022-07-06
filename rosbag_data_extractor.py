#!/usr/bin/env python3
import cv2
import rospy
import numpy as np

import rosbag
#from sensor_msgs.msg import Image, LaserScan
#from grid_map_msgs.msg import GridMap
from cv_bridge import CvBridge

from scipy.spatial.transform import Rotation as R

class DataExtractor:
    def __init__(self, args, bag_file):
        if hasattr(args, 'left_image_topic'):
            left_image_topic = args.left_image_topic
            # Also, check if compressed topics are available
            left_image_compressed_topic = left_image_topic + "/compressed"
        else:
            left_image_topic = None
            left_image_compressed_topic = None

        if hasattr(args, 'left_depth_topic'):
            left_depth_topic = args.left_depth_topic
        else:
            left_depth_topic = None

        if hasattr(args, 'center_image_topic'):
            center_image_topic = args.center_image_topic
            # Also, check if compressed topics are available
            center_image_compressed_topic = center_image_topic + "/compressed"
        else:
            center_image_topic = None
            center_image_compressed_topic = None

        if hasattr(args, 'center_depth_topic'):
            center_depth_topic = args.center_depth_topic
        else:
            center_depth_topic = None
        
        if hasattr(args, 'right_image_topic'):
            right_image_topic = args.right_image_topic
            # Also, check if compressed topics are available
            right_image_compressed_topic = right_image_topic + "/compressed"
        else:
            right_image_topic = None
            right_image_compressed_topic = None

        if hasattr(args, 'right_depth_topic'):
            right_depth_topic = args.right_depth_topic
        else:
            right_depth_topic = None

        if hasattr(args, 'elevation_map_topic'):
            elevation_map_topic = args.elevation_map_topic
        else:
            elevation_map_topic = None

        if hasattr(args, 'vio_pose_topic'):
            vio_pose_topic = args.vio_pose_topic
        else:
            vio_pose_topic = None

        if hasattr(args, 'center_cam_info'):
            center_cam_info = args.center_cam_info
        else:
            center_cam_info = None
        
        lidar_topic = args.lidar_topic 
        odom_topic = args.odom_topic 
        imu_topic = args.imu_topic 
        motion_cmd_topic = args.motion_cmd_topic
        mhe_topic = args.mhe_topic
        
        bridge = CvBridge()

        print("Extract data from", bag_file)
    
        bag = rosbag.Bag(bag_file, "r")

        count = 0
        left_t_img = 0
        left_t_depth = 0
        center_t_img = 0
        center_t_depth = 0
        right_t_img = 0
        right_t_depth = 0

        self.x_path = np.array([])
        self.y_path = np.array([])
        self.z_path = np.array([])
        self.w_hist = np.array([])
        self.x_hist = np.array([])
        self.y_hist = np.array([])
        self.z_hist = np.array([])
        self.odom_stamp = np.array([])
        self.vio_stamp = np.array([])
        self.hor_std = np.array([])
        self.left_cv_img = []
        self.left_cv_depth = []
        self.center_cv_img = []
        self.center_cv_depth = []
        self.center_intrinsics = []
        self.center_resolution = []
        self.right_cv_img = []
        self.right_cv_depth = []
        self.data_elevation_map = []
        self.lidar = []
        self.imu = []
        self.cmd_vel = []
        self.odom_position = []
        self.odom_orientation = []
        self.vio_position = []
        self.vio_orientation = []
        self.traversability = []

        # Timestamp arrays for synchronization
        self.t_traversability = np.array([])
        self.t_lidar = np.array([])
        self.t_imu = np.array([])
        self.t_cmd_vel = np.array([])
        self.left_t_img = np.array([])
        self.left_t_depth = np.array([])
        self.center_t_img = np.array([])
        self.center_t_depth = np.array([])
        self.right_t_img = np.array([])
        self.right_t_depth = np.array([])
        self.t_elevation_map = np.array([])        
        
        # Split bags name to use as frame name
        bag_name = bag_file.split('/')
        self.bag_name = bag_name[-1].split('.')[0]

        topics_list = [
            left_image_topic,
            left_image_compressed_topic,
            left_depth_topic,
            center_image_topic,
            center_image_compressed_topic,
            center_depth_topic,
            center_image_compressed_topic,
            center_cam_info,
            right_image_topic,
            right_image_compressed_topic,
            right_depth_topic,
            lidar_topic,
            odom_topic,
            vio_pose_topic,
            imu_topic,
            motion_cmd_topic,
            mhe_topic,
            elevation_map_topic]

        init_time = bag.get_start_time()
        start_time = rospy.Time(init_time + args.start_after)

        for topic, msg, t in bag.read_messages(topics=topics_list, start_time=start_time):
            #print('topic:', topic)
            t = t.to_sec()
            
            # Left image
            if topic == left_image_topic: # and (count % int(30/args.get_freq)) < 0.001:
                cv_img = bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
                left_cv_img = cv2.resize(cv_img, args.image_size, interpolation = cv2.INTER_CUBIC)
                left_cv_img = cv2.rotate(left_cv_img, cv2.ROTATE_180)
                left_t_img = t
                
            elif topic == left_image_compressed_topic: # and (count % int(30/args.get_freq)) < 0.001:
                cv_img = bridge.compressed_imgmsg_to_cv2(msg, desired_encoding='bgr8')
                left_cv_img = cv2.resize(cv_img, args.image_size, interpolation = cv2.INTER_CUBIC)
                left_cv_img = cv2.rotate(left_cv_img, cv2.ROTATE_180)
                left_t_img = t
                
            elif topic == left_depth_topic:
                cv_depth = bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
                left_cv_depth = cv2.resize(cv_depth, args.image_size, interpolation = cv2.INTER_NEAREST)
                left_cv_depth = cv2.rotate(left_cv_depth, cv2.ROTATE_180)
                left_t_depth = t

            # Center image
            elif topic == center_image_topic:
                np_arr = np.frombuffer(msg.data, np.uint8)
                center_cv_img = np_arr.reshape(msg.height, msg.width)
                if hasattr(args, 'image_size'):
                    center_cv_img = cv2.resize(center_cv_img, args.image_size, interpolation = cv2.INTER_CUBIC)
                center_t_img = t
                
            elif topic == center_image_compressed_topic:
                np_arr = np.frombuffer(msg.data, np.uint8)
                center_cv_img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
                if hasattr(args, 'image_size'):
                    center_cv_img = cv2.resize(center_cv_img, args.image_size, interpolation = cv2.INTER_CUBIC)
                center_t_img = t
                
            elif topic == center_depth_topic:
                np_arr = np.frombuffer(msg.data, np.float32)
                center_cv_depth = np_arr.reshape(msg.height, msg.width)
                if hasattr(args, 'image_size'):
                    center_cv_depth = cv2.resize(center_cv_depth, args.image_size, interpolation = cv2.INTER_NEAREST)
                center_t_depth = t

            elif topic == center_cam_info:
                intrinsics = np.array(msg.K).reshape(3,3)
                self.center_intrinsics.append(intrinsics)
                self.center_resolution.append([msg.width, msg.height])

            # Right image
            elif topic == right_image_topic: # and (count % int(30/args.get_freq)) < 0.001:
                cv_img = bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
                right_cv_img = cv2.resize(cv_img, args.image_size, interpolation = cv2.INTER_CUBIC)
                right_t_img = t
            elif topic == right_image_compressed_topic: # and (count % int(30/args.get_freq)) < 0.001:
                cv_img = bridge.compressed_imgmsg_to_cv2(msg, desired_encoding='bgr8')
                right_cv_img = cv2.resize(cv_img, args.image_size, interpolation = cv2.INTER_CUBIC)
                right_t_img = t
                
            elif topic == right_depth_topic:
                cv_depth = bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
                right_cv_depth = cv2.resize(cv_depth, args.image_size, interpolation = cv2.INTER_NEAREST)
                right_t_depth = t

            elif topic == elevation_map_topic:
                #print('grid layers:', msg.layers == 'elevation')
                elevation_idx = 0
                grid_data = np.asarray(msg.data[elevation_idx].data)
                grid_dim = (msg.data[elevation_idx].layout.dim[0].size, msg.data[elevation_idx].layout.dim[1].size)
                grid_data = grid_data.reshape(grid_dim, order='F')
                grid_img = np.roll(grid_data, (-msg.inner_start_index, -msg.outer_start_index), axis=(1, 0))
                self.data_elevation_map.append(grid_img)
                self.t_elevation_map = np.append(self.t_elevation_map, t)

            elif topic == lidar_topic:
                self.t_lidar = np.append(self.t_lidar, t)
                self.lidar.append(np.asarray(msg.ranges))
                        
            elif topic == odom_topic:
                px = msg.pose.pose.position.x
                py = msg.pose.pose.position.y
                pz = msg.pose.pose.position.z
                
                qw = msg.pose.pose.orientation.w
                qx = msg.pose.pose.orientation.x
                qy = msg.pose.pose.orientation.y
                qz = msg.pose.pose.orientation.z

                self.odom_position.append([px, py, pz])
                self.odom_orientation.append([qw, qx, qy, qz])
                self.odom_stamp = np.append(self.odom_stamp, t)
            
            elif topic == vio_pose_topic:
                px = msg.pose.pose.position.x
                py = msg.pose.pose.position.y
                pz = msg.pose.pose.position.z
                
                qw = msg.pose.pose.orientation.w
                qx = msg.pose.pose.orientation.x
                qy = msg.pose.pose.orientation.y
                qz = msg.pose.pose.orientation.z

                self.vio_position.append([px, py, pz])
                self.vio_orientation.append([qw, qx, qy, qz])
                self.vio_stamp = np.append(self.vio_stamp, t)

            elif topic == imu_topic:
                ax = msg.angular_velocity.x
                ay = msg.angular_velocity.y
                az = msg.angular_velocity.z
                lx = msg.linear_acceleration.x
                ly = msg.linear_acceleration.y
                lz = msg.linear_acceleration.z
                self.t_imu = np.append(self.t_imu, t)
                self.imu.append([ax,ay,az,lx,ly,lz])

            elif topic == motion_cmd_topic:
                vx = msg.linear.x
                wz = msg.angular.z
                self.t_cmd_vel = np.append(self.t_cmd_vel, t)
                self.cmd_vel.append([vx, -wz])

            elif topic == mhe_topic:
                self.t_traversability = np.append(self.t_traversability, t)
                self.traversability.append([msg.mu, msg.nu])

            # Left image
            if (topic==left_image_topic or topic==left_image_compressed_topic or topic==left_depth_topic) and abs(left_t_img-left_t_depth) < 0.02:
                if len(self.left_t_img) == 0:
                    self.left_cv_img.append(left_cv_img)
                    self.left_cv_depth.append(left_cv_depth)
                    self.left_t_img = np.append(self.left_t_img, t)
                    self.left_t_depth = np.append(self.left_t_depth, t)
                elif (left_t_img-self.left_t_img[0])/len(self.left_t_img) >= (1.0/args.get_freq):
                    self.left_cv_img.append(left_cv_img)
                    self.left_cv_depth.append(left_cv_depth)
                    self.left_t_img = np.append(self.left_t_img, t)
                    self.left_t_depth = np.append(self.left_t_depth, t)
            
            # Center image
            if (topic==center_image_topic or topic==center_image_compressed_topic or topic==center_depth_topic) and abs(center_t_img-center_t_depth) < 0.02:
                if len(self.center_t_img) == 0:
                    self.center_cv_img.append(center_cv_img)
                    self.center_cv_depth.append(center_cv_depth)
                    self.center_t_img = np.append(self.center_t_img, t)
                    self.center_t_depth = np.append(self.center_t_depth, t)
                elif (center_t_img-self.center_t_img[0])/len(self.center_t_img) >= (1.0/args.get_freq):
                    self.center_cv_img.append(center_cv_img)
                    self.center_cv_depth.append(center_cv_depth)
                    self.center_t_img = np.append(self.center_t_img, t)
                    self.center_t_depth = np.append(self.center_t_depth, t)

            # Right image
            if (topic==right_image_topic or topic==right_image_compressed_topic or topic==right_depth_topic) and abs(right_t_img-right_t_depth) < 0.02:
                if len(self.right_t_img) == 0:
                    self.right_cv_img.append(right_cv_img)
                    self.right_cv_depth.append(right_cv_depth)
                    self.right_t_img = np.append(self.right_t_img, t)
                    self.right_t_depth = np.append(self.right_t_depth, t)
                elif (right_t_img-self.right_t_img[0])/len(self.right_t_img) >= (1.0/args.get_freq):
                    self.right_cv_img.append(right_cv_img)
                    self.right_cv_depth.append(right_cv_depth)
                    self.right_t_img = np.append(self.right_t_img, t)
                    self.right_t_depth = np.append(self.right_t_depth, t)
            
            if topic == center_image_topic or topic==center_image_compressed_topic:
                count += 1
                if count%1000 == 0:
                    print('{} images read from rosbag'.format(count))

        print('Done reading rosbag')

        # Close rosbag file
        bag.close()

    def get_dict(self):

        bag_dict = {'bag_name': self.bag_name,
                    'cmd_vel': {'stamp': self.t_cmd_vel, 'data': self.cmd_vel},
                    'odom': {'stamp': self.odom_stamp,
                             'position': self.odom_position,
                             'orientation':self.odom_orientation},
                    'vio_pose': {'stamp': self.vio_stamp,
                                 'position': self.vio_position,
                                 'orientation':self.vio_orientation},
                    'traversability': {'stamp': self.t_traversability, 'data': self.traversability}}

        return bag_dict

    def get_images(self):

        images_dict = {'left': {'rgb': {'stamp': self.left_t_img, 'data': self.left_cv_img},
                                'depth': {'stamp': self.left_t_depth, 'data': self.left_cv_depth}},
                       'center': {'rgb': {'stamp': self.center_t_img, 'data': self.center_cv_img},
                                  'depth': {'stamp': self.center_t_depth, 'data': self.center_cv_depth},
                                  'intrinsics': self.center_intrinsics,
                                  'resolution': self.resolution},
                       'right': {'rgb': {'stamp': self.right_t_img, 'data': self.right_cv_img},
                                 'depth': {'stamp': self.right_t_depth, 'data': self.right_cv_depth}}}
                        
        return images_dict

    def get_maps(self):

        maps_dict = {'elevation_map': {'stamp': self.t_elevation_map, 'data': self.data_elevation_map}}

        return maps_dict