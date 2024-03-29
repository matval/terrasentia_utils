import copy
import time
import numpy as np
from numpy.core.function_base import linspace
import open3d as o3d
import matplotlib.pyplot as plt

from scipy.ndimage import gaussian_filter
from scipy.spatial.transform import Rotation as R

def depth_to_pcloud(depth, voxel_size=0.02, min_bound=None, max_bound=None):
    Min = np.array([[610.2332763671875, 0.0, 419.8548583984375],
                    [0.0, 610.2238159179688, 249.6531219482422],
                    [0.0, 0.0, 1.0]])

    #rgbd = create_rgbd_image_from_color_and_depth(color, depth, convert_rgb_to_intensity = False)
    depth = o3d.geometry.Image(depth.astype('uint16'))
    cam = o3d.camera.PinholeCameraIntrinsic()
    cam.intrinsic_matrix = Min
    depth_pcd = o3d.geometry.PointCloud.create_from_depth_image(depth, cam)

    depth_pcd.transform([[0,0,1,0],[-1,0,0,0],[0,-1,0,0],[0,0,0,1]])

    if (min_bound is not None) and (max_bound is not None):
        # Crop Point Cloud
        bbox = o3d.geometry.AxisAlignedBoundingBox(min_bound=min_bound,max_bound=max_bound)
        depth_pcd = depth_pcd.crop(bbox)

    downpcd = depth_pcd.voxel_down_sample(voxel_size=voxel_size)
    
    return downpcd

def multi_depth_to_pcloud(depth_left, depth_center, depth_right, voxel_size=0.02, min_bound=None, max_bound=None):
    Min = np.array([[305.7901306152344, 0.0, 215.58926391601562],
                    [0.0, 305.8052062988281, 117.26675415039062],
                    [0.0, 0.0, 1.0]])

    cam = o3d.camera.PinholeCameraIntrinsic()
    cam.intrinsic_matrix = Min
    T = [[0,0,1,0],[-1,0,0,0],[0,-1,0,0],[0,0,0,1]]

    # Create pointclouds for the 3 cameras
    depth_left = o3d.geometry.Image(depth_left.astype('uint16'))
    pcl_left = o3d.geometry.PointCloud.create_from_depth_image(depth_left, cam)

    depth_center = o3d.geometry.Image(depth_center.astype('uint16'))
    pcl_center = o3d.geometry.PointCloud.create_from_depth_image(depth_center, cam)

    depth_right = o3d.geometry.Image(depth_right.astype('uint16'))
    pcl_right = o3d.geometry.PointCloud.create_from_depth_image(depth_right, cam)

    '''
    # Downsample the pointclouds
    pcl_left = pcl_left.voxel_down_sample(voxel_size=0.02)
    pcl_center = pcl_center.voxel_down_sample(voxel_size=0.02)
    pcl_right = pcl_right.voxel_down_sample(voxel_size=0.02)
    '''

    # Transform the pointclouds
    pcl_left = pcl_left.transform(T)
    pcl_center = pcl_center.transform(T)
    pcl_right = pcl_right.transform(T)

    T_left = np.eye(4)
    R1 = R.from_euler('y', 5, degrees=True)
    R2 = R.from_euler('z', 70, degrees=True)
    T_left[:3, :3] = (R2*R1).as_matrix()
    #T_left[:3, :3] = R.from_euler('xyz', [0, 5, 60], degrees=True).as_matrix()
    T_left[0,3] = -0.04512
    T_left[1,3] = 0.02924
    pcl_left = pcl_left.transform(T_left)

    T_center = np.eye(4)
    T_center[:3, :3] = R.from_euler('xyz', [0, 5, 0], degrees=True).as_matrix()
    pcl_center = pcl_center.transform(T_center)

    T_right = np.eye(4)
    R1 = R.from_euler('y', 5, degrees=True)
    R2 = R.from_euler('z', -70, degrees=True)
    T_right[:3, :3] = (R2*R1).as_matrix()
    #T_right[:3, :3] = R.from_euler('xyz', [0, 5, -60], degrees=True).as_matrix()
    T_right[0,3] = -0.04512
    T_right[1,3] = -0.09424
    pcl_right = pcl_right.transform(T_right)

    # Concatenate pointclouds
    points = np.concatenate((pcl_left.points, pcl_center.points, pcl_right.points), axis=0)

    pcl = o3d.geometry.PointCloud()
    pcl.points = o3d.utility.Vector3dVector(points)    #pcl_left.points

    if (min_bound is not None) and (max_bound is not None):
        # Crop Point Cloud
        bbox = o3d.geometry.AxisAlignedBoundingBox(min_bound=min_bound,max_bound=max_bound)
        depth_pcd = pcl.crop(bbox)

    downpcd = depth_pcd.voxel_down_sample(voxel_size=voxel_size)
    
    return downpcd

def draw_registration_result(source, target, transformation):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    #source_temp.paint_uniform_color([1, 0.706, 0])
    #target_temp.paint_uniform_color([0, 0.651, 0.929])
    source_temp.transform(transformation)
    
    plt.rcParams['figure.figsize'] = [12, 8]
    plt.rcParams['figure.dpi'] = 100
    
    plt.figure()
    ax = plt.axes(projection='3d')
    np_source = np.asarray(source_temp.points).T
    np_target = np.asarray(target_temp.points).T
    print('np_source.shape:', np_source.shape)
    #ax.plot_surface(np_source[0], np_source[1], np_source[2], rstride=1, cstride=1, cmap='viridis', edgecolor='none')
    
    ax.scatter(np_source[0], np_source[1], np_source[2], s=1, color=[1, 0.706, 0])
    ax.scatter(np_target[0], np_target[1], np_target[2], s=1, c=np_target[2])
    #ax.axis('equal')
    ax.set_box_aspect(aspect = (2,1,1))
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    #ax.set_xlim(0, 4)
    ax.set_ylim(-1, 1)
    ax.set_zlim(-1, 2)
    ax.view_init(5, 190)

def convert_to_pixels(X, Y, Z):
    # Pinhole camera
    #f = 1.93*10**-3
    x = X/Z
    y = Y/Z
    z = Z/Z
    xc = np.array([[x, y, z]]).T

    Min = np.array([[610.2332763671875, 0.0, 419.8548583984375],
                    [0.0, 610.2238159179688, 249.6531219482422],
                    [0.0, 0.0, 1.0]])
    #p = 1/f*np.dot(Min, xc)
    p = np.dot(Min, xc)
    #print('p[0]:', p[0,:])
    u = p[0,:].astype(int)
    v = p[1,:].astype(int)
    return u, v

def depth_to_gridmap(depth, resolution, grid_size, center, g_transform, min_bound, max_bound):    
    downpcd = depth_to_pcloud(depth, voxel_size=0.01, min_bound=min_bound, max_bound=max_bound)

    # Rotate pointcloud to align with gravity
    #downpcd.transform(g_transform)
    
    # Rotate pointcloud to match he grid image
    T = R.from_euler('z', 180, degrees=True)
    transform = np.eye(4)
    transform[0:3,0:3] = T.as_matrix()
    downpcd.transform(transform)
    x = np.asarray(downpcd.points).T

    # Convert meters to grid coordinates
    x = np.ceil(x/resolution).astype(int)
    x[0] += center[0]
    x[1] += center[1]

    x = x[:, (x[0] >= 0) & (x[1] >= 0) & (x[0] < grid_size[0]) & (x[1] < grid_size[1])]

    if x.size == 0:
        new_grid = np.zeros(grid_size)
        return new_grid, downpcd

    grid = np.zeros((max(x[0])+1, max(x[1])+1))
    for i in range(x.shape[1]):
        grid[x[0,i], x[1,i]] += 1

    #print("grid:", grid_size)
    new_grid = np.zeros(grid_size)
    new_grid[:grid.shape[0],:grid.shape[1]] = grid
    return new_grid, downpcd

def multi_depth_to_gridmap(depth_left, depth_center, depth_right, resolution, grid_size,
    center, g_transform, min_bound, max_bound):
    
    downpcd = multi_depth_to_pcloud(depth_left, depth_center, depth_right,
        voxel_size=0.01, min_bound=min_bound, max_bound=max_bound)

    # Rotate pointcloud to align with gravity
    downpcd.transform(g_transform)
    
    # Rotate pointcloud to match he grid image
    T = R.from_euler('z', 180, degrees=True)
    transform = np.eye(4)
    transform[0:3,0:3] = T.as_matrix()
    downpcd.transform(transform)
    x = np.asarray(downpcd.points).T

    # Convert meters to grid coordinates
    x = np.ceil(x/resolution).astype(int)
    x[0] += center[0]
    x[1] += center[1]

    x = x[:, (x[0] >= 0) & (x[1] >= 0) & (x[0] < grid_size[0]) & (x[1] < grid_size[1])]

    if x.size == 0:
        new_grid = np.zeros(grid_size)
        return new_grid, downpcd

    grid = np.zeros((max(x[0])+1, max(x[1])+1))
    for i in range(x.shape[1]):
        grid[x[0,i], x[1,i]] += 1

    #print("grid:", grid_size)
    new_grid = np.zeros(grid_size)
    new_grid[:grid.shape[0],:grid.shape[1]] = grid
    return new_grid, downpcd

def path_to_gridmap(path, traversability, resolution, map_size, map_center):
    path[0] = -path[0]/resolution + map_center[1]
    path[1] = -path[1]/resolution + map_center[0] 

    # Filter out path outside the map   
    path_ids = ((path[0] >= 0) &
                (path[1] >= 0) &
                (path[1] < map_size[0]) &
                (path[0] < map_size[1]))
    
    path = path[:, path_ids].astype('int')
    traversability = traversability[path_ids]

    # Verify if path is empty
    if path.shape[1] == 0:
        mu_img = np.full(map_size, np.nan)
        return mu_img

    mu_img = np.zeros(map_size)
    sum_map = np.zeros(map_size)

    # for i in range(path.shape[1]):
    #     mu_img[path[1,i], path[0,i]] += traversability[i]
    #     sum_map[path[1,i], path[0,i]] += 1

    mu_img[path[0], path[1]] += traversability
    sum_map[path[0], path[1]] += 1

    mask = 1*(sum_map == 0)
    mu_img /= (sum_map + mask)
    #mu_img[mask==1] = 0
    path_img = (1-mask)
    
    return mu_img, path_img

def lidar_to_gridmap(lidar, resolution, grid_size, center):
    grid = np.zeros(grid_size)
    angles = linspace(-2.35619449, 2.35619449, len(lidar))

    for idx in range(len(lidar)):
        x = np.round(center[0] - lidar[idx]*np.cos(angles[idx])/resolution - 0.5).astype(int)
        y = np.round(center[1] - lidar[idx]*np.sin(angles[idx])/resolution - 0.5).astype(int)
        if x > 0 and x < grid_size[0] and y > 0 and y < grid_size[1]:
            grid[x,y] = 1

    #print("path grid:", grid.shape)
    return grid

def project_to_image(path, traversability, image_size, patch_size, K):
    # First, we filter out the points with negative z
    traversability = traversability[path[2]>0]
    path = path[:, path[2]>0]

    # Then we project the path onto the image
    proj_path = K @ path
    proj_path = (proj_path/proj_path[2,:]).astype(int)

    path_ids = (proj_path[0] >= 0) & (proj_path[1] >= 0) & \
               (proj_path[1] < image_size[0]) & (proj_path[0] < image_size[1])
    
    proj_path = proj_path[:, path_ids].astype('int')
    traversability = traversability[path_ids]

    # Verify if proj_path is empty
    if proj_path.shape[1] == 0:
        mu_img = np.zeros(image_size)
        nu_img = np.zeros(image_size)
        path_img = np.zeros(image_size)

        return mu_img, nu_img, path_img

    mu_img = np.zeros(image_size)
    nu_img = np.zeros(image_size)
    path_img = np.zeros(image_size)
    sum_map = np.zeros(image_size)

    for i in range(proj_path.shape[1]):
        lower_idx_u = proj_path[0,i]-int(patch_size/2)
        upper_idx_u = proj_path[0,i]+int(patch_size/2)
        lower_idx_v = proj_path[1,i]-int(patch_size/2)
        upper_idx_v = proj_path[1,i]+int(patch_size/2)

        mu_img[lower_idx_v:upper_idx_v, lower_idx_u:upper_idx_u] += traversability[i,0]
        nu_img[lower_idx_v:upper_idx_v, lower_idx_u:upper_idx_u] += traversability[i,1]
        path_img[lower_idx_v:upper_idx_v, lower_idx_u:upper_idx_u] = 1
        sum_map[lower_idx_v:upper_idx_v, lower_idx_u:upper_idx_u] += 1

    mask = 1*(sum_map == 0)
    mu_img /= (sum_map + mask)
    nu_img /= (sum_map + mask)
    
    return mu_img, nu_img, path_img

def dilate_robot_trace(path, orientation, traversability, resolution, robot_size):
    x = np.arange(-robot_size[0]/2, robot_size[0]/2, resolution)
    y = np.arange(-robot_size[1]/2, robot_size[1]/2, resolution)
    xx, yy = np.meshgrid(x, y)
    xx = xx.flatten()
    yy = yy.flatten()
    robots_step = np.vstack((xx, yy, np.ones(len(xx))))

    new_path = []#np.array([[],[],[]]).T
    new_traversability = []#np.array([[],[]]).T

    # convert to numpy array
    np_orientation = np.asarray(orientation)

    # Swap first and last columns, because Rotation lib uses w last
    np_orientation = np_orientation[:,[1,2,3,0]]

    T = R.from_quat(np_orientation)

    for i in range(len(path)):
        curr_trace = robots_step
        curr_trace[2] = 0 #path[i][2]*robots_step[2]

        # Rotate robot's step by its current rotation
        curr_trace = T[i].as_matrix() @ curr_trace
        # And translate to its current location
        curr_trace = curr_trace + np.array([path[i]]).T
        # Concatenate points to array
        #new_path = np.concatenate((new_path, curr_trace.T))
        new_path.append(curr_trace.T)
        # And dilate traversability measurements accordingly
        curr_traversability = np.array([traversability[i]]).T*np.ones(len(xx))
        #new_traversability = np.concatenate((new_traversability, curr_traversability.T))
        new_traversability.append(curr_traversability.T)

    new_path = np.concatenate(new_path)
    new_traversability = np.concatenate(new_traversability)
        
    return new_path, new_traversability