# terrasentia_utils
Python script for TerraSentia rosbag data extraction

To use the package, import `DataExtractor`
```
from terrasentia_utils.rosbag_data_extractor import DataExtractor
```

Create a configuration class data contains the parameters for data extraction
```
class Config:
    def __init__(self):
        self.center_image_topic = "/terrasentia/zed2/zed_node/left/image_rect_color"
        self.center_depth_topic = "/terrasentia/zed2/zed_node/depth/depth_registered"
        self.center_cam_info    = "/terrasentia/zed2/zed_node/left/camera_info"
        self.odom_topic         = "/terrasentia/ekf"
        self.imu_topic          = "/terrasentia/imu"
        self.motion_cmd_topic   = "/terrasentia/motion_command"
        self.mhe_topic          = "/terrasentia/mhe_output"

        # Define sensors transformations
        self.lidar_offset       = [0.17, 0, 0.35]
        self.camera_offset      = [0.17, 0, 0.37]

        self.cam2base = np.array([
            [ 0.0, 0.0, 1.0, self.camera_offset[0]], 
            [-1.0, 0.0, 0.0, self.camera_offset[1]],
            [ 0.0,-1.0, 0.0, self.camera_offset[2]],
            [ 0.0, 0.0, 0.0, 1.0 ]])

        self.get_freq           = 2             # Frequency we get the image frames from rosbag
```

Create a `DataExtractor` object that contains the rosbag data
```
configs = Config()
data_obj = DataExtractor(configs, rosbag_path)
```

Retrieve the rosbag data
```
bag_dict    = data_obj.get_dict()
images_dict = data_obj.get_images()
maps_dict   = data_obj.get_maps()
```
