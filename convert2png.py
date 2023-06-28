import os
from PIL import Image

input_path = r"/home/mateus/catkin_ws/src/traversability_research/dataset/supervised/data_train/ts_2022_07_14_16h50m41s/image"
output_path = r"/home/mateus/catkin_ws/src/traversability_research/dataset/supervised/data_train/ts_2022_07_14_16h50m41s/image_png"

for root, dirs, files in os.walk(input_path, topdown=False):
    for name in files:
        print(os.path.join(root, name))
        if os.path.splitext(os.path.join(root, name))[1].lower() == ".tif":
            outfile = os.path.splitext(os.path.join(output_path, name))[0] + ".png"
            try:
                im = Image.open(os.path.join(root, name))
                print("Generating png for %s" % name)
                im.thumbnail(im.size)
                im.save(outfile, "PNG")
            except Exception:
                print('Exception')