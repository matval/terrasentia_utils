#!/usr/bin/env python3
import os
import bisect
import numpy as np
from itertools import compress
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from salem import GoogleVisibleMap, Map
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar

# custom packages
from utils.rosbag_data_extractor import DataExtractor

def plot_map(configs, experiments):
    # Download map
    g = GoogleVisibleMap(configs.map_lon, configs.map_lat, size_x=640, size_y=320, scale=2, maptype='satellite')  # try out also: 'terrain'

    # the google static image is a standard rgb image
    ggl_img = g.get_vardata()

    ggl_img = ggl_img * 0.3 + (1 - 0.3)
    # ggl_img[ggl_img > 1] = 255

    sm = Map(g.grid, factor=1, countries=False)
    # sm.set_scale_bar((0.5, configs.scalebar_height), length=50, linewidth=10)
    # plt.rcParams.update({'font.size': 25})
    sm.set_rgb(ggl_img)
    f, ax1 = plt.subplots(1, 1, figsize=(20, 12))
    sm.visualize(ax = ax1)

    interventions_lat = []
    interventions_lon = []

    recoveries_lat = []
    recoveries_lon = []

    # for bags_list in collections_list:
    for count, bags_list in enumerate(experiments):
        print('------------------------------')

        exp_lat_points = []
        exp_lon_points = []
        for rosbag_path in bags_list:
            path = os.path.join(configs.data_dir, rosbag_path)
            data_obj = DataExtractor(configs, path)
            bag_dict = data_obj.get_dict()
            # plot_map(configs, bag_dict)

            if configs.plot_recovery:
                idx_collision = [bisect.bisect_left(bag_dict['collision']['stamp'], i) if i<bag_dict['collision']['stamp'][-1] else len(bag_dict['collision']['stamp'])-1 for i in bag_dict['gps']['stamp']]

                is_collision = np.asarray(bag_dict['collision']['data'])[idx_collision]
                is_collision2 = np.roll(is_collision, 1)
                is_collision2[0] = is_collision[0]

                is_collision = (is_collision==True) * (is_collision2==False)

            lat_points = [x[0] for x in bag_dict['gps']['lat_lon']]
            lon_points = [x[1] for x in bag_dict['gps']['lat_lon']]

            # Concat path points
            exp_lat_points.extend(lat_points)
            exp_lon_points.extend(lon_points)
            # Concat interventions
            interventions_lat.append(lat_points[-1])
            interventions_lon.append(lon_points[-1])

            if configs.plot_recovery:
                recoveries_lat.append(list(compress(lat_points, is_collision)))
                recoveries_lon.append(list(compress(lon_points, is_collision)))

        # Remove last point, because that is not an intervention
        interventions_lat.pop()
        interventions_lon.pop()

        print('Experiment {} had {} interventions'.format(count+1, len(bags_list)-1))
        if configs.plot_recovery:
            print('Experiment {} had {} recoveries'.format(count+1, len(recoveries_lat[-1])))

        # Plot path points
        x, y = sm.grid.transform(exp_lon_points, exp_lat_points)
        if configs.multiple_runs:
            ax1.plot(x, y, linewidth=3, zorder=count+1, label='Run {}'.format(count+1))
        else:
            ax1.plot(x, y, linewidth=3, zorder=count+1, label='Navigated path')

    ax1.scatter(x[0], y[0], c='green', s=150, zorder=100, label='Start')
    ax1.scatter(x[-1], y[-1], c='red', s=150, zorder=100, label='End')

    # goals_lat = bag_dict['goals']['data'][0]
    # goals_lon = bag_dict['goals']['data'][1]
    # x, y = sm.grid.transform(goals_lon, goals_lat)
    # ax1.scatter(x, y, c='orange', s=100, zorder=count+2, label='Waypoints')

    # Remove last point, because that is not an intervention
    # interventions_lat.pop()
    # interventions_lon.pop()

    if configs.plot_recovery:
        recoveries_lat = sum(recoveries_lat, [])
        recoveries_lon = sum(recoveries_lon, [])
        x, y = sm.grid.transform(recoveries_lon, recoveries_lat)
        ax1.scatter(x, y, c='black', marker='*', s=150, zorder=count+3, label='Recoveries')

    if configs.plot_intervention:
        x, y = sm.grid.transform(interventions_lon, interventions_lat)
        ax1.scatter(x, y, c='red', marker='*', s=150, zorder=count+4, label='Interventions')

    x, _ = sm.grid.transform([configs.map_lon[0]-0.0001/0.787, configs.map_lon[0]+0.0001/0.787], [configs.map_lat[0], configs.map_lat[0]])
    fontprops = fm.FontProperties(size=20)
    scalebar = AnchoredSizeBar(ax1.transData,
                            x[1]-x[0], '20 m', 'lower left',
                            sep=5,
                            pad=0.8,
                            #    color='white',
                            frameon=False,
                            size_vertical=5,
                            fontproperties=fontprops)
    ax1.add_artist(scalebar)
    # ax1.legend(loc="lower right",prop={'size': 30})
    ax1.axis('on')
    ax1.get_xaxis().set_ticks([])
    ax1.get_yaxis().set_ticks([])

    box = ax1.get_position()
    ax1.set_position([box.x0, box.y0 , box.width, box.height*0.9])
    lgnd = ax1.legend(loc='lower center', bbox_to_anchor=(0.5, 1.), fontsize=20, ncol=configs.legend_columns)

    plt.show()