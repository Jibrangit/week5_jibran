"""week5_jibran controller."""

from controller import Robot, Supervisor, Motor, PositionSensor
import numpy as np
from enum import Enum
from copy import deepcopy
from typing import List, Tuple
from scipy import signal
import matplotlib.pyplot as plt

from bresenham import plot_line
from robot_controller import Controller


LIDAR_NUM_READINGS = 667
LIDAR_ACTUAL_NUM_READINGS = 530
LIDAR_FIRST_READING_INDEX = 57
LIDAR_LAST_READING_INDEX = -80
TOP_LEFT_X = -2.5
TOP_LEFT_Y = 1.8
ARENA_WIDTH = 4.7
ARENA_LENGTH = 5.9
LIDAR_ROBOT_X_OFFSET = 0.202
BALL_DIAMETER = 0.0399
WHEEL_MAX_SPEED_RADPS = 10.15
OCCUPANCY_GRID_THRESHOLD = 0.7
MAP_LENGTH = 300
KERNEL_SIZE = 56


def world2map(x,y) -> Tuple[float]:
    px = np.round(((x - TOP_LEFT_X) / ARENA_WIDTH)  * MAP_LENGTH)
    py = np.round(((TOP_LEFT_Y - y) / ARENA_LENGTH) * MAP_LENGTH)
    return int(px), int(py)

def map2world(px, py) -> Tuple[float]:
    x = ((px / MAP_LENGTH) * ARENA_WIDTH) + TOP_LEFT_X
    y = TOP_LEFT_Y - ((py / MAP_LENGTH) * ARENA_LENGTH)
    return x, y

def compute_cspace(map):
    kernel= np.ones((KERNEL_SIZE, KERNEL_SIZE))  
    cmap = map
    cmap = signal.convolve2d(map,kernel,mode='same')
    cmap = np.clip(cmap, 0, 1)  # As convolution increases values over 1.
    cspace = cmap > OCCUPANCY_GRID_THRESHOLD
    return cspace
    

def display_cspace(cspace):
    plt.imshow(cspace)
    plt.show()

def save_cspace(cspace):
    np.save('cspace',cspace)

def plan_path(map, px, py):
    pass
    

def main():
    robot = Supervisor()
    
    # get the time step of the current world.
    timestep = int(robot.getBasicTimeStep())
    
    leftMotor = robot.getDevice('wheel_left_joint')
    rightMotor = robot.getDevice('wheel_right_joint')
    leftMotor.setPosition(float('inf'))
    rightMotor.setPosition(float('inf'))

    # leftEncoder = robot.getDevice('wheel_left_joint_sensor')
    # rightEncoder = robot.getDevice('wheel_right_joint_sensor')
    # leftEncoder.enable(timestep)
    # rightEncoder.enable(timestep)

    
    lidar = robot.getDevice('Hokuyo URG-04LX-UG01')
    lidar.enable(timestep)
    lidar.enablePointCloud()
    
    gps = robot.getDevice('gps')
    gps.enable(timestep)
    
    compass = robot.getDevice('compass')
    compass.enable(timestep)
    
    angles = np.linspace(2*np.pi/3, -2*np.pi/3, LIDAR_NUM_READINGS)
    angles = angles[LIDAR_FIRST_READING_INDEX:LIDAR_LAST_READING_INDEX]
    
    map = np.zeros((300, 300), dtype=float)
    
    xr, yr = [], []
    
    map_display = robot.getDevice('map_display')
    cspace_display = robot.getDevice('cspace_display')
    
    home_position = (0.4, -3.1)
    WP = [(-1.65, -3.2),
          (-1.65, 0.35),
          (0.65, 0.35),
          (0.67, -1.65),
          (0.56, -3.3)]
    

    reverse_WP = deepcopy(WP)
    reverse_WP.reverse()
    reverse_WP.append(home_position)
    
    marker = robot.getFromDef("marker").getField("translation")

    controllers = [Controller(WHEEL_MAX_SPEED_RADPS, WP), Controller(WHEEL_MAX_SPEED_RADPS, reverse_WP)]
    controller_idx = 0
    robot_coordinates = []
    
    while robot.step(timestep) != -1:
        xw    = gps.getValues()[0]
        yw    = gps.getValues()[1]
        theta = np.arctan2(compass.getValues()[0],compass.getValues()[1])

        px_robot, py_robot = world2map(xw, yw)
        robot_coordinates.append((px_robot, py_robot))

        if controller_idx >= len(controllers):
            leftMotor.setVelocity(0.0)
            rightMotor.setVelocity(0.0)
            break 

        elif controllers[controller_idx].completed():
            controller_idx += 1
            continue

        marker.setSFVec3f([*controllers[controller_idx].get_current_target(), BALL_DIAMETER])

        vl, vr = controllers[controller_idx].get_input_vels((xw, yw, theta))

        leftMotor.setVelocity(vl)
        rightMotor.setVelocity(vr)

        ranges = lidar.getRangeImage()
        ranges[ranges == np.inf] = 100
        ranges = ranges[LIDAR_FIRST_READING_INDEX:LIDAR_LAST_READING_INDEX]
        
        w_T_r = np.array([[np.cos(theta), -np.sin(theta), xw],
                         [np.sin(theta),   np.cos(theta), yw],
                         [0,                   0,         1]])
        
        
        X_i = np.array([ranges * np.cos(angles) + LIDAR_ROBOT_X_OFFSET,
                        ranges * np.sin(angles), 
                        np.ones(LIDAR_ACTUAL_NUM_READINGS)])
        
        X_w = w_T_r @ X_i
        for i in range(LIDAR_ACTUAL_NUM_READINGS):
            px, py = world2map(X_w[0][i], X_w[1][i])
            if map[px, py] < 1:
                map[px, py] += 0.01

            # Reduce probability of obstacle for all pixels in the laser's line of sight using Bresenham's algorithm.
            laser_line_coordinates = plot_line(px_robot, py_robot, px, py)
            for coordinate in laser_line_coordinates[:-1]:
                px_laser = coordinate[0]
                py_laser = coordinate[1]

                if map[px_laser, py_laser] > 0.01:
                    map[px_laser, py_laser] -= 0.01 

        

        # # Draw configuration map
        # for row in np.arange(0, MAP_LENGTH):
        #     for col in np.arange(0, MAP_LENGTH):
        #         v = min(int((cmap[row, col]) * 255), 255)
        #         if v > 0.01:
        #             map_display.setColor(v*256**2 + v*256 + v)
        #             map_display.drawPixel(row, col)
        
        # # Draw configuration space
        # for row in np.arange(0, MAP_LENGTH):
        #     for col in np.arange(0, MAP_LENGTH):
        #         if cspace[row, col]:
        #             cspace_display.setColor(0xFFFFFF)
        #             cspace_display.drawPixel(row, col)
        #         else:
        #             cspace_display.setColor(0x000000)
        #             cspace_display.drawPixel(row, col)
        
        # cspace_display.setColor(0x00FF00)
        # for coordinate in robot_coordinates:
        #     cspace_display.drawPixel(coordinate[0], coordinate[1])
    cspace = compute_cspace(map)
    display_cspace(cspace)
    save_cspace(cspace)


if __name__ == '__main__':
    main()