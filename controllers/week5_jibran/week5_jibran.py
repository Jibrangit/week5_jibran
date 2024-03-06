"""week5_jibran controller."""

from controller import Robot, Supervisor, Motor, PositionSensor
import numpy as np
from enum import Enum
from copy import deepcopy
from bresenham import plot_line

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

class RobotDriveState(Enum):
    IDLE = 0
    DRIVE = 1
    TURN = 2
    ADJUST_CASTOR = 3

def world2map(x,y):
    px = np.round(((x - TOP_LEFT_X) / ARENA_WIDTH)  * 300)
    py = np.round(((TOP_LEFT_Y - y) / ARENA_LENGTH) * 300)
    return int(px), int(py)

class Controller:
    def __init__(self, waypoints):
        self._waypoints = waypoints
        self._index = 0
        self._robot_state = RobotDriveState.DRIVE
        self._MIN_POSITION_ERROR = 0.3  # metres
        self._MIN_HEADING_ERROR = 0.1   # radians

    def compute_errors(self, pose):
        xw = pose[0]
        yw = pose[1]
        theta = pose[2]

        rho = np.sqrt((xw - self._waypoints[self._index][0])**2 + (yw - self._waypoints[self._index][1])**2)
        alpha = np.arctan2(self._waypoints[self._index][1] - yw, self._waypoints[self._index][0] - xw) - theta
        
        # atan2 discontinuity
        if alpha > np.pi:
            alpha -= 2 * np.pi
        
        if alpha < -np.pi:
            alpha += 2 * np.pi

        return rho, alpha

    def get_input_vels(self, pose):

        rho, alpha = self.compute_errors(pose)

        if self._robot_state == RobotDriveState.DRIVE:

            p_trans, p_rot = 0.3 * WHEEL_MAX_SPEED_RADPS, 0.1 * WHEEL_MAX_SPEED_RADPS
            vl = p_trans * rho - p_rot * alpha 
            vr = p_trans * rho + p_rot * alpha

            if abs(rho) < self._MIN_POSITION_ERROR:
                self._index += 1
                self._robot_state = RobotDriveState.TURN


        elif self._robot_state == RobotDriveState.TURN:
            
            p_trans, p_rot = 0.1 * WHEEL_MAX_SPEED_RADPS, 0.3 * WHEEL_MAX_SPEED_RADPS
            vl = p_trans * rho - p_rot * alpha 
            vr = p_trans * rho + p_rot * alpha

            if abs(alpha) < self._MIN_HEADING_ERROR:
                self._robot_state = RobotDriveState.DRIVE

        else:
            vl, vr = 0.0, 0.0


        vl = max(min(vl, WHEEL_MAX_SPEED_RADPS), -WHEEL_MAX_SPEED_RADPS)
        vr = max(min(vr, WHEEL_MAX_SPEED_RADPS), -WHEEL_MAX_SPEED_RADPS)

        return vl, vr
    
    def completed(self):
        if self._index >= len(self._waypoints):
            return True
        else:
            return False 
        
    def get_index(self):
        return self._index
    
    def get_current_target(self):
        return self._waypoints[self._index]

def main():
    robot = Supervisor()
    
    # get the time step of the current world.
    timestep = int(robot.getBasicTimeStep())
    
    leftMotor = robot.getDevice('wheel_left_joint')
    rightMotor = robot.getDevice('wheel_right_joint')
    leftMotor.setPosition(float('inf'))
    rightMotor.setPosition(float('inf'))

    leftEncoder = robot.getDevice('wheel_left_joint_sensor')
    rightEncoder = robot.getDevice('wheel_right_joint_sensor')
    leftEncoder.enable(timestep)
    rightEncoder.enable(timestep)

    
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

    controllers = [Controller(WP), Controller(reverse_WP)]
    controller_idx = 0
    
    while robot.step(timestep) != -1:
        xw    = gps.getValues()[0]
        yw    = gps.getValues()[1]
        theta = np.arctan2(compass.getValues()[0],compass.getValues()[1])


        px_robot, py_robot = world2map(xw, yw)
        map_display.setColor(0x00FF00)
        map_display.drawPixel(px_robot, py_robot)

        if controller_idx >= len(controllers):
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
            map[px, py] += 0.01

            laser_line_coordinates = plot_line(px_robot, py_robot, px, py)
            for coordinate in laser_line_coordinates[:-1]:
                px_laser = coordinate[0]
                py_laser = coordinate[1]

                if px_laser > 0.01 and py_laser > 0.01:
                    map[px_laser, py_laser] -= 0.01 

        for row in np.arange(0, 300):
            for col in np.arange(0, 300):
                v = min(int((map[row, col]) * 255), 255)
                if v > 0.01:
                    map_display.setColor(v*256**2 + v*256 + v)
                    map_display.drawPixel(row, col)



if __name__ == '__main__':
    main()