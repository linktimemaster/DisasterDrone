import time
import pybullet as p
import pybullet_data
import numpy as np
import random
from load_obj import OBJModel
from enum import Enum
import math
import socket

import tensorflow as tf
import numpy as np
from tensorflow.keras import models

from gym_pybullet_drones.utils.enums import DroneModel, Physics, ActionType, ObservationType, ImageType
from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl
from gym_pybullet_drones.envs.CtrlAviary import CtrlAviary
from gym_pybullet_drones.utils.utils import sync

import matplotlib.pyplot as plt
import numpy as np

class Direction(Enum):
    N = 0
    NE = 45
    E = 90
    SE = 135
    S = 180
    SW = 225
    W = 270
    NW = 315

def yaw_from_direction(d:Direction):
    return math.radians(d.value)

#make new class with obstacles

class NewCA(CtrlAviary):
    def _addObstacles(self):
        """Add obstacles to the environment.

        """
        # p.loadURDF("teddy_vhacd.urdf",
        #            [0, -1, .1],
        #            p.getQuaternionFromEuler([0, 0, 0]),
        #            physicsClientId=self.CLIENT
        #            )

        # monkey = OBJModel("DisasterDrone/assets/monkey.obj", self.CLIENT)
        # monkey.loadObj(self.CLIENT, pos=[2, 2, 2])

        SF = 0.5
        buildDist = 5
        earthquakeRadius = 10
        scale = [SF, SF, SF]
        color_gray = [0.74, 0.74, 0.74, 1]


        normalBuilding = OBJModel("/assets/building.obj", self.CLIENT, meshScale=scale, color=color_gray)
        brokenBuilding = OBJModel("/assets/building_broken.obj", self.CLIENT, meshScale=scale, color=color_gray)

        for i in range(-3, 4):
            for j in range(-3, 4):

                if not (i == 0 and j == 0):

                    x_pos = i * buildDist
                    y_pos = j * buildDist

                    dist = math.sqrt(math.pow(x_pos, 2) + math.pow(y_pos, 2))

                    if dist < earthquakeRadius:
                        # texUid = p.loadTexture("rick.png")
                        brokenBuilding.loadObj(self.CLIENT, pos=[x_pos, y_pos, 2], ori=[1, 0, 0, 1])
                        # p.changeVisualShape(bodyUid, -1, textureUniqueId=texUid)
                    else:
                        normalBuilding.loadObj(self.CLIENT, pos=[x_pos, y_pos, 2], ori=[1, 0, 0, 1])


def line(tidx, start=(0,0,0), end=(1,1,0)):
    npstart, npend = np.array(start), np.array(end)
    return npstart + tidx * (npend - npstart)

def arc(tidx, center=(0, 0, 0), R=1.0, theta_start=0, theta_end=np.pi/2, z=0.5):
    angle = theta_start + tidx * (theta_end - theta_start)
    x = center[0] + R * np.cos(angle)
    y = center[1] + R * np.sin(angle)
    return np.array([x,y,z])

def take_image(client, index):
    drone_pos, drone_quat = p.getBasePositionAndOrientation(
                                                            env.DRONE_IDS[0],
                                                            physicsClientId=client
                                                        )
    rotation_mat = np.array(p.getMatrixFromQuaternion(drone_quat)).reshape(3, 3)

    camera_forward = rotation_mat[:, 0]
    camera_up = rotation_mat[:, 2]

    img_target = drone_pos + -1 * camera_up
    
    view_matrix = p.computeViewMatrix(
                                      cameraEyePosition=drone_pos,
                                      cameraTargetPosition=img_target,
                                      cameraUpVector=camera_up,
                                      physicsClientId=client
                                  )
    projection_matrix = p.computeProjectionMatrixFOV(
                                                     fov=60,
                                                     aspect=16/9,
                                                     nearVal=0.1,
                                                     farVal=100
                                                 )
    width, height = 192, 192
    img_arr = p.getCameraImage(
                               width=width,
                               height=height,
                               viewMatrix=view_matrix,
                               projectionMatrix=projection_matrix,
                               renderer=p.ER_TINY_RENDERER,
                               physicsClientId=client
                           )
    rgb = np.reshape(np.uint8(img_arr[2]), (height, width, 4))[:, :, :3]
    return rgb
    # from PIL import Image
    # image = Image.fromarray(rgb)
    # image.save(f"img_{int(index)}.png")

if __name__ == "__main__":

    host = "127.0.0.1"
    port = 8080

    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    class_names = ["normal", "broken"]
    building_recognizer = tf.keras.models.load_model('Models/kinda_working_building_recognizer.keras')

    num_drones = 1
    
    H = .1
    H_STEP = .05
    R = 2 #determines the radius of the circle

    height = 0.5 #determines the height of the drone

    # oldHeightParam = H+i*H_STEP

    # INIT_XYZS = np.array([[R*np.cos((i/6)*2*np.pi+np.pi/2) , R*np.sin((i/6)*2*np.pi+np.pi/2)-R, height] for i in range(num_drones)])
    # INIT_RPYS = np.array([[0, 0,  i * (np.pi/2)/num_drones] for i in range(num_drones)])
    INIT_XYZS = np.array([[0,0,0]])
    INIT_RPYS = np.array([[0,0,0]])

    #Number of Waypoints: Determines how many waypoints the trajectory is split into
    #When the number is higher the drone moves slower when it is lower the drone moves faster

    DEFAULT_CONTROL_FREQ_HZ = 48


    WP_UP = DEFAULT_CONTROL_FREQ_HZ * 3
    WP_LINE = DEFAULT_CONTROL_FREQ_HZ * 6
    WP_BLINE = DEFAULT_CONTROL_FREQ_HZ * 15
    WP_ARC = DEFAULT_CONTROL_FREQ_HZ * 20

    # WP_LINE = DEFAULT_CONTROL_FREQ_HZ * 3

    # NUM_WP = WP_UP + WP_BLINE + WP_ARC + WP_ARC

    NUM_WP = WP_UP + WP_BLINE + (WP_LINE * 7) * 7

    TARGET_POS = np.zeros((NUM_WP,3))
    TARGET_YAW = np.zeros(NUM_WP)

    #fly overhead

    for i in range(WP_UP):
        tidx = i / WP_UP
        TARGET_POS[i,:] = line(tidx, start=(0,0,0), end=(0,0,6))
        TARGET_YAW[i] = yaw_from_direction(Direction.N)

    for i in range(WP_BLINE):
        tidx = i/WP_BLINE
        TARGET_POS[WP_UP + i, :] = line(tidx, start=(0,0,6), end=(15,15,6))
        TARGET_YAW[WP_UP + i] = yaw_from_direction(Direction.N)



    for i in range(WP_LINE*6):
        tidx = i/(WP_LINE*6)
        TARGET_POS[WP_UP + WP_BLINE + i, :] = line(tidx, start=(15,15,6), end=(-15,15,6))
        TARGET_YAW[WP_UP + WP_BLINE + i] = yaw_from_direction(Direction.N)
    for i in range(WP_LINE):
        tidx = i/WP_LINE
        TARGET_POS[WP_UP + WP_BLINE + (WP_LINE*6) + i, :] = line(tidx, start=(-15,15,6), end=(-15,10,6))
        TARGET_YAW[WP_UP + WP_BLINE + (WP_LINE*6) + i] = yaw_from_direction(Direction.N)



    for i in range(WP_LINE*6):
        tidx = i/(WP_LINE*6)
        TARGET_POS[WP_UP + WP_BLINE + (WP_LINE*7)+ i, :] = line(tidx, start=(-15,10,6), end=(15,10,6))
        TARGET_YAW[WP_UP + WP_BLINE + (WP_LINE*7)+ i] = yaw_from_direction(Direction.N)
    for i in range(WP_LINE):
        tidx = i/WP_LINE
        TARGET_POS[WP_UP + WP_BLINE + (WP_LINE*13) + i, :] = line(tidx, start=(15,10,6), end=(15,5,6))
        TARGET_YAW[WP_UP + WP_BLINE + (WP_LINE*13) + i] = yaw_from_direction(Direction.N)



    for i in range(WP_LINE*6):
        tidx = i/(WP_LINE*6)
        TARGET_POS[WP_UP + WP_BLINE + (WP_LINE*14)+ i, :] = line(tidx, start=(15,5,6), end=(-15,5,6))
        TARGET_YAW[WP_UP + WP_BLINE + (WP_LINE*14)+ i] = yaw_from_direction(Direction.N)
    for i in range(WP_LINE):
        tidx = i/WP_LINE
        TARGET_POS[WP_UP + WP_BLINE + (WP_LINE*20) + i, :] = line(tidx, start=(-15,5,6), end=(-15,0,6))
        TARGET_YAW[WP_UP + WP_BLINE + (WP_LINE*20) + i] = yaw_from_direction(Direction.N)


    
    for i in range(WP_LINE*6):
        tidx = i/(WP_LINE*6)
        TARGET_POS[WP_UP + WP_BLINE + (WP_LINE*21)+ i, :] = line(tidx, start=(-15,0,6), end=(15,0,6))
        TARGET_YAW[WP_UP + WP_BLINE + (WP_LINE*21)+ i] = yaw_from_direction(Direction.N)
    for i in range(WP_LINE):
        tidx = i/WP_LINE
        TARGET_POS[WP_UP + WP_BLINE + (WP_LINE*27) + i, :] = line(tidx, start=(15,0,6), end=(15,-5,6))
        TARGET_YAW[WP_UP + WP_BLINE + (WP_LINE*27) + i] = yaw_from_direction(Direction.N)



    for i in range(WP_LINE*6):
        tidx = i/(WP_LINE*6)
        TARGET_POS[WP_UP + WP_BLINE + (WP_LINE*28)+ i, :] = line(tidx, start=(15,-5,6), end=(-15,-5,6))
        TARGET_YAW[WP_UP + WP_BLINE + (WP_LINE*28)+ i] = yaw_from_direction(Direction.N)
    for i in range(WP_LINE):
        tidx = i/WP_LINE
        TARGET_POS[WP_UP + WP_BLINE + (WP_LINE*34) + i, :] = line(tidx, start=(-15,-5,6), end=(-15,-10,6))
        TARGET_YAW[WP_UP + WP_BLINE + (WP_LINE*34) + i] = yaw_from_direction(Direction.N)


    for i in range(WP_LINE*6):
        tidx = i/(WP_LINE*6)
        TARGET_POS[WP_UP + WP_BLINE + (WP_LINE*35)+ i, :] = line(tidx, start=(-15,-10,6), end=(15,-10,6))
        TARGET_YAW[WP_UP + WP_BLINE + (WP_LINE*35)+ i] = yaw_from_direction(Direction.N)
    for i in range(WP_LINE):
        tidx = i/WP_LINE
        TARGET_POS[WP_UP + WP_BLINE + (WP_LINE*41) + i, :] = line(tidx, start=(15,-10,6), end=(15,-15,6))
        TARGET_YAW[WP_UP + WP_BLINE + (WP_LINE*41) + i] = yaw_from_direction(Direction.N)


    
    for i in range(WP_LINE*6):
        tidx = i/(WP_LINE*6)
        TARGET_POS[WP_UP + WP_BLINE + (WP_LINE*42)+ i, :] = line(tidx, start=(15,-15,6), end=(-15,-15,6))
        TARGET_YAW[WP_UP + WP_BLINE + (WP_LINE*42)+ i] = yaw_from_direction(Direction.N)


    #Sprial Building Trajectory

    # for i in range(WP_UP):
    #     tidx = i / WP_UP
    #     TARGET_POS[i,:] = line(tidx, start=(0,0,0), end=(0,2,1))
    #     TARGET_YAW[i] = yaw_from_direction(Direction.N)

    # for i in range(WP_BLINE):
    #     tidx = i/WP_BLINE
    #     TARGET_POS[WP_UP + i, :] = line(tidx, start=(0,2,1), end=(5,2,1))
    #     TARGET_YAW[WP_UP + i] = yaw_from_direction(Direction.W)
    
    # for i in range(WP_ARC):
    #     tidx = i / WP_ARC
    #     TARGET_POS[WP_BLINE + WP_UP + i, :] = arc(tidx, center=(5,0,1), R=2.0, theta_start=np.pi/2, theta_end=-2*np.pi, z= (1 + (2/WP_ARC)*i))
    #     TARGET_YAW[WP_BLINE + WP_UP + i] = 3*(np.pi/2) + ((-(5/2)*np.pi)/WP_ARC)*i

    # pos = TARGET_POS[WP_BLINE + WP_UP + WP_ARC - 1, :]
    
    # for i in range(WP_ARC):
    #     tidx = i / WP_ARC
    #     TARGET_POS[WP_BLINE + WP_UP + WP_ARC + i, :] = arc(tidx, center=(5,0,1), R=2.0, theta_start=0, theta_end=-(3 * np.pi)/2, z= (3 - (2/WP_ARC)*i))
    #     TARGET_YAW[WP_BLINE + WP_UP + WP_ARC + i] = np.pi + ((-(3/2)*np.pi)/WP_ARC)*i


    #circle code
    
    # for i in range(WP_LINE):
    #     tidx = i / WP_LINE
    #     TARGET_POS[WP_UP + WP_ARC + i, :] = line(tidx, start=pos, end=(1,1,1))
    #     TARGET_YAW[WP_UP + WP_ARC + i] = yaw_from_direction(Direction.W)
   

    # for i in range(NUM_WP):
    #     TARGET_POS[i, :] = R*np.cos((i/NUM_WP)*(2*np.pi)+np.pi/2)+INIT_XYZS[0, 0], R*np.sin((i/NUM_WP)*(2*np.pi)+np.pi/2)-R+INIT_XYZS[0, 1], height
    # wp_counters = np.array([int((i*NUM_WP/6)%NUM_WP) for i in range(num_drones)])
    
    # print(wp_counters)

    env = NewCA(
                initial_xyzs=INIT_XYZS,
                initial_rpys=INIT_RPYS,
                ctrl_freq=48,
                gui=True,
                obstacles=True,
                user_debug_gui=True,
            )

    PYB_CLIENT = env.getPyBulletClient()


    ctrl = DSLPIDControl(DroneModel.CF2X)

    action = np.zeros((1,4))
    START = time.time()

    wp_counters = np.zeros(num_drones, dtype=int)

    s.connect((host,port))

    for i in range(0, int(330*env.CTRL_FREQ)):
        obs, reward, terminated, truncated, info = env.step(action)

        t_yaw = TARGET_YAW[wp_counters[0]]

        action[0,:], _, _ = ctrl.computeControlFromState(
            control_timestep=env.CTRL_TIMESTEP,
            state=obs[0],
            target_pos=TARGET_POS[wp_counters[0], :],
            target_rpy=np.array([0, 0, t_yaw])
        )

        if wp_counters[0] >= (WP_UP + WP_BLINE):
            count = wp_counters - (WP_UP + WP_BLINE)
            total = NUM_WP - (WP_UP + WP_BLINE)

            if (count % WP_LINE) == 0 :
                print(f"TargetPos: {TARGET_POS[wp_counters[0], :]}")
                pos = TARGET_POS[wp_counters[0], :]
                x = int(int(pos[0])/5) + 3
                y = int(int(pos[1])/5) + 3
                # take_image(PYB_CLIENT, count/WP_LINE)
                image = take_image(PYB_CLIENT, count/WP_LINE)

                # from PIL import Image
                # saved_image = Image.fromarray(image)
                # saved_image.save(f"img_{int(count/WP_LINE)}.png")

                predict = building_recognizer.predict(tf.reshape(image, (1, 192, 192, 3)))
                score = tf.nn.softmax(predict[0])
                
                print(predict)
                print(score)
                print(class_names[np.argmax(score)], 100 * np.max(score))
                
                name = class_names[np.argmax(score)]

                data = " ".join((name, str(x), str(y)))
                enc_data = data.encode()
                s.sendall(enc_data)
                
        wp_counters[0] = wp_counters[0] + 1 if wp_counters[0] < (NUM_WP-1) else NUM_WP - 1

        env.render()

        sync(i, START, env.CTRL_TIMESTEP)

    s.sendall('EOF'.encode())
    s.close()   
    env.close()
