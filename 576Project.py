import time
import pybullet as p
import pybullet_data
import numpy as np
import random
from load_obj import OBJModel

from gym_pybullet_drones.utils.enums import DroneModel, Physics, ActionType, ObservationType, ImageType
from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl
from gym_pybullet_drones.envs.CtrlAviary import CtrlAviary
from gym_pybullet_drones.utils.utils import sync

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

        # block = OBJModel("DisasterDrone/assets/block.obj", self.CLIENT)
        # block.loadObj(self.CLIENT, pos=[1, 2, 2])

        # building = OBJModel("assets/block.obj", self.CLIENT)
        # building.loadObj(self.CLIENT, pos=[0, -2, 1])


def line(tidx, start=(0,0,0), end=(1,1,0)):
    npstart, npend = np.array(start), np.array(end)
    return npstart + tidx * (npend - npstart)

def arc(tidx, center=(0, 0, 0), R=1.0, theta_start=0, theta_end=np.pi/2, z=0.5):
    angle = theta_start + tidx * (theta_end - theta_start)
    x = center[0] + R * np.cos(angle)
    y = center[1] + R * np.sin(angle)
    return np.array([x,y,z])

if __name__ == "__main__":
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
    WP_ARC = DEFAULT_CONTROL_FREQ_HZ * 6
    WP_LINE = DEFAULT_CONTROL_FREQ_HZ * 3
    NUM_WP = WP_UP + WP_ARC + WP_LINE

    TARGET_POS = np.zeros((NUM_WP,3))

    for i in range(WP_UP):
        tidx = i / WP_UP
        TARGET_POS[i,:] = line(tidx, start=(0,0,0), end=(0,1,1))
    for i in range(WP_ARC):
        tidx = i / WP_ARC
        TARGET_POS[WP_UP + i, :] = arc(tidx, center=(1,1,1), R=1.0, theta_start=np.pi, theta_end=0, z=1)
    pos = TARGET_POS[WP_UP + WP_ARC - 1, :]
    for i in range(WP_LINE):
        tidx = i / WP_LINE
        TARGET_POS[WP_UP + WP_ARC + i, :] = line(tidx, start=pos, end=(1,1,1))

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
    
    for i in range(0, int(100*env.CTRL_FREQ)):
        obs, reward, terminated, truncated, info = env.step(action)

        action[0,:], _, _ = ctrl.computeControlFromState(
            control_timestep=env.CTRL_TIMESTEP,
            state=obs[0],
            target_pos=TARGET_POS[wp_counters[0], :],
            target_rpy=INIT_RPYS[0, :]
        )

        # monkey.loadObj(PYB_CLIENT)

        # Summons infinite Ducks
        # if i/env.CTRL_FREQ>5 and i%10==0 and i/env.CTRL_FREQ<10: monkey.loadObj(PYB_CLIENT, pos = [0+random.gauss(0, 0.3),-0.5+random.gauss(0, 0.3),3])

        wp_counters[0] = wp_counters[0] + 1 if wp_counters[0] < (NUM_WP-1) else NUM_WP - 1

        env.render()

        sync(i, START, env.CTRL_TIMESTEP)

    env.close()
