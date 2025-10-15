import time
import pybullet as p
import pybullet_data
import numpy as np

from gym_pybullet_drones.utils.enums import DroneModel, Physics, ActionType, ObservationType, ImageType
from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl
from gym_pybullet_drones.envs.CtrlAviary import CtrlAviary
from gym_pybullet_drones.utils.utils import sync

#make new class with obstacles
class NewCA(CtrlAviary):
    def _addObstacles(self):
        """Add obstacles to the environment.

        """
        p.loadURDF("teddy_vhacd.urdf",
                   [0, -1, .1],
                   p.getQuaternionFromEuler([0, 0, 0]),
                   physicsClientId=self.CLIENT
                   )

if __name__ == "__main__":
    num_drones = 1
    
    H = .1
    H_STEP = .05
    R = .3

    INIT_XYZS = np.array([[R*np.cos((i/6)*2*np.pi+np.pi/2), R*np.sin((i/6)*2*np.pi+np.pi/2)-R, H+i*H_STEP] for i in range(num_drones)])
    INIT_RPYS = np.array([[0, 0,  i * (np.pi/2)/num_drones] for i in range(num_drones)])

    NUM_WP = 480
    TARGET_POS = np.zeros((NUM_WP,3))
    for i in range(NUM_WP):
        TARGET_POS[i, :] = R*np.cos((i/NUM_WP)*(2*np.pi)+np.pi/2)+INIT_XYZS[0, 0], R*np.sin((i/NUM_WP)*(2*np.pi)+np.pi/2)-R+INIT_XYZS[0, 1], 0
    wp_counters = np.array([int((i*NUM_WP/6)%NUM_WP) for i in range(num_drones)])
    
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
    
    for i in range(0, int(12*env.CTRL_FREQ)):
        obs, reward, terminated, truncated, info = env.step(action)

        action[0,:], _, _ = ctrl.computeControlFromState(
            control_timestep=env.CTRL_TIMESTEP,
            state=obs[0],
            target_pos=np.hstack([TARGET_POS[wp_counters[0], 0:2], INIT_XYZS[0, 2]]),
            target_rpy=INIT_RPYS[0, :]
        )

        wp_counters[0] = wp_counters[0] + 1 if wp_counters[0] < (NUM_WP-1) else 0

        env.render()

        sync(i, START, env.CTRL_TIMESTEP)

    env.close()
