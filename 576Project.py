import time
import pybullet as p
import pybullet_data
import numpy as np

from gym_pybullet_drones.utils.enums import DroneModel, Physics, ActionType, ObservationType, ImageType
from gym_pybullet_drones.envs.HoverAviary import HoverAviary

#make new class with obstacles
class NewHA(HoverAviary):
    def _addObstacles(self):
        """Add obstacles to the environment.

        Only if the observation is of type RGB, 4 landmarks are added.
        Overrides BaseAviary's method.

        """
        if self.OBS_TYPE == ObservationType.RGB:
            p.loadURDF("teddy_vhacd.urdf",
                       [0, -1, .1],
                       p.getQuaternionFromEuler([0, 0, 0]),
                       physicsClientId=self.CLIENT
                       )
        else:
            pass

if __name__ == "__main__":
    env = NewHA(ctrl_freq=24, gui=True, obs=ObservationType.RGB)

    env.EPISODE_LEN_SEC = 9999
    
    client_id = env.CLIENT

    target_pos = np.array([0, 0, 1])
    obs = None
    done = False

    while True:
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        env.render()
        time.sleep(1.0 / env.PYB_FREQ)

    env.close()
