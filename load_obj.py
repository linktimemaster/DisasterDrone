import time
import pybullet as p
import pybullet_data
import numpy as np
import random

class OBJModel():

    def __init__(self, path, physicsClientId, color = [1, 1, 1, 1], specColor = [0.4, .4, 0], shift = [0, -0.02, 0], meshScale = [1, 1, 1]):
        
        self.visualShapeId = p.createVisualShape(shapeType=p.GEOM_MESH,
                                    fileName=path,
                                    rgbaColor=color,
                                    specularColor=specColor,
                                    visualFramePosition=shift,
                                    meshScale=meshScale,
                                    physicsClientId=physicsClientId)

        self.collisionShapeId = p.createCollisionShape(shapeType=p.GEOM_MESH,
                                          fileName=path,
                                          collisionFramePosition=shift,
                                          meshScale=meshScale,
                                          physicsClientId=physicsClientId)

    def loadObj(self, physicsClientId, pos = [1, 1, 1], inertialPos = [0, 0, 0], mass = 1):
        return p.createMultiBody(baseMass=mass,
                            baseInertialFramePosition=inertialPos,
                            baseCollisionShapeIndex=self.collisionShapeId,
                            baseVisualShapeIndex=self.visualShapeId,
                            basePosition = pos,
                            useMaximalCoordinates=True,
                            physicsClientId=physicsClientId)
        