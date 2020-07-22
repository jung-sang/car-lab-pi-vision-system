#imports
from collections import OrderedDict
import math


class VelocityTracker:
    
    def __init__ (self):
        self.oldCentroids = []
        self.newCentroids = []
        self.totalDistance = [[0,0],[0,0],[0,0]]
        self.average_direction = OrderedDict()
        self.instantDirectionDict = OrderedDict()
    
    def update(self, objects, counter):

        
        objectIDs = list(objects.keys())
        objectCentroids = list(objects.values())
        
        self.oldCentroids = self.newCentroids[:]
        self.newCentroids = objectCentroids
        print(self.oldCentroids)

        for (oldCentroids, newCentroids, i) in zip(self.oldCentroids, self.newCentroids, objectIDs):
            instantDirection = []
            normInstantDirection = []
            # instantaneous motion direction vector
            # instantDirection = (newCentroids - oldCentroids)
            instantDirection = [new - old for (new, old) in zip(newCentroids, oldCentroids)]
#             print(instantDirection)
#             instantDirection[1] = 1*instantDirection[1]
#             print(instantDirection)
#             
            # instantaneous norm of motion direction vector
#             if ((instantDirection[0])**2+(instantDirection[1])**2) != 0:
#                 normInstantDirection = [instantDirection[0]/(math.sqrt((instantDirection[0])**2+(instantDirection[1])**2)), instantDirection[1]/(math.sqrt((instantDirection[0])**2+(instantDirection[1])**2))]
#             else:
#                 normInstantDirection = [0,0]

            normInstantDirection = instantDirection
            
            #print(normInstantDirection)
            
            if len(self.instantDirectionDict) <= i:
                self.instantDirectionDict[i] = normInstantDirection
            elif (counter % 10) == 0:
                self.instantDirectionDict[i] = normInstantDirection
            else:
                self.instantDirectionDict[i] = sum(normInstantDirection, self.instantDirectionDict[i])
            
            
            # for (i, direction) in objects.items():
#             
#             self.totalDistance[i] = sum (normInstantDirection[i], self.totalDistance[i])
#             
#             print(self.totalDistance[i])
            
            self.average_direction[i] = [i/counter for i in self.instantDirectionDict[i]]
            
            #print(self.average_direction)
            
        return self.average_direction
        