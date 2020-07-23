#imports
from collections import OrderedDict
import math


class VelocityTracker:

    def __init__ (self):
        self.oldCentroids = []
        self.newCentroids = []
        self.direction = OrderedDict()
        self.sumDirection = OrderedDict()
        self.counter = 0
    
    def update(self, objects):
        
        self.counter += 1

        # separating dictionary components of objects
        objectIDs = list(objects.keys())
        objectCentroids = list(objects.values())  

        # pulling new (most recent frame) and old (last frame) centroid locations from objects
        self.oldCentroids = self.newCentroids[:]   
        self.newCentroids = objectCentroids

        
        for (oldCentroids, newCentroids, i) in zip(self.oldCentroids, self.newCentroids, objectIDs):
            
            # finding change vector between old and new centroids
            instantDirection = [new - old for (new, old) in zip(newCentroids, oldCentroids)]
            
            # adding instantDirection to self.instantDirectionDict that keeps a running sum of displacement over the last n (n=10) frames
            if (i not in self.sumDirection) or ((self.counter % 50) == 0): #len(self.sumDirection) <= i
                self.counter = 1
                self.sumDirection[i] = instantDirection
            else:
                self.sumDirection[i] = [inst + instDict for (inst, instDict) in zip(instantDirection, self.sumDirection[i])]             
            
            # normalizes sum motion direction vector
            if ((self.sumDirection[i][0])**2+(self.sumDirection[i][1])**2) != 0:
                sumDirectionNorm = [self.sumDirection[i][0]/(math.sqrt((self.sumDirection[i][0])**2+(self.sumDirection[i][1])**2)), self.sumDirection[i][1]/(math.sqrt((self.sumDirection[i][0])**2+(self.sumDirection[i][1])**2))]
            else:
                sumDirectionNorm = [0,0]
            
            #place direction norms in dictionary
            self.direction[i] = sumDirectionNorm            
            
        return self.direction
        