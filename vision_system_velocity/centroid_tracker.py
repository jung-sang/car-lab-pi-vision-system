#imports
from collections import OrderedDict
from scipy.spatial import distance as dist
import numpy as np

class CentroidTracker:
    
    def __init__ (self):
        # list that will hold the centroids of the active objects
        self.objects = OrderedDict()
        # list that will hold how many frames the active objects have been off screen for
        self.disappeared = OrderedDict()
        # initially set the next object to be the 0th
        self.nextObjectID = 0
        # max number of frames an object can be not on screen before its index is deleted
        self.maxDisappeared = 50
        
    def register(self, centroid):
        self.objects[self.nextObjectID] = centroid
        # object has been off screen for zero frames
        self.disappeared[self.nextObjectID] = 0
        # get ready to set next unique object
        self.nextObjectID += 1
        
    def deregister(self, objectID):
        del self.objects[objectID]
        del self.disappeared[objectID]
    
    def update(self, rects):
        
        # if no bounding boxes in frame, add 1 frame to self.disappeared for each object being tracked
        if len(rects) == 0:
            for objectID in list(self.disappeared.keys()):
                self.disappeared[objectID] += 1
                
                if self.disappeared[objectID] > self.maxDisappeared:
                    self.deregister(objectID)
                    
            return self.objects
        
        # create an array of zeros to be filled with bounding box centroids
        inputCentroids = np.zeros((len(rects), 2), dtype = "int")
        
        # loop through the bounding box rects
        for(i, (ymin, xmin, ymax, xmax)) in enumerate(rects):
            # find the center of each bounding box
            cX = int((xmin + xmax) / 2.0)
            cY = int((ymin + ymax) / 2.0)
            # put the center in the inputCentroids[] list
            inputCentroids[i] = (cX, cY)
            
        # if there are currently no objects being tracked    
        if len(self.objects) == 0:
            for i in range(0, len(inputCentroids)):
                self.register(inputCentroids[i])
            
        # if there are objects being tracked
        else:
            # split  self.objects into the IDs and the values
            objectIDs = list(self.objects.keys())
            objectCentroids = list(self.objects.values())

            
            # calculate the distance between the new centroids and the previous centroids  
            D = dist.cdist(np.array(objectCentroids), inputCentroids)
            
            # orderes the matix D and gives the indexes the smallest distance between each old centroid to a new centroid
            # rows would give the current centroid indexes and cols would give the centroid index of the nearest centroid
            rows = D.min(axis=1).argsort()
            cols = D.argmin(axis=1)[rows]
            
            # variables to keep track of which rows and columns already examines
            usedRows = set()
            usedCols = set()
            
            # loop through all combinations of rows and columns
            for(row, col) in zip(rows, cols):
                
                # if row or column has been examined before we want to ignore it
                if row in usedRows or col in usedCols:
                    continue
                
                objectID = objectIDs[row]
                self.objects[objectID] = inputCentroids[col]
                self.disappeared[objectID] = 0
                
                # add the examined row and col
                usedRows.add(row)
                usedCols.add(col)
                
            # keep track of unused rows and columns for the next if statement
            unusedRows = set(range(0, D.shape[0])).difference(usedRows)
            unusedCols = set(range(0, D.shape[1])).difference(usedCols)
                
            # handle situations when the number of new centroids does not equal the number of old centroids
            # objects have potentially disappeared
            if D.shape[0] >= D.shape[1]:
                for row in unusedRows:
                    # index disappeared count for objectID that is missing
                    objectID = objectIDs[row]
                    self.disappeared[objectID] += 1
                    
                    # check to see if consecutive frames that the object has been off screen is greater than threshold and delete if it has been
                    if self.disappeared[objectID] > self.maxDisappeared:
                        self.deregister(objectID)
            
            # objects have appeared
            else:
                for col in unusedCols:
                    self.register(inputCentroids[col])

        return self.objects
    