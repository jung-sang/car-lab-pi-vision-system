#imports

class VelocityTracker:
    
    def __init__ (self):
        self.oldCentroids = []
        self.newCentroids = []
        self.direction = [0,0]
        self.average_direction = [0,0]
    
    def update(self, objects, counter):
        
        self.oldCentroids = self.newCentroids[:]
        self.newCentroids = objects
        
        for (objectID, centroid) in self.oldCentroids:
            for (objectID_2, centroid_2) in self.newCentroids:
                self.direction = self.direction + (centroid - centroid_2)
                self.average_direction = self.direction/counter
        print(type(self.average_direction))
                
                
        
        return(self.average_direction)
        


