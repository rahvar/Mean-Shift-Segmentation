import cv2 
import numpy as np
from copy import deepcopy
from matplotlib import pyplot as plt
from scipy.ndimage import imread
import random
import time
from scipy.spatial import distance
from sklearn.metrics.pairwise import euclidean_distances

img = imread('Image_Hill_House.jpg',mode ='L')
img_copy = deepcopy(img)

# orginal 3d image 
x,y = img.shape

# return indices for row and colum as a,b
a ,b = np.indices((x,y))

# 2d matrix with spatial points in form (x,y)
co = np.dstack((a,b))

# reshapes 2d spatial matrix as a 1-d vector with points (x,y)
spatial = co.reshape((x*y,2))

# contains rgb matrix in with each row representing one point 
pixels = img_copy.reshape((x*y,1))

# matrix with spatial and intensity info as 5-d row vectors
mat = np.concatenate((spatial,pixels),axis=1)


from collections import defaultdict
vis_centroids = defaultdict(int)

def gaussian_kernel_update(x, points, bandwidth):
    distances = np.transpose(euclidean_distances(x,points)).reshape(len(points))
    
    #distances = distances.reshape((len(distances),1))
    
    incl = np.intersect1d(np.where(distances<=bandwidth),np.where(marked==0))
    
    #r_points = points[np.where(distances<=bandwidth)]
    r_points = points[incl]
    distances = distances[incl]
    weights = np.exp(-1 * (distances** 2 / bandwidth ** 2))
    weights = weights.reshape((len(weights),1))
    #print weights.shape,r_points.shape,points.shape
    return np.sum(r_points * weights, axis=0) / np.sum(weights),np.where(distances<=bandwidth)



# returns points with eucledian distance less than threshold from centroid
def get_neighbours(point,points,rang):
    
    distances = np.transpose(euclidean_distances(point,points)).reshape(len(points))
    
    return points[np.where(distances<=rang)],np.where(distances<=rang)

# copy of 5-d matrix
rem_points = deepcopy(mat)
remain_points = deepcopy(mat)

visited = set()

# initial centroid chosen randomly
centroid = rem_points[random.randint(0,len(rem_points)-1)]
centroid = centroid.astype(np.float32)
# contains the resultant image after mean shift
new_centroid = deepcopy(centroid)
results = np.zeros(rem_points.shape)
#print centroid

start = time.time()
print centroid
a,b = rem_points.shape
marked = np.zeros(a)
assign = dict()

x=0
while len(remain_points)>0:
    x+=1
    for i in range(10):
        # indices returns all points (marked and unmarked) 
        #that are in the range of the centroid 
        new_centroid,indices =gaussian_kernel_update(centroid,rem_points,30)
    
        #print new_centroid.shape
        dst = distance.euclidean(centroid,new_centroid)
        centroid = new_centroid  
        if dst<=10:
            break
    # break
    # extra contains unmarked indices that belong in the new cluster
    neigh_points, indices = get_neighbours(new_centroid,rem_points,30)
    #print centroid,len(indices)
    extra = np.intersect1d(np.where(marked==0),indices)
    
    # assigns all points in the cluster to their centroid/mean
   
    # marks all the newly clustered points
    marked[extra]=1
    results[extra] = centroid

    print len(marked[marked>0])
    #visited.update(map(tuple,neigh_points))
    # remain_points contains all the original points
    remain_points = rem_points[marked==0]

    if len(remain_points)>0:
        # picks a new random centroid from the remaining points
        centroid = remain_points[random.randint(0,len(remain_points)-1)]
        if vis_centroids[tuple(centroid)] ==1:
          
            break
        vis_centroids[tuple(centroid)]=1 
    #if x%100==0:
        #break   
    if len(extra) == 0:
        break
    print centroid,len(remain_points),len(extra)
end = time.time()
print(end - start)

res = results[:,2:]
x,y =img.shape
r = res.reshape(x,y)
r = r.astype(np.uint8)

fig = plt.figure(figsize=(15,15))
plt.subplot(1,2,1)
plt.imshow(r,cmap="gray")
plt.subplot(1,2,2)
plt.imshow(img,cmap="gray")
plt.show()