import cv2 
import numpy as np
from copy import deepcopy
from matplotlib import pyplot as plt
from scipy.ndimage import imread
import random
import time
from scipy.spatial import distance
from sklearn.metrics.pairwise import euclidean_distances
from scipy import ndimage
from scipy.misc import imsave

img = imread('original.jpg')
img_copy = deepcopy(img)

# Epanechnikov kernel parameters
hs,hr = 50,90

# Uniform kernel parameter
h = 90

# Kernel Selection parameter - 0 for Uniform Kernel , 1 for  Epanechnikov kernel 
kernel = 1

# max no of iterations for convergence
iterations = 20

#Convergence criterion - distance between old and new mean
convergence = 1

size = img.shape
if len(size)==2:
    x,y = size
    z = 1
if len(size)==3:
    x,y,z = size

a,b = np.indices((x,y))
co = np.dstack((a,b))

spatial = co.reshape((x*y,2))
pixels = img_copy.reshape((x*y,z))
mat = np.concatenate((spatial,pixels),axis=1)


# Computes the kernel function for the epanechnikov kernel
def kernel_function(arr,h):
    return (0.75)*(1 - (arr/float(h))**2)

# Creates a lookup table for the epanechnikov kernel
def kernel_lookup(h):
    table = np.arange(h)
    vector_func = np.vectorize(kernel_function)
    lookup_table = vector_func(table,h)
    return lookup_table

def uniform_kernel(centroid,marked,points, bandwidth):
    distances = np.transpose(euclidean_distances(centroid,points)).reshape(len(points))
    distances = distances.astype('int64')
    incl = np.intersect1d(np.where(distances<bandwidth),np.where(marked==0))
    cand_points = points[incl]
    distances_new = distances[incl]    
    new_centroid  = np.sum(cand_points,axis=0)/len(cand_points)

    return new_centroid,incl

def epanechnikov_kernel(centroid,marked,points,hs,hr,lookup_hs,lookup_hr,z):
    lookup_table_hs = lookup_hs
    lookup_table_hr = lookup_hr
    diff = np.abs(np.subtract(points,centroid))

    hr_points = points[:,:z]
    hs_points = points[:,z:]

    x_hr = centroid[:z]
    x_hs = centroid[z:]
    hs_distances = np.transpose(euclidean_distances(x_hs,hs_points)).reshape(len(hs_points))
    hs_distances = hs_distances.astype('int64')
    
    incl = np.intersect1d(np.where(hs_distances<hs),np.where(marked==0))

    hr_distances = np.transpose(euclidean_distances(x_hr,hr_points)).reshape(len(hr_points))
    hr_distances = hr_distances.astype('int64')

    final = np.intersect1d(np.where(hr_distances<hr),incl)
    
    hr_points = hr_points[final]
    hs_points = hs_points[final] 
    hs_distances_new = hs_distances[final]
    hr_distances_new = hr_distances[final]

    hs_weights = lookup_table_hs[hs_distances_new]
    hs_weights = hs_weights.reshape((len(hs_weights),1))

    hr_weights = lookup_table_hr[hr_distances_new]
    hr_weights = hr_weights.reshape((len(hr_weights),1))

    hs_mean = np.sum(np.multiply(hs_points,hs_weights),axis=0)/np.sum(hs_weights)
    hr_mean = np.sum(np.multiply(hr_points,hr_weights),axis=0)/np.sum(hr_weights)

    new_centroid = np.concatenate((hr_mean,hs_mean))

    return new_centroid,final
    
def mean_shift():
    all_points = deepcopy(mat)
    remain_points = deepcopy(mat)

    # initial centroid chosen randomly
    centroid = all_points[random.randint(0,len(all_points)-1)]
    centroid = centroid.astype(np.float32)

    new_centroid = deepcopy(centroid)
    results = np.zeros(all_points.shape)

    start = time.time()
    prev_time = start
    lookup_hs = kernel_lookup(hs)
    lookup_hr = kernel_lookup(hr)

    a,b = all_points.shape
    marked = np.zeros(a)

    while len(remain_points)>0:
        for i in range(iterations):
            if kernel ==1:
                new_centroid,indices =epanechnikov_kernel(centroid,marked,all_points,hs,hr,lookup_hs,lookup_hr,z)
            else:
                new_centroid,indices = uniform_kernel(centroid,marked,all_points,h)
      
            dst = distance.euclidean(centroid,new_centroid)
            centroid = new_centroid  
            if dst<=convergence:
                break

        # extra contains unmarked indices that belong in the new cluster
        extra = np.intersect1d(np.where(marked==0),indices)
    
        # assigns all points in the cluster to their centroid/mean
        # marks all the newly clustered points
        marked[extra]=1
        results[extra] = centroid

        # remain_points contains all the unclustered points
        remain_points = all_points[marked==0]
        #print centroid
        if len(remain_points)>0:
            # picks a new random centroid from the remaining points
            centroid = remain_points[random.randint(0,len(remain_points)-1)]

        if len(extra) == 0:
            break
        current_time = time.time()
        if current_time-prev_time>1:
            print "Remaining Points: " + str(len(remain_points))
            prev_time = current_time

    res = results[:,2:]

    if z ==1:
        segmented_image = res.reshape(x,y)
    else:
        segmented_image = res.reshape((x,y,z))
 
    segmented_image = segmented_image.astype(np.uint8)
    runtime = current_time -start
    print "Total Runtime: "+ str(runtime)
    return segmented_image

if __name__ == "__main__":
    
    segmented_image = mean_shift()
    
    fig = plt.figure(figsize=(15,15))
    plot =plt.subplot(1,2,1)
    plot.set_title("Original Image")
    plot.imshow(img,cmap="gray")
    
    plot= plt.subplot(1,2,2)
    plot.set_title("Segmented Image")
    plot.imshow(segmented_image,cmap="gray")
    plt.show()