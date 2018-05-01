# coding: utf-8

# In[1]:


import mnist
import numpy as np

# In[2]:


import scipy.misc

##Get the first 20 training images from the library
images = mnist.train_images()[0:20]

# In[3]:


##Convert binary images {-1, 1}
bin_images = np.zeros(images.shape)
for w in range(images.shape[0]):
    for i in range(images.shape[1]):
        for j in range(images.shape[2]):
            if images[w, i, j] < 128:
                bin_images[w, i, j] = -1
            else:
                bin_images[w, i, j] = 1

##Show image
scipy.misc.toimage(bin_images[0])

# In[4]:


##Load up noise coordinates
import csv

with open('./example/NoiseCoordinates.csv') as f:
    reader = csv.reader(f)
    next(reader, None)
    mat = np.zeros((40, 15))
    for ind, row in enumerate(reader):
        if ind < 40:
            mat[ind] = np.array([num for num in row[1:]])
    mat = mat.astype(int)

# In[5]:


###Add noises to the images (just negate the value)
for i in range(20):
    for j in range(15):
        bin_images[i][mat[2 * i][j]][mat[2 * i + 1][j]] = -bin_images[i][mat[2 * i][j]][mat[2 * i + 1][j]]

# In[29]:


scipy.misc.toimage(bin_images[6])

# In[7]:


##Load up Update Order Coordinates
with open('./example/UpdateOrderCoordinates.csv') as f:
    reader = csv.reader(f)
    next(reader, None)
    updateOrderMat = np.zeros((40, 784))
    for ind, row in enumerate(reader):
        if ind < 40:
            updateOrderMat[ind] = np.array([num for num in row[1:]])
    updateOrderMat = updateOrderMat.astype(int)

##Load up Initial Parameters Model. (which is q in the instruction pdf or pi in the book)
with open('./example/InitialParametersModel.csv') as f:
    reader = csv.reader(f)
    InitialParaMat = np.zeros((28, 28))
    for ind, row in enumerate(reader):
        if ind < 28:
            InitialParaMat[ind] = np.array([num for num in row])


# In[37]:


##Conpute the energy. Formula is in the instruction pdf
def computeEnergy(q, X):
    thetaHH = 0.8
    thetaHX = 2
    eps = 10 ** (-10)
    eqlogq = 0
    for i in range(28):
        for j in range(28):
            eqlogq += q[i, j] * np.log(q[i, j] + eps) + (1 - q[i, j]) * np.log((1 - q[i, j] + eps))

    eqlogp = 0
    for i in range(28):
        for j in range(28):
            if i - 1 >= 0:
                eqlogp += thetaHH * (2 * q[i, j] - 1) * (2 * q[i - 1, j] - 1)
            if i + 1 < 28:
                eqlogp += thetaHH * (2 * q[i, j] - 1) * (2 * q[i + 1, j] - 1)
            if j - 1 >= 0:
                eqlogp += thetaHH * (2 * q[i, j] - 1) * (2 * q[i, j - 1] - 1)
            if j + 1 < 28:
                eqlogp += thetaHH * (2 * q[i, j] - 1) * (2 * q[i, j + 1] - 1)
            eqlogp += thetaHX * (2 * q[i, j] - 1) * X[i, j]
    return eqlogq - eqlogp


# In[53]:


imgsParaMat = np.zeros((20, 28, 28))
eMat = np.zeros((20, 11))
thetaHH = 0.8
thetaHX = 2

##Update the pi. Formula is in the book p263
for i in range(20):
    imgsParaMat[i, :] = InitialParaMat
    eMat[i, 0] = computeEnergy(imgsParaMat[i], bin_images[i])
    for ti in range(10):
        for j in range(784):
            row = updateOrderMat[2 * i, j]
            col = updateOrderMat[2 * i + 1, j]
            eSum = 0
            if row - 1 >= 0:
                eSum += thetaHH * (2 * imgsParaMat[i, row - 1, col] - 1)
            if row + 1 < 28:
                eSum += thetaHH * (2 * imgsParaMat[i, row + 1, col] - 1)
            if col - 1 >= 0:
                eSum += thetaHH * (2 * imgsParaMat[i, row, col - 1] - 1)
            if col + 1 < 28:
                eSum += thetaHH * (2 * imgsParaMat[i, row, col + 1] - 1)
            eSum += thetaHX * bin_images[i, row, col]
            numer = np.exp(eSum)
            denom = np.exp(eSum) + np.exp(-eSum)
            imgsParaMat[i, row, col] = numer / denom

        eMat[i, ti + 1] = computeEnergy(imgsParaMat[i], bin_images[i])

# In[65]:


np.savetxt("energy.csv", eMat[10:12, 0:2], delimiter=",")


# In[61]:


##Convert the final pi to binary image
def convertToBinary(im):
    retIm = np.zeros(im.shape)
    for i in range(im.shape[0]):
        for j in range(im.shape[1]):
            if im[i, j] >= 0.5:
                retIm[i, j] = 1
    return retIm


# In[72]:


x_rec = np.zeros((28, 280))
for i in range(10):
    x_rec[:, (i * 28):(i * 28 + 28)] = convertToBinary(imgsParaMat[i + 10])

# In[73]:


scipy.misc.toimage(x_rec)

# In[75]:


np.savetxt("denoised.csv", x_rec, fmt='%d', delimiter=",")

