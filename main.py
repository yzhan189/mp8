import mnist
import numpy as np
import scipy.misc
import plot as plt


# 1. Obtaining the dataset
train_images = mnist.train_images()
train_labels = mnist.train_labels()

test_images = mnist.test_images()
test_labels = mnist.test_labels()

# binarize the first 20 images
bi_images = np.zeros(train_images[0:20].shape)
for i in range(20):
    for j in range(28):
        for k in range(28):
            bi_images[i,j,k] = 1 if train_images[i,j,k]>=255*0.5 else -1

# scipy.misc.imsave('orginal.png', scipy.misc.toimage(train_images[13])  )
# scipy.misc.imsave('binaary.png', scipy.misc.toimage(bi_images[13])  )


# 2. Adding pre-determined noise to the dataset
with open('example/NoiseCoordinates.csv') as f:
    f.readline()

    for line in f:
        img_idx = int(line.split(' ')[1])
        row_indices = line.strip('\n').split(',')[1:]
        row_indices = [int(n) for n in row_indices]
        next_line = f.readline()
        col_indices = next_line.strip('\n').split(',')[1:]
        col_indices = [int(n) for n in col_indices]


        for t in range(len(row_indices)):
            i = row_indices[t]
            j = col_indices[t]
            #print(img_idx,i,j)
            bi_images[img_idx,i,j] = -bi_images[img_idx,i,j]
# scipy.misc.imsave('noise.png', scipy.misc.toimage(bi_images[13])  )


# 3. Building a Boltzman Machine for denoising the images and using Mean-Field Inference
# probability of Hi=1 matrix
initial_Q = []
with open('example/InitialParametersModel.csv') as f:
    for line in f:
        line = line.strip('\n').split(',')
        line = [float(n) for n in line]
        initial_Q.append(line)
initial_Q = np.array(initial_Q)
#print(initial_Q)

theta_HH = 0.8
theta_HX = 2
epsilon = 10 ** (-10)

def compute_EQ(Q,X):
    term1 = sum([ p*np.log(p+epsilon) + (1-p)*np.log((1-p)+epsilon) for p in Q.flat])

    term2 = 0 # neighbor
    for i in range(28):
        for j in range(28):
            # hidden neighbor
            if i - 1 >= 0:
                term2 += theta_HH * (2*Q[i,j]-1) * (2*Q[i-1,j]-1)
            if i + 1 < 28:
                term2 += theta_HH * (2*Q[i,j]-1) * (2*Q[i+1,j]-1)
            if j - 1 >= 0:
                term2 += theta_HH * (2*Q[i,j]-1) * (2*Q[i,j-1]-1)
            if j + 1 < 28:
                term2 += theta_HH * (2*Q[i,j]-1) * (2*Q[i,j+1]-1)
            # one observed neighbor
            term2 += theta_HX* (2*Q[i,j]-1)*X[i,j]
    EQ = term1-term2
    return EQ

def update_Q(Q,X,i,j):
    term1 = 0 # H=1
    term2 = 0 # H=-1

    # hidden neighbor
    if i - 1 >= 0:
        term1 += theta_HH * (2 * Q[i - 1, j] - 1)
        term2 += (-theta_HH) * (2 * Q[i - 1, j] - 1)
    if i + 1 < 28:
        term1 += theta_HH * (2 * Q[i + 1, j] - 1)
        term2 += (-theta_HH) * (2 * Q[i + 1, j] - 1)
    if j - 1 >= 0:
        term1 += theta_HH * (2 * Q[i, j - 1] - 1)
        term2 += (-theta_HH)* (2 * Q[i, j - 1] - 1)
    if j + 1 < 28:
        term1 += theta_HH * (2 * Q[i, j + 1] - 1)
        term2 += (-theta_HH)* (2 * Q[i, j + 1] - 1)
    # one observed neighbor
    term1 += theta_HX * X[i, j]
    term2 += (-theta_HX) * X[i, j]

    term1 = np.exp(term1)
    term2 = np.exp(term2)

    pi = term1/(term1+term2)
    return pi


# 4. Turning in the energy function values computed initially and after each iteration\
# 5. Displaying the reconstructed images

EQ_mat = np.zeros((20,11))
sample = np.zeros((10,28,28)) # 0-9
denoised = np.zeros((10,28,28)) # 10-19
with open('example/UpdateOrderCoordinates.csv') as f:
    f.readline()
    for line in f:
        img_idx = int(line.split(' ')[1])
        row_indices = line.strip('\n').split(',')[1:]
        row_indices = [int(n) for n in row_indices]
        next_line = f.readline()
        col_indices = next_line.strip('\n').split(',')[1:]
        col_indices = [int(n) for n in col_indices]


        Q = np.array(initial_Q, copy=True)
        img = bi_images[img_idx]
        EQ_mat[img_idx,0] = compute_EQ(Q,img) # initial energy!!

        for iteration in range(10): # should be 10
            # for each pixel
            for t in range(len(row_indices)):
                i = row_indices[t]
                j = col_indices[t]
                # updating pi
                Q[i, j] = update_Q(Q,img,i,j)
            # after each iteration (+1 because 0 for initial)
            EQ_mat[img_idx,iteration+1] = compute_EQ(Q,img)

        # print(Q)
        # final Q
        if img_idx <10:
            for i in range(28):
                for j in range(28):
                    sample[img_idx,i,j] = 1 if Q[i,j] >= 0.5 else 0
        else:
            for i in range(28):
                for j in range(28):
                    denoised[img_idx-10,i,j] = 1 if Q[i,j] >= 0.5 else 0

np.savetxt("energy.csv", EQ_mat, delimiter=",")

sample_out = np.zeros((28,280))
for i in range(10):
    sample_out[:,(i*28):(i*28+28)] = sample[i]
np.savetxt("sample.csv", sample_out, fmt='%d', delimiter=",")

denoised_out = np.zeros((28,280))
for i in range(10):
    denoised_out[:,(i*28):(i*28+28)] = denoised[i]
np.savetxt("denoised.csv", denoised_out, fmt='%d', delimiter=",")
# 6. Construction of an ROC curve