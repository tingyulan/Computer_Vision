#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import cv2
from matplotlib import pyplot as plt
import random
from mpl_toolkits.mplot3d import Axes3D


# In[2]:


#FOR IMAGE 1
'''
K1 = np.array([[1421.9, 0.5, 509.2],[ 0, 1421.9, 380.2],[ 0, 0, 1]])
K2 = np.array([[1421.9, 0.5, 509.2],[ 0, 1421.9, 380.2],[ 0, 0, 1]])
'''

#FOR IMAGE 2
# #CAMERA A
K1 = np.array([[5426.566895, 0.678017, 330.096680],
                 [0.000000, 5423.133301, 648.950012],
                 [0.000000,  0.000000, 1.000000]])
R1 = np.array([[0.140626, 0.989027, -0.045273],
                 [0.475766, -0.107607, -0.872965],
               [-0.868258, 0.101223, -0.485678]])
t1 = np.array([[67.479439, -6.020049, 40.224911]])

# #CAMERA B
K2 = np.array([[5426.566895, 0.678017, 387.430023],
                 [0.000000, 5423.133301, 620.616699],
                 [0.000000, 0.000000, 1.000000]])
#R2 = np.array([[0.336455, 0.940689, -0.043627],
#                 [0.446741, -0.200225, -0.871970]
#                 [-0.828988, 0.273889, -0.487611]])
# t2 = np.array([[62.882744, -21.081516, 40.544052]])



# In[3]:


def SIFT(img1_path, img2_path):
    img1 = cv2.imread(img1_path) # queryImage
    img2 = cv2.imread(img2_path) # trainImage
    gray1 = cv2.imread(img1_path,0)
    gray2 = cv2.imread(img2_path,0)
    while(img2.shape[0] > 1000):
        if img1.shape == img2.shape:
            img1 = cv2.resize(img1,None,fx=0.5, fy=0.5)
            gray1 = cv2.resize(gray1,None,fx=0.5, fy=0.5)
        img2 = cv2.resize(img2,None,fx=0.5, fy=0.5)
        gray2 = cv2.resize(gray2,None,fx=0.5, fy=0.5)

    # Initiate SIFT detector
    sift = cv2.xfeatures2d.SIFT_create()
    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(gray1,None)
    kp2, des2 = sift.detectAndCompute(gray2,None)
    return img1, img2, kp1, kp2, des1, des2

def match(img1_path, img2_path, ratio = 0.5, show = False):
    img1, img2, kp1, kp2, des1, des2 = SIFT(img1_path, img2_path)

    match = []
    for i in range(des1.shape[0]):
#         m1, m2 = 5000000, 6000000
#         for j in range(len(features_2)):
#             # Find 2 nearest matches for each point in features_1
#             tmp = np.linalg.norm(features_1[i]-features_2[j])
#             if(tmp<m1):
#                 m1, m2 = tmp, m1
#                 m1_index = j
#             elif(tmp<m2):
#                 m2 = tmp
#         if(m1<m2*0.75):
#             matches.append((m1_index, i))
#             # Test for OpenCV fundamental matrix
#             pts1.append(keypoints_1[i])
#             pts2.append(keypoints_2[m1_index])
        
        des1_ = np.tile(des1[i], (des2.shape[0], 1))
        error = des1_ - des2
        SSD = np.sum((error**2), axis=1)
        idx_sort = np.argsort(SSD)
        if SSD[idx_sort[0]] < ratio * SSD[idx_sort[1]]:
            match.append([kp1[i].pt, kp2[idx_sort[0]].pt])
    line = np.array(match)
    
    #draw result after match
    if show:
        print('match points with ratio = {}'.format(ratio))
        keyPoint1, keyPoint2 = line[:, 0], line[:, 1]
        showMatch(img1, img2, keyPoint1, keyPoint2)
    return line, img1, img2

def showMatch(img1, img2, kp1, kp2):
    kp1 = np.array(kp1)
    kp2 = np.array(kp2)
    line = np.zeros((kp1.shape[0], 2, 2))
    for n in range(kp1.shape[0]):
        line[n, :, :] = np.vstack((kp1[n], kp2[n]))

    plt.figure(figsize=(10,10))
    line_tran = np.transpose(line, axes=(0, 2, 1)) #(x1,y1),(x2,y2) -> (x1,x2),(y1,y2)
    if img1.shape[0] != img2.shape[0]:
        img2 = np.vstack([img2, np.full((np.abs(img1.shape[0] - img2.shape[0]), img2.shape[1], 3), 255)])
    imgStack = np.hstack([img1[:,:,::-1], img2[:,:,::-1]])
    for i in range(line.shape[0]):
        color = np.random.rand(3)
        plt.scatter(line[i][0][0], line[i][0][1],c = 'r')
        plt.scatter(line[i][1][0] + img1.shape[1], line[i][1][1],c = 'r')
        plt.plot(line_tran[i][0] + [0, img1.shape[1]], line_tran[i][1], color = color)
    plt.xlim((0, img1.shape[1] + img2.shape[1]))
    plt.ylim((img1.shape[0], 0))
    plt.imshow(imgStack)
    plt.show()


# In[4]:


def normalize(pts_1, pts_2):  #kp1=(8,2)
    x = np.zeros(shape = (3,1))
    x[2,0] = 1
    for i in range(8):
        x[0,0] = pts_1[i,0]
        x[1,0] = pts_1[i,1]
        pts_1[i,0] = np.dot(T1, x)[0,0]
        pts_1[i,1] = np.dot(T1, x)[1,0]

        x[0,0] = pts_2[i,0]
        x[1,0] = pts_2[i,1]
        pts_2[i,0] = np.dot(T2, x)[0,0]
        pts_2[i,1] = np.dot(T2, x)[1,0]
    return pts_1, pts_2

def affinetransform(p1, p2):
    k2 = p2.T
    p1_homo = np.ones((3,3))
    p1_homo[0:2,:]=p1.T
    p1_inv = np.linalg.inv(p1_homo)
    M = k2.dot(p1_inv)
    return M
    
def ransac(img1, img2, line,error_threshold = 2, show=False):
    better_kp1 = []
    better_kp2 = []
    kp1, kp2 = line[:, 0], line[:, 1]
    iterations = 3000
    maxinlier = 0 #correspond keypoint counter
    kp_num = len(kp1)   
    
    for i in range(iterations):
        kp1_rand = np.zeros((8,2), dtype="f")
        kp2_rand = np.zeros((8,2), dtype="f")

        for j in range(8):
            rand = np.random.randint(0,kp_num-1)
            kp1_rand[j,0]=kp2[rand][0]
            kp1_rand[j,1]=kp2[rand][1]
            kp2_rand[j,0]=kp1[rand][0]
            kp2_rand[j,1]=kp1[rand][1]
            
        pts_1, pts_2 = normalize(kp1_rand, kp2_rand)

        F = findF(pts_1, pts_2)
    
        pts_tmp1,pts_tmp2,inlier = Inliernum(kp2, kp1, F)
        if inlier >= maxinlier:
            maxinlier = inlier
            better_kp1 = pts_tmp1
            better_kp2 = pts_tmp2
            Fstore = F
        
    if show:
        print('match points after RANSAC with threshold = {}'.format(error_threshold))
        showMatch(img1, img2, np.array(better_kp1), np.array(better_kp2))

    return better_kp1, better_kp2, Fstore


# In[5]:


def findF(kp1, kp2):
    A = []
    for j in range (len(kp1)):
        x1 = kp1[j][0]
        y1 = kp1[j][1]
        x2 = kp2[j][0]
        y2 = kp2[j][1]
        A.append(np.asarray([x2*x1, x2*y1, x2, y2*x1, y2*y1, y2, x1, y1, 1]))
    
    _,_,Vt= np.linalg.svd(A)
    Fi = Vt[-1].reshape(3,3)
    U,S,V=np.linalg.svd(Fi)
    S1 = np.zeros((3,3))
    S1[0,0] = S[0]
    S1[1,1] = S[1]
    F = (U.dot(S1)).dot(V)
    F = np.dot( np.transpose(T2),np.dot(F, T1))
    F/=F[2,2]

    #THE "F" IS DENORMALIZED F
    return F

def Inliernum(pts1, pts2, F):
    num = pts1.shape[0]
    inlier = 0
    thDist=2
    pts_tmp1 = []
    pts_tmp2 = []
    for i in range(num):
        x1 = pts1[i,0]
        y1 = pts1[i,1]
        x2 = pts2[i,0]
        y2 = pts2[i,1]
#         [a, b, c] = np.dot(F, np.array([x1, y1, 1]))
        a = F[0,0]*x1 + F[0,1]*y1 + F[0,2]
        b = F[1,0]*x1 + F[1,1]*y1 + F[1,2]
        c = F[2,0]*x1 +F[2,1]*y1 + F[2,2]
        dist = abs(a*x2 + b*y2 + c)/((a**2 + b**2)**(0.5))
        
        if (dist < 1):
            pts_tmp1.append([x1,y1]);
            pts_tmp2.append([x2,y2]);
            inlier = inlier + 1
    return pts_tmp1,pts_tmp2,inlier


# In[6]:


def drawlines(img1, img2, lines, pts1, pts2):
#     img1 - image on which we draw the epilines for the points in img2
#     lines - corresponding epilines
    r, c, channel = img1.shape
#     lines = cv2.computeCorrespondEpilines(pts2[:,:2].reshape(-1,1,2), 2,F)
#     lines = lines.reshape(-1,3)
    for line, pt1, pt2 in zip(lines.T, pts1, pts2):
        color = tuple(np.random.randint(0, 255, 3).tolist())
        x0, y0 = map(int, [0, -(line[2] + line[0] * 0) / line[1] ])
        x1, y1 = map(int, [c, -(line[2] + line[0] * (c)) / line[1] ])

        img2 = cv2.line(img2, (x0, y0), (x1, y1), color, 1)
        img1 = cv2.circle(img1, tuple(pt1[:2]), 5, color, -1)
        img2 = cv2.circle(img2, tuple(pt2[:2]), 5, color, -1)
    return img1, img2


# In[7]:


def linear_triangulation(p1, p2, m1, m2):
    A = np.zeros((4,4))
    A[0,:] = p1[0]*m1[2,:]-m1[0,:]
    A[1,:] = p1[1]*m1[2,:]-m1[1,:]
    A[2,:] = p2[0]*m2[2,:]-m2[0,:]
    A[3,:] = p2[1]*m2[2,:]-m2[1,:]
#     print("A: ", A)
    U, S, V = np.linalg.svd(A)
    X = V[-1]/V[-1,3]
    return X


# In[8]:


#img1_path = 'Mesona1.JPG'
#img2_path = 'Mesona2.JPG'
img1_path = 'Statue1.bmp'
img2_path = 'Statue2.bmp'
line, img1, img2 = match(img1_path, img2_path, ratio = 0.65, show = False)
# line, img1, img2 = match(img2_path, img1_path, ratio = 0.75, show = False)
kp1, kp2 = line[:, 0], line[:, 1]

h1,w1,_=img1.shape
h2,w2,_=img2.shape
T1 = np.array([[2.0/w1, 0, -1], [ 0, 2/h1, -1], [ 0,  0, 1.0]])
T2 = np.array([[2.0/w2, 0, -1], [ 0, 2/h2, -1], [ 0,  0, 1.0]])

better_kp1, better_kp2 , F= ransac(img1, img2, line, error_threshold = 2, show = False)
#better_kp2, better_kp1 , F= ransac(img1, img2, line, error_threshold = 2, show = False)


# In[9]:


F_truth, mask = cv2.findFundamentalMat(kp1, kp2, cv2.FM_8POINT + cv2.FM_RANSAC)
print ("F_truth: ", F_truth )


# In[10]:


#OUR F
print("F: ", F)
#F = F_truth
#CORRECT F
#F = np.array([[ 2.03888664e-07,  5.35681524e-07, -2.75574624e-03],
 #              [-1.90835466e-06, -4.08070635e-08,  2.10520438e-02],
  #             [ 1.60370178e-03, -1.95712042e-02,  1.00000000e+00]])

# # given parameter


# In[11]:


pts1 = np.array(better_kp1).astype(np.int32)
pts1 = np.concatenate((pts1, np.ones(pts1.shape[0]).reshape(-1, 1)), axis=1).astype(np.int32)
pts2 = np.array(better_kp2).astype(np.int32)
pts2 = np.hstack((pts2, np.ones(pts2.shape[0]).reshape(-1, 1))).astype(np.int32)
# print(pts1)


# In[12]:


E = np.dot(K1.T, np.dot(F, K2))
print("E: ",E);
l = F.dot(pts1.T)

output1, output2 = drawlines(img1, img2,l, pts1, pts2)


# cv2.imshow('output1', output1)
# cv2.imshow('output2', output2)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


# In[13]:


better_kp1 = np.concatenate((better_kp1, np.zeros(pts1.shape[0]).reshape(-1, 1)), axis=1)
de_kp1 = np.float32(better_kp1)
de_kp2 = np.float32(better_kp2)
de_kp1 = de_kp1[np.newaxis, :]
# de_kp2 = de_kp2[np.newaxis, np.newaxis,:]
de_kp2 = de_kp2[np.newaxis,:]
# print(de_kp1)
img_size = (img1.shape[1], img1.shape[0])
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(de_kp1, de_kp2, img_size,None, None)
#de_kp1= np.int32(de_kp1)
#de_kp2= np.int32(de_kp2)
#print(mtx)


# In[14]:


U, S, V = np.linalg.svd(E)
m = (S[0]+S[1])/2
E = np.dot(U, np.dot(np.diag([m, m, 0]), V))
U, S, V = np.linalg.svd(E)    
W = np.array([[0,-1,0],[1,0,0],[0,0,1]])

u3 = U[np.newaxis,:, 2].T


# In[15]:


solutions = []
solutions.append(np.vstack((np.dot(U, np.dot(W, V)).T, U[:,2])).T)
solutions.append(np.vstack((np.dot(U, np.dot(W, V)).T, -U[:,2])).T)
solutions.append(np.vstack((np.dot(U, np.dot(W.T, V)).T, U[:,2])).T)
solutions.append(np.vstack((np.dot(U, np.dot(W.T, V)).T, -U[:,2])).T)
print("solutions: ", solutions)


# In[16]:


# solutions = []
# print(solutions)

# solutions.append(np.array([[ 0.99496638, -0.08559688,  0.05210641, -0.92815427],
#        [ 0.08534307,  0.99632646,  0.00708079, -0.08777602],
#        [-0.05252109, -0.00259822,  0.99861643,  0.36169742]]))
# solutions.append(np.array([[ 0.99496638, -0.08559688,  0.05210641,  0.92815427],
#        [ 0.08534307,  0.99632646,  0.00708079,  0.08777602],
#        [-0.05252109, -0.00259822,  0.99861643, -0.36169742]]))
# solutions.append(np.array([[ 0.76847125,  0.10220385, -0.63166946, -0.92815427],
#        [ 0.08142613, -0.99475593, -0.06189036, -0.08777602],
#        [-0.63468238, -0.00387344, -0.7727634 ,  0.36169742]]))
# solutions.append(np.array([[ 0.76847125,  0.10220385, -0.63166946,  0.92815427],
#        [ 0.08142613, -0.99475593, -0.06189036,  0.08777602],
#        [-0.63468238, -0.00387344, -0.7727634 , -0.36169742]]))
# print(solutions)


# In[17]:


x1=de_kp1[0,:,0:2]
x2=de_kp2[0,:,:]
np.savetxt('2D.txt',x1)
# print("-----------------x1----------------")
# print(len(x1))
# print("x1: ", x1)
# print("----------------END x1-------------")

P1 = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0]])
P2_right = solutions[0]
#P1 = np.dot(K2,P1)
#P1 = np.concatenate((R1,np.dot(R1, t1.T)), axis=1)
print("P1 ",P1)
P1 = np.dot(K2,P1)

max_pt = 0
for i in range(4):
    print("i=", i)
#     P2 = solutions
#     P2 = solutions[i]
    P2 = np.dot(K1,solutions[i])
    print ("P2 = ")
    print (P2)
    count = 0
    for j in range(len(x1)):
        x = linear_triangulation(x1[j],x2[j],P1,P2)
        v = np.dot(x,P2.T)
        if v[2]>0:
            count += 1

    print ("count = ",count)
    if count > max_pt:
        print ("max = ",i)
        max_pt = count
        
        P2_right = P2
print("P2_right")
print(P2_right)
X = []
for j in range(len(x1)):
    X.append(linear_triangulation(x1[j],x2[j],P1,P2_right)[0:3])

ax = plt.subplot(111, projection='3d') 
X = np.array(X)
np.savetxt('3D.txt',X)
x,y,z = X[:,0],X[:,1],X[:,2]
ax.scatter(x,y,z)
plt.show()


# In[ ]:





# In[ ]:




