#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 14 17:19:19 2016

@author: renanfagundes

Sera usado o codigo de exemplo do python para fazer o trackyng
Primeiro com o CAMSHIFT
correção da condição inicial
Filtro Kalman corrigido
Variaveis de estado 
x, y, vx, vy, W, H,
Variaveis de saida
x, y,  vx, vy, w,h

O filtro Kalman faz o track, ou seja, determina a 
região que sera buscada a face no proximo quadro
"""

#!/usr/bin/env python

import numpy as np
import cv2
import cv2.cv as cv
from video import create_capture
from common import clock, draw_str

help_message = '''
USAGE: facedetect.py [--cascade <cascade_fn>] [--nested-cascade <cascade_fn>] [<video_source>]
'''

def detect(img, cascade):
    rects = cascade.detectMultiScale(img, scaleFactor=1.3, minNeighbors=4, minSize=(30, 30), flags = cv.CV_HAAR_SCALE_IMAGE)
    if len(rects) == 0:
        return []
    rects[:,2:] += rects[:,:2]
    return rects

def draw_rects(img, rects, color):
    for x1, y1, x2, y2 in rects:
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

#############Definições do Filtro kalman###########################
#dimensôes do sistema
m = 0 #numero de entradas
n = 6 #numero de estados
r = 6 #numero de saidas
kdt = 1
A = np.zeros((n, n))
A = np.matrix([[1, 0, 1, 0, 0, 0],[0, 1, 0, 1, 0, 0],[0, 0, 1, 0, 0, 0],[0, 0, 0, 1, 0, 0], [0, 0, 0, 0, 1, 0],[0, 0, 0, 0, 0, 1] ])
#A = np.matrix([[1, 0, 0, 0, 0, 0],[0, 1, 0, 0, 0, 0],[0, 0, 1, 0, 0, 0],[0, 0, 0, 1, 0, 0], [0, 0, 0, 0, 1, 0],[0, 0, 0, 0, 0, 1] ])
#Matri B
#B = np.zeros((n, m))
#Matriz de entradas
#u = np.zeros((m, 1))
# Matrix C
H = np.eye(n)
#H = np.matrix([[1, 0, 0, 0, 0, 0],[0, 1, 0, 0, 0, 0],[0, 0, 0, 0, 1, 0],[0, 0, 0, 0, 0, 1]])
#Vetor de saida
#y_out = np.zeros((r, 1))
#Vetor de entrada #O vetor de entrada é nulo nesse Processo
#u = np.zeros((m, 1))
#Matriz de ruido sw nedida
#sz = np.zeros((r, r)) #=R
#Matriz de ruido do sistema
#sw = np.zeros((n, n)) #=Q  
###### Ruidos####      
Z = np.zeros((r, 1))
Z = np.matrix([[0.005],[0.005],[0.005],[0.005],[0.005],[0.005]])
W = np.zeros((n, 1))
#por enquanto vou chutar diretos as matrizes, e depois eu penso melhor
Q =  (1e-4)* np.identity(n) # teentei com 1   (1e-1)
R =  (1e-4) * np.identity(r)    # tentei com 10000   (1e-5)

### funções do filtro kalman#######
def predict(x_est, p_est):
#    state_cov_prd = []
    # Predicted state and covariance
    x_prd = A * x_est#+ B*u deixa comentato por enqunto, porque não tem nem B nem u
    p_prd = A * p_est * A.transpose() + Q
    #state_cov_prd.append((x_prd, p_prd)) 
    return [x_prd, p_prd]

def update(x_prd, p_prd, Z):
    #S = H * p_prd.transpose() * H.transpose() + R
    S = H * p_prd * H.transpose() + R
    #L = H * p_prd.transpose()
    L = p_prd* H.transpose()
    klm_gain =  L * np.linalg.inv(S) 
    klm_gain = klm_gain#.transpose()
    x_est = x_prd + klm_gain * (Z - H * x_prd)
    p_est = p_prd - klm_gain * H * p_prd
    #state_cov_est.append((x_est, p_est)) 
    return [x_est, p_est]

##########Definições do CAMSHIFT##########################################
#Enter the number of frames you want to track the face after detection
TRACK = 1
# initialize the termination criteria for CAMSHIFT, indicating
# a maximum of ten iteRATIOns or movement by a least one pixel
# along with the bounding box of the ROI
termination = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)

#Track the faces found using CAMSHIFT algorithm
def trackFace(allRoiPts, allRoiHist):     
    pts =[]
    global orig
    for k in range(0, TRACK):

        #convert the given frame to HSV color space
        hsv = cv2.cvtColor(orig, cv2.COLOR_BGR2HSV)
#        i=0
        #For histogram of each window found, back project them on the current frame and track using CAMSHIFT
#        for roiHist in allRoiHist:            
        # Perform mean shift            
        backProj = cv2.calcBackProject([hsv], [0], allRoiHist, [0, 180], 1)
        # Apply cam shift to the back projection, convert the
        # points to a bounding box, and then draw them            
        #temp = allRoiPts[i]
        (r, allRoiPts) = cv2.CamShift(backProj, allRoiPts, termination)  
        #Error handling for bound exceeding 
#            for j in range(0,4):         
#                if allRoiPts[i][j] < 0:
#                    allRoiPts[i][j] = 0
        pts = np.int0(cv2.cv.BoxPoints(r))        
        #Draw bounding box around new position of the object
#        cv2.polylines(orig, [pts], True, (0, 0, 255), 2)
#            i = i + 1



#        #show the face on the frame
#        cv2.imshow("Faces", orig)
#        cv2.waitKey(1)
    return pts

def calHist(allRoiPts):
    global orig
    allRoiHist = []    
    #For each face found, convert it to HSV and calculate the histogram of
    #that region                                
    for roiPts in allRoiPts:                        
        # Grab the ROI for the bounding box by cropping and convert it
        # to the HSV color space.
        roi = orig[roiPts[1]:roiPts[3], roiPts[0]:roiPts[2]]            
        roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)            

        # compute a HSV histogram for the ROI and store the
        # bounding box
        roiHist = cv2.calcHist([roi], [0], None, [16], [0, 180])
        roiHist = cv2.normalize(roiHist, roiHist, 0, 255, cv2.NORM_MINMAX)
        allRoiHist.append(roiHist);

    return roiHist
    

def clamp(n, minn, maxn):
    return max(min(maxn, n), minn)
    

if __name__ == '__main__':
    import sys, getopt
#    print help_message

    args, video_src = getopt.getopt(sys.argv[1:], '', ['cascade=', 'nested-cascade='])
    try: video_src = video_src[0]
    except: video_src = 0
    args = dict(args)
    #cascade_fn = args.get('--cascade', "../../data/haarcascades/haarcascade_frontalface_default.xml")
    cascade_fn = args.get('--cascade', "haarcascade_frontalface_default.xml")

    cascade = cv2.CascadeClassifier(cascade_fn)

#    cam = create_capture(0)
    cam = cv2.VideoCapture(0)
    x_est = np.zeros((n, 1))
    p_est = np.identity(n)
    w = 0
    h = 0
    allRoiHist = [] 
    frist = 0
    count =0
    global orig
    while(frist == 0): 
        ret, img = cam.read()
        orig = vis = img.copy()
        gray = cv2.cvtColor(vis, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        rects = detect(gray, cascade)
        if(format(len(rects)) == '1'):
            frist = 1
            w = rects[0,2]-rects[0,0]
            h = rects[0,3]-rects[0,1]
            h = rects[0,3]-rects[0,1]
            x = rects[0,0] #x atual
            y = rects[0,1]
            Z[0,0] = rects[0,0] #x atual
            Z[1,0] = rects[0,1]
            Z[2,0] = rects[0,0] #x anterior
            Z[3,0] = rects[0,1]
            Z[4,0] = w
            Z[5,0] = h

            allRoiHist = calHist(rects)
        cv2.imshow('facedetect', vis)
   
    print('aki começou')
    while (cam.isOpened()): 
        ret, img = cam.read()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)

        t = clock()
        yi = clamp(y-h, 0, 480)
        yf = clamp(y+2*h, 0, 480)
        xi = clamp(x-w, 0, 640)
        xf = clamp(x+2*w, 0, 640)
        print(yi,yf,xi,xf)
        if(count >= 3):
            yi = 0
            yf = 480
            xi = 0
            xf = 640
        rects = detect(gray[yi:yf,xi:xf], cascade)
        aux =  gray[yi:yf,xi:xf]
        cv2.imshow('aux1',aux)
        
#        rects = detect(gray[y-w:y+w, x-h: x+h], cascade)
        #print("Found {0} faces!".format(len(rects)))
        if(format(len(rects)) == '1'):     

            vis = img.copy()
#            draw_rects(vis, rects, (0, 255, 0))
            w = rects[0,2]-rects[0,0]
            h = rects[0,3]-rects[0,1]
            x = rects[0,0] + xi#x atual
            y = rects[0,1] + yi
            Z[5,0] = h
            Z[4,0] = w
            Z[3,0] = rects[0,1] + yi - Z[1,0] # velocidade é sf - si
            Z[2,0] = rects[0,0] + xi - Z[0,0]
            Z[1,0] = rects[0,1] + yi
            Z[0,0] = rects[0,0] + xi
#            draw_rects(vis, (x, y, x+w, 
            cv2.rectangle(vis, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
            
            ##chama o Filtro de Kalaman com os pontos achados pelo VJ
            [x_prd, p_prd] = predict(x_est, p_est)
            [x_est, p_est] = update(x_prd, p_prd, Z)
            
            # Calcula o histogram da região da face, 
            # para usar depois no CAMSHIFT 
#            orig = vis.copy()
#            allRoiHist = calHist(rects)
            
            cv2.rectangle(vis, (int(x_est[0,0]), int(x_est[1,0])), (w+int(x_est[0,0]), h + int(x_est[1,0])), (255,0,0), 2)

#            print(rects)
#            print(x_est.transpose())
#            print(x_prd.transpose())
            
            count = 0            

            dt = clock() - t
            draw_str(vis, (20, 20), 'time: %.1f ms' % (dt*1000))
            cv2.imshow('facedetect', vis)
            
        elif(format(len(rects)) == '0'): 
            vis = img.copy()
            orig = vis
#            x_est[0,0] = rects[0,0]
#            x_est[1,0] = rects[0,1]
            
            
            pts = trackFace( (int(x_est[0,0]), int(x_est[1,0]), w, h), allRoiHist )
            cv2.polylines(vis, [pts], True, (0, 0, 255), 2)
            #chama o Filtro de Kalaman com os pontos anteriores do FK

#            h = np.amax(pts[:,0]) - np.amin(pts[:,0])
#            w = np.amax(pts[:,1]) - np.amin(pts[:,1])
            x = int((pts[0,0]+pts[1,0]+pts[2,0]+pts[3,0])/4 - w/2)
            y = int((pts[0,1]+pts[1,1]+pts[2,1]+pts[3,1])/4 - h/2)
            
            Z[5,0] = h
            Z[4,0] = w
            Z[3,0] = y - Z[1,0] # velx velocidade é sf - si
            Z[2,0] = x - Z[0,0] #Vel y
            Z[1,0] = y
            Z[0,0] = x
            
            [x_prd, p_prd] = predict(x_est, p_est)
            [x_est, p_est] = update(x_prd, p_prd,Z)
            

            cv2.rectangle(vis, (int(x_est[0,0]), int(x_est[1,0])), (w+int(x_est[0,0]), h + int(x_est[1,0])), (255,0,0), 2)
            
            count = count + 1
            print(count)

                
            
            
            dt = clock() - t
            draw_str(vis, (20, 20), 'time: %.1f ms' % (dt*1000))
            cv2.imshow('facedetect', vis)
        
        #print(rects)
        if 0xFF & cv2.waitKey(5) == 27:
            break
    cv2.destroyAllWindows()

