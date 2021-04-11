# Counts number of dumbbell curls in the video 

import cv2 
import mediapipe as mp 
import base
import math
import numpy as np

class PoseEstimator(base.PoseDetector): 
    def __init__(self, mode=False, upperBody = False, smooth=True, detectConf=.5, trackConf=.5):
        super().__init__(mode, upperBody, smooth, detectConf, trackConf)
        self.count = 0
        self.dir = 0

    def findAngle(self, img, p1, p2, p3, draw=True): 
        x1,y1 = self.lms[p1][1:]
        x2,y2 = self.lms[p2][1:]
        x3,y3 = self.lms[p3][1:]

        angle = math.degrees(math.atan2(y3-y2,x3-x2) - math.atan2(y1-y2,x1-x2))
        if angle<0: 
            angle += 360

        if draw: 
            cv2.line(img, (x1,y1), (x2,y2), (255,255,255) ,2)
            cv2.line(img, (x3,y3), (x2,y2), (255,255,255) ,2)
            cv2.circle(img, (x1,y1), 8, (0,0,255), cv2.FILLED)
            cv2.circle(img, (x1,y1), 12, (0,0,255), 2)
            cv2.circle(img, (x2,y2), 8, (0,0,255), cv2.FILLED)
            cv2.circle(img, (x2,y2), 12, (0,0,255), 2)
            cv2.circle(img, (x3,y3), 8, (0,0,255), cv2.FILLED)
            cv2.circle(img, (x3,y3), 12, (0,0,255), 2)
            cv2.putText(img, str(int(angle)), (x2-40,y2+50), cv2.FONT_HERSHEY_PLAIN, 2, (255,0,255), 2)

        return angle 
    
    def countReps(self, img, p1, p2, p3): 
        angle = self.findAngle(img, p1, p2, p3) 
        perc = np.interp(angle, (210,310), (0,100))
        
        color = (0,255,0)
        if perc == 100: 
            color = (0,0,255)
            if self.dir == 0: 
                self.count += .5 
                self.dir = 1
        if perc == 0: 
            color = (255,0,0)
            if self.dir == 1: 
                self.count += .5
                self.dir = 0 
        cv2.putText(img, f'{int(self.count)}', (99,99), cv2.FONT_HERSHEY_PLAIN, 9, (255,0,0), 3)

        bar = np.interp(perc, (0,100), (650,100))
        cv2.rectangle(img, (1100,100), (1175,650), color, 3)
        cv2.rectangle(img, (1100,int(bar)), (1175,650), color, cv2.FILLED)
        cv2.putText(img, f'{int(perc)}%', (1100,75), cv2.FONT_HERSHEY_PLAIN, 4, (255,0,0), 4)


cap = cv2.VideoCapture("media/1.mp4") 
estimator = PoseEstimator()

while True: 
    _, img = cap.read()
    img = cv2.resize(img, (1280,768)) 

    img = estimator.findPose(img) 
    lms = estimator.findPosition(img, draw=False) 
    if len(lms)>28: 
        estimator.countReps(img,11,13,15)

    cv2.imshow("Correct Pose Estimation", img)
    cv2.waitKey(1)