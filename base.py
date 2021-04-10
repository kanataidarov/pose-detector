import cv2 
import mediapipe as mp 
import time 

class PoseEstimator():
    def __init__(self, mode=False, upperBody = False, smooth=True, detectConf=.5, trackConf=.5): 
        self.mode = mode 
        self.upperBody = upperBody
        self.smooth = smooth
        self.detectConf = detectConf
        self.trackConf = trackConf
        
        self.cTime = 0
        self.pTime = 0
        self.mpPose = mp.solutions.pose
        self.pose = self.mpPose.Pose(self.mode, self.upperBody, self.smooth, self.detectConf, self.trackConf)
        self.mpDraw = mp.solutions.drawing_utils

    def fps(self, img): 
        self.cTime = time.time() 
        fps = 1/(self.cTime-self.pTime)
        self.pTime = self.cTime 
        cv2.putText(img, str(int(fps)), (50,80), cv2.FONT_HERSHEY_PLAIN, 5, (255,0,0), 5)

    def findPose(self, img, draw=True): 
        imgRgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.pose.process(imgRgb)
        if self.results.pose_landmarks and draw: 
            self.mpDraw.draw_landmarks(img, self.results.pose_landmarks, self.mpPose.POSE_CONNECTIONS)
        return img 

    def findPosition(self, img, draw=True): 
        lms = [] 
        if self.results.pose_landmarks: 
            for id, lm in enumerate(self.results.pose_landmarks.landmark): 
                h,w,c = img.shape 
                cx, cy = int(lm.x * w), int(lm.y * h)
                lms.append([id, cx, cy])
                if draw: 
                    cv2.circle(img, (cx,cy), 9, (255,0,0), cv2.FILLED)
        return lms 
        

cap = cv2.VideoCapture('media/1.mp4')

def main():
    estimator = PoseEstimator()

    while True: 
        success, img = cap.read() 
        estimator.fps(img) 

        img = estimator.findPose(img)
        lms = estimator.findPosition(img, draw=False)
        if len(lms) > 0: 
            cv2.circle(img, (lms[14][1], lms[14][2]), 12, (255,0,255), cv2.FILLED)

        cv2.imshow("Pose Detection", img)
        cv2.waitKey(1) 

if __name__ == "__main__":
    main() 
