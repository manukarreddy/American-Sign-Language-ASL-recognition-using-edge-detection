# -*- coding: utf-8 -*-
"""
Created on Fri Nov 16 19:49:50 2018

@author: manukar
"""
import cv2
import os
import numpy as np


def empty(x):
    pass

def create_directory(sign_name):
    if not os.path.exists("./Data/Train/" +str(sign_name)):
        os.makedirs("./Data/Train/" + str(sign_name))
    if not os.path.exists("./Data/Validation/" + str(sign_name)):
        os.makedirs("./Data/Validation/" + str(sign_name))
    if not os.path.exists("./Data/Test"):
        os.makedirs("./Data/Test")



def get_parameters():
    cap=cv2.VideoCapture(0)    
    cv2.namedWindow("Calibrate")
    cv2.namedWindow("Trackbars")
    cv2.createTrackbar("L - H", "Trackbars", 0, 179, empty)
    cv2.createTrackbar("L - S", "Trackbars", 0, 255, empty)
    cv2.createTrackbar("L - V", "Trackbars", 0, 255, empty)
    cv2.createTrackbar("U - H", "Trackbars", 179, 179, empty)
    cv2.createTrackbar("U - S", "Trackbars", 255, 255, empty)
    cv2.createTrackbar("U - V", "Trackbars", 255, 255, empty)  
    ret, frame = cap.read()
    while(ret):
            ret, frame = cap.read()
            frame = cv2.flip(frame, 1)

            l_h = cv2.getTrackbarPos("L - H", "Trackbars")
            l_s = cv2.getTrackbarPos("L - S", "Trackbars")
            l_v = cv2.getTrackbarPos("L - V", "Trackbars")
            u_h = cv2.getTrackbarPos("U - H", "Trackbars")
            u_s = cv2.getTrackbarPos("U - S", "Trackbars")
            u_v = cv2.getTrackbarPos("U - V", "Trackbars")

            cv2.rectangle(frame, (802, 102), (1102, 402), (0, 255, 0), 3)

            lower = np.array([l_h, l_s, l_v])
            upper = np.array([u_h, u_s, u_v])
            roi = frame[102:402, 802:1102]
            hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
            result = cv2.GaussianBlur(hsv,(3,3),100)
            result = cv2.medianBlur(result,3)
            result = cv2.inRange(result, lower, upper)

            
            result=cv2.resize(result,(200,200))
            cv2.imshow("test", frame)
            cv2.imshow("result", result)
            if (cv2.waitKey(1)==13):
                cv2.destroyAllWindows()
                return l_h,l_s,l_v,u_h,u_s,u_v
                




def save_gestures(sign_name,l_h,l_s,l_v,u_h,u_s,u_v):
    global test_name
    create_directory(sign_name)   
    cap=cv2.VideoCapture(0)
    l_h = l_h
    l_s = l_s
    l_v = l_v
    u_h = u_h
    u_s = u_s
    u_v = u_v
    imgno=1
    validation_name=1
    train_name=1
    ret, frame = cap.read()
    kernel=np.ones((3,3),np.uint8)
    print("Press s to save sign\n")
    print("And make the sign for a few seconds\n")
    print("Press ESC to exit\n")
    while(ret):
            ret, frame = cap.read()
            frame = cv2.flip(frame, 1)
            cv2.rectangle(frame, (802, 102), (1102, 402), (0, 255, 0), 3)
            lower = np.array([l_h, l_s, l_v])
            upper = np.array([u_h, u_s, u_v])
            roi = frame[102:402, 802:1102]
            hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
            result = cv2.inRange(hsv, lower, upper)
            result = cv2.GaussianBlur(result,(3,3),100)
            result = cv2.GaussianBlur(result,(3,3),100)            
            result=cv2.resize(result,(200,200))
            result= cv2.dilate(result,kernel,iterations = 1)
            cv2.imshow("test", frame)
            cv2.imshow("result", result)
            k=cv2.waitKey(1)
            if k==ord('s'):
                while(imgno<=500):
                    print(imgno)
                    if imgno<=300:
                        if (imgno>0) and (imgno<=250): 
                            for i in range(1,51):
                                cv2.imwrite(("./Data/Train/" + str(sign_name) +"/" + str(train_name) + ".jpg"), result)
                                train_name+=1
                        elif (imgno>250) and (imgno<=300):
                            for i in range (1,51):
                                cv2.imwrite(("./Data/Validation/" + str(sign_name) + "/" + str(validation_name) + ".jpg"), result)
                                validation_name+=1
                        imgno+=50
                            
                    elif (imgno>300):
                        for i in range(1,51):
                            cv2.imwrite((".Data/Test/" + str(test_name) + ".jpg"), result)
                            test_name+=1
                        imgno+=50
                        
                    else:
                        cv2.destroyAllWindows()
                        break
                    
                print("Finished capturing sign " + str(sign_name) + "\n")
                print("Would you like to enter another sign\n")
                m=str(input("Press y for yes and n for no\n"))
                if (m=='y'or m=='Y'):
                    sign_name=input("Enter the name of the gesture\n")
                    save_gestures(sign_name,l_h,l_s,l_v,u_h,u_s,u_v)
                else:
                    return
            
            if k==27:
                cv2.destroyAllWindows()
                return


def main():
    global test_name
    test_name=1
    l_h,l_s,l_v,u_h,u_s,u_v=get_parameters()
    sign_name=input("Enter the name of the gesture\n")
    save_gestures(sign_name,l_h,l_s,l_v,u_h,u_s,u_v)
    
    
    
    
if __name__=="__main__":    
    main()
    
    
    
    
    
    
    
    
    
    