

import keras
from keras.models import load_model
from keras.preprocessing import image
import numpy as np
import cv2
import os
def empty(x):
    pass



def get_parameters():
    global l_h
    global l_s
    global l_v
    global u_h
    global u_s
    global u_v
    cap=cv2.VideoCapture(0)    
    cv2.namedWindow("Calibrate")
    print("Use the trackbars to adjust parameters until you see your hand clearly and there is no white noise in the background\n")
    cv2.namedWindow("Trackbars")
    cv2.createTrackbar("L - H", "Trackbars", 0, 179, empty)
    cv2.createTrackbar("L - S", "Trackbars", 0, 255, empty)
    cv2.createTrackbar("L - V", "Trackbars", 0, 255, empty)
    cv2.createTrackbar("U - H", "Trackbars", 179, 179, empty)
    cv2.createTrackbar("U - S", "Trackbars", 255, 255, empty)
    cv2.createTrackbar("U - V", "Trackbars", 255, 255, empty)
    ret, frame = cap.read()
    print("Press enter when you are satisfied with the segmentation\n")
    print("Press h for help\n")
    print("Press ESC to exit at any point\n")
    while (ret):
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
        mask = cv2.inRange(hsv, lower, upper)
        #result = cv2.inRange(hsv, lower, upper)
        result=mask
        result = cv2.GaussianBlur(result, (5, 5), 100)
        result = cv2.GaussianBlur(result, (5, 5), 100)

        result = cv2.resize(result, (200, 200))
        # result= cv2.dilate(result,kernel,iterations = 1)
        cv2.imshow("test", frame)
        # cv2.imshow("mask", mask)
        cv2.imshow("result", result)
        k=cv2.waitKey(1)
        if (k == 13):
            cv2.destroyAllWindows()
            cap.release()
            return l_h, l_s, l_v, u_h, u_s, u_v
        elif (k==27):
            print("Exiting.....\n")
            cv2.destroyAllWindows()
            cap.release()
            return
        elif (k==ord('h')):
            cv2.destroyAllWindows()
            cap.release()
            menu()
            
            









def get_image(l_h, l_s, l_v, u_h, u_s, u_v):
    cap=cv2.VideoCapture(0)    

    l_h=l_h
    l_s=l_s
    l_v=l_v
    u_h=u_h
    u_s=u_s
    u_v=u_v
    ret, frame = cap.read()
    text=""
    kernel=np.ones((3,3),np.uint8)
    print("Make gesture in the rectangle region to view the translation\n")
    print("Press h to view menu\n")
    print("Press ESC to quit\n")
    while(ret):
            ret, frame = cap.read()
            frame = cv2.flip(frame, 1)
            cv2.rectangle(frame, (802, 102), (1102, 402), (0, 255, 0), 3)
            roi = frame[102:402, 802:1102]
            hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
            result = cv2.inRange(hsv, np.array([l_h, l_s, l_v]), np.array([u_h, u_s, u_v]))
            result = cv2.GaussianBlur(result,(5,5),100)
            result = cv2.GaussianBlur(result,(5,5),100)
            result=cv2.resize(result,(200,200))
            cv2.imwrite("./1.jpg",result)
            test_image=image.load_img("./1.jpg")
            text=get_prediction(test_image)
            cv2.putText(frame, text, (30, 400), cv2.FONT_HERSHEY_TRIPLEX, 1.5, (0, 255, 0))
            cv2.imshow('frame',frame)
            cv2.imshow("result", result)
            k=cv2.waitKey(1)
            if(k==ord('h')):
                cv2.destroyAllWindows()
                cap.release()
                menu()            
            elif (k==27):
                print("Exiting.....\n")
                cv2.destroyAllWindows()
                cap.release()
                break
                return


def get_prediction(test_image):
    
    global model
    test_image = image.img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis = 0)
    prediction=model.predict(test_image)

    if prediction[0][0] == 1:
              return 'A'
    elif prediction[0][1] == 1:
              return 'B'
    elif prediction[0][2] == 1:
              return 'C'
    elif prediction[0][3] == 1:
              return 'D'
    elif prediction[0][4] == 1:
              return 'E'
    elif prediction[0][5] == 1:
              return 'F'
    elif prediction[0][6] == 1:
              return 'G'
    elif prediction[0][7] == 1:
              return 'H'
    elif prediction[0][8] == 1:
              return 'I'
    elif prediction[0][9] == 1:
              return 'J'
    elif prediction[0][10] == 1:
              return 'K'
    elif prediction[0][11] == 1:
              return 'L'
    elif prediction[0][12] == 1:
              return 'M'
    elif prediction[0][13] == 1:
              return 'N'
    elif prediction[0][14] == 1:
              return 'O'
    elif prediction[0][15] == 1:
              return 'P'
    elif prediction[0][16] == 1:
              return 'Q'
    elif prediction[0][17] == 1:
              return 'R'
    elif prediction[0][18] == 1:
              return 'S'
    elif prediction[0][19] == 1:
              return 'T'
    elif prediction[0][20] == 1:
              return 'U'
    elif prediction[0][21] == 1:
              return 'V'
    elif prediction[0][22] == 1:
              return 'W'
    elif prediction[0][23] == 1:
              return 'X'
    elif prediction[0][24] == 1:
              return 'Y'




def menu():
    global l_h
    global l_s
    global l_v
    global u_h
    global u_s
    global u_v
    print("\n\nEnter the following keys to perform corresponding tasks\n\n")
    print("1. Enter g to generate the gesture data\n")
    print("2. Enter t to train the model\n")
    print("3. Enter r to readjust the segmentation parameters\n")
    print("4. Enter p to proceed with the prediction\n")
    print("5. Enter q to quit\n")
    c=str(input("Enter your choice\n"))
    if (c=='r') or (c=='R'):
        l_h,l_s, l_v, u_h, u_s, u_v=get_parameters()
        get_image(l_h, l_s, l_v, u_h, u_s, u_v)
    elif (c=='p') or (c=='P'):
        l_h,l_s, l_v, u_h, u_s, u_v=get_parameters()
        get_image(l_h, l_s, l_v, u_h, u_s, u_v)
    elif (c=='t') or (c=='T'):
        os.system("python train.py")
        print("Returned back to the Predict program\n")
        main()
    elif(c=='g') or (c=='G'):
        print("!!!\tBEWARE PREVIOUSLY GENERATED DATA WILL BE OVERWRITTEN\t!!!\n")
        m=str(input("Do you want to proceed? Press y or n\n"))
        if (m=='y') or (m=='Y'):
            os.system("python capture.py")
            print("Returned back to the Predict program\n")
            main()
        elif (m=='n') or (m=='N'):
            menu()
    elif(c=='q') or (c=='Q'):
        print("Exiting....\n")
        return








def main():
    global imgsrc
    global img
    global model
    print("\n\nThis programs detects the ASL gesture made based on a model trained on Convoluted Neural Networks\n\n")
    print("\t\tInstructions of usage:\n")
    print("1. You must first generate the gestures with different ASL alphabet for the model to work best\n")
    print("2. Once the gestures are generated the model must be trained on this data\n")
    print("3. After training the model you can proceed with the prediction\n\n")
    print("4. Enter q to quit\n")
    print("Press h while program window is open to display help\n")
    print("-----------------------------------------------------------------------------------------------------\n")
    print("Have you generated data and trained the  model with gestures?\n")
    x=str(input("Press y for YES , n for NO and q to quit\n"))
    if (x=='y') or (x=='Y'):
        print("Loading the model generated\n")
        model=load_model("./model.hdf5")
        model.load_weights("model_weights.hdf5")
        l_h, l_s, l_v, u_h, u_s, u_v=get_parameters()
        get_image(l_h, l_s, l_v, u_h, u_s, u_v)
    
    elif (x=='n') or (x=='N'):  
        menu()
        
    elif (x=='q') or (x=='Q'):
        print("Exiting....\n")
        return



	
			
if __name__=="__main__":
        main()

    
    
            
