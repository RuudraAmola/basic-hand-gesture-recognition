#TOPIC - HAND GESTURE RECOGNITION

import math
import cv2       
import numpy            

capture = cv2.VideoCapture(0)   #opening web cam

while (1):
    try:
        ret, frame = capture.read()
        frame = cv2.flip(frame, 1)
        
        roi=frame[100:300, 100:300]    #region of interest
        cv2.rectangle(frame, (100, 100), (300, 300), (0, 255, 0), 0)

        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        lower_skin = numpy.array([0, 48, 80])   #skin color range for range of different skin tones
        upper_skin = numpy.array([20, 255, 255])
        mask = cv2.inRange(hsv, lower_skin, upper_skin) #creating a mask image from color range

        kernel = numpy.ones((2, 2))
        mask = cv2.dilate(mask, kernel, iterations=4)   #expanding mask image boundaries
        erosion = cv2.erode(mask, kernel, iterations=4)    #shrinking mask image boundaries
        mask = cv2.GaussianBlur(mask, (5, 5), 100)   #blurring image to remove noise and smoothening
        
        #Each point in the contour is a tuple representing the coordinates of a pixel
        contours, hierarchy= cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)   #contours

        cnt = max(contours, key=lambda x: cv2.contourArea(x))   #contour of max area(hand)
        epsilon = 0.0005*cv2.arcLength(cnt, True)   #approximating the contour
        approx = cv2.approxPolyDP(cnt, epsilon, True)   #creating an approx contour of the hand
        #'True' indicates that the contour is a closed shape

        hull = cv2.convexHull(cnt)   #make convex hull around hand
        areahull = cv2.contourArea(hull)    #calculates area of the hull formed
        areacnt = cv2.contourArea(cnt)   #calculates area of  original contour
      
        #area ratio between the convex hull and the original contour
        arearatio = ((areahull-areacnt)/areacnt)*100
    
        hull = cv2.convexHull(approx, returnPoints=False)
        defects = cv2.convexityDefects(approx, hull)    #finding convexity defects
        
        l=0
        #finding no. of defects due to fingers
        for i in range(defects.shape[0]):
            s, e, f, d = defects[i, 0]
            start = tuple(approx[s][0])    #storing value of points from the array
            end = tuple(approx[e][0])
            far = tuple(approx[f][0])

            a = math.sqrt((end[0] - start[0])**2 + (end[1] - start[1])**2)
            b = math.sqrt((far[0] - start[0])**2 + (far[1] - start[1])**2)
            c = math.sqrt((end[0] - far[0])**2 + (end[1] - far[1])**2)
            s = (a+b+c)/2
            ar = math.sqrt(s*(s-a)*(s-b)*(s-c))
            
            d = (2*ar)/a    #distance between defect point and convex hull
            
            #finding angle between fingers with cosine function
            angle = math.acos((b**2 + c**2 - a**2)/(2*b*c)) * 57
            
            #consider angles only which are less than 90 degrees
            if angle <= 90 and d > 30:
                l += 1
                cv2.circle(roi, far, 3, [255, 0, 0], -1)
            
            cv2.line(roi, start, end, [0, 255, 0], 2)   #lines around hand
                
        l += 1
        
        #print gestures on the basis of number of fingers
        font = cv2.FONT_HERSHEY_COMPLEX
        if l == 1:
            if areacnt < 2000:
                cv2.putText(frame, 'Show hand in box', (0, 50), font, 2, (0, 0, 255), 3, cv2.LINE_AA)
            else:
                if arearatio < 12:
                    cv2.putText(frame, 'Showing ZERO', (0, 50), font, 2, (0, 0, 255), 3, cv2.LINE_AA)
                elif arearatio < 17.5:
                    cv2.putText(frame, 'Showing SIX', (0, 50), font, 2, (0, 0, 255), 3, cv2.LINE_AA) 
                else:
                    cv2.putText(frame, 'Showing ONE',(0, 50), font, 2, (0, 0, 255), 3, cv2.LINE_AA)
                    
        elif l == 2:
            cv2.putText(frame, 'Showing TWO', (0, 50), font, 2, (0, 0, 255), 3, cv2.LINE_AA)
            
        elif l == 3:
         
              if arearatio < 27:
                    cv2.putText(frame, 'Showing THREE', (0, 50), font, 2, (0, 0, 255), 3, cv2.LINE_AA)
              else:
                    cv2.putText(frame, 'Showing PERFECT', (0, 50), font, 2, (0, 0, 255), 3, cv2.LINE_AA)
                    
        elif l == 4:
            cv2.putText(frame, 'Showing FOUR', (0, 50), font, 2, (0, 0, 255), 3, cv2.LINE_AA)
            
        elif l == 5:
            cv2.putText(frame, 'Showing FIVE', (0, 50), font, 2, (0, 0, 255), 3, cv2.LINE_AA)
            
        elif l == 6:
            cv2.putText(frame, 'please re-position the hand', (0, 50), font, 2, (0, 0, 255), 3, cv2.LINE_AA)
            
        else:
            cv2.putText(frame, 'please re-position the hand', (0, 50), font, 2, (0, 0, 255), 3, cv2.LINE_AA)
            
        #displaying the cam window and mask window
        cv2.imshow('mask', mask)
        cv2.imshow('frame', frame)
    except:
        pass
        
    if cv2.waitKey(5) & 0xFF == ord('s'):  #press "s" to stop web cam after 5 milisec
     break
    
cv2.destroyAllWindows()
capture.release()    