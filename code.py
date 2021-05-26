
import cv2
import numpy as np


def FindGreenWaySign(src):
    hsv = cv2.cvtColor(src, cv2.COLOR_BGR2HSV)
    lowerb1 = (70, 120, 30)
    upperb1 = (95, 255, 255)

    dst1 = cv2.inRange(hsv, lowerb1, upperb1)

    return dst1

def FindBlueWaySign(src):
    hsv = cv2.cvtColor(src, cv2.COLOR_BGR2HSV)
    lowerb1 = (100, 120, 30)
    upperb1 = (130, 255, 255)
    dst1 = cv2.inRange(hsv, lowerb1, upperb1)

    return dst1


clr=[]

for i in range(300):
    clr.append([np.random.randint(256),np.random.randint(256),0])
cap=cv2.VideoCapture("put your video path")

time=20
#cap=cv2.VideoCapture(0)
count=0
#cap=cv2.VideoCapture(0)
while(True):
        retval, frame = cap.read()
        if not retval:
                break
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        GreenWaySign = FindGreenWaySign(frame)
        BlueWaySign=FindBlueWaySign(frame)
        edges=GreenWaySign+BlueWaySign
        ##############
        kernel = cv2.getStructuringElement(shape=cv2.MORPH_RECT, ksize=(3, 3))
        edges = cv2.erode(edges, kernel, iterations=2)
        edges = cv2.dilate(edges, kernel, iterations=8)

        ##############

        ret,label,stats,centroids=cv2.connectedComponentsWithStats(edges)
        print('ret=',ret-1)
        dst=np.zeros(frame.shape,dtype=frame.dtype)
        dst2=np.zeros(frame.shape,dtype=frame.dtype)
        for i in range(1,int(ret)):
            x,y,w,h,a=stats[i]


            if a>3200 and (w*h*0.6)<=a:

                count=count+1
                roi=frame[y:y + h, x:x + w]
                roidhsv=hsv[y:y + h, x:x + w]
                H,S,V=cv2.split(roidhsv)
                H=np.reshape(H,-1)
                S=np.reshape(S,-1)
                V=np.reshape(V,-1)
                #if np.var(V)<1000 or np.var(H)<100 or np.var(S)<2000:
                if np.var(V) < 1000  or np.var(H)>1000:
                    dst[label == i] = [0, 0, 255]
                    continue
                dst2[y:y + h, x:x + w] = frame[y:y + h, x:x + w]
                ####
                imname='./data2/rk'+str(count)+'.png'
                print(imname)
                cv2.imshow('roi',roi);
                #cv2.imwrite(imname,roi)
                ####
                cv2.rectangle(frame,(x,y),(x+w,y+h),color=clr[i],thickness=2)

                c=sum(hsv[y:y + h, x:x + w,0])
                c=sum(c)/len(c)
                c=[c,255,255]
                pixel=np.array([[c]],dtype=np.uint8)
                pixel=cv2.cvtColor(pixel,cv2.COLOR_HSV2BGR)
                dst[label==i]=pixel
                cv2.rectangle(dst, (x, y), (x + w, y + h), color=(0,0,255), thickness=2)
            else:
                dst[label==i]=[0,0,255]
        #############


        cv2.imshow('frame',frame)
        cv2.imshow('sign',dst)
        cv2.imshow('sign_img',dst2)
        key = cv2.waitKey(time)
        print(key)
        if key == 27: # Esc
                break
        elif key == 32:
            if time==20:
                time=400
            elif time==400:
                time=1
            else :
                time=20
cv2.destroyAllWindows()
