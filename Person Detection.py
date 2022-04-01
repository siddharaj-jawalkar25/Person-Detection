import time
import numpy as np
import cv2

whT = 320 #width height Target
confThreshold = 0.5
nmsThreshold = 0.3

classFile = 'coco.names'
classNames = []
with open(classFile,'rt') as f:
    classNames = f.read().rstrip('\n').split('\n')
#print(classNames)

# importing configuration and weights
modelConfig = 'yolov3-tiny.cfg'
modelWeights = 'yolov3-tiny.weights'

net = cv2.dnn.readNetFromDarknet(modelConfig, modelWeights)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
    
def findObjects(outputs, img):
    hT, wT, cT = img.shape
    bbox = []
    classIds = []
    confs = []

    for output in outputs:
        for det in output:
            scores = det[5:]
            classId = np.argmax(scores)
            confidence = scores[classId]
            if confidence > confThreshold and classId == 0:
                w,h = int(det[2]*wT) , int(det[3]*hT)
                x,y = int((det[0]*wT) - w/2), int((det[1]*hT)-h/2)
                bbox.append([x,y,w,h])
                classIds.append(classId)
                confs.append(float(confidence))
    #print(len(bbox))

    indices = cv2.dnn.NMSBoxes(bbox, confs, confThreshold, nmsThreshold)
    for i in indices:
        #i = i[0]
        box = bbox[i]
        x,y,w,h = box[0], box[1], box[2], box[3]
        cv2.rectangle(img, (x,y), (x+w, y+h), (0,255,0), 2)
        cv2.putText(img, f'{classNames[classIds[i]].upper()} {int(confs[i]*100)}%',(x,y-10), cv2.FONT_HERSHEY_COMPLEX, 0.6,(0,255,0),2)

        if x < 0 :
            x = 1
        crop = img[y:y+h, x:x+w]
        #print(crop)
        print(x,y,x+w, y+h)
        #cv2.cvtColor(crop,cv2.COLOR_BGR2RGB)
        #print(type(frames[0]))
        cv2.imshow("cropped image of Cam 1", crop)
        return crop

def main():
    cap = cv2.VideoCapture(0)
    frames = []

    while True:
        success, img = cap.read()
        if success != True:
            cap.release()
            print('THE END.....')
            break

        blob = cv2.dnn.blobFromImage(img, 1/255, (whT,whT), [0,0,0], 1, crop=False)
        net.setInput(blob)

        layerNames = net.getLayerNames()
        outputNames = [layerNames[i-1] for i in net.getUnconnectedOutLayers()]
        outputs = net.forward(outputNames)

        result = findObjects(outputs,img)
        if result is not None:
            frames.append(result)
            print(type(frames))
        cv2.imshow('Video 1',img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()

if __name__ == '__main__':
    main()