Count the number of people in an Event
=======================================
In this activity, you will learn to detect the number of people in the event and display their count.\

<img src= "https://s3.amazonaws.com/media-p.slid.es/uploads/1525749/images/10482752/pasted-from-clipboard.png" width = "480" height = "320">


Follow the given steps to complete this activity:
1. ### Count the number of people
* Open the main.py file.

* Set confidence and NMS thresholds.

  `confidenceThreshold = 0.5 NMSThreshold = 0.3`
  
* Load YOLO object detection network.

  `yoloNetwork = cv2.dnn.readNetFromDarknet(modelConfiguration, modelWeights)`
  
* Get the image dimensions and store it in H and W variables, respectively.

  `dimensions = image.shape[:2] H = dimensions[0] W = dimensions[1]`
  
* Create blob from image and set input for `YOLO` network.

* Syntax: blob = cv2.dnn.blobFromImage(image, scalefactor=1.0, size)

* 1/255 is takes to normalize the pixel value from 0 to 255 to 0 to 1 as the YOLO (other models also) require the pixel to be in range 0 to 1.

* 416,416 is the size of the images taken by the YOLO model.
 
  `blob = cv2.dnn.blobFromImage(image, 1/255, (416, 416)) yoloNetwork.setInput(blob)`
  
* Get names of the unconnected output layers.
  `layerName = yoloNetwork.getUnconnectedOutLayersNames()`
  
* Forward the layerName from the yoloNetwork and store it in the layerOutputs variable.

  `layerOutputs = yoloNetwork.forward(layerName)`
  
* Initialize lists to store bounding boxes, confidences, and class Ids.
 
  `boxes = [] confidences = [] classIds = []`
  
 * Process each output from the YOLO network.
  
  `for output in layerOutputs: for detection in output:`
  
* Get class scores and ID of class with highest score.
 
  `scores = detection[5:] classId = np.argmax(scores) confidence = scores[classId]`
  
* Check if the confidence is greater than confidenceThreshold.
 
  `if confidence > confidenceThreshold:`
  
* Save the box coordinates.
 
  `box = detection[0:4] * np.array([W, H, W, H])`
  
  `(centerX, centerY,  width, height) = box.astype('int')`
  
  `x = int(centerX - (width/2))`
  
  `y = int(centerY - (height/2))`

* Append the X and Y coordinates to the boxes list, append confidence to confidences list, and classId to classIds list.
 
  `boxes.append([x, y, int(width), int(height)])`
  
  `confidences.append(float(confidence))`
  
  `classIds.append(classId)`

* Apply Non Maxima Suppression to remove overlapping bounding boxes.
 
  `indexes = cv2.dnn.NMSBoxes( boxes, confidences, confidenceThreshold, NMSThreshold)`
  
Save and run the code to check the output.

