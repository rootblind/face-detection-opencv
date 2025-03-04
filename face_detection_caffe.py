import numpy as np
import argparse
import imutils
import time
import cv2
 
# Arguments to be parsed
ap= argparse.ArgumentParser()
ap.add_argument("-p", "--prototxt", default="deploy.prototxt",
	help="path to Caffe 'deploy' prototxt file")
ap.add_argument("-m", "--model", default="dnn_model.caffemodel",
	help="path to Caffe pre-trained model")
ap.add_argument("-c", "--confidence", type=float, default=0.6,
	help="minimum probability to filter weak detections")
args = vars(ap.parse_args())

# Loading the model
net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])
 
# Loading the video
vs = cv2.VideoCapture("video1.mp4")

print("Loading the model...")
time.sleep(2.0) # give a delay in order for everything to be ready

loop_time = time.time() # for frames per second measurement

while True:
	# reading the video frame by frame
    ret, frame = vs.read()
    if not ret: # if the video ends or something fails, break the loop
        break
	
    frame = imutils.resize(frame, width=720) # image resize
 
    (h, w) = frame.shape[:2] # height and width

    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (1280, 720)), 1.0,
	    (1280, 720), (104.0, 177.0, 123.0))
 
	# input for the model
    net.setInput(blob)
    detections = net.forward()

    # drawing rectangles over the face detected coordonates
    for i in range(0, detections.shape[2]):
		
        # model confidence over the specific detection
        confidence = detections[0, 0, i, 2]
 
		# ignore detections below the threshold
        if confidence < args["confidence"]:
            continue
 
		# compute coordinates of the bounding box for the object
        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        (startX, startY, endX, endY) = box.astype("int")
 
		# drawing rectangles and probability
        text = "{:.2f}%".format(confidence * 100)
        y = startY - 10 if startY - 10 > 10 else startY + 10
        cv2.rectangle(frame, (startX, startY), (endX, endY),
			(0, 255, 0), 2)
        cv2.putText(frame, text, (startX, y),
			cv2.FONT_HERSHEY_DUPLEX, 0.45, (0, 255, 0), 1)

    # Display the output frame
    cv2.imshow("Video", frame)
    key = cv2.waitKey(1) & 0xFF

    print(f"FPS: {1 / (time.time() - loop_time)}") # display fps
    loop_time = time.time() # update loop time
	
    # press q to end the process
    if key == ord('q'):
        break

cv2.destroyAllWindows()
vs.release()