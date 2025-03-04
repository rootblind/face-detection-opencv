import cv2
import time

# load model
classifier = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

# load video
video = cv2.VideoCapture("video1.mp4")
# resize
video.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
video.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

loop_time = time.time()

while video.isOpened():
    # reading the video frame by frame
    _, frame = video.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # convert frames to grayscale

    # perform detection
    # gray = converted image; scaleFactor = image resized for better detection; minNeighbors to reduce false positives
    # minSize = filter out detections smaller than given input
    faces = classifier.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(40,40))

    # draw rectangles
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x,y), (x + w, y + h), (255, 0, 0), 3)

    cv2.imshow("video", frame)

    print(f"FPS: {1 / (time.time() - loop_time)}")
    loop_time = time.time()

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

video.release()