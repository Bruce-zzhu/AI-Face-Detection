import cv2


# load pre-trained data on face frontals
trained_face_data = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# load an image to detect face
#img = cv2.imread('RDJ.png')
#img = cv2.resize(img, (750, 499))

# cature video from webcam to detect face
webcam = cv2.VideoCapture(0)  # 0 means default camera, can also be a path to a video file

# iterate over frames captured by the camera
while True:
    # read current frame
    successful_frame_read, frame = webcam.read()   # return True/False, frame

    # convert to gray scale
    grayscaled_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # detect faces
    face_coordinates = trained_face_data.detectMultiScale(grayscaled_img)

    # draw rectangle arond the face
    # param: image, top-left coord, bottom-right coord, color(bgr), thickness
    for (x, y, w, h) in face_coordinates:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    cv2.imshow('Face Detector', frame)
    key = cv2.waitKey(1)  # 1 ms to refresh frame

    # quit by pressing 
    if key == 27:   # ASCII code for ESC
        break
    
# release the video object (clean the code)
webcam.release()
