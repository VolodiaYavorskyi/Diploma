from shared import *

# load the detector
detector = dlib.get_frontal_face_detector()

# load the predictor
predictor = dlib.shape_predictor("predictors/shape_predictor_68_face_landmarks.dat")

# numbers of keypoints
nose_kp = range(29, 36)

# scale = 1.25
# padding_x = 10
# padding_y = 3
# blur_padding = 3
# blur_padding_bottom = 1
# blur_ksize = (5, 5)
# blur_sigma = 3
scale = 0.85
padding_x = 20
padding_y = 10
blur_padding = 5
blur_padding_bottom = 5
blur_ksize = (9, 9)
blur_sigma = 3

# read the camera image
cap = cv2.VideoCapture(0)

# press escape to break
while True:
    _, frame = cap.read()
    # convert image into grayscale
    gray = cv2.cvtColor(src=frame, code=cv2.COLOR_BGR2GRAY)

    # use detector to find landmarks
    faces = detector(gray)

    for face in faces:
        # look for the landmarks
        landmarks = predictor(image=gray, box=face)

        for i in nose_kp:
            x = landmarks.part(i).x
            y = landmarks.part(i).y

            # draw a circle
            # cv2.circle(img=elon_nose, center=(x, y), radius=2, color=(0, 255, 0), thickness=-1)

        x_coords = list(map(lambda i: landmarks.part(i).x, nose_kp))
        y_coords = list(map(lambda i: landmarks.part(i).y, nose_kp))

        x1, x2, y1, y2 = min(x_coords), max(x_coords), min(y_coords), max(y_coords)
        x1, x2, y1, y2 = x1 - padding_x, x2 + padding_x, y1 - padding_y, y2 + padding_y
        x1, x2, y1, y2, _ = check_out_of_frame(x1, x2, y1, y2, frame)

        nose = frame[y1:y2, x1:x2]

        nose_scaled = scale_img(nose, scale)
        height, width = nose.shape[:2]
        height_scaled, width_scaled = nose_scaled.shape[:2]

        x1_new = x1 + int(width / 2) - int(width_scaled / 2)
        x2_new = x1_new + width_scaled
        y1_new = y1
        y2_new = y1_new + height_scaled
        x1_new, x2_new, y1_new, y2_new, nose_scaled = check_out_of_frame(x1_new, x2_new, y1_new, y2_new, frame,
                                                                         nose_scaled)

        frame[y1_new:y2_new, x1_new:x2_new] = nose_scaled
        frame = blur_border(frame, x1_new, x2_new, y1_new, y2_new,
                            blur_padding, blur_padding, blur_padding, blur_padding_bottom,
                            blur_ksize, blur_sigma)
        # cv2.rectangle(img=elon_nose, pt1=(x1, y1), pt2=(x2, y2), color=(255, 0, 0), thickness=1)

    # show the image
    cv2.imshow(winname="Face", mat=frame)

    # exit when escape is pressed
    if cv2.waitKey(delay=1) == 27:
        break

# when everything done, release the video capture and video write objects
cap.release()

# close all windows
cv2.destroyAllWindows()
