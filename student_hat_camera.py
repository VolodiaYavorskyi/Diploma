from shared import *

# load the detector
detector = dlib.get_frontal_face_detector()

# read student hat with transparency
student_hat = cv2.imread("images/student_hat.png", cv2.IMREAD_UNCHANGED)
# imshow(student_hat, "gray")

hat_height = student_hat.shape[0]
hat_width = student_hat.shape[1]

# show hat over the face
width_scale = 2.5

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
        # left, top, right, bottom points
        x1, y1, x2, y2 = face.left(), face.top(), face.right(), face.bottom()
        height = x2 - x1
        width = y2 - y1

        scale = width_scale * width / hat_width
        hat_scaled = scale_img(student_hat, scale)

        height_scaled = hat_scaled.shape[0]
        width_scaled = hat_scaled.shape[1]

        hat_x1 = x1 + int(width / 2) - int(width_scaled / 2)
        hat_x2 = hat_x1 + width_scaled
        hat_y1 = y1 - height - 15
        hat_y2 = hat_y1 + height_scaled

        hat_x1, hat_x2, hat_y1, hat_y2, hat_scaled = check_out_of_frame(
            hat_x1, hat_x2, hat_y1, hat_y2, frame, hat_scaled)

        overlay_with_png(frame, hat_scaled, hat_x1, hat_x2, hat_y1, hat_y2)

    # show the image
    cv2.imshow(winname="Face", mat=frame)

    # exit when escape is pressed
    if cv2.waitKey(delay=1) == 27:
        break

# when everything done, release the video capture and video write objects
cap.release()

# close all windows
cv2.destroyAllWindows()
