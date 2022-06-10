from shared import *

# load the detector
detector = dlib.get_frontal_face_detector()

# load the predictor
predictor = dlib.shape_predictor("predictors/shape_predictor_68_face_landmarks.dat")

# face keypoints: left eyebrow corner, right eyebrow corner, nose middle
face_points = [17, 26, 29]

sunglasses = cv2.imread("images/sunglasses.png", cv2.IMREAD_UNCHANGED)
# imshow(sunglasses, "bgr")

glasses_points = np.float32([[75, 50], [600, 50], [325, 200]])

# show with sunglasses on the face rotated
width_scale = 1
padding = 100

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

        # look for the landmarks
        landmarks = predictor(image=gray, box=face)

        glasses_new_points = []
        for i in face_points:
            glasses_new_points.append([landmarks.part(i).x - x1, landmarks.part(i).y - y1])
        glasses_new_points = np.float32(glasses_new_points)

        M = cv2.getAffineTransform(glasses_points, glasses_new_points + padding)
        glasses_warped = cv2.warpAffine(sunglasses, M=M, dsize=(width + 2 * padding, height + 2 * padding))

        glasses_scaled = scale_img(glasses_warped, width_scale)
        height_scaled = glasses_scaled.shape[0]
        width_scaled = glasses_scaled.shape[1]

        g_x1 = x1 + int(width / 2) - int(width_scaled / 2)
        g_x2 = g_x1 + width_scaled
        g_y1 = y1 + int(height / 2) - int(height_scaled / 2)
        g_y2 = g_y1 + height_scaled

        g_x1, g_x2, g_y1, g_y2, glasses_scaled = check_out_of_frame(
            g_x1, g_x2, g_y1, g_y2, frame, glasses_scaled)

        overlay_with_png(frame, glasses_scaled, g_x1, g_x2, g_y1, g_y2)

    # show the image
    cv2.imshow(winname="Face", mat=frame)

    # exit when escape is pressed
    if cv2.waitKey(delay=1) == 27:
        break

# when everything done, release the video capture and video write objects
cap.release()

# close all windows
cv2.destroyAllWindows()
