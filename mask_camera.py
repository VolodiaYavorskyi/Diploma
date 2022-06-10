from shared import *


def get_mask_points(mask):
    h = mask.shape[0]
    w = mask.shape[1]
    left_eye_x = int(0.15 * w)
    left_eye_y = int(0.4 * h)
    right_eye_x = int(0.8 * w)
    right_eye_y = int(0.4 * h)
    lips_x = int(0.5 * w)
    lips_y = int(0.75 * h)
    return np.float32([[left_eye_x, left_eye_y], [right_eye_x, right_eye_y], [lips_x, lips_y]])


# load the detector
detector = dlib.get_frontal_face_detector()

# load the predictor
predictor = dlib.shape_predictor("predictors/shape_predictor_68_face_landmarks.dat")

# read face mask with transparency
face_mask = cv2.imread("images/mask.png", cv2.IMREAD_UNCHANGED)
# imshow(face_mask, "gray")

# read image
people = cv2.imread("images/people.jpg")
# imshow(people, "bgr")

# convert image to grayscale
people_gray = cv2.cvtColor(people, cv2.COLOR_RGB2GRAY)
# imshow(people_gray, "gray")

mask_height = face_mask.shape[0]
mask_width = face_mask.shape[1]

# face keypoints: left eye corner, right eye corner, lips bottom
face_points = [36, 45, 57]

mask_points = get_mask_points(face_mask)

# show with mask on the face rotated
width_scale = 1.25
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

        mask_new_points = []
        for i in face_points:
            mask_new_points.append([landmarks.part(i).x - x1, landmarks.part(i).y - y1])
        mask_new_points = np.float32(mask_new_points)

        M = cv2.getAffineTransform(mask_points, mask_new_points + padding)
        mask_warped = cv2.warpAffine(face_mask, M=M, dsize=(width + 2 * padding, height + 2 * padding))

        mask_scaled = scale_img(mask_warped, width_scale)
        height_scaled = mask_scaled.shape[0]
        width_scaled = mask_scaled.shape[1]

        mask_x1 = x1 + int(width / 2) - int(width_scaled / 2)
        mask_x2 = mask_x1 + width_scaled
        mask_y1 = y1 + int(height / 2) - int(height_scaled / 2)
        mask_y2 = mask_y1 + height_scaled

        mask_x1, mask_x2, mask_y1, mask_y2, mask_scaled = check_out_of_frame(
            mask_x1, mask_x2, mask_y1, mask_y2, frame, mask_scaled)

        overlay_with_png(frame, mask_scaled, mask_x1, mask_x2, mask_y1, mask_y2)

    # show the image
    cv2.imshow(winname="Face", mat=frame)

    # exit when escape is pressed
    if cv2.waitKey(delay=1) == 27:
        break

# when everything done, release the video capture and video write objects
cap.release()

# close all windows
cv2.destroyAllWindows()
