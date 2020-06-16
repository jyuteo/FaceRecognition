import cv2
import dlib
from imutils import face_utils
import face_classifier

face_landmark_path = './shape_predictor_68_face_landmarks.dat'

def main():
    face_locations = []
    num_frame = 1

    video_path = 'test_video/test_bp.mp4'
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Unable to connect to camera.")
        return

    FPS = 8.0
    original_fps = int(cap.get(cv2.CAP_PROP_FPS))
    # print("original FPS:", original_fps)
    frame_height = int(cap.get(4))
    frame_width = int(cap.get(3))

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter('test_video/test_bp' + '_face_recognition.mp4',
                          fourcc, FPS - 2.5, (frame_width, frame_height))

    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(face_landmark_path)

    while cap.isOpened():
        # capture video by frame
        for i in range(int(original_fps / FPS)):
            ret, frame = cap.read()

        if ret:
            res = frame.copy()

            # detect faces from image
            face_rects = detector(frame, 0)
            # cv2.putText(frame, 'Frame: ' + str(num_frame), (10, 30), cv2.FONT_HERSHEY_DUPLEX, 1.0, (255, 0, 0), 1)
            print('No. of faces for frame', str(num_frame), ':', len(face_rects))

            try:
                for i in range(len(face_rects)):
                    # get face landmarks
                    face_rect = face_rects[i]

                    # Finding points for rectangle to draw on face
                    left, top, right, bottom = face_rect.left(), face_rect.top(), face_rect.right(), face_rect.bottom()
                    w = right - left
                    h = bottom - top

                    # Computing larger face co-ordinates
                    H, W, _ = frame.shape

                    left, right = (max(0, left - int(w * 0.2)), min(left + int(1.2 * w), W))
                    top, bottom = (max(0, top - int(0.2 * h)), min(top + int(1.2 * h), H))

                    # Draw rectangle around the face
                    cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

                    # Crop out the face
                    img_cp = frame[top:bottom, left:right].copy()

                    name = face_classifier.classify_face(img_cp)[0]

                    # Draw a label with a name below the face
                    cv2.rectangle(frame, (left, bottom), (right, bottom + 35), (0, 0, 255), cv2.FILLED)
                    cv2.putText(frame, name, (left + 6, bottom + 30), cv2.FONT_HERSHEY_DUPLEX, 0.85, (255, 255, 255), 1)

                    shape = predictor(frame, face_rect)
                    shape = face_utils.shape_to_np(shape)

                    # Outline the face
                    # for (x, y) in shape:
                    #     cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)

                # Save frames as video
                out.write(frame)
                print('Sucessfully saved frame ' + str(num_frame))
                num_frame += 1

                # Display the resulting image
                cv2.imshow('Video', frame)
                key = cv2.waitKey(1) & 0xFF

            except Exception as e:
                print(e)

        else:
            print('done')
            # Release everything if job is finished
            cap.release()
            out.release()
            cv2.destroyAllWindows()
            print('exit')

if __name__ == '__main__':
    main()


