import cv2

for index in range(10):
    cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)

    if not cap.isOpened():
        cap.release()
        continue

    ok, frame = cap.read()

    if ok and frame is not None:
        print(f"Camera index {index}: available")

        cv2.imshow(f"Camera index = {index}", frame)

        print("press any key to continue...")
        cv2.waitKey(0)

    cap.release()
    cv2.destroyAllWindows()