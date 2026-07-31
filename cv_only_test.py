import cv2
cap = cv2.VideoCapture(1, cv2.CAP_AVFOUNDATION)
while True:
    ok, frame = cap.read()
    if not ok:
        break
    cv2.imshow("test", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
cap.release()
cv2.destroyAllWindows()
