import cv2

url = "http://10.87.231.27:81/stream"

cap = cv2.VideoCapture(url)

if not cap.isOpened():
    print("Cannot open ESP32-CAM stream")
    exit()

print("Connected to ESP32-CAM")

while True:
    ret, frame = cap.read()

    if not ret:
        print("Frame not received")
        break

    cv2.imshow("ESP32-CAM", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()