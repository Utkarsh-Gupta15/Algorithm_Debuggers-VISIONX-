import cv2
import smtplib
import os
import time
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime

#Loading and connecting with the dataset
connection = cv2.dnn.readNet('yolov3weapon.weights', 'yolov3weapon.cfg')
layer = connection.getLayerNames()
output = [layer[i - 1] for i in connection.getUnconnectedOutLayers().flatten()]

#creating directory to store the screenshots which will be captured by the model
storeage = "Repulse screenshots"
if not os.path.exists(storeage):
    os.makedirs(storeage)

#creating a function to take the screenshot and store it in the created directory
def screenshot(frame, suffix=""):
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    filename = os.path.join(storeage, f"weapon_detected_{timestamp}{suffix}.png")
    cv2.imwrite(filename, frame)
    print(f"Screenshot saved: {filename}")

#creating a function to send alert emails when a weapon is been detected
def alert():
    sender = " " #Enter your email
    receiver = "u12344321g@gmail.com"
    password = " " #Enter the password of your email
    subject = "Weapon Detected Alert"

    msg = MIMEMultipart()
    msg['From'] = sender
    msg['To'] = receiver
    msg['Subject'] = subject
    msg.attach(MIMEText("A weapon has been detected."))

    try:
        with smtplib.SMTP('smtp.google.com', 587) as server:
            server.starttls()
            server.login(sender_email, password)
            server.sendmail(sender, receiver, msg.as_string())
            print("Alert sent.")
    except:
        print("Error occured while sending the email!")

#using camera as the source to detect the objects
source = cv2.VideoCapture(0)

lastScreenshot = 0
interval = 3

#running the process of capturing the frames in loop
while source.isOpened():
    ret, frame = source.read()
    if not ret:
        print("Couldn't capture the frame")
        break

    height, width, _ = frame.shape
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), swapRB=True, crop=False)
    connection.setInput(blob)
    detections = connection.forward(output)

    detected = False
    for detection in detections:
        for obj in detection:
            scores = obj[5:]
            class_id = scores.argmax()
            con = scores[class_id]
            if con > 0.5:
                detected = True
                o1 = int(obj[0] * width)
                o2 = int(obj[1] * height)
                w = int(obj[2] * width)
                h = int(obj[3] * height)
                x = int(o1 - w / 2)
                y = int(o2 - h / 2)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
                cv2.putText(frame, "Weapon", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    currentTime = time.time()


    if detected:
        if currentTime - lastScreenshot >= interval:#using condional statement to send emails and capture screenshot after 3 seconds to avoid repetitions
            screenshot(frame)
            lastScreenshot = currentTime
            alert()

    cv2.imshow('Detected', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

source.release()
cv2.destroyAllWindows()
