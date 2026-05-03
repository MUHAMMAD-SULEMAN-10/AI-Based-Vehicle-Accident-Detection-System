import cv2
from flask import Response
from flask import Flask
from flask import render_template
import threading
import paho.mqtt.client as mqtt
import requests
import json
import socket
from tensorflow.keras.models import load_model
import numpy as np
from keras.models import Sequential
from keras.layers import Conv2D, BatchNormalization, MaxPooling2D, Flatten, Dense, Dropout


s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
s.connect(("8.8.8.8", 80))
ip_address = s.getsockname()[0]
print('IP Address:', ip_address)

def cnn_model():
    input_shape = (128,128,1)
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3),activation='relu',padding = 'Same',input_shape=input_shape))
    model.add(BatchNormalization())

    model.add(Conv2D(32,kernel_size=(3, 3), activation='relu',padding = 'Same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size = (2, 2)))

    model.add(Conv2D(64, (3, 3), activation='relu',padding = 'Same'))
    model.add(BatchNormalization())

    model.add(Conv2D(64, (3, 3), activation='relu',padding = 'Same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(128,kernel_size=(3, 3), activation='relu',padding = 'Same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size = (2, 2)))
    model.add(Conv2D(128, (3, 3), activation='relu',padding = 'Same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(2, activation='softmax'))

    return model


# model load
model = cnn_model()
model.load_weights("car_accident_model.weights.h5")
classNames = ['Accident', 'Not-Accident']

def on_message(client, userdata, message):
    print("li")
    data1 =[]
    receivedstring = str(message.payload.decode("utf-8"))
    data1=receivedstring.split(",")
    print(data1)
    with open('config.json', 'w') as json_file:
        json.dump(data1, json_file)

broker_address = "broker.hivemq.com"
client = mqtt.Client("PROJECT") 
client.connect(broker_address) 
client.on_message=on_message 
client.subscribe("CAR-NT")
response = 0

with open('config.json') as f:
    data = json.load(f)

serverToken = 'AAAA9myXIrY:APA91bHsrnPMm8TUAk7lfPCdXzdTZdz0riWCQlaoTpWVTCGMiWakwWWEoAFdERvk6LQ8esGb-rRJvFTTY9NRTcVc-O9WrNS_MaE3GNmoJ7tbRrqt46RRXlzIPVO4NBo21LFEA8lRMtSD'
deviceToken = data[0]
headers = {
        'Content-Type': 'application/json',
        'Authorization': 'key=' + serverToken,
      }

body = {
          'notification': {'title': 'Sending push form python script',
                            'body': 'New Message'
                            },
          'to':
              deviceToken,
          'priority': 'high',
        }

outputFrame = None
lock = threading.Lock()

# initialize a flask object
app = Flask(__name__)

@app.route("/")
def index():
    # return the rendered template
    return render_template("index.html")

def web_stream(frameCount):
    global outputFrame, lock

    notification_flag = 0
    normalflag = 0
    cap = cv2.VideoCapture('Accident_Detection.mp4')

    while True:
        client.loop_start()
        has_frame, show = cap.read()
        if has_frame:
            _, frame = cap.read()
            gray = cv2.cvtColor(show, cv2.COLOR_BGR2GRAY)
            image = cv2.resize(gray, (128, 128))
            data = image.astype("float") / 255.0
            data = np.expand_dims(data, axis=0)
            data = np.expand_dims(data, axis=-1)
            pred = model.predict(data)[0]

            if pred.argmax() == 0 and pred[pred.argmax()] > 0.93:
                prob = round(pred[pred.argmax()] * 100, 2)
                txt = f"{classNames[pred.argmax()]} {prob}%"
                cv2.rectangle(show, (0, 0), (280, 40), (0, 0, 0), -1)
                cv2.putText(show, txt, (0, 30), cv2.FONT_HERSHEY_DUPLEX,
                            1, (125, 246, 55), 2)

                if notification_flag == 0:
                    normalflag = 0
                    client.publish("CAR-AT", "1," + str(ip_address))
                    with open('config.json') as f:
                        data = json.load(f)
                    serverToken = 'YOUR_SERVER_TOKEN'
                    deviceToken = data[0]

                    headers = {
                        'Content-Type': 'application/json',
                        'Authorization': 'key=' + serverToken}
                    body = {
                        'notification': {'title': 'Sending push from python script', 'body': 'New Message'},
                        'to': deviceToken,
                        'priority': 'high',
                    }
                    response = requests.post("https://fcm.googleapis.com/fcm/send", headers=headers, data=json.dumps(body))

                    print(response.status_code)
                    try:
                        print(response.json())
                    except requests.exceptions.JSONDecodeError:
                        print("Response is not in JSON format:", response.text)

                    notification_flag = 1
        else:
            if normalflag == 0:
                client.publish("CAR-AT", "0")
                notification_flag = 0
                normalflag = 1

        cv2.imshow("Accident Detection", show)

        with lock:
            image_resized = cv2.resize(show,(640,480))
            outputFrame = image_resized.copy()
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

def generate():
    # grab global references to the output frame and lock variables
    global outputFrame, lock

    # loop over frames from the output stream
    while True:
        # wait until the lock is acquired
        with lock:
            # check if the output frame is available, otherwise skip
            # the iteration of the loop
            if outputFrame is None:
                continue

            # encode the frame in JPEG format
            (flag, encodedImage) = cv2.imencode(".jpg", outputFrame)

            # ensure the frame was successfully encoded
            if not flag:
                continue

        # yield the output frame in the byte format
        yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + 
            bytearray(encodedImage) + b'\r\n')
        
@app.route("/video_feed")
def video_feed():
    # return the response generated along with the specific media
    # type (mime type)
    return Response(generate(),
        mimetype = "multipart/x-mixed-replace; boundary=frame")

# check to see if this is the main thread of execution
if __name__ == '__main__':

    # start a thread
    t = threading.Thread(target=web_stream, args=(32,))
    t.daemon = True
    t.start()

    # start the flask app
    app.run(host="0.0.0.0", port="8000", threaded=True, use_reloader=False)