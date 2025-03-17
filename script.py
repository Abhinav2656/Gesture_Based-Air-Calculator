# import cv2
# import numpy as np
# import streamlit as st
# from cvzone.HandTrackingModule import HandDetector
#
# # Streamlit UI
# st.title("Air-Drawn Calculator")
# st.write("Draw numbers and operators in the air using your index finger!")
#
# # Initialize webcam
# cap = cv2.VideoCapture(0)
# cap.set(3, 640)
# cap.set(4, 480)
#
# detector = HandDetector(staticMode=False, maxHands=1, detectionCon=0.7, minTrackCon=0.5)
# canvas = None
# prev_pos = None
#
# stframe = st.empty()  # Streamlit frame placeholder
#
# while cap.isOpened():
#     success, img = cap.read()
#     if not success:
#         st.error("Failed to access webcam")
#         break
#
#     img = cv2.flip(img, 1)
#     if canvas is None:
#         canvas = np.zeros_like(img)
#
#     hands, img = detector.findHands(img, draw=True, flipType=True)
#     if hands:
#         hand = hands[0]
#         lmList = hand["lmList"]  # List of 21 landmarks
#         fingers = detector.fingersUp(hand)
#
#         if fingers == [0, 1, 0, 0, 0]:  # Draw with index finger
#             current_pos = lmList[8][0:2]
#             if prev_pos is None:
#                 prev_pos = current_pos
#             cv2.line(canvas, tuple(prev_pos), tuple(current_pos), (255, 0, 255), 5)
#             prev_pos = current_pos
#
#         elif fingers == [1, 0, 0, 0, 0]:  # Clear screen with thumb up
#             canvas = np.zeros_like(img)
#             prev_pos = None
#     else:
#         prev_pos = None  # Reset if no hands detected
#
#     image_combined = cv2.addWeighted(img, 0.7, canvas, 0.3, 0)
#     stframe.image(image_combined, channels="BGR")
#
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break
#
# cap.release()
# cv2.destroyAllWindows()


import cv2
import numpy as np
import streamlit as st
from cvzone.HandTrackingModule import HandDetector
import tensorflow as tf
import time

# Load pre-trained digit recognition model (MNIST-based CNN)
model = tf.keras.models.load_model("inception-v3-tensorflow2-classification-v2/saved_model.pb")

# Streamlit UI
st.title("Air-Drawn Calculator")
st.write("Draw numbers and operators in the air using your index finger!")

# Initialize webcam
cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

detector = HandDetector(staticMode=False, maxHands=1, detectionCon=0.7, minTrackCon=0.5)
canvas = np.zeros((480, 640, 3), dtype=np.uint8)
prev_pos = None
drawing = False  # Start with drawing disabled

time_threshold = 2  # Processing interval
last_eval_time = time.time()

stframe = st.empty()  # Streamlit frame placeholder
expression = ""
result = ""


def process_canvas():
    gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (28, 28))
    normalized = resized / 255.0
    reshaped = normalized.reshape(1, 28, 28, 1)
    prediction = model.predict(reshaped)
    return str(np.argmax(prediction))


while cap.isOpened():
    success, img = cap.read()
    if not success:
        st.error("Failed to access webcam")
        break

    img = cv2.flip(img, 1)
    hands, img = detector.findHands(img, draw=True, flipType=True)

    if hands:
        hand = hands[0]
        lmList = hand["lmList"]
        fingers = detector.fingersUp(hand)

        if fingers == [0, 1, 0, 0, 0]:  # Draw with index finger
            drawing = True
            current_pos = lmList[8][0:2]
            if prev_pos is None:
                prev_pos = current_pos
            cv2.line(canvas, tuple(prev_pos), tuple(current_pos), (255, 0, 255), 5)
            prev_pos = current_pos

        elif fingers == [1, 0, 0, 0, 0]:  # Clear screen with thumb up
            canvas.fill(0)
            prev_pos = None
            expression = ""
            result = ""

        elif fingers == [1, 1, 0, 0, 0]:  # Pen-up (stop drawing)
            drawing = False
            prev_pos = None  # Prevent lingering marks

        elif fingers == [1, 1, 1, 1, 1]:  # Erase everything if full hand is used
            canvas.fill(0)
            prev_pos = None
            expression = ""
            result = ""
            drawing = False
        else:
            drawing = False
    else:
        prev_pos = None  # Reset if no hands detected

    # Perform digit recognition every few seconds to reduce lag
    if time.time() - last_eval_time > time_threshold:
        recognized_digit = process_canvas()
        last_eval_time = time.time()

        if recognized_digit:
            expression += recognized_digit
            try:
                result = str(eval(expression))
            except:
                result = "Invalid Expression"

    # Display results in Streamlit
    st.write(f"Expression: {expression}")
    st.write(f"Result: {result}")

    image_combined = cv2.addWeighted(img, 0.7, canvas, 0.3, 0)
    stframe.image(image_combined, channels="BGR")

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
