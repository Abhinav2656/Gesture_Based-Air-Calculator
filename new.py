import cv2
import numpy as np
import streamlit as st
from cvzone.HandTrackingModule import HandDetector

# Streamlit UI
st.title("Air-Drawn Calculator")
st.write("Draw numbers and operators in the air using gestures!")

# Instruction Display
instruction_placeholder = st.empty()
instruction_placeholder.write("Start by drawing numbers")

# Initialize webcam
cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

detector = HandDetector(staticMode=False, maxHands=1, detectionCon=0.7, minTrackCon=0.5)
canvas = np.zeros((480, 640, 3), dtype=np.uint8)
prev_pos = None
expression = ""
result = ""

stframe = st.empty()  # Streamlit frame placeholder

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
            instruction_placeholder.write("Drawing numbers...")
            current_pos = lmList[8][0:2]
            if prev_pos is None:
                prev_pos = current_pos
            cv2.line(canvas, tuple(prev_pos), tuple(current_pos), (255, 0, 255), 5)
            prev_pos = current_pos

        elif fingers == [1, 0, 0, 0, 0]:  # Clear screen with thumb up
            instruction_placeholder.write("Clearing screen...")
            canvas.fill(0)
            prev_pos = None
            expression = ""
            result = ""

        elif fingers == [0, 1, 1, 0, 0]:  # Pen-up (stop drawing)
            instruction_placeholder.write("Pen-up: Stop drawing")
            prev_pos = None  # Prevent lingering marks

        elif fingers == [1, 1, 1, 1, 1]:  # Erase everything if full hand is used
            instruction_placeholder.write("Erasing everything...")
            canvas.fill(0)
            prev_pos = None
            expression = ""
            result = ""

        elif fingers == [0, 1, 0, 1, 0]:  # Addition (+)
            instruction_placeholder.write("Addition: +")
            expression += " + "

        elif fingers == [0, 1, 0, 0, 1]:  # Subtraction (-)
            instruction_placeholder.write("Subtraction: -")
            expression += " - "

        elif fingers == [0, 1, 1, 1, 0]:  # Multiplication (*)
            instruction_placeholder.write("Multiplication: *")
            expression += " * "

        elif fingers == [0, 1, 1, 0, 1]:  # Division (/)
            instruction_placeholder.write("Division: /")
            expression += " / "

        elif fingers == [0, 0, 0, 0, 0]:  # Evaluate expression
            instruction_placeholder.write("Evaluating expression...")
            try:
                result = str(eval(expression))
                st.write(f"Expression: {expression}")
                st.write(f"Result: {result}")
            except:
                result = "Invalid Expression"
                st.write("Invalid Expression")
    else:
        instruction_placeholder.write("Show your hand to start drawing!")
        prev_pos = None  # Reset if no hands detected

    image_combined = cv2.addWeighted(img, 0.7, canvas, 0.3, 0)
    stframe.image(image_combined, channels="BGR")

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
