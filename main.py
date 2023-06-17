import cv2
import numpy as np
import streamlit as st
from iris_tracking import process_frame


def main():
    st.title("Iris Tracking")

    threshold = st.number_input("Threshold", min_value=0.0, max_value=1.0, step=0.01)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        st.error("Failed to open the webcam.")
        return

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        iris_pos, ratio = process_frame(frame, threshold)

        frame_with_info = np.copy(frame)
        cv2.putText(frame_with_info, f"Iris position and ratio: {iris_pos}, {ratio}", (30, 30),
                    cv2.FONT_HERSHEY_PLAIN, 1.2, (255, 0, 0), 1, cv2.LINE_AA)

        st.image(frame_with_info, channels="BGR")

        if st.button("Stop"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

