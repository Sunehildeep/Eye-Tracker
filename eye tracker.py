import cv2
import numpy as np


if __name__=='__main__':
    # Define the video capture object
    cap = cv2.VideoCapture(0)            

    # Keep reading the frames from the webcam 
    # until the user hits the 'Esc' key
    while True:
        # Grab the current frame
        ret, frame = cap.read()

        # Convert the image to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Apply Gaussian Blur to the image to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # Detect eyes in the image
        eyes = cv2.CascadeClassifier('haarcascade_eye.xml').detectMultiScale(blurred, 1.3, 5)

        # Iterate over each detected eye
        for (ex, ey, ew, eh) in eyes:
            # Extract the region of interest (ROI) which is the eye area
            roi_gray = gray[ey:ey+eh, ex:ex+ew]
            roi_color = frame[ey:ey+eh, ex:ex+ew]

            # Apply thresholding to the ROI to extract the pupil
            _, thresh = cv2.threshold(roi_gray, 30, 255, cv2.THRESH_BINARY)

            # Find the contours in the thresholded image
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

            # Find the contour with the largest area which is the pupil
            if len(contours) > 0:
                cnt = max(contours, key=cv2.contourArea)
                (x, y, w, h) = cv2.boundingRect(cnt)

                # Draw a rectangle around the pupil
                cv2.rectangle(roi_color, (x, y), (x + w, y + h), (0, 255, 0), 3)

        # Display the input and output
        cv2.imshow('Eye Detection', frame)

        # Check if the user hit the 'Esc' key
        key = cv2.waitKey(1)
        if key == 27:
            break

    # Close all the windows
    cv2.destroyAllWindows()
