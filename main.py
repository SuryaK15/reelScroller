import cv2
import dlib
import time
import pyautogui

# Load the face detector and shape predictor
detector = dlib.get_frontal_face_detector()
predictor_path = "shape_predictor_68_face_landmarks.dat"  # Ensure this path is correct
predictor = dlib.shape_predictor(predictor_path)

# Scroll down action
def scroll_down():
    display_message("Scrolling Down")
    pyautogui.scroll(-100)  # Scroll down

# Scroll up action
def scroll_up():
    display_message("Scrolling Up")
    pyautogui.scroll(100)  # Scroll up

# Display "Still" condition
def still():
    display_message("Head still: No action")

# Pause video when no face detected
def pause_video():
    global paused
    if not paused:  # Only pause if not already paused
        display_message("No face detected: Pausing video")
        pyautogui.click()  # Simulate click to pause the video
        paused = True  # Set paused state

# Resume video when face detected
def resume_video():
    global paused
    if paused:  # Only resume if currently paused
        display_message("Face detected: Resuming video")
        pyautogui.click()  # Simulate click to resume the video
        paused = False  # Reset paused state

# Detect face and landmarks (nose y-coordinate)
def detect_face_and_nose(gray):
    faces = detector(gray)
    if len(faces) > 0:
        face = faces[0]  # Take the first detected face
        landmarks = predictor(gray, face)
        nose_y = landmarks.part(30).y  # Get y-coordinate of the nose
        return nose_y, face, landmarks
    return None, None, None

# Draw landmarks and rectangle on face
def draw_landmarks(frame, face, landmarks):
    if face is not None:
        cv2.rectangle(frame, (face.left(), face.top()), (face.right(), face.bottom()), (0, 255, 0), 2)
        for n in range(0, 68):
            x = landmarks.part(n).x
            y = landmarks.part(n).y
            cv2.circle(frame, (x, y), 3, (255, 0, 0), -1)

# Display status and instructions in the video modal
def display_message(message):
    global frame
    # Overlay the message on the frame using OpenCV's putText
    font = cv2.FONT_HERSHEY_SIMPLEX
    text_size = cv2.getTextSize(message, font, 0.8, 2)[0]
    text_x = (frame.shape[1] - text_size[0]) // 2  # Center the text
    text_y = frame.shape[0] - 50  # Place near the bottom
    cv2.putText(frame, message, (text_x, text_y), font, 0.8, (0, 255, 0), 2, cv2.LINE_AA)

# Display current nose Y-coordinate and baseline
def display_nose_position(nose_y, initial_nose_y):
    global frame
    if initial_nose_y is not None:
        message = f"Nose Y Position: {nose_y} | Adjust to: {initial_nose_y} for best experience"
    else:
        message = f"Nose Y Position: {nose_y} | Restart to set baseline"
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    text_size = cv2.getTextSize(message, font, 0.6, 2)[0]
    text_x = (frame.shape[1] - text_size[0]) // 2  # Center the text
    text_y = 30  # Display at the top of the frame
    cv2.putText(frame, message, (text_x, text_y), font, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

# Draw a red target circle to guide the user to position their nose
def draw_target_circle(frame, center_y, radius):
    center_x = frame.shape[1] // 2  # Center horizontally
    color = (0, 0, 255)  # Red circle
    thickness = 2  # Circle line thickness
    cv2.circle(frame, (center_x, center_y), radius, color, thickness)

# Countdown to capture initial nose y-coordinate
def countdown_to_capture(seconds):
    global frame
    for i in range(seconds, 0, -1):
        # Show the camera frame with countdown overlay
        display_message(f"Adjust your face... Capturing in {i} seconds")
        cv2.imshow('Image', frame)
        cv2.waitKey(1000)  # Wait for 1 second

# Initialize and run the main loop
def run_main_loop():
    global frame, paused
    paused = False  # Track whether the video is paused or not

    # Open Camera
    camera = cv2.VideoCapture(0)

    # Gesture detection variables
    initial_nose_y = None  # Baseline for nose position (y-coordinate)
    movement_threshold = 13  # Movement threshold for scrolling
    still_threshold = (-movement_threshold, movement_threshold)  # Threshold for stillness
    cooldown_period = 1.5  # Cooldown period in seconds (e.g., 1 second)
    last_scroll_time = time.time()  # Track the last time a scroll action happened
    face_absent_time = None  # Track when the face is not detected

    # Initialization countdown (5 seconds)
    countdown_completed = False
    countdown_time = 4

    while True:
        ret, frame = camera.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect face and get nose position
        nose_y, face, landmarks = detect_face_and_nose(gray)

        if not countdown_completed:
            if nose_y is not None:
                # Show the countdown only if the face is detected
                countdown_to_capture(countdown_time)
                countdown_completed = True  # Complete the countdown

        if nose_y is not None:
            if initial_nose_y is None and countdown_completed:
                # Set the first detected nose position as the baseline (0, 0)
                initial_nose_y = nose_y
                display_message("Baseline set for nose position.")

            # Show current nose y-coordinate and baseline to guide the user
            display_nose_position(nose_y, initial_nose_y)

            # Draw the target circle to guide the user to align their nose
            if initial_nose_y is not None:
                draw_target_circle(frame, initial_nose_y, movement_threshold)

            # Reset face_absent_time because face is detected
            face_absent_time = None

            # Calculate the difference in y position from the baseline
            y_movement = nose_y - initial_nose_y

            # Check the time since the last scroll action
            current_time = time.time()
            time_since_last_scroll = current_time - last_scroll_time

            # Check for scrolling and still conditions
            if y_movement > movement_threshold and time_since_last_scroll > cooldown_period:
                scroll_up()  # Perform scroll up
                last_scroll_time = current_time  # Reset the last scroll time
            elif y_movement < -movement_threshold and time_since_last_scroll > cooldown_period:
                scroll_down()  # Perform scroll down
                last_scroll_time = current_time  # Reset the last scroll time
            elif still_threshold[0] <= y_movement <= still_threshold[1]:
                still()  # Display "still" message

            # Draw face landmarks
            draw_landmarks(frame, face, landmarks)

            # Resume video if paused and face detected
            resume_video()

        else:
            # Always draw the target circle, even if face is not detected
            if initial_nose_y is not None:
                draw_target_circle(frame, initial_nose_y, movement_threshold)
            else:
                draw_target_circle(frame, frame.shape[0] // 2, movement_threshold)  # Default center if no baseline

            if face_absent_time is None:
                # Start the timer when face is not detected
                face_absent_time = time.time()
            elif time.time() - face_absent_time > 5:  # Wait for 2 seconds before pausing
                pause_video()  # Pause the video
                face_absent_time = None  # Reset after pausing

        # Display reset message option if user wants to reset
        if initial_nose_y is not None:
            cv2.putText(frame, "Press 'r' to reset", (10, frame.shape[0] - 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Display the camera frame
        cv2.imshow('Image', frame)

        # Handle user inputs for quitting or resetting the script
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):  # Quit the script
            break
        elif key == ord('r'):  # Reset the script
            display_message("Script reset. Please adjust your face again.")
            initial_nose_y = None  # Reset initial nose position
            countdown_completed = False  # Reset countdown flag

    # Release resources
    camera.release()
    cv2.destroyAllWindows()

# Run the main function
if __name__ == "__main__":
    run_main_loop()