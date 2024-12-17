import os
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
from ultralytics import YOLO
import cv2
import pyttsx3  # Text-to-Speech library
from llama3 import generate_warning

# Initialize text-to-speech engine
tts_engine = pyttsx3.init()

# Function to read text aloud
def read_text_aloud():
    content = result_text.get("1.0", tk.END).strip()
    if content:
        tts_engine.setProperty("rate", 150)  # Set speed (150 WPM)
        tts_engine.say(content)
        tts_engine.runAndWait()


# Function to handle opening and processing an image or video
def open_media(event):
    global last_predict_folder  # Keep track of the last created folder

    file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png"), ("Video files", "*.mp4 *.avi *.mov")])
    if not file_path:
        return

    try:
        if file_path.endswith(('jpg', 'jpeg', 'png')):  # Image file
            im1 = Image.open(file_path)
            model = YOLO("YOLO11.pt")  # Replace with YOLOv11 weights
            results = model.predict(source=im1, imgsz=640, save=False)  # Don't save automatically

            detected_obj_names = []
            for r in results:
                for c in r.boxes.cls:
                    detected_obj_names.append(model.names[int(c)])

            detected_obj_names_string = ', '.join(detected_obj_names)
            print(detected_obj_names_string)

            # Annotate and save the image with bounding boxes
            annotated_img = results[0].plot()  # Annotate the image with bounding boxes
            save_path = os.path.join("runs/detect", "annotated_image.jpg")
            cv2.imwrite(save_path, annotated_img)  # Save the annotated image
            print(f"Annotated image saved at: {save_path}")


        elif file_path.endswith(('mp4', 'avi', 'mov')):  # Video file
            # Create a new 'predict' folder with an incremented number
            predict_dir_base = "runs/detect/predict"
            i = 1
            last_predict_folder = predict_dir_base
            while os.path.exists(f"{predict_dir_base}{i}"):
                i += 1
            last_predict_folder = f"{predict_dir_base}{i}"
            os.makedirs(last_predict_folder)

            cap = cv2.VideoCapture(file_path)
            model = YOLO("YOLO11.pt")  # Replace with YOLOv11 weights

    # Define the codec and create VideoWriter object
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            output_video_path = os.path.join(last_predict_folder, "output_video.mp4")
            out = cv2.VideoWriter(output_video_path, fourcc, fps, (640, 640))  # Set resolution to 640x640

            detected_obj_names = []
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                # Resize the frame to 640x640
                resized_frame = cv2.resize(frame, (640, 640))

                results = model.predict(source=resized_frame, imgsz=640, save=False)  # Do not save intermediate images
                annotated_frame = results[0].plot()  # Annotate the frame with bounding boxes

                out.write(annotated_frame)  # Write the annotated frame to the video

                for r in results:
                    for c in r.boxes.cls:
                        detected_obj_names.append(model.names[int(c)])

            detected_obj_names_string = ', '.join(set(detected_obj_names))  # Removing duplicates
            print(detected_obj_names_string)

            cap.release()
            out.release()

            # Play the output video
            play_video(output_video_path)

        response = generate_warning(detected_obj_names_string)

        print(response)

        # Display the generated response
        result_text.config(state="normal")
        result_text.delete("1.0", tk.END)
        result_text.insert(tk.END, response)
        result_text.config(state="disabled")

        # Display the annotated image in the GUI
        im_annotated = Image.open(save_path).resize((400, 300), Image.LANCZOS)
        annotated_image = ImageTk.PhotoImage(im_annotated)
        global image_label
        image_label = tk.Label(image=annotated_image)
        image_label.image = annotated_image
        image_label.pack(in_=drag_drop_frame)
    except Exception as e:
        print(f"Error processing media: {e}")

# Function to play the result video
def play_video(video_path):
    cap = cv2.VideoCapture(video_path)

    playback_window = tk.Toplevel(root)
    playback_window.title("Result Video Playback")
    playback_canvas = tk.Canvas(playback_window, width=640, height=480)
    playback_canvas.pack()

    def update_frame():
        ret, frame = cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = ImageTk.PhotoImage(Image.fromarray(frame))
            playback_canvas.create_image(0, 0, anchor=tk.NW, image=img)
            playback_window.img = img
            playback_window.after(30, update_frame)
        else:
            cap.release()

    update_frame()

# Function to clear inputs and results
def clear_input_and_results():
    result_text.config(state="normal")
    result_text.delete("1.0", tk.END)
    result_text.config(state="disabled")
    global image_label
    if image_label:
        image_label.pack_forget()  # Remove the image from the GUI
        image_label = None 
# Function to handle live camera feed
def open_camera():
    cap = cv2.VideoCapture(0)  # Open the default camera
    model = YOLO("YOLO11.pt")  # Replace with YOLOv5 weights

    print("Press 'q' to exit the live camera feed.")

    detected_obj_names = set()  # Keep track of detected object names
    last_detected_class = None  # Track the last detected class to trigger prompt update

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame from camera. Exiting...")
            break

        # Resize the frame to match YOLO input requirements
        resized_frame = cv2.resize(frame, (640, 640))

        # Perform inference
        results = model(resized_frame, show=True)  # Display annotated frames live

        # Extract and track detected object names
        current_detected_classes = set()
        for result in results:
            boxes = result.boxes
            classes = result.names
            for box in boxes:
                detected_class = classes[int(box.cls)]
                current_detected_classes.add(detected_class)
        
        # Check if there is a change in the detected class
        if current_detected_classes != detected_obj_names:
            detected_obj_names.update(current_detected_classes)
            last_detected_class = next(iter(current_detected_classes))  # Get the first class (if multiple detected)

            # Generate a warning message for the detected objects if the class has changed
            if last_detected_class:
                detected_obj_names_string = ', '.join(detected_obj_names)
                print(f"Detected objects: {detected_obj_names_string}")

                response = generate_warning(detected_obj_names_string)

                print(response)

                # Display the generated response in the GUI
                result_text.config(state="normal")
                result_text.delete("1.0", tk.END)
                result_text.insert(tk.END, response)
                result_text.config(state="disabled")

        # Exit condition
        if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to quit
            break

    cap.release()
    cv2.destroyAllWindows()

    # Clear results after quitting the camera
    result_text.config(state="normal")
    result_text.delete("1.0", tk.END)
    result_text.config(state="disabled")
# Set up the GUI
root = tk.Tk()
root.geometry("800x600")
root.title("Groq YOLOv11 - Integration Playground")

drag_drop_frame = tk.Frame(root)
drag_drop_frame.pack(side="left", fill="both", expand=True)

# Button for inputting media
label = tk.Button(drag_drop_frame, text="Input Image or Video")
label.pack(pady=10)
label.bind("<Button-1>", open_media)

# Button for live camera test
live_camera_button = tk.Button(drag_drop_frame, text="Live Camera Test")
live_camera_button.pack(pady=10)
live_camera_button.bind("<Button-1>", lambda event: open_camera())


results_frame = tk.Frame(root)
results_frame.pack(side="right", fill="both", expand=True)

result_label = tk.Label(results_frame, text="Results:")
result_label.pack(pady=10)

result_text = tk.Text(results_frame, height=40, width=80, state="disabled")
result_text.pack(pady=30)
# "Read Text" button below the text area
read_button = tk.Button(results_frame, text="Read Text", command=read_text_aloud)
read_button.pack(pady=5)


clear_button = tk.Button(results_frame, text="Clear Input and Results", command=clear_input_and_results)
clear_button.pack(pady=10)

root.mainloop()
