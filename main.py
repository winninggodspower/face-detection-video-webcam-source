import cv2
import datetime
import requests
import tkinter as tk
from tkinter import ttk, filedialog
from PIL import Image, ImageTk

class FaceDetectionApp:
    def __init__(self, window):
        self.window = window
        self.window.title("Face Detection App")
        
        # Variables
        self.video_source = 0  # Default to webcam
        self.cap = None
        self.is_running = False
        self.last_detected = datetime.datetime.now() - datetime.timedelta(seconds=11)
        
        # Face classifier
        self.face_classifier = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        
        # Create GUI elements
        self.create_widgets()
        
    def create_widgets(self):
        # Controls frame
        controls_frame = ttk.Frame(self.window, padding="10")
        controls_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Source selection
        ttk.Label(controls_frame, text="Video Source:").grid(row=0, column=0, padx=5)
        self.source_var = tk.StringVar(value="Webcam")
        source_menu = ttk.OptionMenu(controls_frame, self.source_var, "Webcam", "Webcam", "Video File", command=self.change_source)
        source_menu.grid(row=0, column=1, padx=5)
        
        # File selection button
        self.file_button = ttk.Button(controls_frame, text="Choose File", command=self.choose_file)
        self.file_button.grid(row=0, column=2, padx=5)
        self.file_button.state(['disabled'])
        
        # Start/Stop button
        self.start_stop_button = ttk.Button(controls_frame, text="Start", command=self.toggle_detection)
        self.start_stop_button.grid(row=0, column=3, padx=5)
        
        # Video frame
        self.video_frame = ttk.Frame(self.window, padding="10")
        self.video_frame.grid(row=1, column=0)
        
        self.canvas = tk.Canvas(self.video_frame, width=640, height=480)
        self.canvas.pack()

    def change_source(self, selection):
        if selection == "Video File":
            self.file_button.state(['!disabled'])
        else:
            self.file_button.state(['disabled'])
            self.video_source = 0  # Reset to webcam

    def choose_file(self):
        filename = filedialog.askopenfilename(
            filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv")]
        )
        if filename:
            self.video_source = filename

    def toggle_detection(self):
        if self.is_running:
            self.is_running = False
            self.start_stop_button.config(text="Start")
            if self.cap is not None:
                self.cap.release()
                self.cap = None
        else:
            self.is_running = True
            self.start_stop_button.config(text="Stop")
            self.cap = cv2.VideoCapture(self.video_source)
            self.update()

    def send_push_notification(self):
        print('sending notification')
        res = requests.post('https://api.mynotifier.app', {
            "apiKey": '63e188a7-f596-4a31-872a-6bd7823a9767',
            "message": "Detected Face!! ",
            "description": "Camera Detected Someone's face. Go check who is",
            "type": "info", 
        })
        print(res)

    def update(self):
        if self.is_running and self.cap is not None:
            ret, frame = self.cap.read()
            if ret:
                # Process frame
                gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = self.face_classifier.detectMultiScale(
                    gray_image, scaleFactor=1.3, minNeighbors=5, minSize=(40, 40)
                )
                
                # Draw rectangles and handle notifications
                if len(faces) > 0:
                    if (datetime.datetime.now() - self.last_detected).seconds > 10:
                        self.send_push_notification()
                        self.last_detected = datetime.datetime.now()
                
                for (x, y, w, h) in faces:
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 4)
                
                # Convert frame to PhotoImage
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                photo = ImageTk.PhotoImage(image=Image.fromarray(rgb_frame))
                
                # Update canvas
                self.canvas.create_image(0, 0, image=photo, anchor=tk.NW)
                self.canvas.photo = photo
            
            # Schedule next update
            self.window.after(10, self.update)

    def __del__(self):
        if self.cap is not None:
            self.cap.release()

def main():
    root = tk.Tk()
    app = FaceDetectionApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()