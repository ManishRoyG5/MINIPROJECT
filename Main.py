import numpy as np
import cv2 as cv
import os
from ObjectDetection import detectObject
from gtts import gTTS
from playsound import playsound
from threading import Thread, Lock
import time

global class_labels
global cnn_model
global cnn_layer_names
playcount_lock = Lock()  # Lock for thread-safe playcount access

def deleteDirectory():
    filelist = [f for f in os.listdir('play') if f.endswith(".mp3")]
    for f in filelist:
        os.remove(os.path.join('play', f))

def speak(data):
    # Create and start the audio thread
    def run():
        with playcount_lock:
            audio_file = f"play/temp_{time.time()}.mp3"  # Use a unique filename
        tts = gTTS(text=data, lang='en', slow=False)
        tts.save(audio_file)  # Save the audio file
        playsound(audio_file)  # Play the audio file
        os.remove(audio_file)  # Clean up the audio file after playing

    Thread(target=run).start()

def loadLibraries():
    global class_labels, cnn_model, cnn_layer_names
    class_labels = open('model/yolov3-labels').read().strip().split('\n')
    cnn_model = cv.dnn.readNetFromDarknet('model/yolov3.cfg', 'model/yolov3.weights')
    cnn_layer_names = cnn_model.getLayerNames()
    cnn_layer_names = [cnn_layer_names[i[0] - 1] for i in cnn_model.getUnconnectedOutLayers()]

def detectFromVideo():
    label_colors = np.random.randint(0, 255, size=(len(class_labels), 3), dtype='uint8')
    video = cv.VideoCapture(0)

    if not video.isOpened():
        print("Error: Could not open video.")
        return

    detected_objects = set()  # Set to track currently detected objects
    previously_spoken = set()  # Set to track spoken objects
    speak_gap = 5  # Time gap between speaking detected objects in seconds
    last_detection_time = {}  # Track last detection time for each object

    try:
        while True:
            frame_grabbed, frames = video.read()
            if not frame_grabbed:
                print("Error: Frame not grabbed.")
                break
            
            frame_height, frame_width = frames.shape[:2]
            frames, cls, _, _, _, _ = detectObject(cnn_model, cnn_layer_names, frame_height, frame_width, frames, label_colors, class_labels)

            cv.imshow("Detected Objects", frames)

            current_time = time.time()

            # Speak the detected objects
            for obj in cls:
                detected_objects.add(obj)  # Mark this object as currently detected
                if obj not in previously_spoken or (current_time - last_detection_time.get(obj, 0) >= speak_gap):
                    speak("Detected Object: " + obj)
                    previously_spoken.add(obj)  # Mark this object as spoken
                    last_detection_time[obj] = current_time  # Update the last detection time

            # Handle objects that are no longer in the frame
            for obj in list(previously_spoken):
                if obj not in cls:
                    previously_spoken.remove(obj)  # Remove object if not detected anymore

            if cv.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        video.release()
        cv.destroyAllWindows()

if __name__ == '__main__':
    if not os.path.exists('play'):
        os.makedirs('play')
    loadLibraries()
    deleteDirectory()
    detectFromVideo()