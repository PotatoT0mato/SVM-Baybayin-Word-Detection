#!/usr/bin/env python
# coding: utf-8

# In[1]:


import customtkinter
import tkinter as tk
from threading import Thread
import cv2
from PIL import Image, ImageTk
import os
import numpy as np
from Final_model_draft_16 import predict_character, global_results


# Initialize the detector
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Failed to open the camera.")
    exit()

# Define the directory to save the captured images
imageT_dir = 'captured_images'
if not os.path.exists(imageT_dir):
    os.makedirs(imageT_dir)

# Introduce the running flag
running = True
    
def display_image(file_path):
    img = Image.open(file_path)
    imgtk = ImageTk.PhotoImage(image=img)

    global processed_img_label
    processed_img_label = tk.Label(frame, image=imgtk)
    processed_img_label.imgtk = imgtk
    processed_img_label.pack(pady=10, padx=10, fill="both", expand=True)

def update_image_label():
    global running
    while running and cap.isOpened():
        ret, frame = cap.read()
        if frame is None:
            print("Failed to read from the camera.")
            continue
        
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)
        imgtk = ImageTk.PhotoImage(image=img)
        image_label.configure(image=imgtk)
        image_label.image = imgtk

        
def capture_image():
    global running
    running = False  # Stop the camera feed
    
    ret, frame = cap.read()
    if frame is None:
        print("Failed to read from the camera.")
        return

    # Save the captured image to the directory
    file_path = os.path.join(imageT_dir, 'image.png')
    cv2.imwrite(file_path, frame)

    # Load the captured image
    image_np = cv2.imread(file_path)

    # Convert the input image to grayscale
    image_np_gray = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur to the image
    image_np_blur1 = cv2.medianBlur(image_np_gray, 3)
    image_np_blur2 = cv2.medianBlur(image_np_gray, 51)

    # Divide the blurred images
    divided = np.ma.divide(image_np_blur1, image_np_blur2).data

    # Normalize the divided image
    normed = np.uint8(255 * divided / divided.max())

    # Apply Otsu thresholding to enhance contrast
    _, image_np_threshold = cv2.threshold(normed, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Resize the input image using bicubic interpolation
    image_np_resized = cv2.resize(image_np_threshold, (100, 100), interpolation=cv2.INTER_CUBIC)
    
    # Save the preprocessed image
    preprocessed_file_path = os.path.join(imageT_dir, 'preprocessed_image_test.png')
    cv2.imwrite(preprocessed_file_path, image_np_resized)
    print(f"Saved preprocessed image: {preprocessed_file_path}")
    
    # Hide the video feed label
    image_label.pack_forget()  
    
    # Display the preprocessed image
    display_image(preprocessed_file_path)
    
    # Hide capture and quit buttons
    capture_button.pack_forget()
    quit_button.pack_forget()
    
    # Show convert and back buttons
    convert_button.pack(side="left", padx=5)
    back_button.pack(side="right", padx=5)
    
def go_back():
    global running
    running = True  # Resume the camera feed
    
    # Hide the result labels if they exist
    try:
        predicted_word_label.pack_forget()
        all_possible_words_label.pack_forget()
    except NameError:
        pass  # result labels don't exist yet

    image_label.pack(pady=10, padx=10, fill="both", expand=True)  # Show the video feed label
    
    # Hide convert and back buttons
    convert_button.pack_forget()
    back_button.pack_forget()
    
    # Show capture and quit buttons
    capture_button.pack(side="left", padx=5)
    quit_button.pack(side="right", padx=5)
    
    # Resume camera feed (using threading)
    thread = Thread(target=update_image_label)
    thread.daemon = True
    thread.start()
    
def display_results():
    # Hide the preprocessed image if shown
    try:
        processed_img_label.pack_forget()
    except NameError:
        pass  # processed_img_label doesn't exist yet

    predicted_word = global_results['predicted_word']
    all_possible_words = ', '.join(global_results['all_possible_words'])

    global predicted_word_label
    predicted_word_label = tk.Label(frame, text=f"Predicted Word: {predicted_word}", font=("Roboto", 16))
    predicted_word_label.pack(pady=10, padx=10, fill="both", expand=True)

    global all_possible_words_label
    all_possible_words_label = tk.Label(frame, text=f"All Possible Words: {all_possible_words}", font=("Roboto", 16))
    all_possible_words_label.pack(pady=10, padx=10, fill="both", expand=True)
    
    # Hide the convert button and show only back button
    convert_button.pack_forget()
    
    
def convert_image():
    test_img_path = 'captured_images/preprocessed_image_test.png'
    predict_character(test_img_path)
    
    # Now that the predictions have been stored in global_results, display them on the GUI
    display_results()

customtkinter.set_appearance_mode("dark")
customtkinter.set_default_color_theme("dark-blue")

root = customtkinter.CTk()
root.geometry("600x500")

frame = customtkinter.CTkFrame(master=root)
frame.pack(pady=20, padx=60, fill="both", expand=True)

label = customtkinter.CTkLabel(master=frame, text="Baybayin Converter", font=("Roboto", 24))
label.pack(pady=12, padx=10)

image_label = tk.Label(frame)
image_label.pack(pady=10, padx=10, fill="both", expand=True)

button_frame = customtkinter.CTkFrame(master=frame)
button_frame.pack(pady=10, padx=10)

capture_button = customtkinter.CTkButton(master=button_frame, text="Capture", command=capture_image)
capture_button.pack(side="left", padx=5)

def close_window():
    cap.release()
    root.quit()
    root.destroy()

quit_button = customtkinter.CTkButton(master=button_frame, text="Quit", command=close_window)
quit_button.pack(side="right", padx=5)

convert_button = customtkinter.CTkButton(master=button_frame, text="Convert", command=convert_image)
back_button = customtkinter.CTkButton(master=button_frame, text="Back", command=go_back)

# Use threading to start the video capture loop
thread = Thread(target=update_image_label)
thread.daemon = True
thread.start()

root.mainloop()
cap.release()


# In[ ]:




