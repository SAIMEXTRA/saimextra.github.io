from flask import Flask, render_template, request, redirect, url_for, send_from_directory, flash
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
from tkinter import messagebox
import threading
import time


app = Flask(__name__)

# Set a secret key for secure sessions
app.secret_key = "your_secret_key_here"

# Set upload folder
UPLOAD_FOLDER = 'static/uploads/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def process_tspot_image(image_path, panel_name="Unknown"):
    # Same logic as in your original backend code, modified to return the result as an integer
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("Unable to load the image.")
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    circles = cv2.HoughCircles(
        gray_image, 
        cv2.HOUGH_GRADIENT, 
        dp=1.2, 
        minDist=50, 
        param1=50, 
        param2=30, 
        minRadius=50, 
        maxRadius=150
    )
    if circles is not None:
        circles = np.uint16(np.around(circles))
        for circle in circles[0, :1]:  # Process the first detected circle
            x, y, radius = circle
            mask = np.zeros_like(gray_image)
            cv2.circle(mask, (x, y), radius, (255, 255, 255), -1)
            cropped_image = cv2.bitwise_and(gray_image, gray_image, mask=mask)
            x1, y1 = max(0, x - radius), max(0, y - radius)
            x2, y2 = min(cropped_image.shape[1], x + radius), min(cropped_image.shape[0], y + radius)
            cropped_image = cropped_image[y1:y2, x1:x2]
            kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
            sharpened_image = cv2.filter2D(cropped_image, -1, kernel)
            blurred_image = cv2.GaussianBlur(sharpened_image, (7, 7), 0)
            _, binary_image = cv2.threshold(blurred_image, 100, 255, cv2.THRESH_BINARY_INV)
            contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            min_area = 0
            max_area = 80
            valid_spots = 0
            for contour in contours:
                area = cv2.contourArea(contour)
                if min_area <= area <= max_area:
                    valid_spots += 1
            return valid_spots
    raise ValueError("No circular region detected.")

def process_tspot_image_positive_panel(image_path):
    # Same as process_tspot_image but tailored for positive panel
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("Unable to load the image.")
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    circles = cv2.HoughCircles(
        gray_image, 
        cv2.HOUGH_GRADIENT, 
        dp=1.2, 
        minDist=50, 
        param1=50, 
        param2=30, 
        minRadius=50, 
        maxRadius=150
    )
    if circles is not None:
        circles = np.uint16(np.around(circles))
        for circle in circles[0, :1]:  # Process the first detected circle
            x, y, radius = circle
            mask = np.zeros_like(gray_image)
            cv2.circle(mask, (x, y), radius, (255, 255, 255), -1)
            cropped_image = cv2.bitwise_and(gray_image, gray_image, mask=mask)
            x1, y1 = max(0, x - radius), max(0, y - radius)
            x2, y2 = min(cropped_image.shape[1], x + radius), min(cropped_image.shape[0], y + radius)
            cropped_image = cropped_image[y1:y2, x1:x2]
            kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
            sharpened_image = cv2.filter2D(cropped_image, -1, kernel)
            blurred_image = cv2.GaussianBlur(sharpened_image, (5, 5), 0)
            _, binary_image = cv2.threshold(blurred_image, 50, 255, cv2.THRESH_BINARY_INV)
            contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            return len(contours)
    raise ValueError("No circular region detected.")
def delete_files_after_delay(file_paths, delay=30):
    """Delete files after a delay."""
    time.sleep(delay)
    for file_path in file_paths:
        if os.path.exists(file_path):
            os.remove(file_path)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/uploads', methods=['POST'])
def upload_file():
    try:
        files = request.files.getlist('image_files')
        if len(files) != 4:
            flash('Please upload exactly 4 images: Nil Control, Panel A, Panel B, and Positive Panel.')
            return redirect(url_for('index'))

        filenames = []
        for file in files:
            filename = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filename)
            filenames.append(filename)

        # Process each image and store spot counts
        nil_control_spots = process_tspot_image(filenames[0])
        panel_a_spots = process_tspot_image(filenames[1])
        panel_b_spots = process_tspot_image(filenames[2])
        positive_panel_spots = process_tspot_image_positive_panel(filenames[3])

        # Validate test quality control
        if nil_control_spots >= 10 or positive_panel_spots <= 20:
            flash("Invalid test: Negative Control must have <10 spots, and Positive Panel must have >20 spots.")
            for filename in filenames:
                os.remove(filename)  # Delete the uploaded files
            return redirect(url_for('index'))

        # Calculate results
        panel_a_result = panel_a_spots - nil_control_spots
        panel_b_result = panel_b_spots - nil_control_spots

        result = f"Panel A - NILL Result: {panel_a_result}, Panel B - NILL Result: {panel_b_result}\n"

        if panel_a_result >= 6 or panel_b_result >= 6:
            result += "Test Result: Positive"
        elif panel_a_result < 5 and panel_b_result < 5:
            result += "Test Result: Negative"
        else:
            result += "Test Result: Borderline"

        # Start a background thread to delete files after a delay
        threading.Thread(target=delete_files_after_delay, args=(filenames,)).start()

        # Render the results page
        return render_template(
            'result.html', 
            nil_control_image=filenames[0], 
            panel_a_image=filenames[1], 
            panel_b_image=filenames[2], 
            positive_panel_image=filenames[3],
            nil_control_spots=nil_control_spots,
            panel_a_spots=panel_a_spots,
            panel_b_spots=panel_b_spots,
            positive_panel_spots=positive_panel_spots,
            result=result
        )

    except ValueError as e:
        flash(str(e))
        return redirect(url_for('index'))
    except Exception as e:
        flash(f"Unexpected error: {str(e)}")
        return redirect(url_for('index'))

if __name__ == '__main__':
    app.run(debug=True)