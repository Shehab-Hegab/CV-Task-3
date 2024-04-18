import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog
from PyQt5.uic import loadUi
import cv2
import time
from PyQt5.QtGui import QPixmap, QImage, QPainter, QColor
from PyQt5.QtCore import Qt
import numpy as np


class MyWidget(QMainWindow):
    def __init__(self):
        super().__init__()

        # Load the UI file
        loadUi('task_3.ui', self)

        # Connect button click events to functions
        self.load_button.clicked.connect(self.load_image)
        self.detect_button.clicked.connect(self.detect_corners)

        # Initialize image
        self.image = None

    def load_image(self):
        # Open file dialog to select image
        file_path, _ = QFileDialog.getOpenFileName(self, "Open Image", "D:\\Documents\\term_2\\cv\\tasks\\CV-Task_3", "Image Files (*.jpg *.png *.bmp)")

        if file_path:
            # Load image using OpenCV
            self.image = cv2.imread(file_path)

            # Display original image
            self.display_image(self.image, self.original_image)

    def display_image(self, image, label):
        if len(image.shape) == 3:  # Color image (3 channels)
            h, w, ch = image.shape
            bytes_per_line = ch * w
            q_image = QImage(image.data, w, h, bytes_per_line, QImage.Format_BGR888)
        else:  # Grayscale image (1 channel)
            h, w = image.shape
            q_image = QImage(image.data, w, h, w, QImage.Format_Grayscale8)

        pixmap = QPixmap.fromImage(q_image)
        label.setPixmap(pixmap.scaled(label.size(), aspectRatioMode=Qt.AspectRatioMode.KeepAspectRatio))

    def detect_corners(self):
        if self.image is None:
            return

        start_time = time.time()

        # Get threshold and sensitivity values from LineEdit
        threshold = float(self.lineEdit_2.text())
        sensitivity = float(self.lineEdit.text())

        # Convert image to grayscale
        gray_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)

        # Perform Harris corner detection
        corners = cv2.cornerHarris(gray_image, blockSize=2, ksize=3, k=sensitivity)

        # Apply threshold to corner responses
        thresholded_corners = np.zeros_like(corners)
        thresholded_corners[corners > threshold * corners.max()] = 255

        # Convert to uint8 for display
        thresholded_corners = np.uint8(thresholded_corners)

        # Find corners positions
        corner_positions = np.argwhere(thresholded_corners > 0)

        # Draw red circles at corner positions
        image_with_corners = self.image.copy()
        for corner in corner_positions:
            cv2.circle(image_with_corners, tuple(corner[::-1]), 5, (0, 0, 255), -1)  # Draw filled red circles

        # Display image with corners
        self.display_image(image_with_corners, self.detect_corner)

        end_time = time.time()
        execution_time = end_time - start_time

        # Display execution time
        self.lineEdit_3.setText(f"{execution_time:.5f} seconds")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MyWidget()
    window.show()
    sys.exit(app.exec_())
