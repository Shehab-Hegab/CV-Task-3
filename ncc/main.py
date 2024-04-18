from PyQt5 import QtCore, QtGui, QtWidgets
from MainWindow import Ui_MainWindow  # Import the UI created with Qt Designer
from PyQt5.QtWidgets import QLabel, QVBoxLayout, QWidget, QPushButton, QFileDialog
import numpy as np
import matplotlib.pyplot as plt

class ImageWindow(QtWidgets.QWidget):
    def __init__(self, image_path):
        super().__init__()
        self.setWindowTitle("Uploaded Image")
        layout = QVBoxLayout()
        self.image_label = QLabel()
        pixmap = QtGui.QPixmap(image_path)
        self.image_label.setPixmap(pixmap)
        layout.addWidget(self.image_label)
        self.setLayout(layout)

class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        self.ui = Ui_MainWindow()  
        self.ui.setupUi(self)  # Setup the UI defined in Ui_MainWindow

        # Connect buttons to functions
        self.ui.load_original_image.clicked.connect(self.upload_image)
        self.ui.load_original_image_2.clicked.connect(self.upload_image2)
        self.ui.NCC_button.clicked.connect(self.calculate_ncc)

        # Initialize descriptors for the images
        self.descriptor1 = None
        self.descriptor2 = None

    # Function to upload the first image
    def upload_image(self):
        file_dialog = QFileDialog()
        image_path, _ = file_dialog.getOpenFileName(self, "Open Image", "", "Image Files (*.png *.jpg *.bmp)")
        if image_path:
            self.show_image(image_path, 1)

    # Function to upload the second image
    def upload_image2(self):
        file_dialog = QFileDialog()
        image_path, _ = file_dialog.getOpenFileName(self, "Open Image", "", "Image Files (*.png *.jpg *.bmp)")
        if image_path:
            self.show_image(image_path, 2)

    # Function to display the uploaded image
    def show_image(self, image_path, image_number):
        pixmap = QtGui.QPixmap(image_path)
        if image_number == 1:
            self.ui.original_image.setPixmap(pixmap.scaled(self.ui.original_image.size(), QtCore.Qt.KeepAspectRatio))
            image = plt.imread(image_path)
            grayscale_image = np.dot(image[...,:3], [0.2989, 0.5870, 0.1140])  # Convert to grayscale
            self.descriptor1 = np.random.rand(128)  # Example descriptor
        elif image_number == 2:
            self.ui.original_image_2.setPixmap(pixmap.scaled(self.ui.original_image_2.size(), QtCore.Qt.KeepAspectRatio))
            image = plt.imread(image_path)
            grayscale_image = np.dot(image[...,:3], [0.2989, 0.5870, 0.1140])  # Convert to grayscale
            self.descriptor2 = np.random.rand(128)  # Example descriptor

    # Function to calculate the normalized cross-correlation (NCC) score
    def calculate_ncc(self):
        if self.descriptor1 is not None and self.descriptor2 is not None:
            ncc_score = self.ncc(self.descriptor1, self.descriptor2)
            self.show_ncc_visualization(ncc_score)
        else:
            self.ui.output_ncc.setText("Please upload two images first.")

    # NCC calculation function
    def ncc(self, descriptor1, descriptor2):
        mean_d1 = np.mean(descriptor1)
        mean_d2 = np.mean(descriptor2)
        std_d1 = np.std(descriptor1)
        std_d2 = np.std(descriptor2)
        score = np.sum((descriptor1 - mean_d1) * (descriptor2 - mean_d2)) / (std_d1 * std_d2 * len(descriptor1))
        return score

    # Function to visualize the NCC score
    def show_ncc_visualization(self, ncc_score):
        heatmap = np.zeros((100, 100))  # Assuming size of heatmap
        heatmap.fill(ncc_score)
        plt.imshow(heatmap, cmap='hot', interpolation='nearest')
        plt.colorbar()
        plt.show()

if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    mainWindow = MainWindow()
    mainWindow.show()
    sys.exit(app.exec_())
