from PyQt5 import QtCore, QtGui, QtWidgets
from MainWindow import Ui_MainWindow  # Import the UI created with Qt Designer
from PyQt5.QtWidgets import QLabel, QVBoxLayout, QWidget, QPushButton, QFileDialog
import numpy as np
import matplotlib.pyplot as plt
import cv2
import time

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
        self.ui.comboBox.currentIndexChanged.connect(self.activate_combo_box)

        # Initialize descriptors for the images
        self.descriptor1 = None
        self.descriptor2 = None
        self.selected_value = ""  


    def activate_combo_box(self, index):
        # Enable the combo box if "NCC" is selected, otherwise disable it
        selected_value = self.ui.comboBox.currentText()
        self.selected_value = selected_value

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
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if image_number == 1:
            self.ui.original_image.setPixmap(pixmap.scaled(self.ui.original_image.size(), QtCore.Qt.KeepAspectRatio))
            # Initialize SIFT detector
            sift = cv2.SIFT_create()
            # Detect keypoints and compute descriptors
            keypoints, descriptor1 = sift.detectAndCompute(image, None)
            self.descriptor1 = descriptor1 
        elif image_number == 2:
            self.ui.original_image_2.setPixmap(pixmap.scaled(self.ui.original_image_2.size(), QtCore.Qt.KeepAspectRatio))
            # Initialize SIFT detector
            sift = cv2.SIFT_create()
            # Detect keypoints and compute descriptors
            keypoints, descriptor2 = sift.detectAndCompute(image, None)
            self.descriptor2 = descriptor2

    # Function to calculate the normalized cross-correlation (NCC) score
    def calculate_ncc(self):
        if self.descriptor1 is not None and self.descriptor2 is not None:
            self.show_heatmap()
        else:
            self.ui.output_ncc.setText("Please upload two images first.")

    def SSD(self):
        descriptor1=self.descriptor1
        descriptor2=self.descriptor2
        sum_square= 0
        for m in range(len(descriptor2)-1):
            sum_square +=(descriptor1[m] - descriptor2[m])**2
        
        return sum_square
    
    # NCC calculation function
    def ncc(self):

        descriptor1= self.descriptor1
        descriptor2= self.descriptor2   
        normlized_output1=(descriptor1 - np.mean(descriptor1))/(np.std(descriptor1))
        normlized_output2=(descriptor2 - np.mean(descriptor2))/(np.std(descriptor2))
        correlation_vector= np.multiply(normlized_output1, normlized_output2)
    
        NCC= float(np.mean(correlation_vector))
    
        return NCC

    def feature_matching (self):
            method= self.selected_value
            descriptor1= self.descriptor1
            descriptor2= self.descriptor2
            key_points1= descriptor1.shape[0]
            key_points2= descriptor2.shape[0]
            distance= -np.inf
            y_index= -1    
            
            start_time= time.time()
            time.sleep(0.1)

            for kp1 in range(key_points1):
                for kp2 in range(key_points2):
                    if method=="SSD":
                        score= self.SSD()
                    else: 
                        score= self.ncc()
                    
                    if score > distance:
                        distance= score
                        y_index= kp2
                
                feature= cv2.DMatch()
                feature.queryIdx= kp1
                feature.trainIdx= y_index
                feature.distance= distance
                matched_features=[]
                matched_features.append(feature)

            end_time= time.time()    
            matching_time= end_time-start_time
            self.matched_features= matched_features
            self.matching_time= matching_time
            return matched_features, matching_time

    def show_heatmap(self):
            descriptor1= self.descriptor1
            descriptor2= self.descriptor2
            matched=self.feature_matching()
            matched_features=matched[0]
            heatmap = np.zeros((len(descriptor1), len(descriptor2)))

            # Fill the heatmap with matching scores
            for match in matched_features:
                queryIdx = match.queryIdx
                trainIdx = match.trainIdx
                distance = match.distance
                heatmap[queryIdx][trainIdx] = distance

            # Create the heatmap plot
            plt.figure(figsize=(8, 6))
            plt.imshow(heatmap, cmap='hot', interpolation='nearest')
            plt.colorbar(label='Matching Score')
            plt.xlabel('Descriptor 2 Index')
            plt.ylabel('Descriptor 1 Index')
            plt.title('Feature Matching Heatmap')
            plt.xticks(range(len(descriptor2)))
            plt.yticks(range(len(descriptor1)))
            plt.gca().invert_yaxis()
            plt.colorbar()
            plt.show()
            return None
    
    # Function to visualize the NCC score
    # def show_ncc_visualization(self, ncc_score):
    #     heatmap = np.zeros((100, 100))  # Assuming size of heatmap
    #     heatmap.fill(ncc_score)
    #     plt.imshow(heatmap, cmap='hot', interpolation='nearest')
    #     plt.colorbar()
    #     plt.show()

if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    mainWindow = MainWindow()
    mainWindow.show()
    sys.exit(app.exec_())
