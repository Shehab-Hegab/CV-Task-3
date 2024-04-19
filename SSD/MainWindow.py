# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'Ui_MainWindow.ui'
#
# Created by: PyQt5 UI code generator 5.15.10
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1007, 717)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.tabWidget = QtWidgets.QTabWidget(self.centralwidget)
        self.tabWidget.setGeometry(QtCore.QRect(10, 10, 961, 671))
        self.tabWidget.setObjectName("tabWidget")
        self.SIFT = QtWidgets.QWidget()
        self.SIFT.setObjectName("SIFT")
        self.output_ncc = QtWidgets.QLabel(self.SIFT)
        self.output_ncc.setGeometry(QtCore.QRect(460, 310, 361, 291))
        self.output_ncc.setFrameShape(QtWidgets.QFrame.Box)
        self.output_ncc.setAlignment(QtCore.Qt.AlignLeading|QtCore.Qt.AlignLeft|QtCore.Qt.AlignTop)
        self.output_ncc.setObjectName("output_ncc")
        self.NCC_button = QtWidgets.QPushButton(self.SIFT)
        self.NCC_button.setGeometry(QtCore.QRect(340, 430, 81, 31))
        self.NCC_button.setObjectName("NCC_button")
        self.load_original_image = QtWidgets.QPushButton(self.SIFT)
        self.load_original_image.setGeometry(QtCore.QRect(20, 260, 81, 24))
        self.load_original_image.setObjectName("load_original_image")
        self.original_image = QtWidgets.QLabel(self.SIFT)
        self.original_image.setGeometry(QtCore.QRect(20, 10, 351, 251))
        self.original_image.setFrameShape(QtWidgets.QFrame.Box)
        self.original_image.setAlignment(QtCore.Qt.AlignLeading|QtCore.Qt.AlignLeft|QtCore.Qt.AlignTop)
        self.original_image.setObjectName("original_image")
        self.original_image_2 = QtWidgets.QLabel(self.SIFT)
        self.original_image_2.setGeometry(QtCore.QRect(460, 10, 351, 251))
        self.original_image_2.setFrameShape(QtWidgets.QFrame.Box)
        self.original_image_2.setAlignment(QtCore.Qt.AlignLeading|QtCore.Qt.AlignLeft|QtCore.Qt.AlignTop)
        self.original_image_2.setObjectName("original_image_2")
        self.load_original_image_2 = QtWidgets.QPushButton(self.SIFT)
        self.load_original_image_2.setGeometry(QtCore.QRect(460, 260, 81, 24))
        self.load_original_image_2.setObjectName("load_original_image_2")
        self.comboBox = QtWidgets.QComboBox(self.SIFT)
        self.comboBox.setGeometry(QtCore.QRect(350, 470, 73, 22))
        self.comboBox.setObjectName("comboBox")
        self.comboBox.addItem("")
        self.comboBox.addItem("")
        self.tabWidget.addTab(self.SIFT, "")
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1007, 26))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        self.tabWidget.setCurrentIndex(0)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.output_ncc.setText(_translate("MainWindow", "NCC or SSD"))
        self.NCC_button.setText(_translate("MainWindow", "Match"))
        self.load_original_image.setText(_translate("MainWindow", "Upload"))
        self.original_image.setText(_translate("MainWindow", "Original Image"))
        self.original_image_2.setText(_translate("MainWindow", "Original Image"))
        self.load_original_image_2.setText(_translate("MainWindow", "Upload"))
        self.comboBox.setItemText(0, _translate("MainWindow", "SSD"))
        self.comboBox.setItemText(1, _translate("MainWindow", "NCC"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.SIFT), _translate("MainWindow", "SIFT"))