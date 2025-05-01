import os
import sys
import numpy as np
import pandas as pd
from scipy.ndimage import zoom
from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QPushButton, QLabel, QGridLayout, 
                             QFileDialog, QMessageBox, QGroupBox, QFormLayout, QProgressBar,
                             QHBoxLayout, QComboBox, QSpinBox, QCheckBox, QTabWidget)
from PyQt6.QtCore import Qt, pyqtSignal, QThread

# TensorFlow diagnostics
print(f"Python version: {sys.version}")
print(f"Python executable: {sys.executable}")
print(f"Current working directory: {os.getcwd()}")
print(f"Sys.path: {sys.path}")
print("Trying to import tensorflow...")

try:
    import tensorflow as tf
    print(f"SUCCESS: TensorFlow {tf.__version__} successfully imported.")
    print(f"TensorFlow path: {tf.__file__}")
    from tensorflow.keras import backend as K
    print("Keras backend imported")
except Exception as e:
    print(f"ERROR: TensorFlow import failed: {e}")
    print(f"Error type: {type(e).__name__}")
    print("Using dummy implementation.")
    class tf:
        class keras:
            class models:
                @staticmethod
                def load_model(path, custom_objects=None):
                    print(f"Dummy model loaded from {path}")
                    class DummyModel:
                        def predict(self, x, verbose=0):
                            print(f"Dummy prediction with shape {x.shape if hasattr(x, 'shape') else 'unknown'}")
                            return [np.zeros(x.shape), np.zeros(x.shape)]
                    return DummyModel()
            losses = type('obj', (object,), {
                'binary_crossentropy': lambda y_true, y_pred: 0.0
            })
    class K:
        @staticmethod
        def flatten(x): return x.flatten() if hasattr(x, 'flatten') else x
        @staticmethod
        def cast(x, dtype): return x
        @staticmethod
        def sum(x): return np.sum(x) if hasattr(x, 'sum') else x

def voxel_loss(y_true, y_pred): return 0.0
def folding_loss(y_true, y_pred, alpha=0.25, gamma=2.0): return 0.0
def weighted_folding_loss(y_true, y_pred): return 0.0
def dice_coef(y_true, y_pred): return 1.0

class PreformPredictionModule:
    def __init__(self):
        self.name = "Preform Prediction"
        self.model = None
        self.min_coords = None
        self.max_coords = None
        self.npy_path = 'train_npy'
        self.result_path = './result'
        self.data_path = 'train_data'
        self.model_name = 'unet_model'
        self.worker = None
        self.preview_worker = None

        os.makedirs(self.npy_path, exist_ok=True)
        os.makedirs(os.path.join(self.npy_path, 'train_y'), exist_ok=True)
        os.makedirs(self.result_path, exist_ok=True)

    def get_name(self):
        return self.name

    def create_widget(self, parent):
        self.parent = parent
        self.widget = QWidget(parent)
        main_layout = QVBoxLayout(self.widget)
        self.status_label = QLabel("Status: Ready")
        self.model_status_label = QLabel("Not loaded")
        main_layout.addWidget(self.status_label)
        main_layout.addWidget(self.model_status_label)
        self.refresh_npy_files()
        return self.widget

    def load_model(self):
        file_path, _ = QFileDialog.getOpenFileName(self.widget, "Select Keras Model File", "", "Keras Model (*.h5)")
        if file_path:
            try:
                self.model = tf.keras.models.load_model(
                    file_path,
                    custom_objects={
                        'voxel_loss': voxel_loss,
                        'folding_loss': folding_loss,
                        'weighted_folding_loss': weighted_folding_loss,
                        'dice_coef': dice_coef
                    }
                )
                self.model_status_label.setText("Model loaded")
                self.status_label.setText("Status: Model loaded")
                print(f"Model loaded successfully from {file_path}")
            except Exception as e:
                print(f"Model load failed: {e}")
                QMessageBox.critical(self.widget, "Model Load Error", f"Failed to load model:\n{str(e)}")
                self.model_status_label.setText("Load failed")
                self.status_label.setText("Status: Load failed")
        else:
            self.status_label.setText("Status: Load cancelled")

    def refresh_npy_files(self):
        print("Refreshing NPY files list...")
        self.npy_files = [f for f in os.listdir(self.npy_path) if f.endswith('.npy')]
        print(f"Found {len(self.npy_files)} NPY files.")

    def start_prediction(self):
        if self.model is None:
            QMessageBox.warning(self.widget, "Prediction Error", "Model not loaded.")
            return

        self.status_label.setText("Status: Predicting...")
        self.refresh_npy_files()

        for fname in self.npy_files:
            fpath = os.path.join(self.npy_path, fname)
            try:
                data = np.load(fpath)
                data = np.expand_dims(data, axis=0)  # Add batch dimension
                pred = self.model.predict(data)
                output_path = os.path.join(self.result_path, f"pred_{fname}")
                np.save(output_path, pred[0])
                print(f"Prediction saved: {output_path}")
            except Exception as e:
                print(f"Prediction failed for {fname}: {e}")

        self.status_label.setText("Status: Prediction complete")

    def start_stl_conversion(self):
        print("Starting STL conversion... (stub)")
        self.status_label.setText("Status: Converting to STL...")
        # Placeholder: Perform NPY to STL conversion here
        self.status_label.setText("Status: STL conversion complete")

    def preview_stl(self):
        print("Previewing STL... (stub)")
        self.status_label.setText("Status: Previewing STL")
        # Placeholder: Preview STL visualization here
        self.status_label.setText("Status: Ready")
