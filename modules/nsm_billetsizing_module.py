# modules/nsm_billetsizing_module.py (boshlanishi)
import numpy as np
import pandas as pd
from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QPushButton, QLineEdit, QLabel, 
                            QGridLayout, QFileDialog, QMessageBox, QSpacerItem, 
                            QSizePolicy, QFrame, QHBoxLayout, QGroupBox, QTabWidget,
                            QSlider, QDoubleSpinBox, QSpinBox, QComboBox, QCheckBox,
                            QFormLayout)
from PyQt6.QtGui import QPixmap, QFont
from PyQt6.QtCore import pyqtSignal, Qt
import os
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas

# Dummy classes for testing without scikit-learn and keras
print("Using fallback mode for NSM Billetsizing module")
def load_model(path):
    print(f"Simulating loading model from {path}")
    class DummyModel:
        def predict(self, x, verbose=0):
            print(f"Simulating prediction")
            return np.array([[0.5]])
    return DummyModel()

class StandardScaler:
    def fit(self, X):
        pass
    def transform(self, X):
        return X
        
class SelectKBest:
    def __init__(self, score_func=None, k=10):
        self.k = k
    def fit(self, X, y):
        pass
    def transform(self, X):
        return X

def f_regression(X, y):
    return np.ones(X.shape[1]), np.ones(X.shape[1])

class Bounds:
    def __init__(self, lb, ub):
        self.lb = lb
        self.ub = ub
        
class NonlinearConstraint:
    def __init__(self, fun, lb, ub):
        self.fun = fun
        self.lb = lb
        self.ub = ub
        
def minimize(fun, x0, method='trust-constr', bounds=None, constraints=None, tol=1e-1):
    class Result:
        def __init__(self, x):
            self.x = x
    return Result(x0)

class NSMBilletsizingModule:
    def __init__(self):
        self.name = "NSM Billetsizing"
        self.model = None
        self.dataset = None
        self.X = None
        self.column_names = ['Feed', 'Depth Schedule', 'Number of Rotation', 'Pass1', 'Pass2', 'Pass3', 'Pass4', 'Pass5', 'Pass6', 'Pass7', 'ENE']
        self.scaler = StandardScaler()
        self.selector = SelectKBest(f_regression, k=10)
        self.results_data = {}  # For storing result data
        
        # Qo'shimcha xususiyatlar
        self.cogging_data = None
        self.bqi_weight_factor = 0.5  # Default BQI vazn koeffitsienti
        self.desired_grain_size = 7.0  # Default istalgan don o'lchami (ASTM E112)
        self.show_forging_details = True  # Bolg'alash detallarini ko'rsatish
        
        # Define paths
        self.set_default_paths()

    def get_name(self):
        return self.name
    
    def set_default_paths(self):
        """Set default paths for data files"""
        base_dir = os.getcwd()
        data_dir = os.path.join(base_dir, "data")
        self.cogging_data_path = os.path.join(data_dir, "Cogging data.xlsx")

    def create_widget(self, parent):
        self.parent = parent  # Store reference to main window
        self.widget = QWidget(parent)
        
        # Using VBoxLayout as the main layout
        main_layout = QVBoxLayout(self.widget)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(10)

        # Styles
        button_style = """
            QPushButton {
                background-color: #4a86e8;
                color: white;
                border-radius: 4px;
                padding: 8px 15px;
                font-weight: bold;
                font-size: 12px;
                min-height: 30px;
            }
            QPushButton:hover {
                background-color: #3b73d1;
            }
            QPushButton:disabled {
                background-color: #cccccc;
                color: #666666;
            }
        """
        
        input_style = """
            QLineEdit {
                padding: 6px;
                border: 1px solid #cccccc;
                border-radius: 4px;
                background-color: white;
                min-height: 25px;
            }
        """
        
        label_style = """
            QLabel {
                font-size: 12px;
                font-weight: bold;
                color: #333333;
            }
        """
        
        group_style = """
            QGroupBox {
                border: 1px solid #cccccc;
                border-radius: 5px;
                margin-top: 1ex;
                font-weight: bold;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 3px;
                color: #4a86e8;
            }
        """
        
        # Create tabs for different functions
        self.tabs = QTabWidget()
        main_layout.addWidget(self.tabs)
        
        # Main tab
        self.main_tab = QWidget()
        self.main_layout = QVBoxLayout(self.main_tab)
        self.tabs.addTab(self.main_tab, "Main")
        
        # BQI Analysis tab
        self.bqi_tab = QWidget()
        self.bqi_layout = QVBoxLayout(self.bqi_tab)
        self.tabs.addTab(self.bqi_tab, "BQI Analysis")
        
        # Cogging Simulation tab
        self.cogging_tab = QWidget()
        self.cogging_layout = QVBoxLayout(self.cogging_tab)
        self.tabs.addTab(self.cogging_tab, "Cogging Simulation")
        
        # ---------- MAIN TAB ----------
        # Top buttons row
        buttons_layout = QHBoxLayout()
        buttons_layout.setSpacing(10)

        self.find_min_button = QPushButton("PASS SCHEDULE")
        self.find_min_button.clicked.connect(self.on_find_min)
        self.find_min_button.setStyleSheet(button_style + "background-color: #34a853;")
        buttons_layout.addWidget(self.find_min_button)

        self.load_model_button = QPushButton("Load Model *.h5")
        self.load_model_button.clicked.connect(self.load_model_file)
        self.load_model_button.setStyleSheet(button_style)
        buttons_layout.addWidget(self.load_model_button)

        self.load_data_button = QPushButton("Load Data *.excel")
        self.load_data_button.clicked.connect(self.load_data_file)
        self.load_data_button.setStyleSheet(button_style)
        buttons_layout.addWidget(self.load_data_button)

        self.show_results_button = QPushButton("Show Results")
        self.show_results_button.clicked.connect(self.show_visualization)
        self.show_results_button.setStyleSheet(button_style + "background-color: #e0e0e0; color: #444444;")
        self.show_results_button.setEnabled(False)
        buttons_layout.addWidget(self.show_results_button)

        self.main_layout.addLayout(buttons_layout)

        # Status label (directly below buttons, no overlapping elements)
        self.status_label = QLabel("Status: Ready")
        self.status_label.setStyleSheet("color: #4a86e8; font-weight: bold; margin-top: 5px;")
        self.main_layout.addWidget(self.status_label)

        # Add a separator line
        separator = QFrame()
        separator.setFrameShape(QFrame.Shape.HLine)
        separator.setFrameShadow(QFrame.Shadow.Sunken)
        separator.setStyleSheet("background-color: #cccccc;")
        self.main_layout.addWidget(separator)

        # Content grid for the rest of the interface
        content_layout = QGridLayout()
        content_layout.setVerticalSpacing(15)
        content_layout.setHorizontalSpacing(10)
        
        current_row = 0
        
        # Input Parameters section (moved below status)
        input_header = QLabel("Input Parameters:")
        input_header.setStyleSheet("font-size: 14px; font-weight: bold; color: #333333;")
        content_layout.addWidget(input_header, current_row, 0, 1, 3)
        current_row += 1
        
        # Feed, Depth Schedule, Number of Rotation
        content_layout.addWidget(QLabel("Feed"), current_row, 0)
        content_layout.addWidget(QLabel("Depth Schedule"), current_row, 1)
        content_layout.addWidget(QLabel("Number of Rotation"), current_row, 2)
        current_row += 1
        
        self.entry_vars = [QLineEdit() for _ in range(10)]
        for i, entry in enumerate(self.entry_vars[:3]):
            entry.setStyleSheet(input_style)
            content_layout.addWidget(entry, current_row, i)
        current_row += 1
        
        # Spacer row
        spacer = QSpacerItem(10, 20, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Fixed)
        content_layout.addItem(spacer, current_row, 0, 1, 3)
        current_row += 1
        
        # Pass Schedule section
        pass_header = QLabel("Pass Schedule:")
        pass_header.setStyleSheet("font-size: 14px; font-weight: bold; color: #333333;")
        content_layout.addWidget(pass_header, current_row, 0, 1, 8)
        current_row += 1
        
        # Initial Cross-section and Pass labels
        content_layout.addWidget(QLabel("Initial Cross-section [mm]"), current_row, 0)
        
        self.pass_labels = []
        for i in range(3, 10):
            label = QLabel(self.column_names[i])
            label.setStyleSheet(label_style)
            content_layout.addWidget(label, current_row, i-2)
            self.pass_labels.append(label)
        current_row += 1
        
        # Initial Cross-section entry and Pass entries
        self.radius_entry = QLineEdit("480")
        self.radius_entry.setStyleSheet(input_style)
        content_layout.addWidget(self.radius_entry, current_row, 0)
        
        self.pass_entries = []
        for i in range(1, 8):
            entry = QLineEdit()
            entry.setStyleSheet(input_style)
            content_layout.addWidget(entry, current_row, i)
            self.pass_entries.append(entry)
        current_row += 1
        
        # Forging Ratios
        content_layout.addWidget(QLabel("Forging Ratios:"), current_row, 0)
        
        self.forging_labels = [QLabel("") for _ in range(7)]
        for i, label in enumerate(self.forging_labels):
            label.setStyleSheet("border: 1px solid black; color: red; background-color: #f8f9fa; padding: 5px;")
            content_layout.addWidget(label, current_row, i+1)
        current_row += 1
        
        # Spacer row
        spacer = QSpacerItem(10, 20, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Fixed)
        content_layout.addItem(spacer, current_row, 0, 1, 8)
        current_row += 1
        
        # Length Changes section
        length_header = QLabel("Length Changes:")
        length_header.setStyleSheet("font-size: 14px; font-weight: bold; color: #333333;")
        content_layout.addWidget(length_header, current_row, 0, 1, 8)
        current_row += 1
        
        # Initial Length and Length Values labels
        content_layout.addWidget(QLabel("Initial Length [mm]"), current_row, 0)
        content_layout.addWidget(QLabel("Length Values:"), current_row, 1, 1, 7)
        current_row += 1
        
        # Initial Length entry and Length Values
        self.initial_length_entry = QLineEdit("1500")
        self.initial_length_entry.setStyleSheet(input_style)
        content_layout.addWidget(self.initial_length_entry, current_row, 0)
        
        self.length_change_labels = [QLabel("") for _ in range(7)]
        for i, label in enumerate(self.length_change_labels):
            label.setStyleSheet("border: 1px solid black; color: blue; background-color: #f8f9fa; padding: 5px;")
            content_layout.addWidget(label, current_row, i+1)
        current_row += 1
        
        # Spacer row
        spacer = QSpacerItem(10, 20, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Fixed)
        content_layout.addItem(spacer, current_row, 0, 1, 8)
        current_row += 1
        
        # Cutting Analysis section
        cutting_header = QLabel("Cutting Analysis:")
        cutting_header.setStyleSheet("font-size: 14px; font-weight: bold; color: #333333;")
        content_layout.addWidget(cutting_header, current_row, 0, 1, 8)
        current_row += 1
        
        # Cutting Length and Cutted Length Values labels
        content_layout.addWidget(QLabel("Cutting Length [mm]"), current_row, 0)
        content_layout.addWidget(QLabel("Cutted Length Values:"), current_row, 1, 1, 7)
        current_row += 1
        
        # Cutting Length entry and Cutted Length Values
        self.cutting_length_entry = QLineEdit("3000")
        self.cutting_length_entry.setStyleSheet(input_style)
        content_layout.addWidget(self.cutting_length_entry, current_row, 0)
        
        self.cutted_length_labels = [QLabel("") for _ in range(7)]
        for i, label in enumerate(self.cutted_length_labels):
            label.setStyleSheet("border: 1px solid black; color: blue; background-color: #f8f9fa; padding: 5px;")
            content_layout.addWidget(label, current_row, i+1)
        current_row += 1
        
        # Total cutted counter
        self.yellow_changes_label = QLabel("Total cutted 0 times")
        self.yellow_changes_label.setStyleSheet("font-weight: bold; color: #333333;")
        content_layout.addWidget(self.yellow_changes_label, current_row, 7, Qt.AlignmentFlag.AlignRight)
        
        # Add content layout to main layout
        self.main_layout.addLayout(content_layout)
        
        # Add stretch at the bottom of main tab
        self.main_layout.addStretch()
        
        # ---------- BQI ANALYSIS TAB ----------
        # BQI formula info group
        bqi_info_group = QGroupBox("BQI Formula")
        bqi_info_group.setStyleSheet(group_style)
        bqi_info_layout = QVBoxLayout()
        
        # BQI formula label with formula using LaTeX
        bqi_formula_label = QLabel("BQI = (εs/ε̄) * ds/d̄ + w * (ddes - d̄)²")
        bqi_formula_label.setStyleSheet("font-size: 16px; font-weight: bold; padding: 10px;")
        bqi_formula_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        bqi_info_layout.addWidget(bqi_formula_label)
        
        # BQI parameters description
        bqi_params_label = QLabel(
            "Where: \n"
            "εs - standard deviation effective strain\n"
            "ε̄ - average effective strain\n"
            "ds - standard deviation grain size number\n"
            "d̄ - average grain size number (ASTM E112)\n"
            "ddes - desired grain size number\n"
            "w - weight factor"
        )
        bqi_params_label.setStyleSheet("font-size: 14px; padding: 10px;")
        bqi_info_layout.addWidget(bqi_params_label)
        
        bqi_info_group.setLayout(bqi_info_layout)
        self.bqi_layout.addWidget(bqi_info_group)
        
        # BQI Analysis settings group
        bqi_settings_group = QGroupBox("BQI Analysis Settings")
        bqi_settings_group.setStyleSheet(group_style)
        bqi_settings_layout = QFormLayout()
        
        # Weight factor input
        self.weight_factor_input = QDoubleSpinBox()
        self.weight_factor_input.setRange(0.0, 2.0)
        self.weight_factor_input.setSingleStep(0.1)
        self.weight_factor_input.setValue(self.bqi_weight_factor)
        self.weight_factor_input.valueChanged.connect(self.update_bqi_weight_factor)
        bqi_settings_layout.addRow("Weight Factor (w):", self.weight_factor_input)
        
        # Desired grain size input
        self.desired_grain_size_input = QDoubleSpinBox()
        self.desired_grain_size_input.setRange(1.0, 14.0)
        self.desired_grain_size_input.setSingleStep(0.5)
        self.desired_grain_size_input.setValue(self.desired_grain_size)
        self.desired_grain_size_input.valueChanged.connect(self.update_desired_grain_size)
        bqi_settings_layout.addRow("Desired Grain Size (ASTM E112):", self.desired_grain_size_input)
        
        # Load Cogging data button
        self.load_cogging_data_button = QPushButton("Load Cogging Data")
        self.load_cogging_data_button.clicked.connect(self.load_cogging_data)
        self.load_cogging_data_button.setStyleSheet(button_style)
        bqi_settings_layout.addRow("Cogging Data:", self.load_cogging_data_button)
        
        # Analyze button
        self.analyze_bqi_button = QPushButton("Analyze BQI")
        self.analyze_bqi_button.clicked.connect(self.analyze_bqi)
        self.analyze_bqi_button.setStyleSheet(button_style + "background-color: #34a853;")
        bqi_settings_layout.addRow("", self.analyze_bqi_button)
        
        bqi_settings_group.setLayout(bqi_settings_layout)
        self.bqi_layout.addWidget(bqi_settings_group)
        
        # BQI Results group
        bqi_results_group = QGroupBox("BQI Analysis Results")
        bqi_results_group.setStyleSheet(group_style)
        bqi_results_layout = QVBoxLayout()
        
        # BQI results figure
        self.bqi_figure, self.bqi_ax = plt.subplots(figsize=(8, 4))
        self.bqi_canvas = FigureCanvas(self.bqi_figure)
        bqi_results_layout.addWidget(self.bqi_canvas)
        
        # BQI results label
        self.bqi_results_label = QLabel("No BQI analysis results yet")
        self.bqi_results_label.setStyleSheet("font-size: 14px; padding: 10px;")
        self.bqi_results_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        bqi_results_layout.addWidget(self.bqi_results_label)
        
        bqi_results_group.setLayout(bqi_results_layout)
        self.bqi_layout.addWidget(bqi_results_group)
        
        # Add stretch to the bottom of BQI tab
        self.bqi_layout.addStretch()
        
        # ---------- COGGING SIMULATION TAB ----------
        # Cogging diagram group
        cogging_diagram_group = QGroupBox("Cogging Process Diagram")
        cogging_diagram_group.setStyleSheet(group_style)
        cogging_diagram_layout = QVBoxLayout()
        
        # Cogging process visualization
        self.cogging_figure, self.cogging_ax = plt.subplots(figsize=(8, 6))
        self.cogging_canvas = FigureCanvas(self.cogging_figure)
        cogging_diagram_layout.addWidget(self.cogging_canvas)
        
        cogging_diagram_group.setLayout(cogging_diagram_layout)
        self.cogging_layout.addWidget(cogging_diagram_group)
        
        # Cogging Settings group
        cogging_settings_group = QGroupBox("Cogging Settings")
        cogging_settings_group.setStyleSheet(group_style)
        cogging_settings_layout = QFormLayout()
        
        # Number of rotation selector
        self.rotation_number_input = QSpinBox()
        self.rotation_number_input.setRange(2, 5)
        self.rotation_number_input.setValue(3)
        self.rotation_number_input.valueChanged.connect(self.update_cogging_diagram)
        cogging_settings_layout.addRow("Number of Rotation:", self.rotation_number_input)
        
        # Feed amount input
        self.feed_amount_input = QDoubleSpinBox()
        self.feed_amount_input.setRange(10.0, 200.0)
        self.feed_amount_input.setSingleStep(5.0)
        self.feed_amount_input.setValue(50.0)
        self.feed_amount_input.setSuffix(" mm")
        self.feed_amount_input.valueChanged.connect(self.update_cogging_diagram)
        cogging_settings_layout.addRow("Feed Amount:", self.feed_amount_input)
        
        # Depth schedule input
        self.depth_schedule_input = QDoubleSpinBox()
        self.depth_schedule_input.setRange(0.0, 50.0)
        self.depth_schedule_input.setSingleStep(1.0)
        self.depth_schedule_input.setValue(20.0)
        self.depth_schedule_input.setSuffix(" mm")
        self.depth_schedule_input.valueChanged.connect(self.update_cogging_diagram)
        cogging_settings_layout.addRow("Depth Schedule:", self.depth_schedule_input)
        
        # Overlap option
        self.overlap_checkbox = QCheckBox("Show Overlap")
        self.overlap_checkbox.setChecked(True)
        self.overlap_checkbox.stateChanged.connect(self.update_cogging_diagram)
        cogging_settings_layout.addRow("", self.overlap_checkbox)
        
        # Simulate button
        self.simulate_cogging_button = QPushButton("Simulate Cogging")
        self.simulate_cogging_button.clicked.connect(self.simulate_cogging)
        self.simulate_cogging_button.setStyleSheet(button_style + "background-color: #ea4335;")
        cogging_settings_layout.addRow("", self.simulate_cogging_button)
        
        cogging_settings_group.setLayout(cogging_settings_layout)
        self.cogging_layout.addWidget(cogging_settings_group)
        
        # Initial cogging diagram
        self.update_cogging_diagram()
        
        # Check if we can load Cogging data automatically
        if os.path.exists(self.cogging_data_path):
            try:
                self.cogging_data = pd.read_excel(self.cogging_data_path)
                self.status_label.setText("Status: Cogging data loaded automatically")
            except Exception as e:
                self.status_label.setText(f"Status: Error auto-loading cogging data - {str(e)}")
        
        return self.widget
        
    def update_bqi_weight_factor(self, value):
        """Update BQI weight factor"""
        self.bqi_weight_factor = value
        
    def update_desired_grain_size(self, value):
        """Update desired grain size"""
        self.desired_grain_size = value
        
    def load_cogging_data(self):
        """Load cogging data from Excel file"""
        filepath, _ = QFileDialog.getOpenFileName(self.widget, "Open Cogging Data File", "", "Excel Files (*.xlsx *.xls)")
        if filepath:
            try:
                self.cogging_data = pd.read_excel(filepath)
                self.cogging_data_path = filepath
                self.status_label.setText("Status: Cogging data loaded successfully")
                QMessageBox.information(self.widget, "Success", "Cogging data loaded successfully!")
            except Exception as e:
                self.status_label.setText(f"Status: Error loading cogging data - {str(e)}")
                QMessageBox.critical(self.widget, "Error", f"Error loading cogging data: {str(e)}")
                
    def analyze_bqi(self):
        """Analyze BQI based on the loaded data and settings"""
        if self.cogging_data is None:
            QMessageBox.warning(self.widget, "Warning", "Please load cogging data first!")
            return
            
        if self.results_data is None or not self.results_data:
            QMessageBox.warning(self.widget, "Warning", "Please calculate pass schedule first!")
            return
            
        try:
            # Generate simulated BQI analysis results since we don't have actual strain data
            # In a real application, this would use the actual strain and grain size data
            
            # Clear previous figure
            self.bqi_ax.clear()
            
            # Create sample data for demonstration
            passes = range(1, 8)
            
            # Simulate standard deviation and average effective strain
            std_strain = np.array([0.05, 0.08, 0.12, 0.15, 0.18, 0.22, 0.25])
            avg_strain = np.array([0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
            
            # Simulate standard deviation and average grain size
            std_grain_size = np.array([0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2])
            avg_grain_size = np.array([4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0])
            
            # Calculate BQI components
            strain_component = std_strain / avg_strain * std_grain_size / avg_grain_size
            grain_size_component = self.bqi_weight_factor * (self.desired_grain_size - avg_grain_size)**2
            
            bqi_values = strain_component + grain_size_component
            
            # Plot BQI values
            self.bqi_ax.bar(passes, bqi_values, color='#4a86e8', alpha=0.7)
            self.bqi_ax.plot(passes, bqi_values, 'ro-')
            
            # Add strain component
            self.bqi_ax.plot(passes, strain_component, 'g--', label='Strain Component')
            
            # Add grain size component
            self.bqi_ax.plot(passes, grain_size_component, 'b--', label='Grain Size Component')
            
            # Set graph properties
            self.bqi_ax.set_xlabel('Pass Number')
            self.bqi_ax.set_ylabel('BQI Value')
            self.bqi_ax.set_title(f'BQI Analysis (w={self.bqi_weight_factor}, d_des={self.desired_grain_size})')
            self.bqi_ax.grid(True, linestyle='--', alpha=0.7)
            self.bqi_ax.legend()
            
            # Update canvas
            self.bqi_canvas.draw()
            
            # Update results label
            optimal_pass = np.argmin(bqi_values) + 1
            min_bqi = np.min(bqi_values)
            self.bqi_results_label.setText(f"Optimal Pass: {optimal_pass} (BQI={min_bqi:.4f})\n"
                                          f"Final BQI: {bqi_values[-1]:.4f}\n"
                                          f"Weight Factor: {self.bqi_weight_factor}, Desired Grain Size: {self.desired_grain_size}")
            
            # Show a message
            QMessageBox.information(self.widget, "Success", f"BQI Analysis completed. Optimal pass: {optimal_pass}")
            
        except Exception as e:
            self.status_label.setText(f"Status: Error in BQI analysis - {str(e)}")
            QMessageBox.critical(self.widget, "Error", f"Error in BQI analysis: {str(e)}")
            
    def update_cogging_diagram(self):
        """Update cogging process diagram based on settings"""
        try:
            # Clear previous figure
            self.cogging_ax.clear()
            
            # Get settings
            num_rotation = self.rotation_number_input.value()
            show_overlap = self.overlap_checkbox.isChecked()
            
            # Create a simple cogging process diagram
            # In a real application, this would be a more detailed simulation
            
            # Define workpiece dimensions
            width = 100
            height = 50
            
            # Create figure with appropriate size
            num_steps = num_rotation * 2 + 1
            x_range = width * (num_steps + 2)
            
            # Set axis limits
            self.cogging_ax.set_xlim(0, x_range)
            self.cogging_ax.set_ylim(0, height * 4)
            
            # Initial workpiece
            self.cogging_ax.add_patch(plt.Rectangle((width/2, height*1.5), width, height, fill=False, edgecolor='black', linewidth=2))
            self.cogging_ax.text(width, height*1.3, "Initial Billet", ha='center')
            
            # Draw rotation steps
            x_pos = width * 2
            y_pos = height * 1.5
            
            for i in range(num_rotation):
                # First orientation
                self.cogging_ax.add_patch(plt.Rectangle((x_pos, y_pos), width, height, fill=False, edgecolor='black', linewidth=2))
                
                # 90-degree rotation arrow
                self.cogging_ax.annotate("", xy=(x_pos + width + width/4, y_pos + height/2), 
                                        xytext=(x_pos + width, y_pos + height/2),
                                        arrowprops=dict(arrowstyle="->", color='blue', lw=2))
                self.cogging_ax.text(x_pos + width + width/8, y_pos + height/2 + 10, "90°", color='blue')
                
                # Rotated piece
                x_pos += width * 1.5
                self.cogging_ax.add_patch(plt.Rectangle((x_pos, y_pos), height, width, fill=False, edgecolor='black', linewidth=2))
                
                # If not the last rotation, add another 90-degree rotation
                if i < num_rotation - 1:
                    # 90-degree rotation arrow
                    self.cogging_ax.annotate("", xy=(x_pos + height + width/4, y_pos + width/2), 
                                            xytext=(x_pos + height, y_pos + width/2),
                                            arrowprops=dict(arrowstyle="->", color='blue', lw=2))
                    self.cogging_ax.text(x_pos + height + width/8, y_pos + width/2 + 10, "90°", color='blue')
                    
                    # Rotated piece again
                    x_pos += height * 1.5
                    self.cogging_ax.add_patch(plt.Rectangle((x_pos, y_pos), width, height, fill=False, edgecolor='black', linewidth=2))
                    
                    x_pos += width * 1.5
            
            # Show overlap if checked
            if show_overlap:
                overlap_x = width * 2.5
                overlap_y = height * 3
                
                # Draw overlap illustration
                self.cogging_ax.add_patch(plt.Rectangle((overlap_x, overlap_y), width*1.5, height, fill=False, edgecolor='black', linewidth=2))
                self.cogging_ax.text(overlap_x + width*0.75, overlap_y - 10, "Overlap View", ha='center')
                
                # Draw feed and depth annotations
                feed = self.feed_amount_input.value()
                depth = self.depth_schedule_input.value()
                
                # Draw feed arrow
                feed_arrow_length = min(feed, width)
                self.cogging_ax.arrow(overlap_x, overlap_y - 20, feed_arrow_length, 0, 
                                     head_width=5, head_length=5, fc='red', ec='red', lw=2)
                self.cogging_ax.text(overlap_x + feed_arrow_length/2, overlap_y - 30, f"Feed: {feed} mm", 
                                    color='red', ha='center')
                
                # Draw depth arrow
                self.cogging_ax.arrow(overlap_x + width*1.5 + 20, overlap_y + height/2, 0, -depth,
                                     head_width=5, head_length=5, fc='green', ec='green', lw=2)
                self.cogging_ax.text(overlap_x + width*1.5 + 30, overlap_y + height/2 - depth/2, 
                                    f"Depth: {depth} mm", color='green', ha='left', va='center')
            
            # Set title
            self.cogging_ax.set_title(f"Cogging Process Simulation (Rotations: {num_rotation})")
            
            # Remove axis ticks
            self.cogging_ax.set_xticks([])
            self.cogging_ax.set_yticks([])
            
            # Update canvas
            self.cogging_canvas.draw()
            
        except Exception as e:
            self.status_label.setText(f"Status: Error updating cogging diagram - {str(e)}")
    
    def simulate_cogging(self):
        """Simulate cogging process and estimate results"""
        try:
            # Get input values
            num_rotation = self.rotation_number_input.value()
            feed = self.feed_amount_input.value()
            depth = self.depth_schedule_input.value()
            
            # Update the UI entries with these values
            self.entry_vars[0].setText(str(feed))
            self.entry_vars[1].setText(str(depth))
            self.entry_vars[2].setText(str(num_rotation))
            
            # Run the pass schedule calculation
            self.on_find_min()
            
            # Show a message
            QMessageBox.information(self.widget, "Success", "Cogging simulation completed and values transferred to Pass Schedule")
            
            # Switch to main tab
            self.tabs.setCurrentIndex(0)
            
        except Exception as e:
            self.status_label.setText(f"Status: Error in cogging simulation - {str(e)}")
            QMessageBox.critical(self.widget, "Error", f"Error in cogging simulation: {str(e)}")

    def load_model_file(self):
        filepath, _ = QFileDialog.getOpenFileName(self.widget, "Open Model File", "", "H5 Files (*.h5)")
        if filepath:
            try:
                self.model = load_model(filepath)
                self.status_label.setText("Status: Model loaded successfully")
                QMessageBox.information(self.widget, "Success", "Model loaded successfully!")
            except Exception as e:
                self.status_label.setText(f"Status: Error loading model - {str(e)}")
                QMessageBox.critical(self.widget, "Error", f"Error loading model: {str(e)}")

    def load_data_file(self):
        filepath, _ = QFileDialog.getOpenFileName(self.widget, "Open Data File", "", "Excel Files (*.xlsx)")
        if filepath:
            try:
                rawdata = pd.read_excel(filepath, sheet_name='Sheet1', names=self.column_names)
                self.dataset = rawdata.copy()
                self.dataset = round(self.dataset, 5)
                self.X = self.dataset.drop('ENE', axis=1)
                y = self.dataset['ENE']
                self.scaler.fit(self.X)
                self.selector.fit(self.X, y)
                self.status_label.setText("Status: Data loaded successfully")
                QMessageBox.information(self.widget, "Success", "Data loaded successfully!")
            except Exception as e:
                self.status_label.setText(f"Status: Error loading data - {str(e)}")
                QMessageBox.critical(self.widget, "Error", f"Error loading data: {str(e)}")

    def predict_y(self, x):
        try:
            x_scaled = self.scaler.transform([x])
            x_selected = self.selector.transform(x_scaled)
            return self.model.predict(x_selected, verbose=0)[0, 0]
        except Exception as e:
            QMessageBox.critical(self.widget, "Error", f"Prediction error: {str(e)}")
            return 0.0

    def minimize_y(self, x):
        return self.predict_y(x)

    def constraint_fun(self, x):
        y = self.predict_y(x)
        return y - 0.001

    def on_find_min(self):
        if self.model is None:
            self.status_label.setText("Status: No model loaded")
            QMessageBox.warning(self.widget, "Warning", "Please load model first!")
            return
        
        if self.X is None:
            self.status_label.setText("Status: No data loaded")
            QMessageBox.warning(self.widget, "Warning", "Please load data first!")
            return
        
        try:
            self.status_label.setText("Status: Calculating optimal pass schedule...")
            
            # Get custom input values
            custom_inputs = []
            for i in range(3):
                if self.entry_vars[i].text():
                    try:
                        custom_inputs.append(float(self.entry_vars[i].text()))
                    except ValueError:
                        custom_inputs.append(None)
                else:
                    custom_inputs.append(None)
            
            # Get mean values as starting point
            x0 = np.mean(self.X, axis=0)
            
            # Update with custom inputs if provided
            for i, val in enumerate(custom_inputs):
                if val is not None:
                    x0[i] = val
            
            # Get min/max bounds from data
            lower_bounds = [self.X[column].min() for column in self.X.columns]
            upper_bounds = [self.X[column].max() for column in self.X.columns]
            
            # Adjust bounds for custom inputs
            for i, val in enumerate(custom_inputs):
                if val is not None:
                    lower_bounds[i] = val
                    upper_bounds[i] = val
            
            bounds = Bounds(lower_bounds, upper_bounds)
            
            # Set constraint for optimization
            constraint = NonlinearConstraint(self.constraint_fun, lb=0, ub=np.inf)
            
            # Perform optimization
            result = minimize(self.minimize_y, x0, method='trust-constr', 
                             bounds=bounds, constraints=constraint, tol=1e-1)
            
            # Get optimal values
            x_min = result.x

            # Display values in input fields
            for i, x in enumerate(x_min[:3]):
                if custom_inputs[i] is None:  # Only update if not custom input
                    self.entry_vars[i].setText(f"{x:.5f}")
            
            # Update pass values
            for i, x in enumerate(x_min[3:10]):
                if i < len(self.pass_entries):
                    self.pass_entries[i].setText(f"{x:.5f}")

            # Calculate forging ratios
            radius_value = float(self.radius_entry.text())
            forging_ratios = [np.sqrt((radius_value**2 * np.pi / 4) / x_min[3])]
            for i in range(1, 7):
                ratio = np.sqrt(x_min[3+i-1] / x_min[3+i])
                forging_ratios.append(ratio)

            # Display forging ratios
            for i, forging_ratio in enumerate(forging_ratios):
                self.forging_labels[i].setText(f"{forging_ratio:.2f}×{forging_ratio:.2f}")

            # Calculate length changes
            initial_length = float(self.initial_length_entry.text())
            length_change = [x_min[3]*initial_length]
            for i in range(1, 7):
                length = length_change[-1]/x_min[3+i]
                length_change.append(length)

            # Display length changes
            for i, length_chang in enumerate(length_change):
                self.length_change_labels[i].setText(f"{length_chang:.0f}")

            # Update cutting calculations
            self.update_cutted_length_labels(length_change)
            
            # Save result data
            self.results_data = {
                'forging_ratios': forging_ratios,
                'length_changes': length_change,
                'x_min': x_min.tolist(),
                'initial_radius': radius_value,
                'initial_length': initial_length,
                'bqi_weight_factor': self.bqi_weight_factor,
                'desired_grain_size': self.desired_grain_size
            }
            
            # Enable visualization button
            self.show_results_button.setEnabled(True)
            self.show_results_button.setStyleSheet("""
                QPushButton {
                    background-color: #ea4335;
                    color: white;
                    border-radius: 4px;
                    padding: 8px 15px;
                    font-weight: bold;
                    font-size: 12px;
                    min-height: 30px;
                }
                QPushButton:hover {
                    background-color: #d62516;
                }
            """)
            
            self.status_label.setText("Status: Pass schedule calculation completed successfully")
            QMessageBox.information(self.widget, "Success", "Pass schedule calculated successfully!")
            
        except Exception as e:
            self.status_label.setText(f"Status: Error in calculation - {str(e)}")
            QMessageBox.critical(self.widget, "Error", f"Error in calculation: {str(e)}")

    def update_cutted_length_labels(self, length_changes):
        try:
            cutting_length = float(self.cutting_length_entry.text())
            cutted_length_changes = []
            
            for length_change in length_changes:
                rods = [{'length': length_change, 'quantity': 1}]
                for rod in rods:
                    while rod['length'] > cutting_length:
                        rod['length'] /= 2
                        rod['quantity'] *= 2
                cutted_length_changes.append(rods)

            previous_rods_quantity = 1
            yellow_changes = 0
            
            for idx, rods in enumerate(cutted_length_changes):
                if rods[0]['quantity'] > previous_rods_quantity:
                    bg_color = 'background-color: yellow;'
                    yellow_changes += 1
                else:
                    bg_color = 'background-color: #f8f9fa;'
                
                self.cutted_length_labels[idx].setText(f"{rods[0]['length']:.0f} ({rods[0]['quantity']})")
                self.cutted_length_labels[idx].setStyleSheet(f"border: 1px solid black; color: blue; {bg_color} padding: 5px;")
                previous_rods_quantity = rods[0]['quantity']

            self.yellow_changes_label.setText(f"Total cutted {yellow_changes} times")
            
            # Save additional data
            self.results_data['cutting_length'] = cutting_length
            self.results_data['cutted_length_changes'] = [rod[0]['length'] for rod in cutted_length_changes]
            self.results_data['cutted_quantities'] = [rod[0]['quantity'] for rod in cutted_length_changes]
            
        except Exception as e:
            self.status_label.setText(f"Status: Error in cutting calculation - {str(e)}")
            QMessageBox.warning(self.widget, "Warning", f"Error in cutting calculation: {str(e)}")

    def show_visualization(self):
        """Show visualization results"""
        if not self.results_data:
            self.status_label.setText("Status: No results to visualize")
            QMessageBox.warning(self.widget, "Warning", "Please perform calculation first!")
            return
        
        try:
            # Call visualization manager
            if hasattr(self.parent, 'visualization_manager'):
                self.parent.visualization_manager.display_nsm_billetsizing_results(self.results_data)
                self.status_label.setText("Status: Results visualized successfully")
            else:
                self.status_label.setText("Status: Visualization manager not found")
                QMessageBox.warning(self.widget, "Warning", "Visualization manager not available")
        except Exception as e:
            self.status_label.setText(f"Status: Error in visualization - {str(e)}")
            QMessageBox.critical(self.widget, "Error", f"Error in visualization: {str(e)}")