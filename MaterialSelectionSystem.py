# material_selector_final_ui_intuitive.py

import sys
import os
import pandas as pd
import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

# ---------------- CONFIG ----------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(BASE_DIR, "material_dataset.csv") 
# ----------------------------------------

# ---------------- CORE FUNCTIONS ----------------
def load_database(csv_path):
    if not os.path.exists(csv_path) or not os.path.isfile(csv_path):
        raise FileNotFoundError(f"ملف قاعدة البيانات غير موجود في المسار: {csv_path}\nالرجاء التأكد من وضع ملف material_dataset.csv في نفس مجلد البرنامج.")
    
    df = pd.read_csv(csv_path)
    
    df.columns = df.columns.str.strip()

    name_cols = ['Material Name', 'Name', 'material name', 'Mat_Name']
    found_name_col = None
    for col in name_cols:
        if col in df.columns:
            found_name_col = col
            break
    
    if found_name_col:
        df[found_name_col] = df[found_name_col].fillna('Unknown Material').astype(str).str.strip()
    
    fam_cols = ['Family', 'family', 'Material Family', 'Type']
    found_fam_col = None
    for col in fam_cols:
        if col in df.columns:
            found_fam_col = col
            break
            
    if found_fam_col:
        df[found_fam_col] = df[found_fam_col].astype(str).str.strip().str.lower()
        
    return df

# ---------------- MAIN CLASS ----------------
class ExactMaterialSelector(QtWidgets.QMainWindow):
    def __init__(self, csv_path=CSV_PATH):
        super().__init__()
        
        try:
            self.df = load_database(csv_path)
        except Exception as e:
            QtWidgets.QMessageBox.critical(None, "Database Error", str(e))
            sys.exit(1)
        
        self.col_map = {
            'Material Name': self._find_col(['Material Name', 'Name', 'Mat_Name']),
            'Family': self._find_col(['Family', 'Material Family', 'Type']),
            'Density': self._find_col(['Density_g/cm3', 'Density']),
            'Youngs Modulus': self._find_col(['Youngs_Modulus_GPa', 'Youngs Modulus']),
            'Yield Strength': self._find_col(['Yield_Strength_MPa', 'Yield Strength']),
            'Ultimate Strength': self._find_col(['Ultimate_Strength_MPa', 'Ultimate Strength']),
            'Hardness': self._find_col(['Hardness_HB', 'Hardness']),
            'Thermal Conductivity': self._find_col(['Thermal_Conductivity_W/mK', 'Thermal Conductivity']),
            'Melting Point': self._find_col(['Melting_Point_C', 'Melting Point']), 
            'Electrical Conductivity': self._find_col(['Electrical_Conductivity_IACS', 'Electrical Conductivity']),
            'Corrosion Resistance': self._find_col(['Corrosion_Resistance', 'Corrosion']),
            'Cost': self._find_col(['Cost_per_kg_USD', 'Cost']),
            # أسماء الموردين التي لم تعد تُعرض في اللوحة اليسرى:
            'Supplier 1': self._find_col(['Supplier 1', 'SupplierName1']),
            'Availability 1': self._find_col(['Available 1', 'Stock1']),
            'Actual Price 1': self._find_col(['Price 1', 'Actual_Price1']),
            'Supplier 2': self._find_col(['Supplier 2', 'SupplierName2']),
            'Availability 2': self._find_col(['Available 2', 'Stock2']),
            'Actual Price 2': self._find_col(['Price 2', 'Actual_Price2']),
        }

        self.property_categories = {
            "Mechanical Properties": [
                ("Youngs Modulus", "Youngs Modulus"),
                ("Yield Strength", "Yield Strength"),
                ("Ultimate Strength", "Ultimate Strength"),
                ("Hardness", "Hardness"),
            ],
            "Physical and Thermal Properties": [
                ("Density", "Density"), 
                ("Melting Point", "Melting Point"), 
                ("Thermal Conductivity", "Thermal Conductivity"),
                ("Electrical Conductivity", "Electrical Conductivity"),
                ("Corrosion Resistance", "Corrosion Resistance"), 
            ],
            "Economic Properties": [
                ("Cost per kg (USD)", "Cost"), 
            ]
        }
        
        self.slider_defs = [
            (disp, key) for sublist in self.property_categories.values() 
            for disp, key in sublist
        ]

        self.setWindowTitle("Material Selector")
        self.setGeometry(50, 50, 1280, 780) 
        
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        main_layout = QtWidgets.QVBoxLayout(central)
        main_layout.setContentsMargins(40, 30, 40, 30)
        
        card = QtWidgets.QFrame()
        card.setObjectName("outer_card")
        card_layout = QtWidgets.QHBoxLayout(card)
        card_layout.setContentsMargins(26, 26, 26, 26)

        # LEFT Panel Setup 
        left = QtWidgets.QFrame()
        left.setObjectName("left_card")
        left.setFixedWidth(540)
        left_layout = QtWidgets.QVBoxLayout(left)
        left_layout.setContentsMargins(28, 18, 28, 18)

        title = QtWidgets.QLabel("RECOMMENDED MATERIAL")
        title.setObjectName("title_label")
        title.setAlignment(QtCore.Qt.AlignCenter)

        line = QtWidgets.QFrame()
        line.setFrameShape(QtWidgets.QFrame.HLine)
        line.setObjectName("title_line")
        line.setFixedHeight(2)

        self.material_name = QtWidgets.QLabel("")
        self.material_name.setObjectName("material_name")
        self.material_name.setAlignment(QtCore.Qt.AlignCenter)

        self.left_props = QtWidgets.QWidget()
        self.left_props_layout = QtWidgets.QFormLayout(self.left_props)
        self.left_props_layout.setHorizontalSpacing(40)
        self.left_props_layout.setLabelAlignment(QtCore.Qt.AlignLeft)

        # إعداد الرسم البياني
        self.graph_container = QtWidgets.QWidget()
        self.graph_layout = QtWidgets.QVBoxLayout(self.graph_container)
        self.graph_layout.setContentsMargins(0, 0, 0, 0)
        self.graph_layout.setSpacing(0)
        
        self.figure = Figure(figsize=(5, 4))
        self.canvas = FigureCanvas(self.figure)
        self.graph_layout.addWidget(self.canvas)
        self._setup_graph_theme() 

        left_layout.addWidget(title)
        left_layout.addWidget(line)
        left_layout.addSpacing(8)
        left_layout.addWidget(self.material_name)
        left_layout.addSpacing(12)
        left_layout.addWidget(self.left_props)
        left_layout.addSpacing(20) 
        left_layout.addWidget(self.graph_container) 
        left_layout.addStretch()

        # RIGHT Panel Setup
        right = QtWidgets.QFrame()
        right.setObjectName("right_card")
        right_layout = QtWidgets.QVBoxLayout(right)
        right_layout.setContentsMargins(26, 18, 26, 18)
        
        self.family_combo = QtWidgets.QComboBox()
        self.family_combo.setObjectName("family_combo")
        self.family_combo.addItem("Select Family")

        fam_col = self.col_map.get('Family')
        if fam_col in self.df.columns:
            families_unique = sorted(self.df[fam_col].str.title().unique().tolist())
            for f in families_unique:
                self.family_combo.addItem(f)

        self.sliders_content_widget = QtWidgets.QWidget()
        self.sliders_layout = QtWidgets.QVBoxLayout(self.sliders_content_widget)
        self.sliders_layout.setContentsMargins(0, 0, 0, 0) 
        
        self.slider_objs = []
        self._build_categorized_sliders(500) 

        self.calc_btn = QtWidgets.QPushButton("Calculate Best Material")
        self.calc_btn.setObjectName("calc_button")
        self.calc_btn.setFixedHeight(44)
        self.calc_btn.clicked.connect(self.calculate_best)

        self.results_label = QtWidgets.QLabel("Top Candidates:")
        self.results_list = QtWidgets.QListWidget()
        self.results_list.setObjectName("results_list")
        self.results_list.itemClicked.connect(self._on_result_clicked)
        self.results_list.setSizePolicy(
            QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.MinimumExpanding)

        right_layout.addWidget(self.family_combo, alignment=QtCore.Qt.AlignTop) 
        right_layout.addSpacing(10)
        right_layout.addWidget(self.sliders_content_widget, alignment=QtCore.Qt.AlignTop)
        right_layout.addSpacing(12)
        right_layout.addWidget(self.calc_btn, alignment=QtCore.Qt.AlignTop)
        right_layout.addSpacing(8)
        right_layout.addWidget(self.results_label, alignment=QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter)
        right_layout.addWidget(self.results_list, alignment=QtCore.Qt.AlignTop)
        right_layout.addStretch()

        card_layout.addWidget(left)
        card_layout.addWidget(right)
        main_layout.addWidget(card)

        self.setStyleSheet(self._style())
        self._current_candidates_df = None


    def _build_categorized_sliders(self, group_width):
        SLIDER_LABEL_WIDTH = 220 
        SLIDER_VALUE_WIDTH = 44
        THIN_DECO_WIDTH = 6
        
        for category, defs in self.property_categories.items():
            
            group_box = QtWidgets.QGroupBox(category)
            group_box.setObjectName("category_group")
            group_box.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
            
            group_box_layout = QtWidgets.QFormLayout(group_box)
            group_box_layout.setHorizontalSpacing(10) 
            group_box_layout.setContentsMargins(10, 10, 10, 10)
            
            group_box_layout.setLabelAlignment(QtCore.Qt.AlignLeft) 
            group_box_layout.setFormAlignment(QtCore.Qt.AlignHCenter) 

            for name, _ in defs: 
                label = QtWidgets.QLabel(name)
                label.setFixedWidth(SLIDER_LABEL_WIDTH) 
                
                row = QtWidgets.QWidget()
                row_l = QtWidgets.QHBoxLayout(row)
                row_l.setContentsMargins(0, 0, 0, 0)
                row_l.setSpacing(8)

                thin = QtWidgets.QFrame()
                thin.setFrameShape(QtWidgets.QFrame.VLine)
                thin.setObjectName("thin_deco")
                thin.setFixedWidth(THIN_DECO_WIDTH)

                s = QtWidgets.QSlider(QtCore.Qt.Horizontal)
                s.setObjectName("custom_slider")
                s.setMinimum(0)
                s.setMaximum(100)
                s.setValue(50)
                s.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed) 

                num = QtWidgets.QLabel("0.50")
                num.setFixedWidth(SLIDER_VALUE_WIDTH) 
                num.setAlignment(QtCore.Qt.AlignCenter)
                s.valueChanged.connect(lambda v, lbl=num: lbl.setText(f"{v/100:.2f}"))

                row_l.addWidget(thin)
                row_l.addWidget(s)
                row_l.addWidget(num)

                group_box_layout.addRow(label, row)
                self.slider_objs.append((s, num))
            
            self.sliders_layout.addWidget(group_box)
        
        self.sliders_layout.addStretch()


    def calculate_best(self):
        fam_selected = self.family_combo.currentText().strip().lower()
        if fam_selected == "select family":
            QtWidgets.QMessageBox.warning(self, "No family selected", "الرجاء اختيار فئة المواد.")
            return

        fam_col = self.col_map.get('Family')
        fam_options = [
            fam_selected,
            fam_selected.replace(' ', ''),
            fam_selected.replace('-', ''),
            fam_selected.replace('_', '')
        ]
        
        df_temp = self.df.copy()
        df_temp[fam_col] = df_temp[fam_col].astype(str).str.strip().str.lower()
        df = df_temp[df_temp[fam_col].isin(fam_options)].copy()
        
        if df.empty:
            self.material_name.setText("N/A")
            self.results_list.clear()
            self._setup_graph_theme() 
            QtWidgets.QMessageBox.information(self, "No materials", f"لم يتم العثور على مواد للفئة: {self.family_combo.currentText()}")
            return

        # 1. Get raw weights (0.0 to 1.0) from sliders
        weights = []
        for (sobj, _), (_, _) in zip(self.slider_objs, self.slider_defs):
            weights.append(sobj.value() / 100.0)

        weight_arr = np.array(weights, dtype=float)

        norm_df = pd.DataFrame(index=df.index)
        final_weights = weight_arr.copy()

        # 2. Iterate through properties for normalization and directional scoring
        for idx, (_, col_key) in enumerate(self.slider_defs):
            colname = self.col_map.get(col_key)
            
            if colname in df.columns and colname is not None:
                df[colname] = pd.to_numeric(df[colname], errors='coerce')
                
                data_series = df[colname].dropna()
                
                if data_series.empty or (data_series.max() == data_series.min()):
                    norm_df[col_key] = np.zeros(len(df), dtype=float)
                    final_weights[idx] = 0
                else:
                    # Impute and standardize
                    fill_value = data_series.median()
                    df[colname] = df[colname].fillna(fill_value) 
                    
                    # A. Standard Normalization (P_norm_standard): Highest physical value always gets 1.0
                    mn = df[colname].min()
                    mx = df[colname].max()
                    P_norm_standard = (df[colname] - mn) / (mx - mn)
                    
                    
                    # B. APPLY DIRECTION LOGIC based on the raw slider weight (w_raw)
                    w_raw = weights[idx] # Raw weight (0.0 to 1.0)
                    
                    if w_raw < 0.5:
                        # User wants a LOW value (e.g., Density w=0.10): Use inverted score (1 - P_norm_standard)
                        norm_df[col_key] = 1.0 - P_norm_standard
                    else:
                        # User wants a HIGH value (e.g., Strength w=0.90): Use standard score (P_norm_standard)
                        norm_df[col_key] = P_norm_standard
            else:
                norm_df[col_key] = np.zeros(len(df), dtype=float)
                final_weights[idx] = 0

        # 3. Normalize the actual weights used in the dot product
        if final_weights.sum() > 0:
             final_weights = final_weights / final_weights.sum()
        
        # 4. Calculate scores
        scores = norm_df.values.dot(final_weights)
        
        df = df.assign(**{'_score': scores}) 
        
        # 5. UI Update
        sorted_df = df.sort_values('_score', ascending=False).reset_index(drop=True)
        self._current_candidates_df = sorted_df

        if sorted_df.empty:
            self.material_name.setText("N/A")
            self._setup_graph_theme() 
            return

        top = sorted_df.iloc[0]
        self.update_left(top, sorted_df)
        self._update_graph(sorted_df) 

        self.results_list.clear()
        for i, row in sorted_df.head(10).iterrows():
            name_col = self.col_map.get('Material Name')
            name = str(row.get(name_col, 'Unknown Material')).strip()
            
            cost_col = self.col_map.get("Cost")
            cost_text = ""
            
            if cost_col and cost_col in row:
                 try:
                     cost_val = pd.to_numeric(row[cost_col], errors='coerce') 
                     if not pd.isna(cost_val):
                         cost_text = f" — {cost_val:g}$"
                 except:
                     pass

            self.results_list.addItem(f"{i+1}. {name}{cost_text} — Score: {row['_score']:.4f}")


    def _find_col(self, names):
        for n in names:
            if n in self.df.columns:
                return n
        return None

    def _setup_graph_theme(self):
        BG_COLOR = "#051428"  # Darker Blue/Navy Background
        TEXT_COLOR = "#FFFFFF"            
        GRID_COLOR = "#FFFFFF1A"          # Faint White Grid

        self.figure.patch.set_facecolor(BG_COLOR)
        
        self.ax = self.figure.add_subplot(111)
        self.ax.set_facecolor(BG_COLOR)

        self.ax.spines['top'].set_visible(False)
        self.ax.spines['right'].set_visible(False)
        self.ax.spines['bottom'].set_color(GRID_COLOR)
        self.ax.spines['left'].set_color(GRID_COLOR)

        self.ax.tick_params(axis='x', colors=TEXT_COLOR)
        self.ax.tick_params(axis='y', colors=TEXT_COLOR)
        
        self.ax.set_ylabel("Score", color=TEXT_COLOR, fontsize=12)
        
        self.ax.set_ylim(0, 1)
        self.ax.set_xlim(0, 1)
        self.ax.set_xticks([]) 
        self.ax.set_yticks([]) 
        
        self.ax.grid(False) 
        
        self.figure.tight_layout(pad=2.0)
        self.canvas.draw()
        
    def _update_graph(self, sorted_df):
        self.ax.clear()
        
        BG_COLOR = "#051428" 
        TEXT_COLOR = "#FFFFFF" 
        LINE_COLOR = "#00BFFF"   # Deep Sky Blue (Accent)
        FILL_COLOR = "#00BFFF2A" # Lighter fill with transparency (Matplotlib accepts 8-digit hex for alpha)
        GRID_COLOR = "#FFFFFF1A"
        
        self.ax.set_facecolor(BG_COLOR)

        top_10 = sorted_df.head(10).copy()
        scores = top_10['_score'].values
        num_candidates = len(scores)

        x_data = np.arange(num_candidates)
        
        self.ax.plot(x_data, scores, 
                      color=LINE_COLOR, 
                      linewidth=3, 
                      marker='o', 
                      markersize=8, 
                      markerfacecolor=LINE_COLOR,
                      markeredgecolor=TEXT_COLOR,
                      label='Material Score')
        
        self.ax.fill_between(x_data, scores, color=FILL_COLOR, alpha=0.8)

        self.ax.set_ylim(0, 1.05)
        self.ax.set_xlim(-0.5, num_candidates - 0.5)

        labels = [f"Mat {i+1}" for i in range(num_candidates)]
        
        self.ax.set_xticks(x_data)
        
        self.ax.set_xticklabels(labels, rotation=30, ha='right', color=TEXT_COLOR, fontsize=10)
        self.ax.set_yticks(np.arange(0, 1.1, 0.2)) 

        self.ax.spines['top'].set_visible(False)
        self.ax.spines['right'].set_visible(False)
        self.ax.spines['bottom'].set_color(GRID_COLOR)
        self.ax.spines['left'].set_color(GRID_COLOR)

        self.ax.tick_params(axis='x', colors=TEXT_COLOR)
        self.ax.tick_params(axis='y', colors=TEXT_COLOR)
        self.ax.set_ylabel("Score (0 to 1)", color=TEXT_COLOR, fontsize=12)
        
        self.ax.grid(axis='y', linestyle='--', alpha=0.3, color=GRID_COLOR)
        
        self.figure.tight_layout(pad=2.0)
        self.canvas.draw()


    def _style(self):
        font_family = "Poppins" 
        
        # New BLUE Theme Colors
        WINDOW_GRADIENT_C1 = "rgba(0, 10, 20, 255)"
        WINDOW_GRADIENT_C2 = "rgba(0, 5, 15, 255)"
        OUTER_GRADIENT_C1 = "rgba(10, 20, 30, 255)"
        OUTER_GRADIENT_C2 = "rgba(15, 25, 35, 255)"
        
        # Card Colors (Dark Navy/Deep Blue)
        LEFT_START_COLOR = "rgba(10, 30, 50, 220)"
        LEFT_END_COLOR = "rgba(20, 40, 60, 220)"
        RIGHT_START_COLOR = "rgba(5, 25, 45, 200)"
        RIGHT_END_COLOR = "rgba(15, 35, 55, 200)"
        
        # Accent Colors (Vibrant Blue/Cyan)
        ACCENT_COLOR = "#00BFFF"   # Deep Sky Blue
        ACCENT_DARK = "#008B8B"    # Dark Cyan
        
        GRAPH_BG = "rgba(5, 20, 40, 255)" 

        return f"""
        * {{
            font-family: "{font_family}", sans-serif;
        }}
        
        QMainWindow, QWidget#centralWidget {{
            background: qlineargradient(x1:0,y1:0,x2:1,y2:1,
                stop:0 {WINDOW_GRADIENT_C1}, stop:1 {WINDOW_GRADIENT_C2});
        }}
        
        #outer_card {{
            border-radius: 18px;
            background: qlineargradient(x1:0,y1:0,x2:1,y2:1,
                stop:0 {OUTER_GRADIENT_C1}, stop:1 {OUTER_GRADIENT_C2});
        }}
        
        #left_card {{
            border-radius: 14px;
            background: qlineargradient(x1:0,y1:0,x2:1,y2:1,
                stop:0 {RIGHT_START_COLOR}, stop:1 {LEFT_START_COLOR}); 
        }}
        
        #right_card {{
            border-radius: 14px;
            background: qlineargradient(x1:0,y1:0,x2:1,y2:1,
                stop:0 {LEFT_END_COLOR}, stop:1 {RIGHT_END_COLOR});
        }}
        
        QWidget#graph_container {{
             background-color: {GRAPH_BG};
             border-radius: 8px;
             padding: 5px;
        }}

        QLabel#title_label {{ color: white; font-size: 18px; font-weight: 700; }}
        QLabel#material_name {{ color: {ACCENT_COLOR}; font-size: 20px; font-weight: 700; }}
        #title_line {{ background: rgba(255,255,255,0.06); height:2px; }}

        /* QComboBox Style */
        QComboBox#family_combo {{
            background: qlineargradient(x1:0,y1:0,x2:1,y2:1,
                stop:0 {ACCENT_DARK}, stop:1 {ACCENT_COLOR});
            color: white;
            border-radius: 10px;
            padding: 6px 12px;
            padding-right: 30px; 
            border: 1px solid rgba(255,255,255,0.25);
            font-weight:600;
        }}

        QComboBox#family_combo::drop-down {{
            subcontrol-origin: padding;
            subcontrol-position: top right;
            width: 20px;
            border-left-width: 1px;
            border-left-color: rgba(255, 255, 255, 0.2);
            border-left-style: solid;
            border-top-right-radius: 8px;
            border-bottom-right-radius: 8px;
        }}
        
        QComboBox#family_combo:on {{
            border: 1px solid #1E90FF;
        }}
        
        QFrame#thin_deco {{ background: rgba(255,255,255,0.08); }}

        QPushButton#calc_button {{
            background: qlineargradient(x1:0,y1:0,x2:1,y2:0,
                stop:0 #1E90FF, stop:1 #4682B4);
            color:white;
            border-radius:12px;
            font-weight:700;
            padding:8px 14px;
        }}

        QListWidget#results_list {{
            background: rgba(0,0,0,0.25);
            color: white;
            border-radius: 8px;
            padding:6px;
        }}
        
        QGroupBox#category_group {{
            color: #ADD8E6; 
            font-size: 15px; 
            font-weight: 600; 
            margin-top: 10px; 
            padding-top: 10px;
            border: 1px solid rgba(255,255,255,0.1);
            border-radius: 8px;
        }}
        
        QGroupBox#category_group::title {{
            subcontrol-origin: margin;
            subcontrol-position: top left;
            padding: 0 10px;
            color: {ACCENT_COLOR};
        }}

        QSlider#custom_slider::groove:horizontal {{
            height:8px;
            border-radius:4px;
            background: rgba(255,255,255,0.12);
        }}

        QSlider#custom_slider::handle:horizontal {{
            width: 16px;
            height: 16px;
            margin: -4px 0;
            border-radius: 8px;
            background: {ACCENT_COLOR};
            border: 1px solid #6495ED;
        }}

        QSlider#custom_slider::handle:horizontal:hover {{
            background: #4169E1;
            border: 1px solid #F0FFFF;
        }}

        QLabel {{ color: white; font-size: 13px; }}
        QLabel#score_lbl {{ color:#f0f8ff; margin-top:8px; }}
        QLabel#count_lbl {{ color:#B0C4DE; font-size:11px; }}
        """

    def _on_result_clicked(self, item):
        try:
             idx = int(item.text().split('.')[0].strip()) - 1
        except:
             return
             
        if self._current_candidates_df is None:
            return
        if idx < 0 or idx >= len(self._current_candidates_df):
            return
        row = self._current_candidates_df.iloc[idx]

        self.update_left(row, self._current_candidates_df)

    def update_left(self, top_row, sorted_df):
        lay = self.left_props_layout
        while lay.count():
            it = lay.takeAt(0)
            w = it.widget()
            if w:
                w.deleteLater()

        mat_col_name = self.col_map.get('Material Name')
        mat_name = str(top_row.get(mat_col_name, 'Unknown Material')).strip()
        self.material_name.setText(mat_name)

        display_order = [
            ("Density", 'Density', "g/cm³"),
            ("Youngs Modulus", 'Youngs Modulus', "GPa"),
            ("Yield Strength", 'Yield Strength', "MPa"),
            ("Ultimate Strength", 'Ultimate Strength', "MPa"),
            ("Hardness", 'Hardness', "HB"),
            ("Thermal Conductivity", 'Thermal Conductivity', "W/mK"),
            ("Melting Point", 'Melting Point', "C"), 
            ("Electrical Conductivity", 'Electrical Conductivity', "%IACS"),
            ("Corrosion Resistance", 'Corrosion Resistance', "Rating"),
            ("Design Cost/kg", 'Cost', "$") 
        ]

        for disp_name, key, unit in display_order:
            col = self.col_map.get(key)
            
            if col is None or col not in top_row: 
                disp = '-'
            else:
                val = top_row.get(col, "")
                try:
                    v = pd.to_numeric(val, errors='coerce') 
                    
                    if pd.isna(v):
                        disp = '-'
                    else:
                        if 'Cost' in key:
                            disp = f"{v:g}$"
                        elif 'Corrosion Resistance' in disp_name:
                            disp = str(int(v))
                        else:
                            disp = f"{v:.3g} {unit}"
                except:
                    disp = f"{val} {unit}" if val else '-'

            left_lbl = QtWidgets.QLabel(disp_name)
            right_lbl = QtWidgets.QLabel(disp)
            right_lbl.setStyleSheet("font-weight:700; color:#f6e9ff;")
            right_lbl.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
            lay.addRow(left_lbl, right_lbl)
            
        # ----------------------------------------
        # تم حذف قسم عرض تفاصيل الموردين والأسعار بالكامل
        # ----------------------------------------
        
        score_lbl = QtWidgets.QLabel(f"Score: {top_row['_score']:.4f}")
        score_lbl.setStyleSheet("color:#f0f8ff; margin-top:8px;") # Light Blue
        lay.addRow("", score_lbl)

        count_lbl = QtWidgets.QLabel(f"Candidates considered: {len(sorted_df)}")
        count_lbl.setStyleSheet("color:#B0C4DE; font-size:11px;") # Muted Blue
        lay.addRow("", count_lbl)

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    
    win = ExactMaterialSelector(csv_path=CSV_PATH)
    win.show()
    sys.exit(app.exec_())
