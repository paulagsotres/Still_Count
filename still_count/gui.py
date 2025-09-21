# -*- coding: utf-8 -*-
"""
Created on Sat Sep 20 18:36:17 2025

@author: paula gomez sotres
"""
import tkinter as tk
from tkinter import filedialog, ttk, messagebox, simpledialog, scrolledtext
import cv2
import numpy as np
from PIL import Image, ImageTk
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import os
import json
import random
from collections import deque
import threading
import time
import tkinter.font as tkFont
from datetime import datetime
import requests


from .core import take_all_files, run_background_subtraction_for_analysis, detect_immobility, calculate_immobility_by_bin_core, create_immobility_mark_video,create_csv_immobility


class immobilityAnalyzerGUI:
    def __init__(self, master):
        self.master = master
        master.title("StillCount: Behavioral Immobility Analyzer")
        
         # Hardcoded preconfigured full analysis configs
        self.PRECONFIGURED_CONFIGS = {
            "FC context retrieval white BG": {
                "video_threshold": 75,
                "immobility_threshold": 400,
                "frame_interval_bg_sub": 5,
                "window_size_immobility": 5,
                "bins": 12,
                "time_adjustment": 0,
                "roi_x1": 48,
                "roi_y1": 27,
                "roi_x2": 537,
                "roi_y2": 431,
                "last_video_folder": ""
            },
            "Conditioning chamber TT/TL": {
                "video_threshold": 31,
                "immobility_threshold": 100,
                "frame_interval_bg_sub": 8,
                "window_size_immobility": 4,
                "bins": 12,
                "time_adjustment": 2,
                "roi_x1": 15,
                "roi_y1": 34,
                "roi_x2": 599,
                "roi_y2": 309,
                "last_video_folder": ""
            },

            "Open field": {
                "video_threshold": 30,
                "immobility_threshold": 50,
                "frame_interval_bg_sub": 4,
                "window_size_immobility": 5,
                "bins": 15,
                "time_adjustment": 0,
                "roi_x1": 134,
                "roi_y1": 0,
                "roi_x2": 1101,
                "roi_y2": 1017,
                "last_video_folder": ""
            }, 
            
            "FC context retrieval grid camera 2": {
                "video_threshold": 12,
                "immobility_threshold": 280,
                "frame_interval_bg_sub": 5,
                "window_size_immobility": 5,
                "bins": 12,
                "time_adjustment": 0,
                "roi_x1": 97,
                "roi_y1": 39,
                "roi_x2": 782,
                "roi_y2": 620,
                "last_video_folder": ""
            },
        }

        self.video_files = {}
        self.current_folder_path = ""
        self.config_file = "immobility_analysis_config.json"

        # Thresholds
        self.video_threshold = tk.IntVar(value=130)
        self.immobility_threshold = tk.IntVar(value=50)

        self.frame_interval_bg_sub = tk.IntVar(value=3)
        self.window_size_immobility = tk.IntVar(value=2)
        self.bins = tk.IntVar(value=12)
        self.time_adjustment = tk.IntVar(value=0)

        # ROI variables
        self.roi_x1 = tk.IntVar(value=10)
        self.roi_y1 = tk.IntVar(value=10)
        self.roi_x2 = tk.IntVar(value=110)
        self.roi_y2 = tk.IntVar(value=110)

        self.preview_frame = None
        self.roi_rect_id = None
        self.start_x = None
        self.start_y = None

        self.folder_path_var = tk.StringVar(value="")
        self.output_dir_var = tk.StringVar()

        self.analysis_results_cache = {}

        self.analysis_thread = None
        self.stop_analysis_event = threading.Event()

        # Classification related attributes
        self.file_listbox_files = {}
        self.file_classifications = {}
        self.categories = []
        self.selected_file_paths_for_analysis = []

        self.default_font = tkFont.nametofont("TkDefaultFont")
        self.default_font.configure(size=10, family="Segoe UI")
        self.master.option_add("*Font", self.default_font)

        self.create_widgets()
        self.load_config()
        self.master.after(100, self.load_random_frame_on_startup)


    def create_widgets(self):
        s = ttk.Style()
        s.theme_use('clam')

        s.configure("green.Horizontal.TProgressbar", 
                    troughcolor='lightgray', 
                    background='green', 
                    lightcolor='darkgreen',
                    darkcolor='green', 
                    bordercolor='gray', 
                    gripcount=0)

        self.master.grid_rowconfigure(0, weight=1)
        self.master.grid_rowconfigure(1, weight=0)
        self.master.grid_rowconfigure(2, weight=0)
        self.master.grid_rowconfigure(3, weight=0)

        self.master.grid_columnconfigure(0, weight=1)
        self.master.grid_columnconfigure(1, weight=1)
        self.master.grid_columnconfigure(2, weight=1)

        left_panel_frame = ttk.Frame(self.master, padding="10")
        left_panel_frame.grid(row=0, column=0, padx=5, pady=5, sticky="nsew")
        
        # --- NEW CODE: Add a logo frame at the top of the left panel ---
        logo_frame = ttk.Frame(left_panel_frame)
        logo_frame.grid(row=0, column=0, padx=5, pady=5, sticky="nsew")
    
        # Configure the logo frame cell to expand
        left_panel_frame.rowconfigure(0, weight=1)
        left_panel_frame.columnconfigure(0, weight=1)
    
        try:
            import requests
            from io import BytesIO
        
            logo_url = "https://github.com/paulagsotres/Still_Count/blob/master/Still_count_logo.png"
            response = requests.get(logo_url)
            response.raise_for_status()
        
            img_data = BytesIO(response.content)
            original_logo = Image.open(img_data)
        
            # Resize manually: fixed width = 120, height scaled proportionally
            target_width = 280
            img_ratio = original_logo.width / original_logo.height
            target_height = int(target_width / img_ratio)
            img_resized = original_logo.resize((target_width, target_height), Image.LANCZOS)
        
            logo_image = ImageTk.PhotoImage(img_resized)
        
            # Center the logo in the column
            logo_label = ttk.Label(logo_frame, image=logo_image)
            logo_label.image = logo_image  # keep reference
            logo_label.pack(padx=0, pady=0, expand=False)  # expand=False prevents stretching
            logo_label.pack(anchor="center")  # center horizontally
        
        except Exception as e:
            print(f"Could not load logo image: {e}")
            logo_label = ttk.Label(logo_frame, text="Error loading logo", foreground="red")
            logo_label.pack(fill="both", expand=True)
    
      
        # The following row configurations are re-ordered and adjusted to
        # accommodate the new logo frame.
        left_panel_frame.rowconfigure(0, weight=0)  # For the new logo_frame
        left_panel_frame.rowconfigure(1, weight=0)  # For the old config_frame
        left_panel_frame.rowconfigure(2, weight=0)  # For the old video_output_frame
        left_panel_frame.rowconfigure(3, weight=1)  # For the old params_frame, which should expand
        left_panel_frame.columnconfigure(0, weight=1)
        
        # --- END NEW CODE ---

        # Original frames, with updated row numbers
        config_frame = ttk.LabelFrame(left_panel_frame, text="Configuration", padding="10")
        config_frame.grid(row=1, column=0, padx=5, pady=5, sticky="ew")
        config_frame.columnconfigure(0, weight=0)
        config_frame.columnconfigure(1, weight=1)
        
        
        ttk.Label(config_frame, text="Select Preset:").grid(row=0, column=0, padx=2, pady=2, sticky="w")
        self.preconfig_var = tk.StringVar(self.master)
        self.preconfig_var.set("Select a preset")
        preconfig_options = list(self.PRECONFIGURED_CONFIGS.keys())
        
        # Pass to load_config *with selection*
        self.preconfig_menu = ttk.OptionMenu(
            config_frame,
            self.preconfig_var,
            self.preconfig_var.get(),
            *preconfig_options,
            command=self.load_preset_config  # NEW: separate function
        )
        self.preconfig_menu.grid(row=0, column=1, padx=2, pady=2, sticky="ew")

        ttk.Button(config_frame, text="Load Config", command=self.load_config_dialog).grid(row=1, column=0, padx=2, pady=2, sticky="ew")
        ttk.Button(config_frame, text="Save Config", command=self.save_config_dialog).grid(row=1, column=1, padx=2, pady=2, sticky="ew")
        
    
        
        video_output_frame = ttk.LabelFrame(left_panel_frame, text="Video & Output Folder", padding="10")
        video_output_frame.grid(row=2, column=0, padx=5, pady=5, sticky="ew")
        video_output_frame.columnconfigure(1, weight=1)
        ttk.Label(video_output_frame, text="Folder:").grid(row=0, column=0, padx=5, pady=5, sticky="w")
        self.folder_entry = ttk.Entry(video_output_frame, textvariable=self.folder_path_var, width=30)
        self.folder_entry.grid(row=0, column=1, padx=5, pady=5, sticky="ew")
        ttk.Button(video_output_frame, text="Browse", command=self.select_folder).grid(row=0, column=2, padx=5, pady=5)

        params_frame = ttk.LabelFrame(left_panel_frame, text="Detection Parameters", padding="10")
        params_frame.grid(row=3, column=0, padx=5, pady=5, sticky="nsew")
        params_frame.columnconfigure(1, weight=1)

        ttk.Label(params_frame, text="Video Binarization Threshold:").grid(row=0, column=0, padx=5, pady=2, sticky="w")
        self.video_threshold_scale = ttk.Scale(params_frame, from_=0, to=255, orient="horizontal", variable=self.video_threshold, command=self.update_preview_display)
        self.video_threshold_scale.grid(row=0, column=1, padx=5, pady=2, sticky="ew")
        ttk.Entry(params_frame, textvariable=self.video_threshold, width=5).grid(row=0, column=2, padx=5, pady=2)

        ttk.Label(params_frame, text="immobility Event Threshold (Pixels):").grid(row=1, column=0, padx=5, pady=2, sticky="w")
        self.immobility_threshold_scale = ttk.Scale(params_frame, from_=0, to=2000, orient="horizontal", variable=self.immobility_threshold)
        self.immobility_threshold_scale.grid(row=1, column=1, padx=5, pady=2, sticky="ew")
        ttk.Entry(params_frame, textvariable=self.immobility_threshold, width=5).grid(row=1, column=2, padx=5, pady=2)

        ttk.Label(params_frame, text="Frame Interval (BG Sub.):").grid(row=2, column=0, padx=5, pady=2, sticky="w")
        self.frame_interval_scale = ttk.Scale(params_frame, from_=1, to=20, orient="horizontal", variable=self.frame_interval_bg_sub)
        self.frame_interval_scale.grid(row=2, column=1, padx=5, pady=2, sticky="ew")
        ttk.Entry(params_frame, textvariable=self.frame_interval_bg_sub, width=5).grid(row=2, column=2, padx=5, pady=2)

        ttk.Label(params_frame, text="immobility Window Size (frames):").grid(row=3, column=0, padx=5, pady=2, sticky="w")
        ttk.Scale(params_frame, from_=1, to=60, orient="horizontal", variable=self.window_size_immobility).grid(row=3, column=1, padx=5, pady=2, sticky="ew")
        ttk.Entry(params_frame, textvariable=self.window_size_immobility, width=5).grid(row=3, column=2, padx=5, pady=2)


        ttk.Label(params_frame, text="Time Bins:").grid(row=4, column=0, padx=5, pady=2, sticky="w")
        ttk.Scale(params_frame, from_=1, to=30, orient="horizontal", variable=self.bins).grid(row=4, column=1, padx=5, pady=2, sticky="ew")
        ttk.Entry(params_frame, textvariable=self.bins, width=5).grid(row=4, column=2, padx=5, pady=2)

        ttk.Label(params_frame, text="Time Adjustment (seconds):").grid(row=5, column=0, padx=5, pady=2, sticky="w")
        ttk.Scale(params_frame, from_=0, to=60, orient="horizontal", variable=self.time_adjustment).grid(row=5, column=1, padx=5, pady=2, sticky="ew")
        ttk.Entry(params_frame, textvariable=self.time_adjustment, width=5).grid(row=5, column=2, padx=5, pady=2)
        
        ttk.Label(params_frame, text="Generate binary area plots for videos:").grid(row=6, column=0, padx=5, pady=2, sticky="w")

        self.plot_binary_area_var = tk.BooleanVar(value=False)
        self.plot_binary_area_checkbox = ttk.Checkbutton(params_frame,variable=self.plot_binary_area_var)
        self.plot_binary_area_checkbox.grid(row=6, column=2, padx=5, pady=5, sticky="w")
        
        ttk.Button(params_frame, text="Help", command=self.show_help_window).grid(row=7, column=0, columnspan=3, pady=10, sticky="ew")


        # --- Middle Panel: File List & Classification ---
        classification_frame = ttk.LabelFrame(self.master, text="File List & Classification", padding="10")
        classification_frame.grid(row=0, column=1, padx=5, pady=5, sticky="nsew")
        classification_frame.rowconfigure(1, weight=1)
        classification_frame.columnconfigure(0, weight=1)
        classification_frame.columnconfigure(1, weight=1)

        ttk.Label(classification_frame, text="Loaded Videos:").grid(row=0, column=0, padx=5, pady=5, sticky="w", columnspan=2)
        self.file_listbox = tk.Listbox(classification_frame, selectmode=tk.MULTIPLE, width=60, height=10, relief="flat", borderwidth=0)
        self.file_listbox.grid(row=1, column=0, padx=5, pady=5, sticky="nsew", columnspan=2)
        listbox_scrollbar = ttk.Scrollbar(classification_frame, orient="vertical", command=self.file_listbox.yview)
        listbox_scrollbar.grid(row=1, column=2, sticky="ns")
        self.file_listbox.config(yscrollcommand=listbox_scrollbar.set)

        self.file_listbox.bind("<<ListboxSelect>>", self.on_file_listbox_select)

        ttk.Label(classification_frame, text="Categories:").grid(row=2, column=0, pady=5, sticky="w")
        self.category_dropdown = tk.StringVar(self.master)
        self.category_dropdown.set("Select Category")
        self.category_menu = ttk.OptionMenu(classification_frame, self.category_dropdown, "")
        self.category_menu.grid(row=2, column=1, padx=5, pady=5, sticky="ew")
        self.update_category_dropdown()

        self.add_category_button = ttk.Button(classification_frame, text="Add Category", command=self.add_category)
        self.add_category_button.grid(row=3, column=0, padx=5, pady=2, sticky="ew")

        self.assign_category_button = ttk.Button(classification_frame, text="Assign Selected to Category", command=self.assign_selected_to_category)
        self.assign_category_button.grid(row=3, column=1, padx=5, pady=2, sticky="ew")

        self.show_classification_button = ttk.Button(classification_frame, text="Show Classifications", command=self.show_classifications)
        self.show_classification_button.grid(row=4, column=0, padx=5, pady=5, sticky="ew")

        self.save_classification_button = ttk.Button(classification_frame, text="Save Classifications", command=self.save_classifications)
        self.save_classification_button.grid(row=4, column=1, padx=5, pady=5, sticky="ew")


        # --- Right Panel: Video Preview & ROI Selection ---
        preview_frame_container = ttk.LabelFrame(self.master, text="Video Preview & ROI Selection", padding="10")
        preview_frame_container.grid(row=0, column=2, padx=5, pady=5, sticky="nsew")
        preview_frame_container.grid_rowconfigure(0, weight=1)
        preview_frame_container.grid_columnconfigure(0, weight=1)

        self.preview_canvas = tk.Canvas(preview_frame_container, bg="black", width=480, height=320, relief="sunken", borderwidth=1)
        self.preview_canvas.pack(padx=5, pady=5, fill="both", expand=True)

        self.preview_canvas.bind("<ButtonPress-1>", self.on_button_press)
        self.preview_canvas.bind("<B1-Motion>", self.on_mouse_drag)
        self.preview_canvas.bind("<ButtonRelease-1>", self.on_button_release)

        roi_coords_frame = ttk.Frame(preview_frame_container)
        roi_coords_frame.pack(pady=5)
        ttk.Label(roi_coords_frame, text="ROI (x,y,w,h):").pack(side="left")
        self.roi_display_label = ttk.Label(roi_coords_frame, text="0,0,0,0")
        self.roi_display_label.pack(side="left")

        ttk.Button(preview_frame_container, text="Load Random Frame", command=self.load_random_frame).pack(pady=5)


        # --- Bottom Panel: Action Buttons, Progress Bar, Status ---
        button_frame = ttk.Frame(self.master, padding="10")
        button_frame.grid(row=1, column=0, columnspan=3, padx=5, pady=5, sticky="ew")
        button_frame.columnconfigure(0, weight=1)
        button_frame.columnconfigure(1, weight=1)
        button_frame.columnconfigure(2, weight=1)
        button_frame.columnconfigure(3, weight=1)
        button_frame.columnconfigure(4, weight=1)

        self.run_immobility_csv_button = ttk.Button(button_frame, text="Run immobility Analysis (CSV)", command=self.start_immobility_analysis_thread)
        self.run_immobility_csv_button.pack(side="left", padx=5)

    
        self.export_results_by_categories_button = ttk.Button(button_frame, text="Export Results by Categories", command=self.export_results_by_categories, state=tk.DISABLED)
        self.export_results_by_categories_button.pack(side="left", padx=5)
        
        self.load_csv_export_video_button = ttk.Button(button_frame, text="Load immobility CSVs & Export Videos", command=self.load_csvs_and_export_videos)
        self.load_csv_export_video_button.pack(side="left", padx=5)

        self.export_video_button = ttk.Button(button_frame, text="Export Marked Videos", command=self.export_marked_videos, state=tk.DISABLED)
        self.export_video_button.pack(side="left", padx=5)
        
        self.export_boris_events_button = ttk.Button(button_frame, text="Export BORIS Events",command=self.export_boris_events)
        self.export_boris_events_button.pack(side="left", padx=5)

    
        ttk.Button(button_frame, text="Exit", command=self.master.quit).pack(side="right", padx=5)

        self.progress_bar = ttk.Progressbar(self.master, orient="horizontal", length=300, mode="determinate", style="green.Horizontal.TProgressbar")
        self.progress_bar.grid(row=2, column=0, columnspan=3, sticky="ew", padx=10, pady=(0, 10))
        self.progress_bar.grid_remove()

        self.status_label = ttk.Label(self.master, text="Status: Ready")
        self.status_label.grid(row=3, column=0, columnspan=3, rowspan=2, pady=5)




    def show_help_window(self):
        help_window = tk.Toplevel(self.master)
        help_window.title("immobility Detection Help")
        help_window.geometry("700x500")
        help_window.transient(self.master)
        help_window.grab_set()

        help_text = """
        Welcome to the Video immobility Analyzer!

        This tool detects immobility behavior in your AVI videos by analyzing pixel changes within a defined Region of Interest (ROI).

        1.  Video & Output Folder:
            * **Folder:** Select the directory containing your .avi video files. All output CSVs, the main Excel summary, and marked videos will be saved here.
            * The listbox will automatically populate with detected .avi files.

        2.  File List & Classification:
            * **Loaded Videos:** This list displays all .avi files found in the selected folder.
            * **Selection:** Select one or more videos from this list to be processed by "Run immobility Analysis (CSV)".
            * **Categories:** Define and manage custom categories (e.g., "Control", "Treatment").
            * **Assign Selected to Category:** Assigns the currently selected videos to the chosen category. This helps organize your data but does not affect the analysis logic itself.
            * **Show Classifications:** Opens a window displaying current file-to-category assignments.
            * **Save Classifications:** Saves your category definitions and file assignments to `classifications.json` in the selected video folder for future use.

        3.  Video Preview & ROI Selection:
            * **Load Random Frame:** Displays a random frame from one of your loaded videos. This is crucial for setting your ROI and thresholds accurately.
            * **Interactive ROI:** Click and drag on the black canvas to draw a **rectangle** Region of Interest (ROI). The ROI defines the area where pixel changes will be monitored. The drawing is from your click point to your drag end point.
            * **ROI (x,y,w,h):** Displays the calculated coordinates (x-coordinate of top-left, y-coordinate of top-left, width, height) of your selected ROI in original video resolution pixels.
            * **Visual Feedback:** As you adjust "Video Binarization Threshold" (below), you'll see its immediate effect on the ROI in the preview, showing which pixels are detected as changed.

        4.  Detection Parameters:
            * **Video Binarization Threshold:** This is the pixel intensity difference threshold used during background subtraction. It converts the difference image into a binary (black/white) image. Pixels with changes above this value are marked as "changed." (Range: 0-255, affects `_binary_diff_values.csv` and internal calculation).
            * **immobility Event Threshold (Pixels):** This is the threshold for the *number of changed pixels* (from the binarization step) allowed for a frame to be considered "still" or "immobility." If the total changed pixels in the ROI are *below* this value, the animal is "still." If *above*, the animal is "moving." (Range: 0-2000+, affects `immobility.csv` and overall analysis).
            * **Frame Interval (BG Sub.):** The number of frames between the two frames used for calculating the pixel difference (e.g., 3 means comparing frame N with frame N-3).
            * **immobility Window Size (frames):** The minimum number of *consecutive* "still" frames required for a period to be classified as a immobility bout. Short interruptions in stillness are ignored if they are less than this window size.
            * **Time Bins:** Divides the total video duration into this many equal time bins for summary statistics in the Excel output.
            * **Time Adjustment (seconds):** An offset applied to the start of the video for binning calculations (e.g., to exclude an initial acclimation period or pre-stimulus phase from binning).
            * **Generate Binary Area Plot:** Plot the binary area difference so you can visually set threshold of immobility
        5.  Action Buttons:
            * **Run immobility Analysis (CSV):**
                * Processes the *selected* videos from the list.
                * Performs background subtraction using your ROI and "Video Binarization Threshold".
                * Calculates immobility events using "immobility Event Threshold" and "immobility Window Size".
                * Saves `[mouse_name]_binary_diff_values.csv` (raw pixel change values per frame).
                * Saves `[mouse_name]_immobility.csv` (start/stop times of immobility bouts).
                * Generates a comprehensive summary Excel file: `ALL immobility RESULTS.xlsx` with total immobility and binned immobility data for all selected videos.
                * Displays a live progress bar and a reduced-resolution video preview during processing for visual feedback.
                * **Enables "Export Marked Videos" and "Export Results by Categories" upon successful completion.**
            * **Export Results by Categories:**
                * Loads the `ALL immobility RESULTS.xlsx` (generated by "Run immobility Analysis (CSV)").
                * Groups the results by the categories assigned in the "File List & Classification" section.
                * Exports a new Excel file named `ALL_immobility_RESULTS_BY_CATEGORY.xlsx` with separate sheets for each category.
                * Requires "Run immobility Analysis (CSV)" to have been run previously to generate the main Excel.
            * **Load immobility CSVs & Export Videos:**
                * Allows you to select a folder containing previously generated `immobility_XXXX.csv` files.
                * It parses these CSVs to reconstruct the immobility event data.
                * Then, it enables the "Export Marked Videos" button, allowing you to create marked videos (e.g., if you lost them or want to change their location) without re-running the full analysis.
            * **Export Marked Videos:**
            * Creates `[mouse_name]_MARKED.avi` videos in your output folder.
            * These videos will be at 1/3rd resolution of the original and will have a red circle marking immobility periods.
            * Requires a previous run of "Run immobility Analysis (CSV)" or "Load immobility CSVs & Export Videos" to have populated the internal data cache.
            * **Exit:** Closes the application.
        """

        text_widget = scrolledtext.ScrolledText(help_window, wrap=tk.WORD, padx=10, pady=10, font=self.default_font)
        text_widget.pack(expand=True, fill="both")
        text_widget.insert(tk.END, help_text)
        text_widget.config(state=tk.DISABLED)
        
        close_button = ttk.Button(help_window, text="Close", command=help_window.destroy)
        close_button.pack(pady=10)
        
        help_window.wait_window()


    def update_category_dropdown(self):
        menu = self.category_menu["menu"]
        menu.delete(0, "end")
        if not self.categories:
            self.category_dropdown.set("No Categories")
            return
        if self.categories:
            self.category_dropdown.set(self.categories[0])
        else:
            self.category_dropdown.set("Select Category")
        for category in self.categories:
            menu.add_command(label=category, command=lambda value=category: self.category_dropdown.set(value))

    def add_category(self):
        new_category = simpledialog.askstring("Add Category", "Enter new category name:")
        if new_category and new_category not in self.categories:
            self.categories.append(new_category)
            self.update_category_dropdown()
            messagebox.showinfo("Category Added", f"Category '{new_category}' added.")
        elif new_category:
            messagebox.showwarning("Duplicate Category", f"Category '{new_category}' already exists.")

    def on_file_listbox_select(self, event):
        self.selected_file_paths_for_analysis = []
        selected_indices = self.file_listbox.curselection()
        for i in selected_indices:
            filename_display_with_tag = self.file_listbox.get(i)
            filename_display = filename_display_with_tag.split(" [")[0]
            full_path = self.file_listbox_files.get(filename_display)
            if full_path:
                self.selected_file_paths_for_analysis.append(full_path)

        if not self.analysis_thread or not self.analysis_thread.is_alive():
            if self.selected_file_paths_for_analysis:
                self.run_immobility_csv_button.config(state=tk.NORMAL)
            else:
                self.run_immobility_csv_button.config(state=tk.DISABLED)
            
            self.export_results_by_categories_button.config(state=tk.DISABLED)
            self.export_video_button.config(state=tk.DISABLED)


    def assign_selected_to_category(self):
        selected_category = self.category_dropdown.get()
        if selected_category == "Select Category" or selected_category == "No Categories":
            messagebox.showwarning("No Category Selected", "Please select or add a category first.")
            return

        selected_indices = self.file_listbox.curselection()
        if not selected_indices:
            messagebox.showwarning("No Files Selected", "Please select files from the list to assign a category.")
            return

        updates = []
        for i in selected_indices:
            filename_display_with_tag = self.file_listbox.get(i)
            filename_display = filename_display_with_tag.split(" [")[0]

            found_filepath = self.file_listbox_files.get(filename_display)
            
            if found_filepath:
                self.file_classifications[found_filepath] = selected_category
                updates.append((i, f"{filename_display} [{selected_category}]"))
            else:
                print(f"Warning: Could not find full path for {filename_display} to assign category.")

        for i, new_text in sorted(updates, key=lambda item: item[0], reverse=True):
            self.file_listbox.delete(i)
            self.file_listbox.insert(i, new_text)
            self.file_listbox.selection_set(i)

        messagebox.showinfo("Assignment Complete", f"Selected files assigned to '{selected_category}'.")
        self.save_classifications() # Call save after assignment

    def show_classifications(self):
        if not self.file_classifications:
            messagebox.showinfo("No Classifications", "No files have been classified yet.")
            return

        classification_text = "Current File Classifications:\n\n"
        sorted_classifications = sorted(self.file_classifications.items(), key=lambda item: os.path.basename(item[0]))
        for filepath, category in sorted_classifications:
            classification_text += f"{os.path.basename(filepath)}: {category}\n"

        info_window = tk.Toplevel(self.master)
        info_window.title("File Classifications")
        text_widget = scrolledtext.ScrolledText(info_window, width=60, height=20, wrap=tk.WORD)
        text_widget.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)
        text_widget.insert(tk.END, classification_text)
        text_widget.config(state=tk.DISABLED)

    def save_classifications(self):
        if not self.file_classifications and not self.categories:
            messagebox.showinfo("Nothing to Save", "No classifications or categories to save.")
            return

        save_path = ""
        if self.current_folder_path:
            save_path = os.path.join(self.current_folder_path, "classifications.json")
        else:
            save_path = filedialog.asksaveasfilename(
                defaultextension=".json",
                filetypes=[("JSON files", "*.json")],
                title="Save Classifications As"
            )
            if not save_path:
                return

        try:
            save_data = {
                "categories": self.categories,
                "file_assignments": self.file_classifications
            }
            with open(save_path, "w") as f:
                json.dump(save_data, f, indent=4)
            messagebox.showinfo("Save Successful", f"Classifications saved to {save_path}")
            self._check_enable_export_categories_button() # Re-check enable state after saving
        except Exception as e:
            messagebox.showerror("Save Error", f"Failed to save classifications: {e}")

    def load_classifications(self, folder_path):
        classification_file_to_load = os.path.join(folder_path, "classifications.json")
        if not os.path.exists(classification_file_to_load):
            print(f"No classifications.json found in {folder_path}. Starting fresh.")
            self.file_classifications = {}
            self.categories = []
            self.update_category_dropdown()
            return

        try:
            with open(classification_file_to_load, "r") as f:
                loaded_data = json.load(f)

            if "categories" in loaded_data:
                self.categories = loaded_data["categories"]
                self.update_category_dropdown()

            if "file_assignments" in loaded_data:
                self.file_classifications = {
                    fp: category for fp, category in loaded_data["file_assignments"].items()
                    if os.path.exists(fp) and os.path.basename(fp) in self.file_listbox_files.keys()
                }
            messagebox.showinfo("Classifications Loaded", f"Loaded classifications from {classification_file_to_load}")
        except Exception as e:
            messagebox.showwarning("Load Error", f"Failed to load classifications from {classification_file_to_load}: {e}")
            self.file_classifications = {}
            self.categories = []
            self.update_category_dropdown()

    def apply_config(self, config):
        """Helper function to set all parameters from config dict"""
        self.video_threshold.set(config.get('video_threshold', self.video_threshold.get()))
        self.immobility_threshold.set(config.get('immobility_threshold', self.immobility_threshold.get()))
        self.frame_interval_bg_sub.set(config.get('frame_interval_bg_sub', self.frame_interval_bg_sub.get()))
        self.window_size_immobility.set(config.get('window_size_immobility', self.window_size_immobility.get()))
        self.bins.set(config.get('bins', self.bins.get()))
        self.time_adjustment.set(config.get('time_adjustment', self.time_adjustment.get()))
    
        self.roi_x1.set(config.get('roi_x1', self.roi_x1.get()))
        self.roi_y1.set(config.get('roi_y1', self.roi_y1.get()))
        self.roi_x2.set(config.get('roi_x2', self.roi_x2.get()))
        self.roi_y2.set(config.get('roi_y2', self.roi_y2.get()))
        
        self._update_roi_rectangle_from_vars()



    def _update_roi_rectangle_from_vars(self):
        """Draw/update ROI rectangle on preview canvas from roi_x1/y1/x2/y2 variables"""
        if self.preview_frame is None:
            return
    
        x1, y1 = self.roi_x1.get(), self.roi_y1.get()
        x2, y2 = self.roi_x2.get(), self.roi_y2.get()
    
        if self.roi_rect_id:
            self.preview_canvas.coords(self.roi_rect_id, x1, y1, x2, y2)
        else:
            self.roi_rect_id = self.preview_canvas.create_rectangle(x1, y1, x2, y2, outline="red", width=2)
    
        # Update ROI label
        self.roi_display_label.config(text=f"{x1},{y1},{abs(x2-x1)},{abs(y2-y1)}")
    
    def select_folder(self):
        folder_selected = filedialog.askdirectory()
        if folder_selected:
            self.current_folder_path = folder_selected
            self.folder_path_var.set(folder_selected)
            self.output_dir_var.set(folder_selected)
            
            
            videos, sample_fps = take_all_files(folder_selected)
        
            self.video_files = videos
            self.framerate = sample_fps if sample_fps is not None else 30 
            self.file_listbox_files = {}

            self.file_listbox.delete(0, tk.END)

            if not self.video_files:
                messagebox.showinfo("No Videos", "No video files found in the selected folder. Please choose a folder containing videos.")
                self.preview_frame = None
                self.update_preview_display()
                self.file_classifications = {}
                self.categories = []
                self.update_category_dropdown()
                self.selected_file_paths_for_analysis = []
                self._check_enable_export_categories_button() 
                return
            
            for mouse_name_key, filepath in sorted(self.video_files.items()):
                base_filename = os.path.basename(filepath)
                self.file_listbox_files[base_filename] = filepath
            
            self.load_classifications(folder_selected)

            for mouse_name_key, filepath in sorted(self.video_files.items()):
                category = self.file_classifications.get(filepath, "Unassigned")
                display_name = f"{os.path.basename(filepath)} [{category}]"
                self.file_listbox.insert(tk.END, display_name)


            self.load_random_frame()
            self.export_results_by_categories_button.config(state=tk.DISABLED)
            self.export_video_button.config(state=tk.DISABLED)
            self.analysis_results_cache = {}
            self.selected_file_paths_for_analysis = []
            self._check_enable_export_categories_button()

    def load_preset_config(self, selection):
        """Called when selecting from OptionMenu"""
        if selection in self.PRECONFIGURED_CONFIGS:
            config = self.PRECONFIGURED_CONFIGS[selection]
            self.apply_config(config)
            self.status_label.config(text=f"Status: Loaded preset '{selection}'")
        else:
            self.status_label.config(text="Status: Invalid preset selected.")
        
    def load_config(self):
        try:
            with open(self.config_file, 'r') as f:
                config = json.load(f)
            self.apply_config(config)
            self.status_label.config(text=f"Status: Configuration loaded from {self.config_file}")
        except FileNotFoundError:
            self.status_label.config(text="Status: No configuration file found. Using default settings.")
        except json.JSONDecodeError:
            self.status_label.config(text="Status: Error reading configuration file. Using default settings.")
        except Exception as e:
            self.status_label.config(text=f"Status: Error loading config: {e}. Using default settings.")
        
        
        
    def save_config(self):
        config = {
            'video_threshold': self.video_threshold.get(),
            'immobility_threshold': self.immobility_threshold.get(),
            'frame_interval_bg_sub': self.frame_interval_bg_sub.get(),
            'window_size_immobility': self.window_size_immobility.get(),
            'bins': self.bins.get(),
            'time_adjustment': self.time_adjustment.get(),
            'roi_x1': self.roi_x1.get(),
            'roi_y1': self.roi_y1.get(),
            'roi_x2': self.roi_x2.get(),
            'roi_y2': self.roi_y2.get(),
            'last_video_folder': self.current_folder_path,
        }
        try:
            with open(self.config_file, 'w') as f:
                json.dump(config, f, indent=4)
            self.status_label.config(text=f"Status: Configuration saved to {self.config_file}")
            self._check_enable_export_categories_button() # Re-check enable state after saving
        except Exception as e:
            messagebox.showerror("Save Error", f"Error saving config: {e}")

    def save_config_dialog(self):
        file_path = filedialog.asksaveasfilename(
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
            initialfile=self.config_file
        )
        if file_path:
            self.config_file = file_path
            self.save_config()

    def load_config_dialog(self):
        """Called when pressing 'Load Config' button"""
        file_path = filedialog.askopenfilename(
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
            initialfile=self.config_file
        )
        if file_path:
            self.config_file = file_path
            self.load_config()


    def load_random_frame_on_startup(self):
        if self.current_folder_path:
            self.video_files = take_all_files(self.current_folder_path)
            if self.video_files:
                self.file_listbox_files = {}
                for mouse_name_key, filepath in sorted(self.video_files.items()):
                    self.file_listbox_files[os.path.basename(filepath)] = filepath
                self.load_random_frame()
            else:
                self.status_label.config(text="Status: No videos found in configured folder. Select a folder.")
        else:
            self.status_label.config(text="Status: Please select a video folder to load a frame.")


    def load_random_frame(self):
        dir_path = self.current_folder_path
        if not dir_path:
            self.status_label.config(text="Status: Please select a video folder first!")
            return
    
        if not self.video_files:
            self.video_files = take_all_files(dir_path)
    
        if not self.video_files:
            self.status_label.config(text="Status: No video files found in the selected folder.")
            self.preview_frame = None
            self.bg_sub_frame = None
            self.update_preview_display()
            return
    
        random_video_key = random.choice(list(self.video_files.keys()))
        video_path_to_preview = self.video_files[random_video_key]
    
        cap = cv2.VideoCapture(video_path_to_preview)
        if not cap.isOpened():
            self.status_label.config(text=f"Error: Could not open video {video_path_to_preview}")
            self.preview_frame = None
            self.bg_sub_frame = None
            self.update_preview_display()
            return
    
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames == 0:
            self.status_label.config(text="Status: Video has no frames.")
            cap.release()
            self.preview_frame = None
            self.bg_sub_frame = None
            self.update_preview_display()
            return
    
        random_frame_idx = random.randint(0, total_frames - 1)
        cap.set(cv2.CAP_PROP_POS_FRAMES, random_frame_idx)
        ret1, frame1 = cap.read()
    
        # try to load frame for background subtraction
        frame_interval = self.frame_interval_bg_sub.get()
        bg_idx = min(random_frame_idx + frame_interval, total_frames - 1)
        cap.set(cv2.CAP_PROP_POS_FRAMES, bg_idx)
        ret2, frame2 = cap.read()
        cap.release()
    
        if ret1 and ret2:
            # background subtraction
            bg_sub = cv2.absdiff(frame1, frame2)
    
            # blended frame (original + bg_sub)
            blended = cv2.addWeighted(frame1, 1.0, bg_sub, 0.5, 0)
    
            self.preview_frame = blended               # what you display
            self.bg_sub_frame = bg_sub                 # what you threshold ROI on
            self.status_label.config(
                text=f"Status: Loaded random frame {random_frame_idx} with BG subtraction (interval {frame_interval})"
            )
            self.update_preview_display()
        else:
            self.status_label.config(text="Status: Failed to load random frame or background frame.")
            self.preview_frame = None
            self.bg_sub_frame = None
            self.update_preview_display()
    
    
    def update_preview_display(self, event=None):
        if self.preview_frame is None:
            self.preview_canvas.delete("all")
            self.photo = None
            self.roi_display_label.config(text="N/A")
            return
    
        display_frame = self.preview_frame.copy()
    
        x_min_final, y_min_final, x_max_final, y_max_final = self.get_current_roi_coords()
    
        if x_max_final > x_min_final and y_max_final > y_min_final:
            # --- use background subtraction for thresholding ---
            temp_gray = cv2.cvtColor(self.bg_sub_frame, cv2.COLOR_BGR2GRAY)
            roi_gray = temp_gray[y_min_final:y_max_final, x_min_final:x_max_final]
    
            if roi_gray.shape[0] > 0 and roi_gray.shape[1] > 0:
                _, binary_roi = cv2.threshold(roi_gray, self.video_threshold.get(), 255, cv2.THRESH_BINARY)
                binary_roi = cv2.cvtColor(binary_roi, cv2.COLOR_GRAY2BGR)
    
                display_frame[y_min_final:y_max_final, x_min_final:x_max_final] = binary_roi
            else:
                self.roi_display_label.config(text="Invalid ROI (Zero Area)")
                cv2.rectangle(display_frame, (x_min_final, y_min_final), (x_max_final, y_max_final), (0, 0, 255), 2)
    
            cv2.rectangle(display_frame, (x_min_final, y_min_final), (x_max_final, y_max_final), (0, 255, 0), 2)
    
            roi_width = abs(x_max_final - x_min_final)
            roi_height = abs(y_max_final - y_min_final)
            self.roi_display_label.config(text=f"{x_min_final},{y_min_final},{roi_width},{roi_height}")
        else:
            self.roi_display_label.config(text="Invalid ROI (Zero Area)")
            cv2.rectangle(display_frame, (x_min_final, y_min_final), (x_max_final, y_max_final), (0, 0, 255), 2)
    
        img = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)
    
        canvas_width = self.preview_canvas.winfo_width()
        canvas_height = self.preview_canvas.winfo_height()
        if canvas_width == 0 or canvas_height == 0:
            canvas_width, canvas_height = 480, 320
    
        img.thumbnail((canvas_width, canvas_height), Image.LANCZOS)
    
        self.photo = ImageTk.PhotoImage(image=img)
        self.preview_canvas.delete("all")
        self.preview_canvas.create_image(0, 0, image=self.photo, anchor="nw")


    def on_button_press(self, event):
        self.start_x = self.preview_canvas.canvasx(event.x)
        self.start_y = self.preview_canvas.canvasy(event.y)
        
        if self.roi_rect_id:
            self.preview_canvas.delete(self.roi_rect_id)
        self.roi_rect_id = self.preview_canvas.create_rectangle(self.start_x, self.start_y, self.start_x, self.start_y, outline="red", width=2)

    def on_mouse_drag(self, event):
        if self.start_x is None or self.start_y is None:
            return

        cur_x = self.preview_canvas.canvasx(event.x)
        cur_y = self.preview_canvas.canvasy(event.y)

        canvas_width = self.preview_canvas.winfo_width()
        canvas_height = self.preview_canvas.winfo_height()

        cur_x_clamped = max(0, min(cur_x, canvas_width))
        cur_y_clamped = max(0, min(cur_y, canvas_height))

        x1_rect = min(self.start_x, cur_x_clamped)
        y1_rect = min(self.start_y, cur_y_clamped)
        x2_rect = max(self.start_x, cur_x_clamped)
        y2_rect = max(self.start_y, cur_y_clamped)


        if self.roi_rect_id:
            self.preview_canvas.coords(self.roi_rect_id, x1_rect, y1_rect, x2_rect, y2_rect)

        self.update_roi_vars_from_canvas(x1_rect, y1_rect, x2_rect, y2_rect)
        self.update_preview_display()

    def on_button_release(self, event):
        if self.start_x is None or self.start_y is None:
            return

        final_x = self.preview_canvas.canvasx(event.x)
        final_y = self.preview_canvas.canvasy(event.y)

        canvas_width = self.preview_canvas.winfo_width()
        canvas_height = self.preview_canvas.winfo_height()

        final_x_clamped = max(0, min(final_x, canvas_width))
        final_y_clamped = max(0, min(final_y, canvas_height))
        
        x1_rect = min(self.start_x, final_x_clamped)
        y1_rect = min(self.start_y, final_y_clamped)
        x2_rect = max(self.start_x, final_x_clamped)
        y2_rect = max(self.start_y, final_y_clamped) # Corrected: Was using y1_clamped_rect incorrectly.


        if self.roi_rect_id:
            self.preview_canvas.coords(self.roi_rect_id, x1_rect, y1_rect, x2_rect, y2_rect)

        self.update_roi_vars_from_canvas(x1_rect, y1_rect, x2_rect, y2_rect)
        self.update_preview_display()

        self.start_x = None
        self.start_y = None
    
    

            
    def update_roi_vars_from_canvas(self, x1_canvas, y1_canvas, x2_canvas, y2_canvas):
        if self.preview_frame is None:
            return

        original_h, original_w, _ = self.preview_frame.shape
        canvas_w = self.preview_canvas.winfo_width()
        canvas_h = self.preview_canvas.winfo_height()

        if canvas_w == 0 or canvas_h == 0:
            return

        img_pil_original = Image.fromarray(cv2.cvtColor(self.preview_frame, cv2.COLOR_BGR2RGB))
        temp_img = img_pil_original.copy()
        temp_img.thumbnail((canvas_w, canvas_h), Image.LANCZOS)
        
        display_w, display_h = temp_img.size

        offset_x = (canvas_w - display_w) / 2
        offset_y = (canvas_h - display_h) / 2

        x1_img = max(0, min(x1_canvas - offset_x, display_w))
        y1_img = max(0, min(y1_canvas - offset_y, display_h))
        x2_img = max(0, min(x2_canvas - offset_x, display_w))
        y2_img = max(0, min(y2_canvas - offset_y, display_h))


        scale_w = original_w / display_w
        scale_h = original_h / display_h

        x1_original = int(x1_img * scale_w)
        y1_original = int(y1_img * scale_h)
        x2_original = int(x2_img * scale_w)
        y2_original = int(y2_img * scale_h)

        self.roi_x1.set(min(x1_original, x2_original))
        self.roi_y1.set(min(y1_original, y2_original))
        self.roi_x2.set(max(x1_original, x2_original))
        self.roi_y2.set(max(y1_original, y2_original))


    def get_current_roi_coords(self):
        x1 = self.roi_x1.get()
        y1 = self.roi_y1.get()
        x2 = self.roi_x2.get()
        y2 = self.roi_y2.get()
        return min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)

    def _update_progressbar_per_frame(self, current_frame, total_frames):
        if self.stop_analysis_event.is_set():
            return
            
        progress_value = (current_frame / total_frames) * 100
        self.master.after_idle(lambda: self.progress_bar.config(value=progress_value))

    def _update_live_video_preview(self, frame):
        if self.stop_analysis_event.is_set():
            return

        display_width = self.preview_canvas.winfo_width()
        display_height = self.preview_canvas.winfo_height()
        
        if display_width == 0 or display_height == 0:
            display_width, display_height = 480, 320

        target_preview_w = display_width // 3
        target_preview_h = display_height // 3
        
        if target_preview_w == 0: target_preview_w = 1
        if target_preview_h == 0: target_preview_h = 1

        frame_resized = cv2.resize(frame, (target_preview_w, target_preview_h), interpolation=cv2.INTER_AREA)
        
        img = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)
        
        self.photo = ImageTk.PhotoImage(image=img)
        
        self.master.after_idle(lambda: self.preview_canvas.create_image(0, 0, image=self.photo, anchor="nw"))

    

    def plot_binary_diff(self, mouse_name, binary_area_series, immobility_threshold, output_folder):
        """
        Plots the binary difference values over time and highlights immobility periods.
        Opens a standard Matplotlib window with zoom & pan.
        """
        if binary_area_series.empty:
            print(f"No binary difference data to plot for {mouse_name}.")
            return
    
        plt.figure(figsize=(10, 6))
        plt.plot(binary_area_series.index, binary_area_series.values,
                 label='Binary Diff Pixel Count', color='blue')
        plt.axhline(y=immobility_threshold, color='red', linestyle='--',
                    label=f'Immobility Threshold ({immobility_threshold} pixels)')
    
        plt.title(f'Binary Difference Pixel Count for {mouse_name}')
        plt.xlabel('Frame Number')
        plt.ylabel('Pixel Count')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
    
        plt.show()  # Opens interactive window with zoom/pan toolbar


    def start_immobility_analysis_thread(self):
        dir_path = self.current_folder_path
        output_folder = self.output_dir_var.get()

        if not dir_path or not os.path.isdir(dir_path):
            messagebox.showwarning("Input Error", "Please select a valid video folder!")
            return
        if not output_folder or not os.path.isdir(output_folder):
            messagebox.showwarning("Input Error", "Please select a valid output folder!")
            return
        
        if not self.selected_file_paths_for_analysis:
            messagebox.showwarning("Selection Error", "Please select one or more video files from the list to analyze!")
            return

        self.analysis_results_cache = {}
        self.export_results_by_categories_button.config(state=tk.DISABLED)
        self.export_video_button.config(state=tk.DISABLED)
        
        self.run_immobility_csv_button.config(state=tk.DISABLED)
        self.load_csv_export_video_button.config(state=tk.DISABLED)
        self.stop_analysis_event.clear()

        self.analysis_thread = threading.Thread(target=self._run_immobility_analysis_csv_threaded)
        self.analysis_thread.start()

    def _run_immobility_analysis_csv_threaded(self):
        dir_path = self.current_folder_path
        output_folder = self.output_dir_var.get()

        os.makedirs(output_folder, exist_ok=True)
        self.save_config()

        if not self.selected_file_paths_for_analysis:
            self.master.after_idle(lambda: messagebox.showwarning("Selection Error", "No files selected for analysis."))
            self.master.after_idle(self._analysis_finished_callback)
            return


        video_threshold = float(self.video_threshold.get())
        immobility_threshold = float(self.immobility_threshold.get())
        window_size_immobility = int(self.window_size_immobility.get())
        frame_interval_bg_sub = int(self.frame_interval_bg_sub.get())
        bins = int(self.bins.get())
        time_adjustment = int(self.time_adjustment.get())
        
        roi_x, roi_y, roi_x2_val, roi_y2_val = self.get_current_roi_coords()
        roi_width = abs(roi_x2_val - roi_x)
        roi_height = abs(roi_y2_val - roi_y)

        if roi_width <= 0 or roi_height <= 0:
            self.master.after_idle(lambda: messagebox.showerror("ROI Error", "Invalid ROI dimensions (width or height is zero). Please draw a valid ROI."))
            self.master.after_idle(self._analysis_finished_callback)
            return

 

        self.master.after_idle(lambda: self.status_label.config(text="Status: Starting immobility analysis (CSV generation)..."))
        self.master.after_idle(lambda: self.progress_bar.grid())
        self.master.after_idle(lambda: self.progress_bar.config(value=0, maximum=100))

        self.analysis_results_cache = {}
        all_results_excel_combined = pd.DataFrame()
        total_videos = len(self.selected_file_paths_for_analysis)
        processed_count = 0

        for video_path in self.selected_file_paths_for_analysis:
            if self.stop_analysis_event.is_set():
                break

            processed_count += 1
            mouse = Path(video_path).stem.split('-')[0]
            self.master.after_idle(lambda m=mouse, pc=processed_count, tv=total_videos: self.status_label.config(text=f"Status: Processing ({pc}/{tv}) {m}..."))
            
            framerate, binary_area_series = run_background_subtraction_for_analysis(
                video_path, roi_x, roi_y, roi_width, roi_height, video_threshold, frame_interval_bg_sub,
                progress_callback=self._update_progressbar_per_frame,
                frame_display_callback=self._update_live_video_preview,
                stop_event=self.stop_analysis_event
            )

            if self.stop_analysis_event.is_set():
                break

            if binary_area_series.empty:
                self.master.after_idle(lambda m=mouse: messagebox.showwarning("Warning", f"Could not get binary area data for {m}. Skipping."))
                continue


            persistent_immobility, frame_events, seconds_immobility = detect_immobility(
                binary_area_series, immobility_threshold, window_size_immobility, framerate
            )
            
            self.analysis_results_cache[mouse] = {
                'persistent_immobility': persistent_immobility,
                'frame_events': frame_events,
                'seconds_immobile': seconds_immobility,
                'binary_area_series': binary_area_series # Store binary_area_series here
            }

            create_csv_immobility(persistent_immobility, mouse, framerate, output_folder)

            immobility_by_bins_df = calculate_immobility_by_bin_core(
                persistent_immobility, bins, framerate, time_adjustment
            )
            total_immobility_df = pd.DataFrame({'total_immobility': [seconds_immobility]})
            current_mouse_results = pd.concat([total_immobility_df, immobility_by_bins_df], axis=1)
            current_mouse_results.index = [mouse]
            all_results_excel_combined = pd.concat([all_results_excel_combined, current_mouse_results], axis=0, sort=False)

            if self.plot_binary_area_var.get():
                self.master.after_idle(
                    lambda m=mouse, bas=binary_area_series, ft=immobility_threshold, op=output_folder:
                        self.plot_binary_diff(m, bas, ft, op)
                )

            self.master.after_idle(lambda pc=processed_count, tv=total_videos: self.progress_bar.config(value=(pc / tv) * 100))

        if not all_results_excel_combined.empty:
            final_excel_path = os.path.join(output_folder, "ALL SUBJECTS RESULTS.xlsx")
            all_results_excel_combined.to_excel(final_excel_path)
            self.master.after_idle(lambda: messagebox.showinfo("Analysis Complete", f"Immobility analysis CSVs and ALL immobility RESULTS.xlsx generated successfully."))
        else:
            self.master.after_idle(lambda: messagebox.showwarning("Analysis Result", "No immobility data generated for any videos."))

        self.master.after_idle(self._analysis_finished_callback)

    def _analysis_finished_callback(self):
        self.progress_bar.grid_remove()
        self.progress_bar.config(value=0)
        
        self.run_immobility_csv_button.config(state=tk.NORMAL)
        self.load_csv_export_video_button.config(state=tk.NORMAL)

        if not self.analysis_results_cache:
            self.status_label.config(text="Status: No immobility data generated for any videos.")
            self.export_results_by_categories_button.config(state=tk.DISABLED)
            self.export_video_button.config(state=tk.DISABLED)
            messagebox.showwarning("Analysis Result", "No immobility data generated for any videos.")
        else:
            self.status_label.config(text="Status: immobility analysis (CSV & Excel) Complete!")
            self.export_results_by_categories_button.config(state=tk.NORMAL)
            self.export_video_button.config(state=tk.NORMAL)
        print("immobility analysis (CSV & Excel) complete.")

    def _check_enable_export_categories_button(self):
        """Checks if ALL_immobility_RESULTS.xlsx and classifications.json exist to enable the button."""
        if not self.current_folder_path or not os.path.isdir(self.current_folder_path):
            self.export_results_by_categories_button.config(state=tk.DISABLED)
            return

        main_excel_exists = os.path.exists(os.path.join(self.current_folder_path, "ALL SUBJECTS RESULTS.xlsx"))
        classifications_json_exists = os.path.exists(os.path.join(self.current_folder_path, "classifications.json"))

        if main_excel_exists and classifications_json_exists:
            self.export_results_by_categories_button.config(state=tk.NORMAL)
        else:
            self.export_results_by_categories_button.config(state=tk.DISABLED)


    def export_results_by_categories(self):
        output_folder = self.output_dir_var.get()
        if not output_folder or not os.path.isdir(output_folder):
            messagebox.showwarning("Input Error", "Please select a valid output folder!")
            return

        main_excel_path = os.path.join(output_folder, "ALL SUBJECTS RESULTS.xlsx")
        if not os.path.exists(main_excel_path):
            messagebox.showwarning("File Missing", "ALL SUBJECTS RESULTS.xlsx not found. Please run 'Run immobility Analysis (CSV)' first.")
            return
        
        self.status_label.config(text="Status: Exporting results by categories...")
        self.progress_bar.grid()
        self.progress_bar.config(value=0, maximum=100)
        self.master.update_idletasks()

        try:
            all_results_df = pd.read_excel(main_excel_path, index_col=0)

            if all_results_df.empty:
                messagebox.showwarning("Data Empty", "The loaded ALL SUBJECTS RESULTS.xlsx is empty.")
                self.progress_bar.grid_remove()
                self.status_label.config(text="Status: Ready")
                return
            
            output_excel_path = os.path.join(output_folder, "ALL_SUBJECTS_RESULTS_BY_CATEGORY.xlsx")

            with pd.ExcelWriter(output_excel_path, engine='xlsxwriter') as writer:
                all_results_df['Category'] = 'Unassigned'
                
                for mouse_name_from_excel in all_results_df.index:
                    found_video_path = None
                    if mouse_name_from_excel in self.video_files:
                        found_video_path = self.video_files[mouse_name_from_excel]
                    else:
                        for original_key, full_path_from_video_files in self.video_files.items():
                             if Path(full_path_from_video_files).stem.split('-')[0] == mouse_name_from_excel:
                                found_video_path = full_path_from_video_files
                                break

                    if found_video_path and found_video_path in self.file_classifications:
                        all_results_df.loc[mouse_name_from_excel, 'Category'] = self.file_classifications[found_video_path]
                
                cols = ['Category'] + [col for col in all_results_df.columns if col != 'Category']
                all_results_df = all_results_df[cols]
                
                all_results_df.to_excel(writer, sheet_name='Overall Summary', index=True)

                for category in sorted(self.categories + ["Unassigned"]):
                    category_df = all_results_df[all_results_df['Category'] == category].copy()
                    if not category_df.empty:
                        category_df_to_export = category_df.drop(columns=['Category']) 
                        category_df_to_export.to_excel(writer, sheet_name=category, index=True)
            
            self.progress_bar.grid_remove()
            self.progress_bar["value"] = 0
            self.status_label.config(text="Status: Results by Categories Export Complete!")
            messagebox.showinfo("Export Complete", f"Results by category exported to {output_excel_path}")

        except Exception as e:
            self.progress_bar.grid_remove()
            self.progress_bar["value"] = 0
            self.status_label.config(text="Status: Error during category export.")
            messagebox.showerror("Export Error", f"Failed to export results by categories: {e}")


    def export_marked_videos(self):
        output_folder = self.output_dir_var.get()
        if not output_folder or not os.path.isdir(output_folder):
            messagebox.showwarning("Input Error", "Please select a valid output folder!")
            return
    
        if not self.analysis_results_cache:
            messagebox.showwarning(
                "No Data",
                "No immobility calculation data found. Please run 'Run immobility Analysis (CSV)' first or 'Load immobility CSVs & Export Videos'."
            )
            return
    
        total_videos_to_export = len(self.analysis_results_cache)
        exported_count = 0
    
        for mouse, data in self.analysis_results_cache.items():
            exported_count += 1
            self.status_label.config(
                text=f"Status: Exporting video ({exported_count}/{total_videos_to_export}) for {mouse}..."
            )
            self.master.update_idletasks()
    
            video_path = self.video_files.get(mouse)
            if not video_path:
                messagebox.showwarning(
                    "Video Not Found",
                    f"Original video file for '{mouse}' not found. Skipping this video."
                )
                continue
    
            frame_events = data['frame_events']
            if len(frame_events) == 0:
                messagebox.showinfo(
                    "No immobility Events",
                    f"No immobility events detected for {mouse}. Skipping this video."
                )
                continue
    
            output_video_name = Path(video_path).stem + "_MARKED.avi"
            output_file_path = os.path.join(output_folder, output_video_name)
    
            # Define a local callback that prints every 100 frames
            def local_frame_progress_callback(current_frame, total_frames):
                if current_frame % 1000 == 0 or current_frame == total_frames - 1:
                    print(f"[{mouse}] Processing frame {current_frame+1}/{total_frames}")
    
            # Call your existing function but pass the local callback
            create_immobility_mark_video(
                video_path,
                output_file_path,
                frame_events,
                frame_progress_callback=local_frame_progress_callback  # if your function accepts it
            )
    
        self.status_label.config(text="Status: Video Export Complete!")
        messagebox.showinfo("Video Export Complete", "All marked videos exported successfully.")
        print("Video export complete.")

    
    def export_boris_events(self):
        folder = self.folder_path_var.get()
        if not folder:
            folder = filedialog.askdirectory(title="Select folder containing immobility CSVs")
            if not folder:
                return
            self.folder_path_var.set(folder)
    
        # Create output folder
        output_folder = os.path.join(folder, "BORIS_compatible")
        os.makedirs(output_folder, exist_ok=True)
    
        # Find CSVs containing "immobility"
        csv_files = [f for f in os.listdir(folder) if f.lower().endswith(".csv") and f.startswith("immobility_")]
        if not csv_files:
            messagebox.showinfo("Info", "No CSV files starting with 'immobility_' found in the folder.")
            return
    
        video_extensions = [".avi", ".mp4", ".mov", ".mkv", ".wmv"]
    
        for csv_file in csv_files:
            try:
                csv_path = os.path.join(folder, csv_file)
                df = pd.read_csv(csv_path, sep=",")  # adjust separator if needed
    
                # --- Avoid duplicate time/frame entries ---
                times = df['Time'].tolist()
                frames = df['Image index'].tolist()
                behaviors = df['Behavior'].tolist()
    
                new_rows = []
                for i in range(len(times)):
                    time_val = times[i]
                    frame_val = frames[i]
    
                    # Check if same as previous row
                    if new_rows and time_val == new_rows[-1][0] and frame_val == new_rows[-1][5]:
                        time_val += 0.1
                        frame_val += 1
    
                    new_rows.append([time_val, '', behaviors[i], '', '', frame_val])
    
                # --- Export as BORIS-compatible TXT without trailing tabs ---
                txt_filename = os.path.splitext(csv_file)[0] + ".txt"
                txt_path = os.path.join(output_folder, txt_filename)
    
                with open(txt_path, 'w', newline='') as f:
                    for row_idx, row in enumerate(new_rows):
                        line = '\t'.join(str(val) for val in row)
                        f.write(line)
                        if row_idx < len(new_rows) - 1:  # add newline only between rows
                            f.write('\n')
    
            except Exception as e:
                messagebox.showerror("Error", f"Error processing {csv_file}: {e}")
    
        messagebox.showinfo("Done", f"BORIS-compatible files saved in:\n{output_folder}")

    

    def load_csvs_and_export_videos(self):
        folder_selected = filedialog.askdirectory(title="Select folder containing immobility_XXXX.csv files and original videos")
        if not folder_selected:
            return

        self.current_folder_path = folder_selected
        self.folder_path_var.set(folder_selected)
        self.output_dir_var.set(folder_selected)

        self.status_label.config(text="Status: Loading immobility CSVs...")
        self.progress_bar.grid()
        self.progress_bar.config(value=0, maximum=100)
        self.master.after_idle(lambda: self.progress_bar.config(value=0))
        self.master.update_idletasks()

        self.analysis_results_cache = {}
        csv_files_found = list(Path(folder_selected).glob("immobility_*.csv"))
        self.video_files = take_all_files(folder_selected)

        if not csv_files_found:
            messagebox.showwarning("No CSVs Found", f"No 'immobility_*.csv' files found in {folder_selected}.")
            self.progress_bar.grid_remove()
            self.status_label.config(text="Status: Ready")
            return
        
        if not self.video_files:
            messagebox.showwarning("No Videos", f"No matching original video files found in {folder_selected}. Cannot export videos.")
            self.progress_bar.grid_remove()
            self.status_label.config(text="Status: Ready")
            return


        processed_csv_count = 0
        total_csvs = len(csv_files_found)
        errors_found = False


        for csv_path in csv_files_found:
            processed_csv_count += 1
            mouse_name = csv_path.stem.replace('immobility_', '')
            self.status_label.config(text=f"Status: Parsing CSV ({processed_csv_count}/{total_csvs}) for {mouse_name}...")
            self.progress_bar.config(value=(processed_csv_count / total_csvs) * 100)
            self.master.update_idletasks()

            try:
                df_csv = pd.read_csv(csv_path)
                
                frame_events = []
                if 'Image index' in df_csv.columns and 'Behavior type' in df_csv.columns:
                    immobility_events_df = df_csv[df_csv['Behavior'] == 'immobility'].copy()
                    
                    if immobility_events_df.empty:
                        print(f"No immobility events found in CSV: {csv_path}")
                        continue
                    
                    immobility_events_df['Image index'] = immobility_events_df['Image index'].astype(int)

                    temp_frame_events_set = set()
                    
                    immobility_events_df_sorted = immobility_events_df.sort_values(by=['Image index', 'Behavior type'], ascending=[True, True])
                    
                    active_bouts = {}
                    
                    for index, row in immobility_events_df_sorted.iterrows():
                        img_idx = row['Image index']
                        event_type = row['Behavior type']
                        
                        if event_type == 'START':
                            active_bouts[img_idx] = True
                        elif event_type == 'STOP':
                            matching_start = None
                            for start_i in sorted(active_bouts.keys()):
                                if active_bouts[start_i]:
                                    matching_start = start_i
                                    break
                            
                            if matching_start is not None:
                                for frame_num in range(matching_start, img_idx + 1):
                                    temp_frame_events_set.add(frame_num)
                                del active_bouts[matching_start]
                            else:
                                print(f"Warning: STOP event at {img_idx} without a matching START in {csv_path}. Skipping this event.")
                                errors_found = True
                    
                    frame_events = sorted(list(temp_frame_events_set))

                    if frame_events:
                        max_frame = max(frame_events) if frame_events else 0
                        persistent_immobility_arr = np.zeros(max_frame + 1, dtype=int)
                        for f_idx in frame_events:
                            if f_idx <= max_frame:
                                persistent_immobility_arr[f_idx] = 1
                        
                        self.analysis_results_cache[mouse_name] = {
                            'frame_events': frame_events,
                            'persistent_immobility': persistent_immobility_arr,
                            'seconds_immobility': len(frame_events) / self.framerate
                        }
                    else:
                        print(f"No valid immobility frames reconstructed from {csv_path}. Skipping.")
                else:
                    print(f"Warning: Missing 'Image index' or 'Behavior type' in {csv_path}. Skipping.")
                    errors_found = True

            except Exception as e:
                print(f"Error reading or parsing {csv_path}: {e}")
                errors_found = True

        self.progress_bar.grid_remove()
        self.progress_bar.config(value=0)
        
        if self.analysis_results_cache:
            messagebox.showinfo("CSV Load Complete", "immobility CSVs loaded successfully. You can now 'Export Marked Videos'.")
            self.export_video_button.config(state=tk.NORMAL)
            self.export_results_by_categories_button.config(state=tk.DISABLED)
        else:
            messagebox.showwarning("No Data Loaded", "No valid immobility data could be loaded from CSVs.")
            self.export_video_button.config(state=tk.DISABLED)
            self.export_results_by_categories_button.config(state=tk.DISABLED)

        self.status_label.config(text="Status: Ready (CSVs loaded)")
        if errors_found:
            messagebox.showwarning("Load with Errors", "Some CSV files could not be processed correctly. Check console for details.")
