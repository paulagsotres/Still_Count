# -*- coding: utf-8 -*-
"""
Created on Sat Sep 20 18:41:05 2025

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


from still_count import immobilityAnalyzerGUI, take_all_files, run_background_subtraction_for_analysis, detect_immobility, calculate_immobility_by_bin_core, create_immobility_mark_video,create_csv_immobility


if __name__ == "__main__":
    root = tk.Tk()
    app = immobilityAnalyzerGUI(root)
    app.master.protocol("WM_DELETE_WINDOW", app.master.destroy)
    root.mainloop()
