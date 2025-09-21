# -*- coding: utf-8 -*-
"""
Created on Sat Sep 20 18:35:36 2025

@author: paula gomez sotres
"""
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from collections import deque
import os


# --- Core immobility Detection Functions ---

import cv2
from pathlib import Path

def take_all_files(dir_path):
    """
    Returns:
        dictionary_all_files: dict of key -> video path
        fps_of_sample_video: float (FPS of one selected video)
    Raises:
        FileNotFoundError: if no video files are found in the folder
    """
    dictionary_all_files = {}
    fps = None

    dir_path = Path(dir_path)
    if not dir_path.exists():
        raise FileNotFoundError(f"Directory does not exist: {dir_path}")
    
    video_extensions = ['.avi', '.mp4', '.mkv', '.mov', '.wmv']

    # Collect all video files
    video_files = []
    for ext in video_extensions:
        video_files.extend(dir_path.glob(f"*{ext}"))

    if not video_files:
        raise FileNotFoundError(f"No video files found in directory: {dir_path}")

    # Populate dictionary
    for file_path in video_files:
        key_name = file_path.stem.split('-')[0]
        dictionary_all_files[key_name] = str(file_path)

    # Get FPS from the first video file
    cap = cv2.VideoCapture(str(video_files[0]))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()

    return dictionary_all_files, fps



def run_background_subtraction_for_analysis(video_path, roi_x, roi_y, roi_width, roi_height, video_threshold, frame_interval,
                                            progress_callback=None, frame_display_callback=None, stop_event=None):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path} for background subtraction.")
    
    fps = cap.get(cv2.CAP_PROP_FPS)

    frame_buffer = deque(maxlen=frame_interval)
    binary_areas = []

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    current_frame_idx = 0

    live_preview_target_fps = 2
    display_interval = max(1, int(fps/ live_preview_target_fps))

    while True:
        ret, current_frame = cap.read()
        if not ret:
            remaining_frames = total_frames - len(binary_areas)
            binary_areas.extend([0] * remaining_frames)
            break

        h, w, _ = current_frame.shape
        actual_roi_x = max(0, min(roi_x, w))
        actual_roi_y = max(0, min(roi_y, h))
        actual_roi_width = min(roi_width, w - actual_roi_x)
        actual_roi_height = min(roi_height, h - actual_roi_y)

        binary_diff = None
        if actual_roi_width <= 0 or actual_roi_height <= 0:
            binary_areas.append(0)
            frame_buffer.append(np.zeros((10,10), dtype=np.uint8))
        else:
            roi_frame = current_frame[actual_roi_y:actual_roi_y + actual_roi_height,
                                      actual_roi_x:actual_roi_x + actual_roi_width]
            roi_frame_gray = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2GRAY)
            frame_buffer.append(roi_frame_gray)

        if len(frame_buffer) == frame_interval:
            frame_diff = cv2.absdiff(frame_buffer[0], frame_buffer[-1])
            
             # Threshold the difference to get the binary image
            _, binary_diff = cv2.threshold(frame_diff, video_threshold, 255, cv2.THRESH_BINARY)
        

            binary_area_size = np.sum(binary_diff == 255)
            
            binary_areas.append(binary_area_size)
            
        
            
        else:
            binary_areas.append(0)

        if frame_display_callback and current_frame_idx % display_interval == 0:
            display_frame = current_frame.copy()
            cv2.rectangle(display_frame, (actual_roi_x, actual_roi_y),(actual_roi_x + actual_roi_width, actual_roi_y + actual_roi_height), (0, 255, 0), 2)
            
            if len(frame_buffer) == frame_interval and binary_diff is not None:
                binary_diff_bgr = cv2.cvtColor(binary_diff, cv2.COLOR_GRAY2BGR)
                alpha = 0.5
                display_frame[actual_roi_y:actual_roi_y + actual_roi_height,
                              actual_roi_x:actual_roi_x + actual_roi_width] = cv2.addWeighted(
                                  display_frame[actual_roi_y:actual_roi_y + actual_roi_height,
                                                actual_roi_x:actual_roi_x + actual_roi_width],
                                  1 - alpha,
                                  binary_diff_bgr,
                                  alpha,
                                  0)

            frame_display_callback(display_frame)

        if progress_callback:
            progress_callback(current_frame_idx + 1, total_frames)

        current_frame_idx += 1

    cap.release()
    cv2.destroyAllWindows()

    

    return fps, pd.Series(binary_areas)

def detect_immobility(frames_moving, immobility_threshold, window_size, frame_rate):
    if frames_moving.empty:
        return np.array([]), [], 0.0

    is_instantly_still = (frames_moving < immobility_threshold).astype(int)
    is_persistently_still = is_instantly_still.rolling(window=window_size, min_periods=window_size).sum() == window_size
    is_persistently_still = is_persistently_still.shift(-(window_size - 1)).fillna(False)
    persistent_immobility_bool = is_persistently_still.astype(bool)
    frame_events = np.where(persistent_immobility_bool.values)[0]
    
    seconds_immobility = len(frame_events) / frame_rate if frame_rate != 0 else 0.0
    
    return persistent_immobility_bool.values, frame_events, seconds_immobility


def calculate_immobility_by_bin_core(persistent_immobility, num_bins, framerate, time_adjustment):
    if len(persistent_immobility) == 0:
        return pd.DataFrame()

    immobility_series = pd.Series(persistent_immobility.flatten())
    
    offset_frames = int(time_adjustment * framerate) if framerate != 0 else 0
    effective_length = len(immobility_series) - offset_frames
    
    if effective_length <= 0:
        return pd.DataFrame([np.zeros(num_bins)], columns=[f'Bin_{i+1}' for i in range(num_bins)])

    bin_size = effective_length // num_bins
    true_counts = []
    
    for i in range(num_bins):
        start_index = i * bin_size + offset_frames
        end_index = start_index + bin_size
        
        start_index = max(0, min(int(start_index), len(immobility_series)))
        end_index = max(0, min(int(end_index), len(immobility_series)))
        
        if start_index >= end_index:
            true_counts.append(0)
        else:
            bin_slice = immobility_series.iloc[start_index:end_index]
            true_count = np.sum(bin_slice.values)
            true_counts.append(true_count)
    
    if effective_length % num_bins != 0:
        remainder_start_index = num_bins * bin_size + offset_frames
        remainder_start_index = max(0, min(int(remainder_start_index), len(immobility_series)))
        
        if remainder_start_index < len(immobility_series):
            remainder_slice = immobility_series.iloc[remainder_start_index:]
            true_count_remainder = np.sum(remainder_slice.values)
            true_counts.append(true_count_remainder)

    true_counts_seconds = np.array(true_counts) / framerate if framerate != 0 else np.array(true_counts) * 0.0
    
    column_names = [f'Bin_{i+1}' for i in range(len(true_counts))]
    df_bins = pd.DataFrame([true_counts_seconds], columns=column_names)
    return df_bins


def create_immobility_mark_video(input_video_path, output_video_path, frame_events, frame_progress_callback=None):
    cap = cv2.VideoCapture(input_video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not cap.isOpened():
        print(f"Error: Could not open input video {input_video_path}")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    new_width = width // 3
    new_height = height // 3
    
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (new_width, new_height))
    if not out.isOpened():
        print(f"Error: Could not open output video {output_video_path}")
        cap.release()
        return

    frame_indices_to_plot = set(frame_events)
    frame_number = 0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
    
        frame_resized = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_AREA)
        
        if frame_number in frame_indices_to_plot:
            cv2.circle(frame_resized, (20, 20), 5, (0, 0, 255), -1)
    
        out.write(frame_resized)

        # Call the progress callback if provided
        if frame_progress_callback is not None:
            frame_progress_callback(frame_number, total_frames)
        
        frame_number += 1

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"Marked video created at: {output_video_path}")

    
def create_csv_immobility(persistent_immobility, mouse, framerate, dir_path):
    if len(persistent_immobility) == 0:
        print(f"No immobility detected for {mouse}, skipping CSV creation.")
        return

    if persistent_immobility.ndim > 1:
        persistent_immobility = persistent_immobility.flatten()
    persistent_immobility = persistent_immobility.astype(int)

    behav_indices = np.where(persistent_immobility > 0)[0]

    if len(behav_indices) == 0:
        print(f"No immobility frames found for {mouse}.")
        return

    behavioral_events = []
    if len(behav_indices) > 0:
        start = behav_indices[0]
        for i in range(len(behav_indices) - 1):
            if behav_indices[i + 1] != behav_indices[i] + 1:
                behavioral_events.append((start, behav_indices[i]))
                start = behav_indices[i + 1]
        behavioral_events.append((start, behav_indices[-1]))

    if not behavioral_events:
        print(f"Could not form behavioral events for {mouse}.")
        return

    rows_list = []
    for start_idx, stop_idx in behavioral_events:
        rows_list.append({'Behavior': 'immobility', 'Behavior type': 'START', 'Time': start_idx / framerate if framerate !=0 else 0.0, 'Image index': start_idx})
        rows_list.append({'Behavior': 'immobility', 'Behavior type': 'STOP', 'Time': stop_idx / framerate if framerate !=0 else 0.0, 'Image index': stop_idx})
    
    immobility_df = pd.DataFrame(rows_list)
    immobility_df['Image index'] = immobility_df['Image index'].astype(int)


    csv_output_path = os.path.join(dir_path, f'immobility_{mouse}.csv')
    immobility_df.to_csv(csv_output_path, index=False)
    print(f"immobility CSV created at: {csv_output_path}")
