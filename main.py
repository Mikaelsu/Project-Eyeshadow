import cv2
import numpy as np
import pandas as pd

# CONFIGURATION
csv_path = '/Users/msukoine/Desktop/Varjo Eye/varjo_gaze_output_2025-06-02_10-27-05-285.csv'
video_path = '/Users/msukoine/Desktop/Varjo Eye/varjo_capture_2025-06-02_10-27-05-285.mp4'
output_path = '/Users/msukoine/Desktop/Varjo Eye/varjo_capture_2025-06-02_10-27-05-285.mp4output_with_heatmap.mp4'

time_col = 'relative_to_video_first_frame_timestamp'
x_col = 'gaze_projected_to_left_view_x'
y_col = 'gaze_projected_to_left_view_y'

# Loading gaze data
df = pd.read_csv(csv_path)

# Converting timestamp into seconds
if df[time_col].max() > 1e6:
    df[time_col] = df[time_col] / 1_000_000_000

df[time_col] -= df[time_col].min()

# Cleaning and validating data
df[x_col] = pd.to_numeric(df[x_col], errors='coerce')
df[y_col] = pd.to_numeric(df[y_col], errors='coerce')
df = df[(df[x_col].between(0, 1)) & (df[y_col].between(0, 1))].dropna()

# Setting up video
cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

# HEATMAP SETTINGS
radius = 15
intensity = 0.3
time_window = 0.2  # seconds
global_max = 1e-6
last_valid_heatmap = None

frame_idx = 0

print("🎥 Generating heatmap overlay video...")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_time = frame_idx / fps
    gaze_window = df[
        (df[time_col] >= frame_time - time_window / 2) &
        (df[time_col] <= frame_time + time_window / 2)
    ]

    if not gaze_window.empty:
        heatmap = np.zeros((frame_height, frame_width), dtype=np.float32)

        for _, row in gaze_window.iterrows():
            x = int(row[x_col] * frame_width)
            y = int(row[y_col] * frame_height)
            if 0 <= x < frame_width and 0 <= y < frame_height:
                cv2.circle(heatmap, (x, y), radius, intensity, -1)

        # Applying proportional Gaussian blur
        ksize = int(max(frame_width, frame_height) * 0.03)
        ksize = ksize if ksize % 2 == 1 else ksize + 1
        heatmap = cv2.GaussianBlur(heatmap, (ksize, ksize), sigmaX=0, sigmaY=0)

        # Normalising heatmap for consistent scaling
        global_max = max(global_max, heatmap.max())
        heatmap /= global_max
        last_valid_heatmap = heatmap.copy()
    else:
        heatmap = last_valid_heatmap.copy() if last_valid_heatmap is not None else None

    # Overlaying heatmap on input video
    if heatmap is not None:
        heatmap_norm = cv2.normalize(heatmap, None, 0, 255, cv2.NORM_MINMAX)
        heatmap_color = cv2.applyColorMap(heatmap_norm.astype(np.uint8), cv2.COLORMAP_TURBO)
        overlayed = cv2.addWeighted(frame, 0.7, heatmap_color, 0.5, 0)
    else:
        overlayed = frame

    out.write(overlayed)
    frame_idx += 1

cap.release()
out.release()
print(f"🎬 ...and cut! Heatmap overlay video saved in: {output_path}")
