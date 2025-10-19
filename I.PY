import cv2
import numpy as np
import os
import json 

# --- Configuration & File Paths ---
WINDOW_NAME = 'Hawk-Eye 3D Pitch Mapper'
PITCH_POINTS_FILE = 'pitch_points_2D.json' # Saved 2D coordinates
POSE_FILE = 'camera_pose.npy' # Saved rvec and tvec

# --- 3D World Model (Real-world measurements in meters) ---
# These 3D points (X, Y, Z) correspond EXACTLY to the 2D points A, B, C, D in PITCH_POINTS
# X=side-to-side, Y=down-pitch, Z=vertical
OBJ_POINTS_3D = np.array([
    [-0.22, 0.00, 0.00],  # A (Off Stump Base) - Used for alignment
    [ 0.22, 0.00, 0.00],  # B (Leg Stump Base) - Used for alignment
    [ 0.60, 20.12, 0.00], # C (Bowler End Right, far down pitch) - Used for alignment
    [-0.60, 20.12, 0.00]  # D (Bowler End Left, far down pitch) - Used for alignment
], dtype=np.float32)

# --- Camera Intrinsics (Estimated based on 1920x1080, adjust if needed) ---
# Assuming a full HD video resolution
W, H = 1920, 1080 
focal_length = 2000  # Initial estimate for a wide field of view

CAMERA_MATRIX = np.array([
    [focal_length, 0, W / 2],
    [0, focal_length, H / 2],
    [0, 0, 1]
], dtype=np.float32)
DIST_COEFFS = np.zeros((4, 1), dtype=np.float32) # Assuming zero distortion

# --- Tracking Configuration (Adjust these for ball color!) ---
# Range for a bright white ball in HSV: [H_low, S_low, V_low], [H_high, S_high, V_high]
LOWER_COLOR = np.array([0, 0, 180])   
UPPER_COLOR = np.array([179, 30, 255]) 
MIN_BALL_RADIUS = 5 # Minimum pixel radius to be considered a ball

# --- Drawing Constants ---
POINT_SIZE = 10
POINT_COLOR = (0, 255, 255)  # Yellow
LINE_COLOR = (255, 150, 0)   # Blue/Teal (Outline)
FILL_COLOR = (255, 100, 0)   # Blue/Teal (Fill)
FILL_OPACITY = 0.4           # 40% transparency
LINE_THICKNESS = 3

# --- Global State Variables ---
video_path = ''
cap = None
is_playing = False
is_layer_active = False # For 2D/3D Pitch Layer display
is_tracking_display = False # For Ball Trajectory display
is_dragging = False
active_point_index = -1
current_frame = None

PITCH_POINTS = [] # Will be initialized or loaded
BALL_TRAJECTORY = [] # Stores 2D (x, y) coordinates of the ball

# Camera Pose Variables (The core of the "tracking")
rvec = None
tvec = None

# --- Helper Functions (Save/Load/Init) ---

def save_points():
    """Saves the current 2D pitch points to a JSON file."""
    try:
        with open(PITCH_POINTS_FILE, 'w') as f:
            json.dump(PITCH_POINTS, f)
        print(f"\n[SAVE] 2D coordinates saved to {PITCH_POINTS_FILE}")
    except Exception as e:
        print(f"\n[ERROR] Could not save points: {e}")

def load_points(width, height):
    """Loads 2D pitch points from a JSON file if it exists."""
    global PITCH_POINTS
    if os.path.exists(PITCH_POINTS_FILE):
        try:
            with open(PITCH_POINTS_FILE, 'r') as f:
                loaded_points = json.load(f)
                if isinstance(loaded_points, list) and len(loaded_points) == 4:
                    PITCH_POINTS = loaded_points
                    print(f"\n[LOAD] Successfully loaded 2D coordinates from {PITCH_POINTS_FILE}")
                    return True
        except Exception as e:
            print(f"\n[ERROR] Could not load points: {e}")
    # Default initialization
    initialize_pitch_points(width, height)
    return False

def initialize_pitch_points(width, height):
    """Initializes the pitch points to a sensible size in the center."""
    global PITCH_POINTS
    center_x, center_y = width // 2, height // 2
    w = width * 0.4
    h = height * 0.4
    
    PITCH_POINTS = [
        [int(center_x - w / 4), int(center_y - h / 2)],
        [int(center_x + w / 4), int(center_y - h / 2)],
        [int(center_x + w / 2), int(center_y + h / 2)],
        [int(center_x - w / 2), int(center_y + h / 2)]
    ]

# --- BALL TRACKING CORE FUNCTION ---

def detect_ball(frame):
    """
    Detects the center of the largest object matching the color range (the ball).
    Returns the center coordinates (x, y) or None if no ball is found.
    """
    if frame is None:
        return None
        
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, LOWER_COLOR, UPPER_COLOR)
    
    # Clean up noise
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)

    # Find contours (blobs)
    contours, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) > 0:
        largest_contour = max(contours, key=cv2.contourArea)
        ((x, y), radius) = cv2.minEnclosingCircle(largest_contour)
        
        # Check if the blob is large enough to be a ball
        if radius > MIN_BALL_RADIUS:
            # Return the center coordinates
            return (int(x), int(y), int(radius))
            
    return None

# --- HAWK-EYE 3D PROJECTION CORE FUNCTIONS ---

def calculate_camera_pose(img_points_2D):
    """
    Uses solvePnP (Perspective-n-Point) to calculate the camera's rotation 
    (rvec) and translation (tvec) relative to the 3D world points.
    """
    global rvec, tvec
    
    img_points = np.array(img_points_2D, dtype=np.float32)
    
    if len(img_points) < 4:
        print("[ERROR] Need at least 4 points for solvePnP.")
        return False

    success, r, t = cv2.solvePnP(OBJ_POINTS_3D, img_points, CAMERA_MATRIX, DIST_COEFFS, flags=cv2.SOLVEPNP_ITERATIVE)
    
    if success:
        rvec, tvec = r, t
        np.save(f'{POSE_FILE}_rvec', rvec)
        np.save(f'{POSE_FILE}_tvec', tvec)
        print("\n[SUCCESS] Camera Pose calculated (rvec/tvec saved). Projection is ready!")
        return True
    else:
        print("\n[FAILURE] solvePnP failed. Check your 2D alignments carefully.")
        rvec, tvec = None, None
        return False

def draw_hawk_eye_projection(frame, r_vec, t_vec, color, thickness):
    """
    Projects a 3D line (the stump line) onto the 2D frame using the calculated pose.
    """
    if r_vec is None or t_vec is None:
        return frame
    
    # 1. Define the line in 3D space (from middle stump to far bowler end)
    pitch_line_3D = np.array([
        [0.0, y, 0.0] for y in np.linspace(0.0, 20.0, 100) # 100 points from 0m to 20m
    ], dtype=np.float32)
    
    # 2. Project the 3D points to 2D screen coordinates
    projected_points, _ = cv2.projectPoints(pitch_line_3D, r_vec, t_vec, CAMERA_MATRIX, DIST_COEFFS)
    
    # Reshape the projected points for cv2.polylines
    points_2D = np.int32(projected_points).reshape(-1, 2)
    
    # 3. Draw the line on the frame
    cv2.polylines(frame, [points_2D], isClosed=False, color=color, thickness=thickness)
    
    return frame

# --- Drawing and UI Functions ---

def draw_overlay(frame):
    """Draws all elements: Pitch, Controls, Ball Path."""
    
    # 1. Draw 3D Pitch Projection (Red Line) if calculated and layer is active
    if rvec is not None and tvec is not None and is_layer_active:
        frame = draw_hawk_eye_projection(frame, rvec, tvec, (0, 0, 255), LINE_THICKNESS + 1) # Red line
        
        # Draw a small 2D trapezoid outline for reference
        points_np = np.array(PITCH_POINTS, np.int32).reshape((-1, 1, 2))
        cv2.polylines(frame, [points_np], isClosed=True, color=(255, 255, 255), thickness=1)
        
    # 2. Draw 2D Alignment Guides (Blue Fill) if only aligning
    elif is_layer_active and frame is not None:
        points_np = np.array(PITCH_POINTS, np.int32).reshape((-1, 1, 2))
        mask = np.zeros_like(frame, dtype=np.uint8)
        cv2.fillPoly(mask, [points_np], FILL_COLOR)
        frame = cv2.addWeighted(frame, 1.0, mask, FILL_OPACITY, 0) 
        cv2.polylines(frame, [points_np], isClosed=True, color=LINE_COLOR, thickness=LINE_THICKNESS) 
    
    # 3. Draw control points (always visible when layer is active)
    if is_layer_active:
        for i, (x, y) in enumerate(PITCH_POINTS):
            color = (0, 0, 255) if i == active_point_index else POINT_COLOR
            cv2.circle(frame, (x, y), POINT_SIZE, color, -1)
            label = chr(ord('A') + i)
            cv2.putText(frame, label, (x + POINT_SIZE + 2, y + 5), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # 4. Draw Ball Trajectory if active
    if is_tracking_display and len(BALL_TRAJECTORY) > 1:
        # Draw the path using polylines
        traj_np = np.array([[x, y] for x, y, _ in BALL_TRAJECTORY], np.int32).reshape((-1, 1, 2))
        cv2.polylines(frame, [traj_np], isClosed=False, color=(0, 255, 0), thickness=2) # Green Path
        
        # Draw a circle on the most recent ball position
        last_x, last_y, last_r = BALL_TRAJECTORY[-1]
        cv2.circle(frame, (last_x, last_y), last_r, (0, 255, 0), 2) # Green circle on current ball
        cv2.circle(frame, (last_x, last_y), 3, (0, 0, 255), -1) # Red center dot

    return frame


def mouse_callback(event, x, y, flags, param):
    """Handles mouse clicks and dragging for point manipulation."""
    global is_dragging, active_point_index
    
    if not is_layer_active:
        return

    if event == cv2.EVENT_LBUTTONDOWN:
        is_dragging = True
        active_point_index = -1
        for i, (px, py) in enumerate(PITCH_POINTS):
            distance = np.sqrt((x - px)**2 + (y - py)**2)
            if distance < POINT_SIZE * 3: 
                active_point_index = i
                break
        
    elif event == cv2.EVENT_LBUTTONUP:
        is_dragging = False
        active_point_index = -1 

    elif event == cv2.EVENT_MOUSEMOVE:
        if is_dragging and active_point_index != -1:
            PITCH_POINTS[active_point_index][0] = x
            PITCH_POINTS[active_point_index][1] = y


def display_controls(frame):
    """Adds persistent instructions and status to the video window."""
    h, w = frame.shape[:2]
    
    # Background box for text
    cv2.rectangle(frame, (5, 5), (w - 5, 120), (30, 30, 30), -1)
    
    # Control instructions
    cv2.putText(frame, 'CONTROLS:', (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    controls_text_1 = 'P: Play/Pause | L: Toggle Pitch Layer (2D/3D) | S: Save 2D | R: CALIBRATE (3D Pose)'
    cv2.putText(frame, controls_text_1, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 255, 150), 1)
    
    controls_text_2 = 'C: Clear Trajectory | T: Toggle Trajectory Display | ESC: Exit'
    cv2.putText(frame, controls_text_2, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 255, 150), 1)
    
    # Status
    status = "PLAYING" if is_playing else "PAUSED"
    
    if rvec is not None:
        layer_status = "3D PROJECTION ACTIVE (RED)"
    elif is_layer_active:
        layer_status = "2D ALIGNMENT MODE (BLUE)"
    else:
        layer_status = "INACTIVE"
    
    tracking_status = f"TRAJECTORY: {'ON' if is_tracking_display else 'OFF'} | Points: {len(BALL_TRAJECTORY)}"
        
    cv2.putText(frame, f'STATUS: {status} | LAYER: {layer_status}', (10, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 150), 1)
    cv2.putText(frame, tracking_status, (10, 115), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 150, 255), 1)
    
    
def main():
    """Main application loop."""
    global video_path, cap, is_playing, is_layer_active, is_tracking_display, current_frame, rvec, tvec, BALL_TRAJECTORY

    # 1. Get video file path from user
    while not video_path:
        video_path = input("Enter the path to your video file (e.g., cricket.mp4): ")
        if not os.path.exists(video_path):
            print(f"Error: File not found at '{video_path}'. Try again.")
            video_path = ''
        
    # 2. Initialize video capture
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    # 3. Setup window and mouse callback
    cv2.namedWindow(WINDOW_NAME)
    cv2.setMouseCallback(WINDOW_NAME, mouse_callback)

    # Initialize state variables
    fps = cap.get(cv2.CAP_PROP_FPS)
    wait_time = int(1000 / fps) if fps > 0 else 30 
    is_playing = True
    
    # Read the first frame to get dimensions for initialization
    ret, frame = cap.read()
    if ret:
        load_points(frame.shape[1], frame.shape[0])
        # Try to load existing 3D pose data if it exists
        try:
            if os.path.exists(f'{POSE_FILE}_rvec.npy'):
                rvec = np.load(f'{POSE_FILE}_rvec.npy')
                tvec = np.load(f'{POSE_FILE}_tvec.npy')
                print("[LOAD] Found and loaded previous Camera Pose.")
        except Exception:
            pass
    
    print("\n--- PROFESSIONAL HAWK-EYE MAPPING CONTROLS ---")
    print("1. Align points (L key) & CALIBRATE (R key) first.")
    print("2. Press 'P' to play and record the ball's trajectory.")
    print("3. Press 'T' to display the recorded path.")
    print("Press 'ESC' to close.")
    
    # 4. Main Loop
    while cap.isOpened():
        key = cv2.waitKey(wait_time) & 0xFF
        
        # --- Handle Key Presses ---
        if key == 27: # ESC key to exit
            break
        elif key == ord('p') or key == ord('P'):
            is_playing = not is_playing
        elif key == ord('l') or key == ord('L'):
            is_layer_active = not is_layer_active
            if is_layer_active:
                is_playing = False
                load_points(current_frame.shape[1], current_frame.shape[0])
        elif key == ord('s') or key == ord('S'):
            save_points()
        elif key == ord('r') or key == ord('R'):
            calculate_camera_pose(PITCH_POINTS)
        elif key == ord('c') or key == ord('C'):
            # Clear the recorded trajectory
            BALL_TRAJECTORY = []
            print("[INFO] Ball trajectory cleared.")
        elif key == ord('t') or key == ord('T'):
            is_tracking_display = not is_tracking_display
            
        # --- Read Frame ---
        if is_playing:
            ret, frame = cap.read()
            if not ret:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0) # Loop video
                BALL_TRAJECTORY = [] # Clear trajectory on loop start
                ret, frame = cap.read()
                if not ret: break
            current_frame = frame
            
            # 5. Continuous Ball Detection while playing
            ball_result = detect_ball(current_frame)
            if ball_result:
                # Store (x, y, radius) of the detected ball
                BALL_TRAJECTORY.append(ball_result)
        
        if current_frame is not None:
            # 6. Draw all elements
            display_frame = current_frame.copy()
            display_frame = draw_overlay(display_frame)
            display_controls(display_frame) 
            
            cv2.imshow(WINDOW_NAME, display_frame)

    # --- Cleanup and Output ---
    cap.release()
    cv2.destroyAllWindows()
    
    print("\n--- CALIBRATION RESULTS ---")
    if rvec is not None:
        print("3D Camera Pose SUCCESSFULLY CALCULATED.")
        print(f"Recorded Trajectory Points: {len(BALL_TRAJECTORY)}")
    else:
        print("Calibration not complete. Run the 'R' key step next time.")
    
if __name__ == '__main__':
    main()
