import streamlit as st
import tempfile
import os
import cv2
import numpy as np
import pandas as pd
import mediapipe as mp
import math
import base64
import streamlit.components.v1 as components
import imageio
from typing import List, Dict, Tuple, Optional, Any
import json
from dataclasses import dataclass
from datetime import datetime
import time
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ——— Constants & Configuration ———
DETECTION_CONFIDENCE = 0.5
TRACKING_CONFIDENCE = 0.5
FPS_DEFAULT = 30.0
SPLASH_THRESHOLD_FACTOR = 1.0  # Multiplier for standard deviation
CACHE_DIR = ".dive_cache"  # Cache directory for processed videos

# Create cache directory if it doesn't exist
os.makedirs(CACHE_DIR, exist_ok=True)

# ——— Data Models ———
@dataclass
class DiveMetrics:
    """Data class for dive metrics"""
    entry_angle: float
    straightness: float
    splash_area: int
    peak_speed: float
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'entry_angle': self.entry_angle,
            'straightness': self.straightness,
            'splash_area': self.splash_area,
            'peak_speed': self.peak_speed
        }

@dataclass
class DiveResult:
    """Data class for a complete dive result"""
    dive_number: int
    takeoff_frame: int
    entry_frame: int
    metrics: DiveMetrics
    score: float
    annotated_path: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "Dive": f"Dive {self.dive_number}",
            "Takeoff": self.takeoff_frame,
            "Entry": self.entry_frame,
            "Angle (°)": round(self.metrics.entry_angle, 2),
            "Straight": round(self.metrics.straightness, 3),
            "Splash px": self.metrics.splash_area,
            "Speed": self.metrics.peak_speed,
            "Score": round(self.score, 2)
        }

# ——— MediaPipe Setup ———
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# ——— Helper Functions ———
def get_video_info(video_path: str) -> Dict[str, Any]:
    """Extract video metadata"""
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or FPS_DEFAULT
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps
    cap.release()
    
    return {
        "fps": fps,
        "width": width,
        "height": height,
        "total_frames": total_frames,
        "duration": duration
    }

def extract_frames(video_path: str, start_frame: int = 0, end_frame: Optional[int] = None) -> List[np.ndarray]:
    """Extract frames from video with optional start/end frames"""
    cap = cv2.VideoCapture(video_path)
    frames = []
    
    # Set starting position
    if start_frame > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret or (end_frame is not None and frame_count >= (end_frame - start_frame)):
            break
        frames.append(frame)
        frame_count += 1
    
    cap.release()
    return frames

def get_landmarks(frames: List[np.ndarray]) -> List[Optional[Dict[str, Tuple[float, float]]]]:
    """Process frames to extract pose landmarks"""
    with mp_pose.Pose(
        min_detection_confidence=DETECTION_CONFIDENCE,
        min_tracking_confidence=TRACKING_CONFIDENCE) as pose:
        
        all_lm = []
        for f in frames:
            rgb = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
            res = pose.process(rgb).pose_landmarks
            if not res:
                all_lm.append(None)
            else:
                pts = {mp_pose.PoseLandmark(i).name: (lm.x, lm.y)
                       for i, lm in enumerate(res.landmark)}
                all_lm.append(pts)
        
        return all_lm

def detect_takeoff_and_entry(frames: List[np.ndarray], landmarks: List[Optional[Dict]]) -> Tuple[Optional[int], Optional[int]]:
    """Detect takeoff and entry frames based on vertical velocity and frame differences"""
    if len(frames) < 2:
        return None, None

    # 1) Hip vertical velocity for takeoff detection
    y_positions = []
    for pts in landmarks:
        if pts and 'LEFT_HIP' in pts and 'RIGHT_HIP' in pts:
            y_positions.append((pts['LEFT_HIP'][1] + pts['RIGHT_HIP'][1]) / 2)
        else:
            y_positions.append(None)
    
    velocities = []
    for i in range(1, len(y_positions)):
        if y_positions[i] is not None and y_positions[i-1] is not None:
            velocities.append(y_positions[i-1] - y_positions[i])
        else:
            velocities.append(0)
    
    if not velocities:
        return None, None
        
    takeoff = int(np.argmax(velocities))

    # 2) Splash detection via frame difference
    diffs = []
    for i in range(1, len(frames)):
        g1 = cv2.cvtColor(frames[i-1], cv2.COLOR_BGR2GRAY)
        g2 = cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY)
        diffs.append(np.sum(cv2.absdiff(g2, g1)))
    
    if not diffs:
        return takeoff, None
        
    entry = int(np.argmax(diffs))

    return takeoff, entry

def find_valid_frame(landmarks: List[Optional[Dict]], idx: Optional[int]) -> Optional[int]:
    """Find nearest valid frame with landmarks if the specified frame is invalid"""
    n = len(landmarks)
    if idx is None or idx < 0 or idx >= n:
        return None
    
    if landmarks[idx] is not None:
        return idx
    
    for offset in range(1, n):
        prev, next_idx = idx - offset, idx + offset
        if prev >= 0 and landmarks[prev] is not None:
            return prev
        if next_idx < n and landmarks[next_idx] is not None:
            return next_idx
    
    return None

def compute_metrics(frames: List[np.ndarray], landmarks: List[Optional[Dict]], 
                   takeoff_idx: Optional[int], entry_idx: Optional[int], 
                   fps: float) -> Optional[DiveMetrics]:
    """Compute dive metrics based on landmark data"""
    if takeoff_idx is None or entry_idx is None:
        return None
    
    if entry_idx >= len(landmarks) or landmarks[entry_idx] is None:
        return None
    
    # Entry angle calculation
    entry_pose = landmarks[entry_idx]
    shoulder = (np.array(entry_pose['LEFT_SHOULDER']) + np.array(entry_pose['RIGHT_SHOULDER'])) / 2
    hip = (np.array(entry_pose['LEFT_HIP']) + np.array(entry_pose['RIGHT_HIP'])) / 2
    vec = hip - shoulder
    
    # Calculate angle between body vector and vertical axis
    entry_angle = math.degrees(
        math.acos(np.dot(vec/np.linalg.norm(vec), [0, 1]))
    )
    
    # Body straightness calculation
    deviations = []
    for frame_landmarks in landmarks[takeoff_idx:entry_idx]:
        if not frame_landmarks:
            continue
            
        ankle = (np.array(frame_landmarks['LEFT_ANKLE']) + np.array(frame_landmarks['RIGHT_ANKLE'])) / 2
        shoulder_pos = (np.array(frame_landmarks['LEFT_SHOULDER']) + np.array(frame_landmarks['RIGHT_SHOULDER'])) / 2
        hip_pos = (np.array(frame_landmarks['LEFT_HIP']) + np.array(frame_landmarks['RIGHT_HIP'])) / 2
        
        # Calculate how far ankle deviates from the shoulder-hip line
        deviation = np.linalg.norm(np.cross(hip_pos - shoulder_pos, shoulder_pos - ankle)) / np.linalg.norm(hip_pos - shoulder_pos)
        deviations.append(deviation)
    
    straightness = max(0, 1 - (np.mean(deviations) if deviations else 0) * 10)
    
    # Splash area calculation
    if entry_idx > 0 and entry_idx < len(frames):
        g1 = cv2.cvtColor(frames[entry_idx-1], cv2.COLOR_BGR2GRAY)
        g2 = cv2.cvtColor(frames[entry_idx], cv2.COLOR_BGR2GRAY)
        diff = cv2.absdiff(g2, g1)
        _, thresh = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)
        splash_area = int(np.sum(thresh > 0))
    else:
        splash_area = 0
    
    # Peak speed calculation
    peak_speed = compute_peak_speed(landmarks, fps)
    
    return DiveMetrics(
        entry_angle=entry_angle,
        straightness=straightness,
        splash_area=splash_area,
        peak_speed=peak_speed
    )

def compute_peak_speed(landmarks: List[Optional[Dict]], fps: float) -> float:
    """Calculate peak vertical speed of the hips"""
    speeds = []
    for i in range(1, len(landmarks)):
        prev, curr = landmarks[i-1], landmarks[i]
        if prev and curr and 'LEFT_HIP' in prev and 'LEFT_HIP' in curr:
            y0 = (prev['LEFT_HIP'][1] + prev['RIGHT_HIP'][1]) / 2
            y1 = (curr['LEFT_HIP'][1] + curr['RIGHT_HIP'][1]) / 2
            speeds.append(abs(y1 - y0) * fps)
    
    return round(max(speeds) if speeds else 0, 1)

def score_dive(metrics: DiveMetrics, weights: Tuple[float, float, float] = (0.4, 0.3, 0.3)) -> float:
    """Calculate overall dive score based on weighted metrics"""
    # Normalize each metric to 0-1 scale
    angle_score = max(0, 1 - metrics.entry_angle / 30)
    straight_score = metrics.straightness
    splash_score = max(0, 1 - metrics.splash_area / 50000)
    
    # Weighted sum scaled to 10
    return 10 * (
        weights[0] * angle_score + 
        weights[1] * straight_score + 
        weights[2] * splash_score
    )

def detect_dive_segments(frames: List[np.ndarray], landmarks: List[Optional[Dict]], 
                        fps: float) -> List[Tuple[int, int]]:
    """Detect multiple dive segments within a video"""
    if not frames:
        return []

    # Track hip positions
    positions = []
    for i, lm in enumerate(landmarks):
        if lm and 'LEFT_HIP' in lm and 'RIGHT_HIP' in lm:
            positions.append((i, (lm['LEFT_HIP'][1] + lm['RIGHT_HIP'][1]) / 2))
        else:
            positions.append(None)

    # Detect dive starts by upward motion
    dive_starts = []
    for i in range(1, len(positions)):
        prev = positions[i-1]
        curr = positions[i]
        if prev and curr and (prev[1] - curr[1] > 0.01):  # Upward movement
            dive_starts.append(curr[0])

    # Calculate frame differences for splash detection
    diffs = []
    for i in range(1, len(frames)):
        g1 = cv2.cvtColor(frames[i-1], cv2.COLOR_BGR2GRAY)
        g2 = cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY)
        diffs.append(np.sum(cv2.absdiff(g2, g1)))
    
    if not diffs:
        return []
        
    # Find splash frames above threshold
    mean_diff, std_diff = np.mean(diffs), np.std(diffs)
    splash_frames = [i+1 for i, d in enumerate(diffs) 
                    if d > mean_diff + SPLASH_THRESHOLD_FACTOR * std_diff]

    # Pair dive starts with splashes
    dives = []
    for splash in splash_frames:
        # Find closest preceding dive start
        starts = [s for s in dive_starts if s < splash]
        if starts:
            start = max(starts)
        else:
            # Default to 2 seconds before splash
            start = max(0, splash - int(fps * 2))
        
        # Add 1 second after splash for complete sequence
        end = min(len(frames) - 1, splash + int(fps * 1))
        dives.append((start, end))

    # Merge overlapping segments
    if not dives:
        return []
        
    dives.sort()
    merged = [dives[0]]
    
    for start, end in dives[1:]:
        prev_start, prev_end = merged[-1]
        if start <= prev_end:
            merged[-1] = (prev_start, max(prev_end, end))
        else:
            merged.append((start, end))
    
    return merged

def export_annotated_video(video_path: str, frames: List[np.ndarray], 
                          takeoff_idx: Optional[int], entry_idx: Optional[int],
                          score: float, dive_number: Optional[int] = None) -> str:
    """Create annotated video with pose landmarks and metrics"""
    # Get video properties
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or FPS_DEFAULT
    cap.release()
    
    # Create annotated frames
    annotated_frames = []
    with mp_pose.Pose(
        min_detection_confidence=DETECTION_CONFIDENCE,
        min_tracking_confidence=TRACKING_CONFIDENCE) as pose:
        
        for i, frame in enumerate(frames):
            annotated = frame.copy()
            rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
            result = pose.process(rgb)
            
            # Draw pose landmarks
            if result.pose_landmarks:
                mp_drawing.draw_landmarks(
                    annotated, 
                    result.pose_landmarks, 
                    mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
                )
            
            # Add dive number
            if dive_number is not None:
                cv2.putText(
                    annotated, 
                    f"Dive {dive_number}", 
                    (annotated.shape[1] // 2 - 50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 0), 3
                )
            
            # Mark takeoff
            if i == takeoff_idx:
                cv2.putText(
                    annotated, 
                    "TAKEOFF", 
                    (15, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2
                )
            
            # Mark entry
            if i == entry_idx:
                cv2.putText(
                    annotated, 
                    "ENTRY", 
                    (15, 95),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2
                )
            
            # Add score
            cv2.putText(
                annotated,
                f"Score: {score:.1f}",
                (annotated.shape[1] - 200, annotated.shape[0] - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2
            )
            
            annotated_frames.append(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB))
    
    # Export video
    output_path = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False).name
    writer = imageio.get_writer(
        output_path, 
        fps=fps,
        codec="libx264",
        ffmpeg_params=["-pix_fmt", "yuv420p"]
    )
    
    for img in annotated_frames:
        writer.append_data(img)
    
    writer.close()
    return output_path

def generate_coach_feedback(dive_result: DiveResult) -> str:
    """Generate personalized feedback based on dive metrics"""
    score = dive_result.score
    angle = dive_result.metrics.entry_angle
    straightness = dive_result.metrics.straightness
    splash = dive_result.metrics.splash_area
    speed = dive_result.metrics.peak_speed
    
    feedback = []
    
    # Overall assessment
    if score >= 8.5:
        feedback.append(f"### {dive_result.dive_number}  —  Overall Score: **{score:.1f}/10** 🏆")
        feedback.append("> **Excellent dive!** This was a high-quality performance with very good technique.")
    elif score >= 7:
        feedback.append(f"### {dive_result.dive_number}  —  Overall Score: **{score:.1f}/10** 👍")
        feedback.append("> **Solid dive!** Good overall execution with some areas for improvement.")
    else:
        feedback.append(f"### {dive_result.dive_number}  —  Overall Score: **{score:.1f}/10** 🔍")
        feedback.append("> **Room for improvement.** Let's work on key areas to enhance your technique.")
    
    # Entry angle feedback
    if angle > 20:
        feedback.append(
            f"> **Entry angle needs work:** Your torso was at **{angle:.1f}°** off vertical. "
            f"Aim for under **10°** for a pencil-straight entry. Try focusing on maintaining "
            f"vertical alignment as you approach the water."
        )
    elif angle > 10:
        feedback.append(
            f"> **Decent entry angle:** Your torso was at **{angle:.1f}°** off vertical. "
            f"Getting closer to that ideal vertical entry! Focus on keeping your head in line "
            f"with your spine at entry."
        )
    else:
        feedback.append(
            f"> **Excellent entry angle!** You achieved **{angle:.1f}°** off vertical - "
            f"nearly perfect alignment at entry. This vertical positioning is exactly what judges look for."
        )
    
    # Straightness feedback
    if straightness < 0.6:
        feedback.append(
            f"> **Body alignment needs work:** Your straightness score was **{straightness:.2f}**. "
            f"Focus on tightening your core and maintaining alignment between shoulders, hips, and ankles "
            f"throughout the dive. Dryland practice with hollow body holds can help."
        )
    elif straightness < 0.8:
        feedback.append(
            f"> **Decent body alignment:** Straightness score **{straightness:.2f}**. "
            f"You're showing good control but could tighten up the line from shoulders through "
            f"toes for a more streamlined position."
        )
    else:
        feedback.append(
            f"> **Excellent body alignment!** Straightness score **{straightness:.2f}** - "
            f"you maintained great form throughout the dive with minimal body deviation."
        )
    
    # Splash feedback
    if splash > 50000:
        feedback.append(
            f"> **Large splash area:** Measured at **{splash:,} px**. "
            f"Work on a cleaner entry by entering water with fingertips first, "
            f"keeping legs together, and maintaining a tight streamline position."
        )
    elif splash > 30000:
        feedback.append(
            f"> **Moderate splash:** Splash area **{splash:,} px**. "
            f"You're creating a relatively controlled entry. Continue focusing on "
            f"hand positioning at entry and maintaining tight body position."
        )
    else:
        feedback.append(
            f"> **Minimal splash!** Splash area only **{splash:,} px** - "
            f"excellent water entry technique with very little disturbance. "
            f"This is what we call 'ripping' the entry!"
        )
    
    # Speed feedback
    if speed < 5:
        feedback.append(
            f"> **Low dive velocity:** Peak speed **{speed} px/s**. "
            f"Work on generating more power from your takeoff to improve height and rotation."
        )
    elif speed < 10:
        feedback.append(
            f"> **Good dive velocity:** Peak speed **{speed} px/s**. "
            f"You're generating decent momentum. Focus on controlling this speed through entry."
        )
    else:
        feedback.append(
            f"> **Excellent dive velocity!** Peak speed **{speed} px/s** - "
            f"powerful takeoff while maintaining control throughout the dive."
        )
    
    # Additional technique tips based on overall score
    if score < 7:
        feedback.append(
            "> **Training focus:** Practice diving drills that emphasize body alignment and "
            "vertical entry. Video review sessions will help you identify the precise moments "
            "to make adjustments."
        )
    
    return "\n\n".join(feedback)

def cache_key(file_path: str) -> str:
    """Generate a unique cache key for a video file"""
    file_stats = os.stat(file_path)
    return f"{os.path.basename(file_path)}_{file_stats.st_size}_{file_stats.st_mtime}"

def get_cached_results(cache_key: str) -> Optional[Dict]:
    """Retrieve cached results if available"""
    cache_file = os.path.join(CACHE_DIR, f"{cache_key}.json")
    if os.path.exists(cache_file):
        try:
            with open(cache_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Failed to load cache: {e}")
    return None

def save_to_cache(cache_key: str, results: Dict) -> None:
    """Save results to cache"""
    cache_file = os.path.join(CACHE_DIR, f"{cache_key}.json")
    try:
        with open(cache_file, 'w') as f:
            json.dump(results, f)
    except Exception as e:
        logger.warning(f"Failed to save cache: {e}")

# ——— Main Streamlit App ———
def main():
    st.set_page_config(
        page_title="Dive Analyzer Pro",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Create sidebar for options
    st.sidebar.title("Dive Analyzer Options")
    
    # Scoring weights
    st.sidebar.subheader("Scoring Weights")
    weight_angle = st.sidebar.slider("Entry Angle Weight", 0.1, 0.8, 0.4, 0.1)
    weight_straight = st.sidebar.slider("Body Straightness Weight", 0.1, 0.8, 0.3, 0.1)
    weight_splash = st.sidebar.slider("Splash Size Weight", 0.1, 0.8, 0.3, 0.1)
    
    # Normalize weights to sum to 1
    total_weight = weight_angle + weight_straight + weight_splash
    weights = (
        weight_angle / total_weight,
        weight_straight / total_weight,
        weight_splash / total_weight
    )
    
    # Advanced options
    st.sidebar.subheader("Advanced Settings")
    detection_confidence = st.sidebar.slider("Pose Detection Confidence", 0.1, 0.9, 0.5, 0.1)
    splash_threshold = st.sidebar.slider("Splash Detection Sensitivity", 0.5, 2.0, 1.0, 0.1)
    
    # Update global settings
    global DETECTION_CONFIDENCE, TRACKING_CONFIDENCE, SPLASH_THRESHOLD_FACTOR
    DETECTION_CONFIDENCE = detection_confidence
    TRACKING_CONFIDENCE = detection_confidence
    SPLASH_THRESHOLD_FACTOR = splash_threshold
    
    # Cache control
    use_cache = st.sidebar.checkbox("Use cached results (faster)", True)
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### About")
    st.sidebar.info(
        "This app analyzes dive technique using computer vision and provides personalized feedback. "
        "Upload your dive video to get started!"
    )
    
    # Main content
    st.title("Swim Dive Video Analyzer Pro")
    st.markdown(
        """
        Upload a video of your dive to receive detailed analysis and coaching feedback.
        The analyzer will detect takeoff and entry points, measure critical metrics, and score your dive.
        """
    )
    
    # File uploader
    uploaded_file = st.file_uploader("Upload a dive clip (*.mp4)", type=["mp4", "mov", "avi"])
    
    if uploaded_file:
        # Save upload to temporary file
        temp_file = tempfile.NamedTemporaryFile(suffix=f".{uploaded_file.name.split('.')[-1]}", delete=False)
        temp_file.write(uploaded_file.read())
        video_path = temp_file.name
        
        # Check cache
        key = cache_key(video_path) if use_cache else None
        cached_data = get_cached_results(key) if key and use_cache else None
        
        if cached_data:
            st.success("✅ Using cached analysis results")
            
            # Show original video
            st.subheader("Original Video")
            with open(video_path, "rb") as f:
                video_bytes = f.read()
                st.video(video_bytes)
            
            # Load results from cache
            results = []
            for r in cached_data["results"]:
                dive_number = r["dive_number"]
                metrics = DiveMetrics(
                    entry_angle=r["metrics"]["entry_angle"],
                    straightness=r["metrics"]["straightness"],
                    splash_area=r["metrics"]["splash_area"],
                    peak_speed=r["metrics"]["peak_speed"]
                )
                result = DiveResult(
                    dive_number=dive_number,
                    takeoff_frame=r["takeoff_frame"],
                    entry_frame=r["entry_frame"],
                    metrics=metrics,
                    score=r["score"],
                    annotated_path=r["annotated_path"]
                )
                results.append(result)
            
            # Display results
            if results:
                # Summary results table
                st.subheader("Dive Results Summary")
                df = pd.DataFrame([r.to_dict() for r in results])
                st.dataframe(df, use_container_width=True)
                
                # Individual dive analyses
                for result in results:
                    st.subheader(f"Dive {result.dive_number} Analysis")
                    col1, col2 = st.columns([2, 1])
                    
                    # Video in col1
                    with col1:
                        if os.path.exists(result.annotated_path):
                            with open(result.annotated_path, "rb") as f:
                                annotated_bytes = f.read()
                                st.video(annotated_bytes)
                        else:
                            st.warning("Annotated video file not found in cache")
                    
                    # Metrics in col2
                    with col2:
                        st.markdown(f"**Score: {result.score:.1f}/10**")
                        st.markdown(f"Entry Angle: {result.metrics.entry_angle:.1f}°")
                        st.markdown(f"Body Straightness: {result.metrics.straightness:.2f}")
                        st.markdown(f"Splash Area: {result.metrics.splash_area:,} px")
                        st.markdown(f"Peak Speed: {result.metrics.peak_speed} px/s")
                    
                    # Coach feedback
                    st.markdown(generate_coach_feedback(result))
                    st.markdown("---")
            else:
                st.warning("No valid dives found in the cached analysis")
        
        else:
            # Process video from scratch
            with st.spinner("🔄 Processing your dive... please wait"):
                start_time = time.time()
                
                # Get video information
                video_info = get_video_info(video_path)
                fps = video_info["fps"]
                
                # Show progress indicators
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                # Display original video
                st.subheader("Original Video")
                with open(video_path, "rb") as f:
                    video_bytes = f.read()
                    st.video(video_bytes)
                
                # Extract all frames
                status_text.text("Extracting frames...")
                frames = extract_frames(video_path)
                progress_bar.progress(0.2)
                
                # Process landmarks
                status_text.text("Detecting pose landmarks...")
                landmarks = get_landmarks(frames)
                progress_bar.progress(0.4)
                
                # Detect dive segments
                status_text.text("Analyzing dive segments...")
                segments = detect_dive_segments(frames, landmarks, fps)
                if not segments:
                    segments = [(0, len(frames)-1)]  # Default to full video if no segments detected
                progress_bar.progress(0.6)
                
                # Process each dive segment
                results = []
                cache_data = {"results": []}
                
                for dive_number, (start_idx, end_idx) in enumerate(segments, start=1):
                    status_text.text(f"Processing dive {dive_number}...")
                    
                    # Extract segment frames
                    segment_frames = frames[start_idx:end_idx+1]
                    segment_landmarks = landmarks[start_idx:end_idx+1]
                    
                    # Detect key moments
                    takeoff_idx, entry_idx = detect_takeoff_and_entry(segment_frames, segment_landmarks)
                    valid_takeoff = find_valid_frame(segment_landmarks, takeoff_idx)
                    valid_entry = find_valid_frame(segment_landmarks, entry_idx)
                    
                    if valid_takeoff is None or valid_entry is None:
                        logger.warning(f"Dive {dive_number}: Missing key frames")
                        continue
                    
                    # Compute metrics
                    metrics = compute_metrics(
                        segment_frames, segment_landmarks, 
                        valid_takeoff, valid_entry, fps
                    )
                    
                    if metrics is None:
                        logger.warning(f"Dive {dive_number}: Failed to compute metrics")
                        continue
                    
                    # Score the dive
                    score = score_dive(metrics, weights)
                    
                    # Create annotated video
                    annotated_path = export_annotated_video(
                        video_path, segment_frames,
                        valid_takeoff, valid_entry, 
                        score, dive_number
                    )
                    
                    # Create result object
                    result = DiveResult(
                        dive_number=dive_number,
                        takeoff_frame=valid_takeoff + start_idx,
                        entry_frame=valid_entry + start_idx,
                        metrics=metrics,
                        score=score,
                        annotated_path=annotated_path
                    )
                    
                    results.append(result)
                    
                    # Add to cache data
                    cache_data["results"].append({
                        "dive_number": dive_number,
                        "takeoff_frame": valid_takeoff + start_idx,
                        "entry_frame": valid_entry + start_idx,
                        "metrics": metrics.to_dict(),
                        "score": score,
                        "annotated_path": annotated_path
                    })
                
                # Save to cache
                if use_cache and key:
                    save_to_cache(key, cache_data)
                
                progress_bar.progress(1.0)
                status_text.text("Analysis complete!")
                
                processing_time = time.time() - start_time
                st.success(f"✅ Done! Processing took {processing_time:.1f} seconds")
                
                # Display results
                if results:
                    # Summary results table
                    st.subheader("Dive Results Summary")
                    result_dicts = [r.to_dict() for r in results]
                    df = pd.DataFrame(result_dicts)
                    st.dataframe(df, use_container_width=True)
                    
                    # Create downloadable CSV
                    csv_data = df.to_csv(index=False)
                    st.download_button(
                        "Download Results as CSV",
                        csv_data,
                        f"dive_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        "text/csv",
                        key="download-csv"
                    )
                    
                    # Individual dive analyses
                    for result in results:
                        st.subheader(f"Dive {result.dive_number} Analysis")
                        col1, col2 = st.columns([2, 1])
                        
                        # Video in col1
                        with col1:
                            with open(result.annotated_path, "rb") as f:
                                annotated_bytes = f.read()
                                st.video(annotated_bytes)
                        
                        # Metrics in col2
                        with col2:
                            st.markdown(f"### Score: {result.score:.1f}/10")
                            
                            # Visual gauge for entry angle
                            st.markdown("#### Entry Angle")
                            angle_pct = max(0, min(100, 100 - (result.metrics.entry_angle / 30 * 100)))
                            st.progress(angle_pct/100)
                            st.markdown(f"{result.metrics.entry_angle:.1f}° (lower is better)")
                            
                            # Visual gauge for straightness
                            st.markdown("#### Body Straightness")
                            st.progress(result.metrics.straightness)
                            st.markdown(f"{result.metrics.straightness:.2f} (higher is better)")
                            
                            # Visual gauge for splash
                            st.markdown("#### Splash Control")
                            splash_pct = max(0, min(100, 100 - (result.metrics.splash_area / 50000 * 100)))
                            st.progress(splash_pct/100)
                            st.markdown(f"{result.metrics.splash_area:,} px (lower is better)")
                            
                            # Peak speed
                            st.markdown(f"#### Peak Speed: {result.metrics.peak_speed} px/s")
                        
                        # Coach feedback
                        st.markdown("### Coach Feedback")
                        st.markdown(generate_coach_feedback(result))
                        
                        # Comparison with previous dives (placeholder)
                        if dive_number > 1:
                            st.markdown("### Improvement vs Previous Dive")
                            prev_result = results[dive_number-2]
                            score_change = result.score - prev_result.score
                            if score_change > 0:
                                st.success(f"⬆️ Score improved by {score_change:.1f} points")
                            elif score_change < 0:
                                st.error(f"⬇️ Score decreased by {abs(score_change):.1f} points")
                            else:
                                st.info("Score unchanged from previous dive")
                        
                        st.markdown("---")
                else:
                    st.warning("No valid dives detected in the video. Try adjusting the sensitivity settings in the sidebar.")

if __name__ == "__main__":
    main()