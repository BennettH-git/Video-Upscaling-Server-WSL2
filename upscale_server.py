import os
import sys
import subprocess
import logging
import shutil
import time
import math
import collections
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from PIL import Image

### --- PROJECT CONFIGURATION & CONSTANTS ---
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
INPUT_DIR = os.path.join(PROJECT_ROOT, 'input')
OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'output')
TEMP_DIR = os.path.join(PROJECT_ROOT, 'temp')
LOGS_DIR = os.path.join(PROJECT_ROOT, 'logs')

# Section 3: New constants for Job Queue and Tiling
QUEUE_FILE = os.path.join(PROJECT_ROOT, 'queue.txt')
TILING_THRESHOLD_WIDTH = 2560  # Activate tiling for videos wider than 1440p
TILE_SIZE = 1024               # Input tile size for FFmpeg crop
UPSCALE_FACTOR = 4             # REAL-ESRGAN model is 4x (unfortunately)

# Temporary subdirectories
AUDIO_DIR = os.path.join(TEMP_DIR, 'audio')
RAW_FRAMES_DIR = os.path.join(TEMP_DIR, 'frames_raw')
TILED_RAW_FRAMES_DIR = os.path.join(TEMP_DIR, 'frames_raw_tiled')
TILED_UPSCALED_FRAMES_DIR = os.path.join(TEMP_DIR, 'frames_upscaled_tiled')
UPSCALED_FRAMES_DIR = os.path.join(TEMP_DIR, 'frames_upscaled')

# Path to the REAL-ESRGAN inference script
REALESRGAN_SCRIPT_PATH = os.path.join(PROJECT_ROOT, 'Real-ESRGAN', 'inference_realesrgan.py')

# Supported video file extensions
SUPPORTED_VIDEO_FORMATS = ('.mp4', '.mkv', '.mov', '.avi', '.webm')

### --- CORE PROCESSING FUNCTIONS ---
def setup_directories():
    """Create all necessary temporary and output directories if they don't exist."""
    logging.info("Setting up directories...")
    # Clean up temp directory from previous runs
    if os.path.exists(TEMP_DIR):
        shutil.rmtree(TEMP_DIR)
    for dir_path in [INPUT_DIR, OUTPUT_DIR, LOGS_DIR, AUDIO_DIR, RAW_FRAMES_DIR, UPSCALED_FRAMES_DIR, TILED_RAW_FRAMES_DIR, TILED_UPSCALED_FRAMES_DIR]:
        os.makedirs(dir_path, exist_ok=True)
    logging.info("Directory setup complete.")

def get_video_properties(input_video_path):
    """Section 3.1: Extracts video resolution and frame rate using ffprobe."""
    logging.info(f"Extracting video properties for {os.path.basename(input_video_path)}...")
    try:
        # Get frame rate
        ffprobe_rate_command = [
            'ffprobe', '-v', 'error', '-select_streams', 'v:0',
            '-show_entries', 'stream=r_frame_rate', '-of',
            'default=noprint_wrappers=1:nokey=1', input_video_path
        ]
        rate_result = subprocess.run(ffprobe_rate_command, capture_output=True, text=True, check=True)
        frame_rate = rate_result.stdout.strip()

        # Get resolution
        ffprobe_res_command = [
            'ffprobe', '-v', 'error', '-select_streams', 'v:0',
            '-show_entries', 'stream=width,height', '-of',
            'csv=s=x:p=0', input_video_path
        ]
        res_result = subprocess.run(ffprobe_res_command, capture_output=True, text=True, check=True)
        width, height = map(int, res_result.stdout.strip().split('x'))

        if not frame_rate or frame_rate == "0/0" or width == 0 or height == 0:
            raise ValueError("Invalid video properties detected.")
            
        logging.info(f"Properties found: {width}x{height} @ {frame_rate} fps")
        return {'frame_rate': frame_rate, 'width': width, 'height': height}

    except (subprocess.CalledProcessError, ValueError) as e:
        logging.error(f"Failed to get video properties: {e}", exc_info=True)
        return None

def deconstruct_video(input_video_path, temp_audio_path, raw_frames_dir):
    """Deconstructs the video into its audio and frame components."""
    logging.info("Stage 1: Deconstructing video...")
    try:
        logging.info("Extracting audio...")
        audio_extract_command = ['ffmpeg', '-hide_banner', '-loglevel', 'error', '-i', input_video_path, '-vn', '-c:a', 'copy', '-y', temp_audio_path]
        subprocess.run(audio_extract_command, check=True, capture_output=True, text=True)

        logging.info("Extracting frames...")
        frames_extract_command = ['ffmpeg', '-hide_banner', '-loglevel', 'error', '-i', input_video_path, os.path.join(raw_frames_dir, 'frame_%06d.png')]
        subprocess.run(frames_extract_command, check=True, capture_output=True, text=True)
        logging.info("Deconstruction complete.")
        return True
    except subprocess.CalledProcessError as e:
        logging.error("FFmpeg command failed during deconstruction.", exc_info=True)
        return False

def tile_frames(raw_frames_dir, tiled_raw_frames_dir, tile_size):
    """Section 3.2: Slices each raw frame into smaller tiles using FFmpeg crop filter."""
    logging.info(f"Tiling raw frames into {tile_size}x{tile_size} chunks...")
    raw_frame_files = sorted([f for f in os.listdir(raw_frames_dir) if f.endswith('.png')])
    for frame_file in raw_frame_files:
        base_name = os.path.splitext(frame_file)[0]
        input_frame_path = os.path.join(raw_frames_dir, frame_file)
        
        tiling_command = [
            'ffmpeg', '-hide_banner', '-loglevel', 'error', '-i', input_frame_path,
            '-f', 'image2', '-vf', f'crop={tile_size}:{tile_size}',
            os.path.join(tiled_raw_frames_dir, f'{base_name}_tile_%02d.png')
        ]
        subprocess.run(tiling_command, check=True, capture_output=True, text=True)
    logging.info("Frame tiling complete.")

def stitch_frames(tiled_upscaled_dir, final_upscaled_dir, original_width, original_height):
    """Section 3.2: Stitches upscaled tiles back into full frames using Pillow."""
    logging.info("Stitching upscaled tiles into final frames...")
    
    grouped_tiles = collections.defaultdict(list)
    for tile_file in sorted(os.listdir(tiled_upscaled_dir)):
        parts = tile_file.split('_')
        frame_num = parts[1]
        grouped_tiles[frame_num].append(os.path.join(tiled_upscaled_dir, tile_file))

    final_width = original_width * UPSCALE_FACTOR
    final_height = original_height * UPSCALE_FACTOR
    upscaled_tile_size = TILE_SIZE * UPSCALE_FACTOR

    for frame_num, tiles in grouped_tiles.items():
        stitched_image = Image.new('RGB', (final_width, final_height))
        num_tiles_x = math.ceil(original_width / TILE_SIZE)
        
        for i, tile_path in enumerate(tiles):
            with Image.open(tile_path) as tile_image:
                x = (i % num_tiles_x) * upscaled_tile_size
                y = (i // num_tiles_x) * upscaled_tile_size
                stitched_image.paste(tile_image, (x, y))
        
        output_path = os.path.join(final_upscaled_dir, f"frame_{frame_num}_out.png")
        stitched_image.save(output_path)
        
    logging.info("Frame stitching complete.")

def upscale_frames(input_dir, output_dir):
    """
    Upscales all frames by calling the inference script once for the entire directory.
    This is the most efficient method as it loads the AI model only once.
    """
    logging.info(f"Stage 2: Upscaling frames from '{os.path.basename(input_dir)}'...")
    try:
        upscale_command = [
            sys.executable, REALESRGAN_SCRIPT_PATH,
            '-i', input_dir,
            '-o', output_dir,
            '-n', 'realesr-general-x4v3',
            '--fp32'
        ]
        subprocess.run(upscale_command, check=True, capture_output=True, text=True)
        logging.info("Frame upscaling complete.")
    except subprocess.CalledProcessError as e:
        logging.error("REAL-ESRGAN script failed.", exc_info=True)
        # Log stderr for better debugging
        if e.stderr:
            logging.error(f"REAL-ESRGAN stderr: {e.stderr}")
        raise

def reconstruct_video(frame_rate, upscaled_frames_dir, temp_audio_path, output_video_path):
    """Reconstructs the final video using GPU-accelerated encoding (h264_nvenc)."""
    logging.info("Stage 3: Reconstructing video...")
    temp_upscaled_video_path = os.path.join(TEMP_DIR, 'upscaled_silent.mp4')
    try:
        logging.info("Creating silent upscaled video with GPU acceleration (h264_nvenc)...")
        video_reconstruct_command = [
            'ffmpeg', '-hide_banner', '-loglevel', 'error', 
            '-framerate', frame_rate, 
            '-i', os.path.join(upscaled_frames_dir, 'frame_%06d_out.png'), 
            '-c:v', 'h264_nvenc',  # Used to be a different encoder, need a GPU with this capability anyways to run ESRGAN model and it is faster so why not
            '-pix_fmt', 'yuv420p', 
            '-y', temp_upscaled_video_path
        ]
        subprocess.run(video_reconstruct_command, check=True, capture_output=True, text=True)

        logging.info("Merging audio and video...")
        merge_command = ['ffmpeg', '-hide_banner', '-loglevel', 'error', '-i', temp_upscaled_video_path, '-i', temp_audio_path, '-c', 'copy', '-y', output_video_path]
        subprocess.run(merge_command, check=True, capture_output=True, text=True)

        logging.info("Video reconstruction complete.")
    except subprocess.CalledProcessError as e:
        logging.error("FFmpeg command failed during reconstruction.", exc_info=True)
        if e.stderr:
             logging.error(f"FFmpeg stderr: {e.stderr}")
        raise

def process_video(input_video_path):
    """Main orchestration function that runs the entire upscaling pipeline for a single video."""
    filename = os.path.basename(input_video_path)
    logging.info(f"--- Starting processing for: {filename} ---")
    
    base_filename, _ = os.path.splitext(filename)
    output_video_path = os.path.join(OUTPUT_DIR, f"{base_filename}_upscaled.mp4")
    temp_audio_path = os.path.join(AUDIO_DIR, f'{base_filename}_audio.aac')

    if not os.path.exists(input_video_path):
        logging.error(f"Input file not found: {input_video_path}")
        return

    try:
        setup_directories()
        
        properties = get_video_properties(input_video_path)
        if not properties:
            raise ValueError("Failed to get video properties. Aborting.")

        if not deconstruct_video(input_video_path, temp_audio_path, RAW_FRAMES_DIR):
             raise RuntimeError("Failed to deconstruct video.")

        if properties['width'] > TILING_THRESHOLD_WIDTH:
            logging.info("High-resolution source detected. Engaging tiling workflow.")
            tile_frames(RAW_FRAMES_DIR, TILED_RAW_FRAMES_DIR, TILE_SIZE)
            upscale_frames(TILED_RAW_FRAMES_DIR, TILED_UPSCALED_FRAMES_DIR)
            stitch_frames(TILED_UPSCALED_FRAMES_DIR, UPSCALED_FRAMES_DIR, properties['width'], properties['height'])
        else:
            logging.info("Standard resolution source. Using direct upscaling workflow.")
            upscale_frames(RAW_FRAMES_DIR, UPSCALED_FRAMES_DIR)

        reconstruct_video(properties['frame_rate'], UPSCALED_FRAMES_DIR, temp_audio_path, output_video_path)
        logging.info(f"--- Successfully processed {filename}. Final video saved to: {output_video_path} ---")

    except (subprocess.CalledProcessError, ValueError, RuntimeError, Exception) as e:
        logging.error(f"An error occurred while processing {filename}: {e}", exc_info=True)
    finally:
        logging.info(f"Cleaning up temporary files for {filename}...")
        if os.path.exists(TEMP_DIR):
            shutil.rmtree(TEMP_DIR)
        logging.info("Cleanup complete.")

### --- AUTOMATION & FILE MONITORING ---
class VideoHandler(FileSystemEventHandler):
    """Section 3.3: Enqueues new video files instead of processing directly."""
    def on_created(self, event):
        if event.is_directory or not event.src_path.lower().endswith(SUPPORTED_VIDEO_FORMATS):
            return
        
        logging.info(f"New video file detected: {event.src_path}")
        
        try:
            initial_size = -1
            time.sleep(2)
            while initial_size != os.path.getsize(event.src_path):
                initial_size = os.path.getsize(event.src_path)
                time.sleep(5)
            logging.info(f"File size stable. Enqueuing job for {os.path.basename(event.src_path)}")
            with open(QUEUE_FILE, 'a') as f:
                f.write(f"{event.src_path}\n")
        except (FileNotFoundError, Exception) as e:
            logging.error(f"Error while enqueuing {event.src_path}: {e}", exc_info=True)

### --- MAIN EXECUTION BLOCK ---
if __name__ == "__main__":
    log_file = os.path.join(LOGS_DIR, 'server_operations.log')
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - [%(levelname)s] - %(message)s', handlers=[logging.FileHandler(log_file), logging.StreamHandler(sys.stdout)])
    
    setup_directories()

    logging.info("--- AI Video Upscaling Server ---")
    
    observer = Observer()
    observer.schedule(VideoHandler(), INPUT_DIR, recursive=True)
    observer.start()
    logging.info(f"Monitoring for new videos in: {INPUT_DIR}")
    
    logging.info("Starting main worker loop...")
    try:
        while True:
            if os.path.exists(QUEUE_FILE) and os.path.getsize(QUEUE_FILE) > 0:
                with open(QUEUE_FILE, 'r+') as f:
                    lines = f.readlines()
                    f.seek(0)
                    next_job_path = lines[0].strip()
                    f.writelines(lines[1:])
                    f.truncate()
                
                if next_job_path:
                    logging.info(f"Dequeued job: {next_job_path}")
                    process_video(next_job_path)
            time.sleep(5)
    except KeyboardInterrupt:
        logging.info("Shutdown signal received. Stopping server...")
    finally:
        observer.stop()
        observer.join()
        logging.info("Server has been shut down successfully.")