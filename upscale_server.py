import os
import sys
import subprocess
import logging
import shutil
import time
import math
import datetime
import json
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from basicsr.utils.download_util import load_file_from_url
from PIL import Image

# --- PROJECT CONFIGURATION & CONSTANTS ---
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
REALESRGAN_DIR = os.path.join(PROJECT_ROOT, 'Real-ESRGAN')
INPUT_DIR = os.path.join(PROJECT_ROOT, 'input')
OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'output')
LOGS_DIR = os.path.join(PROJECT_ROOT, 'logs')
TEMP_DIR_BASE = os.path.join(PROJECT_ROOT, 'temp')

# --- Just-in-Time RAM Disk Configuration ---
USE_RAM_DISK = False                       # RAM_DISK_SIZE is important to change based upon your systems memory. 
RAM_DISK_MOUNT_POINT = "/mnt/ramdisk"      # More ram means bigger batches and less I/O processing time to let the GPU do the heavy lifting.
RAM_DISK_SIZE_GB = 32                      # Off by default because it is complicated to set up, but absolutely necessary to prevent SSD wear if youre using this script often.

# --- Model & Tiling Configuration ---
MODEL_NAME = 'realesr-general-x4v3'                                                    # Note: I tried an older model that was able to be preloaded into VRAM but it sucked.
MODEL_PATH = os.path.join(REALESRGAN_DIR, 'weights', f'{MODEL_NAME}.pth')              # It was 100% GPU usage for similar performance and slightly worse results.
REALESRGAN_SCRIPT_PATH = os.path.join(REALESRGAN_DIR, 'inference_realesrgan.py')       
TILING_THRESHOLD_WIDTH = 1920                                                          # This should probably be decreased because of WSL2 I/O overhead and resource draw
TILE_SIZE = 512                                                                        # I am afraid it will introduce artifacts though
UPSCALE_FACTOR = 4

# --- Adaptive Batch Processing & Job Control ---                 #Note: the batch processing architecture changed from the previous version to prevent crashes caused by running out of VRAM
ETA_BATCH_SIZE = 30                                                   # Smaller first batch, ETA is a worst case estimate.
MIN_BATCH_SIZE = 50                                                   
MAX_BATCH_SIZE = 4000
RAM_DISK_UTILIZATION_TARGET = 0.75
CLEANUP_ON_COMPLETE = True                                            # Set to False to keep intermediate clips for debugging. SAM2 implementation would be cool, but is another massive project.

# --- Stability Configuration ---
RECONSTRUCTION_RETRY_ATTEMPTS = 3             # Typically either works or it doesnt. CPU fallback is good enough for right now.
UPSCALE_PROCESS_TIMEOUT_SECONDS = 7200        # After 1 hour of script hang on a single batch, the script aborts.

# --- Directory Definitions ---
QUEUE_FILE = os.path.join(PROJECT_ROOT, 'queue.txt')
TEMP_DIR_FRAMES = RAM_DISK_MOUNT_POINT if USE_RAM_DISK else os.path.join(PROJECT_ROOT, 'temp_frames')

SUPPORTED_VIDEO_FORMATS = ('.mp4', '.mkv', '.mov', '.avi', '.webm')

# --- CORE PROCESSING & HELPER FUNCTIONS ---

def check_dependencies():
    """Performs pre-flight checks for critical files."""
    logging.info("Performing dependency pre-flight checks...")
    if not os.path.isfile(REALESRGAN_SCRIPT_PATH):
        logging.critical(f"REAL-ESRGAN script not found at: {REALESRGAN_SCRIPT_PATH}")
        return False
    
    if not os.path.exists(MODEL_PATH):
        logging.warning(f"Model weights not found at {MODEL_PATH}. Attempting to download...")
        model_url = f'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/{MODEL_NAME}.pth'
        model_dir = os.path.dirname(MODEL_PATH)
        os.makedirs(model_dir, exist_ok=True)
        try:
            load_file_from_url(url=model_url, model_dir=model_dir, progress=True, file_name=f'{MODEL_NAME}.pth')
            logging.info("Model downloaded successfully.")
        except Exception as download_error:
            logging.critical(f"Failed to download model weights: {download_error}", exc_info=True)
            return False
            
    logging.info("All dependencies found.")
    return True

def manage_ramdisk(mount=True):
    """Mounts or unmounts the RAM disk using sudo."""
    if not USE_RAM_DISK: return
    mount_point = RAM_DISK_MOUNT_POINT
    is_mounted = os.path.ismount(mount_point)
    try:
        if mount and not is_mounted:
            logging.info(f"Mounting RAM disk ({RAM_DISK_SIZE_GB}GB) at '{mount_point}'...")
            subprocess.run(['sudo', 'mkdir', '-p', mount_point], check=True)
            subprocess.run(['sudo', 'chown', f'{os.getuid()}:{os.getgid()}', mount_point], check=True)
            
            mount_command = ['sudo', 'mount', '-t', 'tmpfs', '-o', f'size={RAM_DISK_SIZE_GB}G', 'tmpfs', mount_point]
            subprocess.run(mount_command, check=True, capture_output=True, text=True)
            logging.info("RAM disk mounted.")
        elif not mount and is_mounted:
            logging.info(f"Unmounting RAM disk at '{mount_point}'...")
            unmount_command = ['sudo', 'umount', mount_point]
            subprocess.run(unmount_command, check=True, capture_output=True, text=True)
            logging.info("RAM disk unmounted.")
    except subprocess.CalledProcessError as e:
        logging.critical(f"Failed to manage RAM disk! Configure passwordless sudo. Error: {e.stderr}", exc_info=True)
        raise

def setup_job_directories(job_dir):
    """Creates all necessary directories for a specific job."""
    os.makedirs(job_dir, exist_ok=True)
    os.makedirs(os.path.join(job_dir, 'final_clips'), exist_ok=True)
    
    os.makedirs(TEMP_DIR_FRAMES, exist_ok=True)
    for dir_name in ['audio', 'frames_raw', 'frames_raw_tiled', 'frames_upscaled_tiled', 'frames_upscaled']:
        os.makedirs(os.path.join(TEMP_DIR_FRAMES, dir_name), exist_ok=True)

def cleanup_batch_frames():
    """Cleans out the ephemeral frame directories between batches."""                                       # Prevents running out of space on mounted drive
    for dir_name in ['frames_raw', 'frames_raw_tiled', 'frames_upscaled_tiled', 'frames_upscaled']:
        dir_path = os.path.join(TEMP_DIR_FRAMES, dir_name)
        if os.path.exists(dir_path):
            shutil.rmtree(dir_path)
            os.makedirs(dir_path)

def get_video_properties(input_video_path):
    """Extracts video properties."""                                                                       # Added a fallback and JSON transcription of the relevant properties to boot faster
    logging.info(f"Extracting video properties for {os.path.basename(input_video_path)}...")               # Certain file types were previously breaking the script since FPS wasnt a value, it now calculates FPS
    try:
        ffprobe_command = ['ffprobe', '-v', 'error', '-select_streams', 'v:0', '-show_entries', 'stream=width,height,r_frame_rate,nb_frames', '-of', 'default=noprint_wrappers=1:nokey=1', input_video_path]
        result = subprocess.run(ffprobe_command, capture_output=True, text=True, check=True)
        output = result.stdout.strip().split('\n')
        
        if len(output) < 4: raise ValueError(f"ffprobe returned incomplete data: {output}")
        width_str, height_str, frame_rate_str, total_frames_str = output[:4]
        
        if 'N/A' in total_frames_str:
            logging.warning("Metadata frame count missing. Performing stream scan...")
            count_command = ['ffprobe', '-v', 'error', '-count_frames', '-select_streams', 'v:0', '-show_entries', 'stream=nb_read_frames', '-of', 'default=noprint_wrappers=1:nokey=1', input_video_path]
            count_result = subprocess.run(count_command, capture_output=True, text=True, check=True)
            total_frames_str = count_result.stdout.strip()
            if not total_frames_str or 'N/A' in total_frames_str:
                raise ValueError("Robust frame count also failed.")
            logging.info(f"Accurate frame count found: {total_frames_str}")

        num, den = map(int, frame_rate_str.split('/'))
        if den == 0: raise ValueError("Invalid frame rate denominator.")

        properties = {'width': int(width_str), 'height': int(height_str), 'frame_rate_str': frame_rate_str, 'frame_rate_float': num / den, 'total_frames': int(total_frames_str)}
        logging.info(f"Properties found: {properties['width']}x{properties['height']} @ {properties['frame_rate_str']} fps, {properties['total_frames']} total frames")
        return properties
    except Exception as e:
        logging.error(f"Failed to get video properties: {e}", exc_info=True)
        return None

def calculate_disk_aware_batch_size(width, height):
    """Calculates batch size based on available disk space and frame dimensions."""
    try:
        _, _, free = shutil.disk_usage(TEMP_DIR_FRAMES)
        usable_space = free * RAM_DISK_UTILIZATION_TARGET
        
        bytes_per_pixel = 3 # RGB
        raw_size = width * height * bytes_per_pixel
        upscaled_size = (width * UPSCALE_FACTOR) * (height * UPSCALE_FACTOR) * bytes_per_pixel
        space_per_frame = raw_size + upscaled_size

        if space_per_frame == 0: return MIN_BATCH_SIZE

        calculated_size = int(usable_space / space_per_frame)
        
        adaptive_size = max(MIN_BATCH_SIZE, min(calculated_size, MAX_BATCH_SIZE))
        
        logging.info(f"Disk-aware batch size calculation: Available space={free/(1024**3):.2f}GB, Final={adaptive_size}")
        return adaptive_size

    except Exception as e:
        logging.error(f"Could not determine disk-aware batch size: {e}. Falling back to MIN_BATCH_SIZE.", exc_info=True)
        return MIN_BATCH_SIZE

def log_gpu_status():
    """Logs the output of nvidia-smi."""
    try:
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True, check=True)
        logging.warning(f"--- NVIDIA-SMI Output on Final Failure ---\n{result.stdout}\n--- End NVIDIA-SMI Output ---")
    except Exception as e:
        logging.error(f"Failed to execute 'nvidia-smi': {e}")

def deconstruct_video_batch(input_video_path, start_frame, num_frames, frame_rate_float):
    """Deconstructs a specific batch of frames."""
    logging.info(f"Deconstructing batch: starting at frame {start_frame}, for {num_frames} frames...")
    try:
        raw_frames_dir = os.path.join(TEMP_DIR_FRAMES, 'frames_raw')
        start_time = start_frame / frame_rate_float
        command = ['ffmpeg', '-hide_banner', '-loglevel', 'error', '-ss', str(start_time), '-i', input_video_path, '-frames:v', str(num_frames), '-start_number', str(start_frame + 1), os.path.join(raw_frames_dir, 'frame_%06d.png')]
        subprocess.run(command, check=True, capture_output=True, text=True)
        return True
    except subprocess.CalledProcessError as e:
        logging.error(f"FFmpeg failed during deconstruction. Stderr: {e.stderr}", exc_info=True)
        return False

def upscale_frames_controller(width):                                                          # I/O overhead is still an issue in WSL2, when I was looking into it I was seeing that it may also be a problem on bare metal linux
    """Decides whether to use standard or tiled upscaling based on frame width."""            
    raw_frames_dir = os.path.join(TEMP_DIR_FRAMES, 'frames_raw')
    tiled_raw_frames_dir = os.path.join(TEMP_DIR_FRAMES, 'frames_raw_tiled')
    tiled_upscaled_frames_dir = os.path.join(TEMP_DIR_FRAMES, 'frames_upscaled_tiled')
    upscaled_frames_dir = os.path.join(TEMP_DIR_FRAMES, 'frames_upscaled')
    
    if width >= TILING_THRESHOLD_WIDTH:
        logging.info(f"Frame width ({width}px) exceeds threshold. Activating manual tiling.")
        slice_frames_into_tiles(raw_frames_dir, tiled_raw_frames_dir)
        upscale_frames_subprocess(tiled_raw_frames_dir, tiled_upscaled_frames_dir)
        stitch_tiles_into_frames(tiled_upscaled_frames_dir, upscaled_frames_dir)
    else:
        logging.info("Frame width is within limits. Using standard upscaling.")
        upscale_frames_subprocess(raw_frames_dir, upscaled_frames_dir)

def slice_frames_into_tiles(raw_dir, tiled_dir):
    """Uses Pillow to slice each raw frame into smaller tiles."""
    logging.info("Slicing full-resolution frames into tiles...")
    frames = sorted(os.listdir(raw_dir))
    for frame_name in frames:
        try:
            with Image.open(os.path.join(raw_dir, frame_name)) as img:
                width, height = img.size
                frame_base, ext = os.path.splitext(frame_name)
                for i in range(0, width, TILE_SIZE):
                    for j in range(0, height, TILE_SIZE):
                        box = (i, j, i + TILE_SIZE, j + TILE_SIZE)
                        tile = img.crop(box)
                        tile.save(os.path.join(tiled_dir, f"{frame_base}_tile_{i}_{j}{ext}"))
        except Exception as e:
            logging.error(f"Failed to slice frame {frame_name}: {e}", exc_info=True)

def stitch_tiles_into_frames(tiled_upscaled_dir, final_dir):
    """Uses Pillow to reassemble upscaled tiles into full frames."""
    logging.info("Stitching upscaled tiles back into full-resolution frames...")
    grouped_tiles = {}
    for tile_name in os.listdir(tiled_upscaled_dir):
        try:
            parts = os.path.splitext(tile_name)[0].split('_')
            frame_base = parts[0] + '_' + parts[1]
            if frame_base not in grouped_tiles:
                grouped_tiles[frame_base] = []
            grouped_tiles[frame_base].append(tile_name)
        except IndexError:
            logging.warning(f"Could not parse tile name: {tile_name}. Skipping.")
            continue

    for frame_base, tiles in grouped_tiles.items():
        try:
            max_x, max_y = 0, 0
            for tile_name in tiles:
                parts = os.path.splitext(tile_name)[0].split('_')
                max_x = max(max_x, int(parts[3]))
                max_y = max(max_y, int(parts[4]))
            
            with Image.open(os.path.join(tiled_upscaled_dir, tiles[0])) as sample_tile:
                tile_w, tile_h = sample_tile.size
            
            full_width = (max_x // TILE_SIZE + 1) * tile_w
            full_height = (max_y // TILE_SIZE + 1) * tile_h
            
            full_image = Image.new('RGB', (full_width, full_height))
            
            for tile_name in tiles:
                parts = os.path.splitext(tile_name)[0].split('_')
                x_offset = int(parts[3]) // TILE_SIZE * tile_w
                y_offset = int(parts[4]) // TILE_SIZE * tile_h
                
                with Image.open(os.path.join(tiled_upscaled_dir, tile_name)) as tile:
                    full_image.paste(tile, (x_offset, y_offset))
            
            full_image.save(os.path.join(final_dir, f"{frame_base}_out.png"))
        except Exception as e:
            logging.error(f"Failed to stitch tiles for frame {frame_base}: {e}", exc_info=True)

def upscale_frames_subprocess(input_dir, output_dir):                                                                                   # The resource intensive part for the GPU.
    """Upscales frames using a robust subprocess call."""
    logging.info(f"Starting subprocess upscaling for frames in '{os.path.basename(input_dir)}'...")
    try:
        command = [sys.executable, REALESRGAN_SCRIPT_PATH, '-i', input_dir, '-o', output_dir, '-n', MODEL_NAME, '--suffix', 'out']
        
        logging.info(f"Executing command: {' '.join(command)}")
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        
        logging.info(f"Waiting for REAL-ESRGAN subprocess (PID: {process.pid}) to complete...")
        stdout, stderr = process.communicate(timeout=UPSCALE_PROCESS_TIMEOUT_SECONDS)
        
        if process.returncode != 0:
            logging.error(f"REAL-ESRGAN script failed with exit code {process.returncode}.")
            if stdout: logging.error(f"REAL-ESRGAN stdout:\n{stdout}")
            if stderr: logging.error(f"REAL-ESRGAN stderr:\n{stderr}")
            log_gpu_status()
            raise subprocess.CalledProcessError(process.returncode, command, output=stdout, stderr=stderr)
        
        logging.info("Subprocess upscaling completed successfully.")
        if stderr and stderr.strip() and 'FutureWarning' not in stderr:
             logging.warning(f"REAL-ESRGAN stderr (warnings):\n{stderr}")

    except subprocess.TimeoutExpired:
        logging.error(f"REAL-ESRGAN subprocess timed out after {UPSCALE_PROCESS_TIMEOUT_SECONDS} seconds. Killing process.")
        process.kill()
        stdout, stderr = process.communicate()
        if stdout: logging.error(f"REAL-ESRGAN stdout on timeout:\n{stdout}")
        if stderr: logging.error(f"REAL-ESRGAN stderr on timeout:\n{stderr}")
        log_gpu_status()
        raise
    except Exception as e:
        logging.error(f"An exception occurred while running the upscaling subprocess.", exc_info=True)
        raise

def reconstruct_and_spill_batch(batch_num, frame_rate_str, start_frame_num, job_dir):
    """Reconstructs a video clip and saves it directly to the physical disk."""                                   # For practicality sake
    logging.info(f"Reconstructing video clip for batch {batch_num}...")
    final_clip_path = os.path.join(job_dir, 'final_clips', f'clip_{batch_num:04d}.mp4')
    
    input_pattern = os.path.join(TEMP_DIR_FRAMES, 'frames_upscaled', 'frame_%06d_out.png')
    
    # --- ARCHITECTURE FIX: Write directly to the final destination ---
    gpu_command = ['ffmpeg', '-hide_banner', '-loglevel', 'error', '-framerate', frame_rate_str, '-start_number', str(start_frame_num), '-i', input_pattern, '-c:v', 'h264_nvenc', '-pix_fmt', 'yuv420p', '-y', final_clip_path]
    
    for attempt in range(RECONSTRUCTION_RETRY_ATTEMPTS):
        try:
            if attempt > 0:
                delay = attempt + 1
                logging.warning(f"GPU encoding attempt {attempt + 1}/{RECONSTRUCTION_RETRY_ATTEMPTS}: Retrying after {delay}s...")
                time.sleep(delay)
            logging.info(f"Attempting GPU encoding (h264_nvenc)... Attempt {attempt + 1}")
            subprocess.run(gpu_command, check=True, capture_output=True, text=True)
            logging.info(f"GPU encoding successful. Clip saved to: {final_clip_path}")
            return
        except subprocess.CalledProcessError as e:
            logging.warning(f"GPU encoding attempt {attempt + 1} failed.")
            if (attempt + 1) == RECONSTRUCTION_RETRY_ATTEMPTS:
                logging.error(f"All GPU encoding attempts have failed. Logging final GPU status. Stderr: {e.stderr}")
                log_gpu_status()
                break
            else:
                 logging.warning(f"Transient GPU error detected. Stderr: {e.stderr}")

    logging.warning("Falling back to CPU encoding (libx264).")
    cpu_command = ['ffmpeg', '-hide_banner', '-loglevel', 'error', '-framerate', frame_rate_str, '-start_number', str(start_frame_num), '-i', input_pattern, '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-preset', 'fast', '-y', final_clip_path]
    try:
        subprocess.run(cpu_command, check=True, capture_output=True, text=True)
        logging.info(f"CPU encoding successful. Clip saved to: {final_clip_path}")
    except Exception as e:
        logging.error(f"FATAL: CPU encoding fallback failed. Stderr: {getattr(e, 'stderr', e)}", exc_info=True)
        raise

def concatenate_clips(temp_audio_path, output_video_path, job_dir):
    """Concatenates clips from the physical disk and merges audio."""
    logging.info("Concatenating all final video clips from physical disk...")
    final_clips_dir = os.path.join(job_dir, 'final_clips')
    clip_files = sorted([f for f in os.listdir(final_clips_dir) if f.endswith('.mp4')])
    if not clip_files: raise ValueError("No clips to concatenate.")

    concat_list_path = os.path.join(job_dir, 'concat_list.txt')
    with open(concat_list_path, 'w') as f:
        for clip in clip_files:
            f.write(f"file '{os.path.abspath(os.path.join(final_clips_dir, clip))}'\n")

    final_silent_video = os.path.join(job_dir, 'final_silent.mp4')
    try:
        concat_command = ['ffmpeg', '-hide_banner', '-loglevel', 'error', '-f', 'concat', '-safe', '0', '-i', concat_list_path, '-c', 'copy', '-y', final_silent_video]
        subprocess.run(concat_command, check=True, capture_output=True, text=True)
        logging.info("Merging final video with audio...")
        merge_command = ['ffmpeg', '-hide_banner', '-loglevel', 'error', '-i', final_silent_video, '-i', temp_audio_path, '-c', 'copy', '-y', output_video_path]
        subprocess.run(merge_command, check=True, capture_output=True, text=True)
        logging.info("Final video reintegration complete.")
    except subprocess.CalledProcessError as e:
        logging.error(f"FFmpeg failed during final merge. Stderr: {e.stderr}", exc_info=True)
        raise

def process_video(input_video_path):
    """Main orchestration function for a single video file."""
    filename = os.path.basename(input_video_path)
    base_filename, _ = os.path.splitext(filename)
    job_dir = os.path.join(OUTPUT_DIR, base_filename)
    status_file = os.path.join(job_dir, 'status.json')
    output_video_path = os.path.join(OUTPUT_DIR, f"{base_filename}_upscaled.mp4")
    
    logging.info(f"--- Starting processing for: {filename} ---")
    
    try:
        manage_ramdisk(mount=True)
        setup_job_directories(job_dir)
        
        if os.path.exists(status_file):
            with open(status_file, 'r') as f: status = json.load(f)
            logging.info(f"Resuming job for {filename}. Status file found.")
        else:
            status = {'completed_batches': []}
            logging.info(f"Starting new job for {filename}.")

        properties = status.get('properties')
        if not properties:
            logging.warning("Video properties not found in status file. Re-analyzing video.")
            properties = get_video_properties(input_video_path)
            if properties: status['properties'] = properties
            else: raise ValueError("Failed to get video properties for a new or incomplete job.")

        adaptive_batch_size = status.get('adaptive_batch_size')
        if not adaptive_batch_size:
            logging.warning("Batch size not found in status file. Re-calculating.")
            adaptive_batch_size = calculate_disk_aware_batch_size(properties['width'], properties['height'])
            if adaptive_batch_size: status['adaptive_batch_size'] = adaptive_batch_size
            else: raise ValueError("Failed to calculate batch size.")
        
        with open(status_file, 'w') as f: json.dump(status, f, indent=4)

        total_frames = properties['total_frames']
        audio_dir_frames = os.path.join(TEMP_DIR_FRAMES, 'audio')
        temp_audio_path = os.path.join(audio_dir_frames, f'{base_filename}_audio.aac')
        
        if not os.path.exists(temp_audio_path):
            logging.info("Extracting full audio track...")
            audio_extract_command = ['ffmpeg', '-hide_banner', '-loglevel', 'error', '-i', input_video_path, '-vn', '-c:a', 'copy', '-y', temp_audio_path]
            subprocess.run(audio_extract_command, check=True)
        else:
            logging.info("Audio track already exists. Skipping extraction.")

        # Batching logic
        if total_frames <= ETA_BATCH_SIZE:
            batch_definitions = [(0, total_frames)]
        else:
            batch_definitions = [(0, ETA_BATCH_SIZE)]
            remaining_frames = total_frames - ETA_BATCH_SIZE
            current_frame = ETA_BATCH_SIZE
            while remaining_frames > 0:
                frames_in_this_batch = min(adaptive_batch_size, remaining_frames)
                batch_definitions.append((current_frame, frames_in_this_batch))
                remaining_frames -= frames_in_this_batch
                current_frame += frames_in_this_batch
        
        num_total_batches = len(batch_definitions)
        job_start_time = time.time()
        frames_processed_so_far = sum(status.get('frames_in_batch', {}).values())

        for i, (start_frame, num_frames_in_batch) in enumerate(batch_definitions):
            batch_num = i + 1
            if str(batch_num) in status.get('completed_batches', []):
                logging.info(f"--- Skipping Batch {batch_num}/{num_total_batches} (already complete) ---")
                continue

            logging.info(f"--- Processing Batch {batch_num}/{num_total_batches} ---")
            
            cleanup_batch_frames()
            if not deconstruct_video_batch(input_video_path, start_frame, num_frames_in_batch, properties['frame_rate_float']):
                raise RuntimeError("Failed to deconstruct video batch.")

            upscale_frames_controller(properties['width'])
            reconstruct_and_spill_batch(batch_num, properties['frame_rate_str'], start_frame + 1, job_dir)
            
            status['completed_batches'].append(str(batch_num))
            status.setdefault('frames_in_batch', {})[str(batch_num)] = num_frames_in_batch
            with open(status_file, 'w') as f: json.dump(status, f, indent=4)
            
            if i == 0 and 'eta_logged' not in status:
                first_batch_duration = time.time() - job_start_time
                time_per_frame = first_batch_duration / num_frames_in_batch if num_frames_in_batch > 0 else 0
                if time_per_frame > 0:
                    estimated_total_seconds = time_per_frame * total_frames
                    logging.info(f"--- ETA Calculation (based on {num_frames_in_batch} frames) ---")
                    logging.info(f"Estimated total job duration: {datetime.timedelta(seconds=int(estimated_total_seconds))}")
                    status['eta_logged'] = True
                    with open(status_file, 'w') as f: json.dump(status, f, indent=4)

            frames_processed_so_far += num_frames_in_batch
            logging.info(f"--- Batch {batch_num} Complete ---")
            progress_percent = (frames_processed_so_far / total_frames) * 100
            logging.info(f"Overall Progress: {frames_processed_so_far}/{total_frames} frames processed ({progress_percent:.2f}%)")
        
        concatenate_clips(temp_audio_path, output_video_path, job_dir)
        logging.info(f"--- Successfully processed {filename}. Final video saved to: {output_video_path} ---")

        if CLEANUP_ON_COMPLETE:
            logging.info("Job complete. Cleaning up persistent job directory...")
            shutil.rmtree(job_dir)

    except Exception as e:
        logging.error(f"An error occurred while processing {filename}: {e}", exc_info=True)
    finally:
        logging.info("Cleaning up ephemeral frame directories...")
        if USE_RAM_DISK and os.path.exists(TEMP_DIR_FRAMES):
            for dir_name in ['audio', 'frames_raw', 'frames_raw_tiled', 'frames_upscaled_tiled', 'frames_upscaled']:
                dir_path = os.path.join(TEMP_DIR_FRAMES, dir_name)
                if os.path.exists(dir_path): shutil.rmtree(dir_path)
        manage_ramdisk(mount=False)
        logging.info("Cleanup complete.")

def scan_for_incomplete_jobs():
    """Scans the input and output directories on startup to find jobs to resume."""
    logging.info("Scanning for incomplete jobs to resume...")
    resumed_jobs = 0
    
    current_queue = []
    if os.path.exists(QUEUE_FILE):
        with open(QUEUE_FILE, 'r') as f: current_queue = [line.strip() for line in f.readlines()]

    if not os.path.exists(INPUT_DIR): os.makedirs(INPUT_DIR)

    for video_file in os.listdir(INPUT_DIR):
        if not video_file.lower().endswith(SUPPORTED_VIDEO_FORMATS): continue
            
        input_video_path = os.path.join(INPUT_DIR, video_file)
        base_filename, _ = os.path.splitext(video_file)
        job_dir = os.path.join(OUTPUT_DIR, base_filename)
        final_output_file = os.path.join(OUTPUT_DIR, f"{base_filename}_upscaled.mp4")

        if not os.path.exists(final_output_file):
             if input_video_path not in current_queue:
                logging.info(f"Found incomplete or new job for '{video_file}'. Adding to queue.")
                current_queue.insert(0, input_video_path) # Prioritize resuming
                resumed_jobs += 1
    
    if resumed_jobs > 0:
        with open(QUEUE_FILE, 'w') as f:
            for job_path in current_queue: f.write(f"{job_path}\n")
    
    logging.info(f"Scan complete. Found {resumed_jobs} job(s) to queue.")

# --- AUTOMATION & FILE MONITORING ---

class VideoHandler(FileSystemEventHandler):
    """Enqueues new video files after ensuring they are fully written."""
    def on_created(self, event):
        if event.is_directory or not event.src_path.lower().endswith(SUPPORTED_VIDEO_FORMATS):
            return
        
        logging.info(f"New video file detected: {event.src_path}")
        try:
            time.sleep(2)
            last_size = -1
            while last_size != os.path.getsize(event.src_path):
                last_size = os.path.getsize(event.src_path)
                time.sleep(3)
            
            logging.info(f"File size stable. Enqueuing job for {os.path.basename(event.src_path)}")
            with open(QUEUE_FILE, 'a') as f:
                f.write(f"{os.path.abspath(event.src_path)}\n")
        except (FileNotFoundError, Exception) as e:
            logging.error(f"Error while enqueuing {event.src_path}: {e}", exc_info=True)

# --- MAIN EXECUTION BLOCK ---

if __name__ == "__main__":
    os.makedirs(LOGS_DIR, exist_ok=True)
    log_file = os.path.join(LOGS_DIR, 'server_operations.log')
    
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - [%(levelname)s] - %(message)s', handlers=[logging.FileHandler(log_file), logging.StreamHandler(sys.stdout)])
    
    logging.info("--- AI Video Upscaling Server Starting Up ---")
    
    if not check_dependencies():
        logging.critical("Dependency check failed. The server cannot start.")
        sys.exit(1)
    
    scan_for_incomplete_jobs()
    
    observer = Observer()
    observer.schedule(VideoHandler(), INPUT_DIR, recursive=False)
    observer.start()
    logging.info(f"Monitoring for new videos in: {INPUT_DIR}")
    
    logging.info("Starting main worker loop...")
    try:
        while True:
            if os.path.exists(QUEUE_FILE) and os.path.getsize(QUEUE_FILE) > 0:
                with open(QUEUE_FILE, 'r+') as f:
                    lines = f.readlines()
                    next_job_path = lines[0].strip()
                    f.seek(0)
                    f.writelines(lines[1:])
                    f.truncate()
                
                if next_job_path and os.path.isfile(next_job_path):
                    logging.info(f"Dequeued job: {next_job_path}")
                    process_video(next_job_path)
                elif next_job_path:
                    logging.error(f"Job dequeued, but file not found: {next_job_path}")

            time.sleep(6)
    except KeyboardInterrupt:
        logging.info("Shutdown signal received. Stopping server...")
    finally:
        observer.stop()
        observer.join()
        logging.info("Server has been shut down successfully.")

