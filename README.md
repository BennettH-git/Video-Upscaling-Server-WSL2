# Automated AI Video Upscaling Server

An automated AI video upscaling pipeline orchestrated by Python. Manages a sequential job queue for stability and system resources via VRAM-aware tiling. Features a filesystem watcher and GPU-accelerated (NVENC) encoding in a Linux environment. Built with FFmpeg & PyTorch/REAL-ESRGAN.
Key Features

This project is designed as a robust, hands-off service for AI-powered video super-resolution. It emphasizes stability and automation, making it suitable for continuous operation.

  - Fully Automated Workflow: A watchdog-based sentinel monitors a directory for new video files and automatically triggers the entire processing pipeline.

  - Stable Job Queuing: A simple but effective file-based job queue ensures that videos are processed sequentially, preventing resource contention and system overloads when multiple files are added simultaneously.

  - VRAM-Aware Tiling: The server automatically detects high-resolution input videos (above 1080p) and engages a manual tiling-and-stitching workflow to process frames in manageable chunks, preventing CUDA out-of-memory errors on consumer GPUs.

  - GPU-Accelerated Encoding: The final video reconstruction stage uses FFmpeg's h264_nvenc encoder to offload the computationally expensive encoding task from the CPU to the NVIDIA GPU's dedicated hardware, significantly improving performance and freeing up system resources.



## The Workflow

When a new video is detected, the server executes the following multi-stage pipeline:

  1. Enqueue & Dequeue: The video path is added to queue.txt. A worker loop pulls the next available job from the queue.
  2. Inspect: ffprobe extracts the video's resolution and frame rate to inform later stages of the pipeline.
  3. Deconstruct: FFmpeg extracts the original audio track (preserving its quality) and splits the video into a sequence of lossless PNG frames.
  4. Upscale (with Tiling Logic):
        If the video is high-resolution, frames are tiled into smaller pieces. These tiles are upscaled by REAL-ESRGAN. The upscaled tiles are then stitched back together into full frames.
        If the video is standard resolution, the frames are passed directly to REAL-ESRGAN for upscaling.
  5. Reconstruct: The sequence of upscaled frames is encoded into a new, high-resolution H.264 video stream using the GPU's NVENC hardware encoder.
  6. Reintegrate: The new video stream is combined with the original audio track.
  7. Cleanup: All temporary files (frames, tiles, audio clips) are automatically deleted.

## Technology Stack

  - Orchestration: Python 3.10+
  - Media Processing: FFmpeg
  - AI Model: REAL-ESRGAN (via PyTorch)
  - Filesystem Monitoring: Watchdog
  - Image Manipulation: Pillow
  - Environment: WSL2 (Ubuntu 22.04) or a dedicated Linux server

## Project Structure

```video_upscaler/
│
├── .gitignore
├── input/                  # Drop source videos here
├── logs/                   # Server operation logs are stored here
├── output/                 # Final upscaled videos are saved here
├── queue.txt               # The file-based job queue
├── Real-ESRGAN/            # The cloned REAL-ESRGAN repository (dependency)
├── README.md
├── requirements.txt        # Python dependencies
└── upscale_server.py       # The main orchestration script
```

## Setup and Installation
### Prerequisites

  - An NVIDIA GPU with CUDA support (12GB+ VRAM recommended for 4K).
  - A configured WSL2/Ubuntu 22.04 environment or a dedicated Ubuntu server.
  - Correctly installed NVIDIA drivers and the CUDA Toolkit.
  - FFmpeg installed (can be done in terminal).
    
### Installation Steps 
Note: The script assumes this directory is named Real-ESRGAN and resides in the project root.
  1. Clone the Repository
  
    git clone [https://github.com/BennettH-git/Video-Upscaling-Server-WSL2.git](https://github.com/BennettH-git/Video-Upscaling-Server-WSL2.git)
    
    cd BennettH-git/Video-Upscaling-Server-WSL2
    
  2. Set up Python Environment: It is highly recommended to use a virtual environment.
  
    python3 -m venv venv
    
    source venv/bin/activate

  3. Install Python Dependencies: The requirements.txt file contains all necessary packages.
  
    pip install -r requirements.txt

  4. Download REAL-ESRGAN: Clone the model repository, which contains the inference script and pre-trained models.
    
    git clone [https://github.com/xinntao/Real-ESRGAN.git](https://github.com/xinntao/Real-ESRGAN.git)


### Usage

To start the server, simply run the main Python script from the project's root directory:

    python upscale_server.py

The server will initialize and begin monitoring the input/ directory.

To process a video, place a video file into the input/ folder. The server will automatically detect it based upon file type, add it to the queue, and begin processing. The final, upscaled video will appear in the output/ directory when the job is complete.

To shut down the server press Ctrl+C in the terminal where it is running.


PS, this will likely make the room quite warm while upscale videos continuously if you dont crack a window or something.
