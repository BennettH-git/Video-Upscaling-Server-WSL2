# AI Video Upscaling Server (Public Version 2.2 - Resumable)

This server provides a fully automated pipeline for upscaling video files using the REAL-ESRGAN deep learning model. It is designed for stable, hands-off, and fault-tolerant operation within a WSL2/Ubuntu environment.

This project has undergone significant iterative development to solve performance bottlenecks and environmental instabilities. The final architecture is a robust system that prioritizes stability, performance, and recoverability.
## Key Features

  - Automated Job Ingestion & Queuing: Uses watchdog to monitor an input/ directory and automatically adds new video files to a persistent queue.txt file for sequential processing.
  
  - Fault Tolerance & Job Resumption: This is the most critical feature. The server creates a persistent status.json file for each job. If the server is shut down or crashes, it will automatically scan the input directory upon restart and add any unfinished jobs back to the queue, seamlessly resuming from the last completed batch. This can and will save an incredible amount of reprocessing time.
  
  - RAM Disk: To protect SSD longevity and maximize I/O performance, the server automatically creates a RAM disk at the start of each job for all temporary frame processing and destroys it upon completion, freeing all RAM when idle.
  
  - RAM Disk Spillover: For long jobs, completed intermediate video clips are automatically moved from the RAM disk to a persistent folder on the physical disk to prevent the RAM disk from overflowing.
  
  - VRAM-Aware Manual Tiling: The server includes a robust, automated pre-processing step. For high-resolution frames (wider than 1920px), it uses the Pillow library to slice them into smaller tiles before upscaling and stitches them back together afterward.

  - CPU encoding Fallback: The server attempts to use the h264_nvenc GPU-accelerated video encoder. If this fails due to unavoidable driver instability in WSL2, it will automatically fall back to the reliable (but slower) libx264 CPU encoder to ensure the job always completes.

  - ETA Calculation: The server processes a small, 30-frame initial batch to provide a quick and reasonably accurate Estimated Time of Arrival for the entire job. This batch size can be adjusted at the start of the code for a more accurate, but less rapid, ETA.
  
## Project Structure

```
video-upscaling-server/
│
├── .git/
├── .gitignore
├── input/                  # Drop source videos here for automatic processing.
├── logs/                   # Server operation logs are stored here.
├── output/                 # Final upscaled videos & persistent job directories.
├── queue.txt               # The file-based job queue.
├── Real-ESRGAN/            # Cloned dependency, contains the AI model code.
├── README.md
├── requirements.txt        # Python dependencies for a reproducible environment.
├── upscale_server.py       # The main script
└── venv/                   
```

## Technology Stack & Environment

  - Operating System: Ubuntu 24.04 LTS (Recommended, I am running on WSL2)
  - NVIDIA Driver: Studio Driver (v581.29+)
  - Python Version: 3.12+
  - PyTorch CUDA Toolkit Version: 12.1. This is the most critical environmental parameter. To ensure compatibility with Ubuntu's pre-compiled FFmpeg and achieve stable nvenc encoding, the Python environment must use the PyTorch libraries built for the CUDA 12.1 toolkit.
  - Core Python Dependencies: torch==2.5.1+cu121, torchvision==0.20.1+cu121, torchaudio==2.5.1+cu121, opencv-python==4.12.0.88, watchdog==6.0.0, Pillow==11.3.0. A complete list is in requirements.txt.


    
## Usage
  1. Activate the Virtual Environment:
    
    source venv/bin/activate
    
  2. Start the Server:

    python upscale_server.py

  3. Process Videos: Simply copy or move your video files (.mp4, .mkv, .mov, etc.) into the input/ directory. The server will automatically detect them, add them to the queue, and begin processing.

  4. Find Output: Completed upscaled videos will appear in the output/ directory. For each job, a corresponding sub-directory (e.g., output/my_video/) is created to hold the persistent status.json file and intermediate clips. By default, this directory is deleted upon successful completion to save space.
