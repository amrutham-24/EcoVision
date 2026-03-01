# EcoVision: Sustainable Smart Surveillance with Motion‑Based Video Optimization

# Problem Statemen
Traditional CCTV systems record continuously, storing large amounts of static footage with no useful activity. This leads to:
Excessive storage consumption
Increased energy usage
Faster hardware wear
Difficulty in reviewing important events
Most stored footage contains no meaningful motion.

# Proposed solution
Our system uses computer vision and AI‑based background subtraction (MOG2) to detect motion and automatically trim videos.
Key functions:
Detects motion using OpenCV MOG2 algorithm
Extracts only motion‑containing video segments
Removes unnecessary static footage
Stores processed videos in a separate folder
Generates logs with timestamps and storage saved
Deletes original files safely after processing
This significantly reduces storage usage while preserving important events.

# Technology Stack
Programming Language
-Python
Libraries
-OpenCV (motion detection, video processing)
-NumPy (array processing)
-OS, JSON (file handling and logging)

# Setup Instructions
1. Install Python (3.8 or above)

2. Install required libraries:
pip install opencv-python numpy

3. Create folder structure:
project/
│
├── main.py
├── cctv/        (input videos)
├── processed/   (output videos)

4. Run the program:
python main.py

5.Output will be saved in:
processed folder → trimmed videos
log file → motion timestamps and storage saved

# Team Members
Amrutha M
Sanal Sivakumar
Megha Suresh
Agnivesh S


 
