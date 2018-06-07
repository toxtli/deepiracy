# Deepiracy

Deepary is a tool that is able to find a source video on a taget video. It is capable to detect subsequences of the source video, even if it is highly distorted. You can read a complete explanation of how this work by reading [this article](https://medium.com/hci-wvu/piracy-detection-using-longest-common-subsequence-and-neural-networks-a6f689a541a6).

## Installation

Run

> pip install -r requirements.txt

## Quickstart

Parameters:

python app.py [from_frame_number] [to_frame_number] [how_many_frames] [video_path_1_or_url] [video_path_2_or_url]

Example how to run it with files

> python app.py 1 1 -1 video1.mp4 video1.mp4

Example how to run it with youtube URLs

> python app.py 1 1 -1 https://www.youtube.com/watch?v=E5K_Ug0Gq0Y https://www.youtube.com/watch?v=E5K_Ug0Gq0Y

Example how to run the real-time detection from webcam

> python app.py 1 1 -1 video1.mp4 0

The required files can be downloaded from here:

https://drive.google.com/drive/folders/1BPR6j-3xc0NnlbmO96LD55tRV7e07Ynp?usp=sharing

The results can be found here:

https://drive.google.com/open?id=1iyquDYv1o48mA_ymI7AEjrlZZtqXOOz1
