# Baseball-Vision
A baseball tracking system built with Python and OpenCV and SciPy to analyze video and estimate exit velocity, launch angle and distance traveled using computer vision and projectile motion modeling.


**Dependencies**
The following are required to run this project:

Python 3.9 or later
OpenCV (opencv-python) — Video processing and computer vision.
NumPy (numpy) — Numerical computations and array operations.
SciPy (scipy) — Numerical integration for projectile motion calculations.
A video file containing a baseball in motion shot at low shutter speeds to reduce motion blur(all demo videos were made using an iphone 15 and the Black Magic Camera app at 1/4000 or 1/8000 shutter speed)
An estimate of the camera's horizontal field of view (HFOV).
The program also uses Python's built-in math and random modules.


**Installing**
Download the repository from GitHub.
Install the required dependencies using:

pip install opencv-python numpy scipy

Place your video in an accessible directory.
Open the Python script and update the video_file variable to the path of your video file.
Verify that the HFOV constant matches your camera's horizontal field of view.


**Executing the Program**
Open a terminal in the project directory.
Run the main Python script:

Baseball_Tracking_Algorithm.py
When prompted, click approximately on the center of the baseball, avoiding the red seams, to select its color characteristics.
Use the space bar to pause or resume video playback, press q to proceed to the tracking stage.
Click the top-left and bottom-right corners of the region where the baseball will travel, as prompted.
Allow the program to process the video and track the baseball across consecutive frames.
The estimated exit velocity, launch angle, and distance will be displayed by the program when.
