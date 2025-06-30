WINDOWS:- 
Steps to execute on your Local machine: (Install Conda and Python)
1. bash this command :- conda env create -f environment.yml
2. bash :- conda activate gesture_env
3. Run the python in the terminal :- python main.py

MacOS :- 
Installation Instructions for macOS
Step 1: Create a Conda Environment
conda create --name gesture_env python=3.13
conda activate gesture_env

Step 2: Install Dependencies
conda install -c conda-forge opencv
conda install -c conda-forge dlib
Step 3: Install Other Required Packages
pip install pyautogui
Step 4: Run Your Script
Now, you should be able to run your script in the activated environment:
python /path/to/your/script/main.py
