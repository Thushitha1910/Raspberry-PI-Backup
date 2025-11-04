import subprocess
import time
import os

# --- Configuration ---
IMAGE_FILENAME = "fswebcam_test.jpg"
WEBCAM_DEVICE = "/dev/video0"
RESOLUTION = "1280x720"

def test_webcam_fswebcam(filename):
    """
    Captures a single image using the fswebcam command.
    """
    print(f"Attempting to capture image from {WEBCAM_DEVICE}...")
    
    # The command to run
    command = [
        "fswebcam",
        "-d", WEBCAM_DEVICE,
        "-r", RESOLUTION,
        "--no-banner",
        filename
    ]
    
    try:
        # Run the command
        subprocess.run(command, check=True)
        
        # Check if the file was created
        if os.path.exists(filename):
            print(f"\n✅ Success! Image saved as '{filename}'")
        else:
            print(f"\n❌ Error: Command ran, but file '{filename}' was not created.")
            
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Error capturing image: {e}")
        print("Please check that 'fswebcam' is installed (sudo apt-get install fswebcam)")
        print(f"and that the camera is correctly connected at {WEBCAM_DEVICE}.")
        
    except FileNotFoundError:
        print("\n❌ Error: 'fswebcam' command not found.")
        print("Please install it first using: sudo apt-get install fswebcam")

# --- Main execution ---
if __name__ == "__main__":
    test_webcam_fswebcam(IMAGE_FILENAME)