# main.py

import subprocess

subprocess.run(["python", "body_detection.py"])
subprocess.run(["python", "body_plane_animation.py"])
subprocess.run(["python", "body_angle.py"])

subprocess.run(["python", "face_detection.py"])
subprocess.run(["python", "face_plane_animation.py"])
subprocess.run(["python", "face_angle.py"])


subprocess.run(["python", "track_shoulder.py"])


