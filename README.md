<h1> Detect body and face stability </h1>
<hr/>
Assessment of postural stability using Google MediaPipe by analyzing changes in the angular displacement of the body and facial planes
<br/><br/>
<h1>💻 Project Introduction </h1>
<hr/>

Rotation of the body forward/backward: Angle between the normal vector of the body plane and the z-axis.<br/>
Rotation of the body left/right: Angle between the normal vector of the body plane and the x-axis.<br/>
<br/>
Rotation of the head forward/backward: Angle between the normal vector of the head plane and the z-axis.<br/>
Rotation of the head left/right: Angle between the normal vector of the head plane and the x-axis.<br/>

<br/><br/>
<h1>:calendar: When? </h1>
<hr/>
240102-240229, KIST_Creamo<br/>
<br/><br/>
<h1>🙂 Members </h1>
<hr/>

|Members|
|------|
|Seungyeon JI|

<h1>:gear: Environment Setting</h1>
<hr/>
<ul>
  <li><b>Python: </b> 3.11.8</li>
  The project utilizes a virtual environment in Visual Studio Code to run Python 3.11.8
<br/><br/>

    python3.11.8 -m venv 'your_env_name'
    
<br/>

    'your_env_name'\Scripts\activate

<br/>
  <li><b>IDE: </b> VSCode, Google Colab</li>
</ul>
<br/><br/>
<h1>✅ Version</h1>
<hr/>
Requirements.txt<br/><br/>


    pip install -r requirements.txt

<br/>

    opencv-python==4.9.0.80
    mediapipe==0.10.9
    tqdm==4.65.0
    numpy==1.23.5
    matplotlib==3.7.1

<br/>

<h1>▶ How to RUN? </h1>
<hr/>

<ul>
  <li><b>PreProcessing</b><br/></li>
  <br/>
  <a target="_blank" href="https://colab.research.google.com/github/jisally/detect_body_N_face_stability/blob/main/preprocess_the_video.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>
  <br/><br/><br/>
  <li><b>Calculate the Angle</b><br/></li>
  <br/>
  Calculate the body angle


    python body_main.py

<br/>  
  Calculate the face angle


    python face_main.py


<br/>  
  Track standing and sitting


    python track_shoulder.py
    
</ul>


<br/>
<h1>🏆 Results </h1>
<hr/>

```
ADDI_POSE
|   .gitignore
|   body_main.py
|   face_main.py
|   preprocess_the_video.ipynb
|   README.md
|   requirements.txt
|   track_shoulder.py
|   
+---body_landmark
|       body_angle_front_back.json
|       body_angle_right_left.json
|       body_normal_line.json
|       left_ear.json
|       left_eye.json
|       left_shoulder.json
|       nose.json
|       origin.json
|       right_ear.json
|       right_eye.json
|       right_shoulder.json
|       
+---face_landmark
|       face_angle_front_back.json
|       face_angle_right_left.json
|       face_normal_line.json
|       left_face.json
|       right_face.json
|       top_face.json
|       
\---video
```
<br/>

<h1> 📄 Docstring </h1>
<hr/>

<details>
<summary><code>body_detection()</code></summary>

Detects body landmarks from a video using MediaPipe Pose.

### Parameters:
- `video_path` (str): Path to the input video file.

### Notes:
- Saves the detected landmark coordinates as JSON files in the 'body_landmark' directory.

</details>

<details>
<summary><code>body_plane_animation()</code></summary>

Creates a 3D animation of body planes based on the detected shoulder landmarks.

### Notes:
- Reads the shoulder landmark coordinates from JSON files and generates the animation.
- Saves the computed body planes and the origin coordinates as JSON files.

</details>

<details>
<summary><code>body_angle()</code></summary>

Calculates body angles from the detected body planes.

### Notes:
- Reads the computed body planes from a JSON file and calculates the front-back and right-left angles.
- Saves the calculated angles as JSON files.

</details>

<details>
<summary><code>calculate_angle(a, b)</code></summary>

Calculates the angle between two vectors.

### Parameters:
- `a` (list): First vector.
- `b` (list): Second vector.

### Returns:
- `float`: Angle between the two vectors in degrees.

</details>

<details>
<summary><code>face_detection()</code></summary>

Detects facial landmarks from a video using MediaPipe FaceMesh.

### Parameters:
- `video_path` (str): Path to the input video file.

### Notes:
- Saves the detected facial landmark coordinates as JSON files in the 'face_landmark' directory.

</details>

<details>
<summary><code>face_plane_animation()</code></summary>

Generates a 3D animation of facial planes based on the detected facial landmarks.

### Notes:
- Reads the facial landmark coordinates from JSON files and generates the animation.
- Saves the computed facial planes as JSON files.

</details>

<details>
<summary><code>face_angle()</code></summary>

Calculates facial angles from the detected facial planes.

### Notes:
- Reads the computed facial planes from a JSON file and calculates the front-back and right-left angles.
- Saves the calculated angles as JSON files.

</details>

<details>
<summary><code>calculate_angle(a, b)</code></summary>

Calculates the angle between two vectors.

### Parameters:
- `a` (list): First vector.
- `b` (list): Second vector.

### Returns:
- `float`: Angle between the two vectors in degrees.

</details>
<br/>


<h1> ➕ Details </h1>
<hr/>
  
`preprocess_the_video.ipynb`
<br/>
: Enter the path of the video file in input_your_mp4
<br/>
: crop to focus child
<br/>
: 30fps(original) to 10fps (if needed)
<br/><br/>

    ffmpeg -i input.mp4 -r 10 output.mp4

 <br/>
 
`body_main.py`
<br/>
: ectract each landmark's (x,y,z) in video
<br/>
: Enter the path of the video file in input_your_mp4
<br/>

: extract normal line in body and visualization<br/>

: extract the angle between the normal vector and the z-axis<br/>
: extract the angle between the normal vector and the x-axis

`face_main.py`<br/>
: ectract each landmark's (x,y,z) in video
<br/>
: Enter the path of the video file in input_your_mp4
<br/>

: extract normal line in face and visualization

: extract the angle between the normal vector and the z-axis(ex. nod)<br/>
: extract the angle between the normal vector and the x-axis(ex. shake head)
<br/>

`shoulder_track.py`<br/>
: Track the midpoint between shoulder_right and shoulder_left
