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
0102-0229, KIST_Creamo<br/>
<br/><br/>
<h1>:gear: Environment Setting</h1>
<hr/>
<ul>
  <li><b>Python: </b> 3.11.1</li>
  <li><b>IDE: </b> VSCode, Google Colab</li>
</ul>
<br/><br/>
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
  RUN main.py
</ul>


<br/>
<h1> ➕ Details </h1>
<hr/>
<b>preprocess_the_video.ipynb</b>
<br/>
: Enter the path of the video file in input_your_mp4
<br/>
: crop to focus child
<br/>
: 30fps(original) to 10fps (if needed)
<br/><br/>

    ffmpeg -i input.mp4 -r 10 output.mp4

 <br/> <br/>
<b>body_detection.py</b> <br/>
: ectract each landmark's (x,y,z) in video

<b>body_plane_animation.py</b> <br/>
: extract normal line in body and visualization

<b>body_angle.py</b> <br/>
: extract the angle between the normal vector and the z-axis
: extract the angle between the normal vector and the x-axis

<b>face_detection.py</b> <br/>
: ectract each landmark's (x,y,z) in video

<b>face_plane_animation.py</b> <br/>
: extract normal line in face and visualization

<b>face_angle.py</b> <br/>
: extract the angle between the normal vector and the z-axis(ex. nod)
: extract the angle between the normal vector and the x-axis(ex. shake head)

<b>shoulder_track.py</b> <br/>
: Track the midpoint between shoulder_right and shoulder_left
