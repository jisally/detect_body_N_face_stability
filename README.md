<h1> Detect body and face stability </h1>
<hr/>
Postural stability assessment through changes in the angular displacement of the body and face planes
<br/><br/>
<h1>:calendar: When? </h1>
<hr/>
0102~0229, KIST_Creamo
<br/><br/>
<h1>:gear: Environment Setting</h1>
<hr/>
<ul>
  <li><b>Python: </b> 3.11.1</li>
  <li><b>IDE: </b> VSCode</li>
</ul>
<br/><br/>
<h1>▶ How to RUN? </h1>
<hr/>
[Open in Colab](https://colab.research.google.com/github/jisally/detect_body_N_face_stability/blob/master//content/drive/MyDrive/KIST_CREAMO/preprocess_the_video.ipynb)
<br/>
RUN main.py
<br/><br/>
<h1> ➕ Details </h1>
<hr/>
preprocess the video
<br/>
: crop to focus child
<br/>
: 30fps(original) to 10fps
<br/>

    ffmpeg -i input.mp4 -r 10 output.mp4

 <br/>
Enter the path of the video file in input_your_mp4
 <br/> <br/>
body_detection.py
: ectract each landmark's (x,y,z) in video

body_plane_animation.py
: extract normal line in body and visualization

body_angle.py
: extract the angle between the normal vector and the z-axis
: extract the angle between the normal vector and the x-axis

face_detection.py
: ectract each landmark's (x,y,z) in video

face_plane_animation.py
: extract normal line in face and visualization

face_angle.py
: extract the angle between the normal vector and the z-axis(ex. nod)
: extract the angle between the normal vector and the x-axis(ex. shake head)

shoulder_track.py
: Track the midpoint between shoulder_right and shoulder_left
