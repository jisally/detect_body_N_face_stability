<Detect body&face stability>

0102~0229, KIST_Creamo 

preprocess the video
: crop to focus child
: 30fps(original) to 10fps

body_detection.py
: ectract each landmark's (x,y,z) in video

body_plane_animation.py
: extract normal line in body and visualization

&

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
