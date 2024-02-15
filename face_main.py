import cv2
import json
import numpy as np
import mediapipe as mp
from tqdm import tqdm
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation


def face_detection(video_path='input_your_mp4'):
    """
    Detects facial landmarks from a video using MediaPipe FaceMesh.

    Parameters:
        video_path (str): Path to the input video file.

    Saves the detected facial landmark coordinates as JSON files.
    """
    mp_face_mesh = mp.solutions.face_mesh
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    target_landmarks = [10, 150, 379]
    landmark_names = ['top_face', 'right_face', 'left_face']
    landmark_map = dict(zip(target_landmarks, landmark_names))
    landmark_data = {landmark_map[i]: [] for i in target_landmarks}

    with mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            min_detection_confidence=0.5) as face_mesh:
        for _ in tqdm(range(total_frames), desc="Processing video"):
            success, image = cap.read()
            if not success:
                break
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            results = face_mesh.process(image)

            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    for i in target_landmarks:
                        landmark = face_landmarks.landmark[i]
                        landmark_data[landmark_map[i]].append([landmark.x, landmark.y, landmark.z])
                        x, y = int(landmark.x * image.shape[1]), int(landmark.y * image.shape[0])
                        cv2.circle(image, (x, y), 5, (0, 0, 255), -1)
            else:
                for i in target_landmarks:
                    landmark_data[landmark_map[i]].append([None, None, None])

            cv2.imshow('MediaPipe FaceMesh', image)
            if cv2.waitKey(5) & 0xFF == 27:
                break

    cap.release()
    cv2.destroyAllWindows()

    for i in target_landmarks:
        with open(f'face_landmark/{landmark_map[i]}.json', 'w') as f:
            json.dump(landmark_data[landmark_map[i]], f)


def face_plane_animation():
    """
    Generates a 3D animation of facial planes based on the detected facial landmarks.

    Reads the facial landmark coordinates from JSON files and generates the animation.
    Saves the computed facial planes as JSON files.
    """
    with open('face_landmark/top_face.json') as f:
        top_face = json.load(f)
    with open('face_landmark/right_face.json') as f:
        right_face = json.load(f)
    with open('face_landmark/left_face.json') as f:
        left_face = json.load(f)

    planes = []
    frames = []
    top_face_filtered = []
    right_face_filtered = []
    left_face_filtered = []

    for i, (t, r, l) in enumerate(zip(top_face, right_face, left_face)):
        if None in t or None in r or None in l:
            planes.append(None)
            frames.append(i)
            top_face_filtered.append(None)
            right_face_filtered.append(None)
            left_face_filtered.append(None)
            continue

        t = np.array(t)
        r = np.array(r)
        l = np.array(l)
        plane = np.cross(t - r, l - r)
        plane = -plane
        plane /= np.linalg.norm(plane)

        planes.append(plane.tolist())
        frames.append(i)
        top_face_filtered.append(t)
        right_face_filtered.append(r)
        left_face_filtered.append(l)

    with open('face_landmark/face_normal_line.json', 'w') as f:
        json.dump(planes, f)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    def update(i):
        ax.cla()

        if planes[i] is not None:
            ax.quiver(right_face_filtered[i][0], right_face_filtered[i][1], right_face_filtered[i][2],
                      planes[i][0], planes[i][1], planes[i][2])
            ax.plot([right_face_filtered[i][0], top_face_filtered[i][0]],
                    [right_face_filtered[i][1], top_face_filtered[i][1]],
                    [right_face_filtered[i][2], top_face_filtered[i][2]], 'r-')
            ax.plot([right_face_filtered[i][0], left_face_filtered[i][0]],
                    [right_face_filtered[i][1], left_face_filtered[i][1]],
                    [right_face_filtered[i][2], left_face_filtered[i][2]], 'r-')
            ax.plot([top_face_filtered[i][0], left_face_filtered[i][0]],
                    [top_face_filtered[i][1], left_face_filtered[i][1]],
                    [top_face_filtered[i][2], left_face_filtered[i][2]], 'r-')

        ax.quiver(0, 0, 0, 1, 0, 0, color='b', alpha=0.5, label='x')
        ax.quiver(0, 0, 0, 0, 1, 0, color='g', alpha=0.5, label='y')
        ax.quiver(0, 0, 0, 0, 0, 1, color='r', alpha=0.5, label='z')

        ax.set_xlim([0, 1])
        ax.set_ylim([1, 0])
        ax.set_zlim([0, -1])
        ax.legend()
        ax.set_title(f"Frame: {frames[i]}")

    ani = FuncAnimation(fig, update, frames=len(planes), interval=200, repeat=False)
    plt.show()


def face_angle():
    """
    Calculates facial angles from the detected facial planes.

    Reads the computed facial planes from a JSON file and calculates the front-back and right-left angles.
    Saves the calculated angles as JSON files.
    """
    with open('face_landmark/face_normal_line.json', 'r') as f:
        planes = json.load(f)

    face_angle_front_back = []
    face_angle_right_left = []

    for plane in planes:
        if plane is None:
            face_angle_front_back.append(None)
            face_angle_right_left.append(None)
        else:
            front_back_angle = calculate_angle(plane, [0, 0, 1])
            face_angle_front_back.append(front_back_angle)

            left_right_angle = calculate_angle(plane, [1, 0, 0])
            face_angle_right_left.append(left_right_angle)

    with open('face_landmark/face_angle_front_back.json', 'w') as f:
        json.dump(face_angle_front_back, f)
    with open('face_landmark/face_angle_right_left.json', 'w') as f:
        json.dump(face_angle_right_left, f)


def calculate_angle(a, b):
    """
    Calculates the angle between two vectors.

    Parameters:
        a (list): First vector.
        b (list): Second vector.

    Returns:
        float: Angle between the two vectors in degrees.
    """
    dot_product = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    cos_theta = dot_product / (norm_a * norm_b)
    theta = np.arccos(cos_theta)
    return np.degrees(theta)


if __name__ == "__main__":
    video_path = 'input_your_mp4'
    face_detection(video_path)
    face_plane_animation()
    face_angle()
