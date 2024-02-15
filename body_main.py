import cv2
import json
import numpy as np
import matplotlib.pyplot as plt
import mediapipe as mp
from tqdm import tqdm
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D

def body_detection(video_path='input_your_mp4'):
    """
    Detects body landmarks from a video using MediaPipe Pose.

    Parameters:
        video_path (str): Path to the input video file.

    Saves the detected landmark coordinates as JSON files in the 'pose_landmark' directory.
    """
    mp_drawing = mp.solutions.drawing_utils
    mp_pose = mp.solutions.pose

    cap = cv2.VideoCapture(video_path)

    landmarks = {
        0: 'nose',
        2: 'left_eye',
        5: 'right_eye',
        7: 'left_ear',
        8: 'right_ear',
        11: 'left_shoulder',
        12: 'right_shoulder'
    }

    landmark_dict = {name: [] for name in landmarks.values()}

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        for frame_no in tqdm(range(total_frames), desc="Processing frames"):
            success, image = cap.read()

            if success:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image.flags.writeable = False
                results = pose.process(image)

                for name in landmarks.values():
                    if results.pose_landmarks:
                        for idx, landmark in enumerate(results.pose_landmarks.landmark):
                            if idx in landmarks and landmarks[idx] == name:
                                landmark_dict[name].append([landmark.x, landmark.y, landmark.z])
                                break
                        else:
                            landmark_dict[name].append([None, None, None])
                    else:
                        landmark_dict[name].append([None, None, None])

                image.flags.writeable = True
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

                if results.pose_landmarks:
                    for idx, landmark in enumerate(results.pose_landmarks.landmark):
                        if idx in landmarks:
                            image_hight, image_width, _ = image.shape
                            landmark_px = mp_drawing._normalized_to_pixel_coordinates(landmark.x, landmark.y, image_width, image_hight)
                            cv2.circle(image, landmark_px, 3, (255, 0, 0), 2)
                            landmark_coords = f"{landmarks[idx]}: x: {landmark.x:.2f}, y: {landmark.y:.2f}, z: {landmark.z:.2f}"
                            cv2.putText(image, landmark_coords, (10, 30 + 20 * idx),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                cv2.putText(image, f'Frame: {frame_no}', (10, 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                cv2.imshow('MediaPipe Pose', image)
                if cv2.waitKey(5) & 0xFF == 27:
                    break
            else:
                break

    cap.release()
    cv2.destroyAllWindows()

    for name, coordinates in landmark_dict.items():
        with open(f'pose_landmark/{name}.json', 'w') as f:
            json.dump(coordinates, f)

def body_plane_animation():
    """
    Creates a 3D animation of body planes based on the detected shoulder landmarks.

    Reads the shoulder landmark coordinates from JSON files and generates the animation.
    Saves the computed body planes and the origin coordinates as JSON files.
    """
    with open('pose_landmark/left_shoulder.json') as f:
        left_shoulder = json.load(f)
    with open('pose_landmark/right_shoulder.json') as f:
        right_shoulder = json.load(f)

    center = (np.array(left_shoulder[0]) + np.array(right_shoulder[0])) / 2
    origin = np.copy(center)
    origin[1] += (1 - origin[1]) / 2
    planes = []
    frames = []
    left_shoulder_filtered = []
    right_shoulder_filtered = []

    for i, (l, r) in enumerate(zip(left_shoulder, right_shoulder)):
        frames.append(i)
        if None in l or None in r:
            planes.append(None)
            left_shoulder_filtered.append(None)
            right_shoulder_filtered.append(None)
            continue
        l = np.array(l)
        r = np.array(r)
        plane = np.cross(l - origin, r - origin)
        plane /= np.linalg.norm(plane)
        planes.append(plane.tolist())
        left_shoulder_filtered.append(l)
        right_shoulder_filtered.append(r)

    with open('pose_landmark/body_normal_line.json', 'w') as f:
        json.dump(planes, f)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    def update(i):
        print(f"Processing frame {i} / {len(planes)}")  # Print progress
        ax.cla()
        if planes[i] is not None and left_shoulder_filtered[i] is not None and right_shoulder_filtered[i] is not None:
            ax.quiver(origin[0], origin[1], origin[2], planes[i][0], planes[i][1], planes[i][2])
            ax.plot([origin[0], left_shoulder_filtered[i][0]], [origin[1], left_shoulder_filtered[i][1]],
                    [origin[2], left_shoulder_filtered[i][2]], 'r-')
            ax.plot([origin[0], right_shoulder_filtered[i][0]], [origin[1], right_shoulder_filtered[i][1]],
                    [origin[2], right_shoulder_filtered[i][2]], 'r-')
            ax.plot([left_shoulder_filtered[i][0], right_shoulder_filtered[i][0]],
                    [left_shoulder_filtered[i][1], right_shoulder_filtered[i][1]],
                    [left_shoulder_filtered[i][2], right_shoulder_filtered[i][2]], 'r-')
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

    with open('pose_landmark/origin.json', 'w') as f:
        json.dump(origin.tolist(), f)

def body_angle():
    """
    Calculates body angles from the detected body planes.

    Reads the computed body planes from a JSON file and calculates the front-back and right-left angles.
    Saves the calculated angles as JSON files.
    """
    with open('pose_landmark/body_normal_line.json', 'r') as f:
        planes = json.load(f)

    body_angle_front_back = []
    body_angle_right_left = []

    for plane in planes:
        if plane is not None:
            front_back_angle = calculate_angle(plane, [0, 0, 1])

            body_angle_front_back.append(front_back_angle)
            left_right_angle = calculate_angle(plane, [1, 0, 0])
            body_angle_right_left.append(left_right_angle)
        else:
            body_angle_front_back.append(None)
            body_angle_right_left.append(None)

    with open('pose_landmark/body_angle_front_back.json', 'w') as f:
        json.dump(body_angle_front_back, f)

    with open('pose_landmark/body_angle_right_left.json', 'w') as f:
        json.dump(body_angle_right_left, f)

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
    body_detection()
    body_plane_animation()
    body_angle()
