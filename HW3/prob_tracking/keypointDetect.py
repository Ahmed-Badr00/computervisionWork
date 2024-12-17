import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

def detect_corners(image, threshold=1e-4, k=0.04, window_size=5):
    """Detect corners using the Harris corner detection method."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0

    # Compute image gradients
    Ix = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    Iy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)

    # Compute second moment matrix components
    Ixx = Ix * Ix
    Iyy = Iy * Iy
    Ixy = Ix * Iy

    # Smooth components with Gaussian filter
    Ixx = cv2.GaussianBlur(Ixx, (3, 3), 1)
    Iyy = cv2.GaussianBlur(Iyy, (3, 3), 1)
    Ixy = cv2.GaussianBlur(Ixy, (3, 3), 1)

    # Compute Harris response
    det_M = Ixx * Iyy - Ixy ** 2
    trace_M = Ixx + Iyy
    R = det_M - k * (trace_M ** 2)

    # Apply thresholding
    R[R < threshold * R.max()] = 0

    # Perform non-maximal suppression
    corners = np.zeros_like(R)
    for y in range(window_size // 2, R.shape[0] - window_size // 2):
        for x in range(window_size // 2, R.shape[1] - window_size // 2):
            local_window = R[y - window_size // 2:y + window_size // 2 + 1,
                             x - window_size // 2:x + window_size // 2 + 1]
            if R[y, x] == local_window.max():
                corners[y, x] = R[y, x]

    keypoints = np.argwhere(corners > 0)
    return keypoints

def lucas_kanade_tracker(I1, I2, keypoints, window_size=15, max_iterations=10, epsilon=1e-3):
    """Track keypoints using Kanade-Lucas-Tomasi optical flow."""
    I1 = cv2.cvtColor(I1, cv2.COLOR_BGR2GRAY).astype(np.float32)
    I2 = cv2.cvtColor(I2, cv2.COLOR_BGR2GRAY).astype(np.float32)

    Ix = cv2.Sobel(I1, cv2.CV_64F, 1, 0, ksize=3)
    Iy = cv2.Sobel(I1, cv2.CV_64F, 0, 1, ksize=3)
    It = I2 - I1

    tracked_points = []
    half_window = window_size // 2

    for y, x in keypoints:
        # Initialize the displacement
        u, v = 0.0, 0.0

        for _ in range(max_iterations):
            # Extract window of gradients and temporal differences
            x_min, x_max = int(x + u - half_window), int(x + u + half_window + 1)
            y_min, y_max = int(y + v - half_window), int(y + v + half_window + 1)

            if (x_min < 0 or x_max > I1.shape[1] or y_min < 0 or y_max > I1.shape[0]):
                break  # Skip points near the boundary

            Ix_window = Ix[y_min:y_max, x_min:x_max].flatten()
            Iy_window = Iy[y_min:y_max, x_min:x_max].flatten()
            It_window = It[y_min:y_max, x_min:x_max].flatten()

            # Formulate the system of equations
            A = np.stack([Ix_window, Iy_window], axis=1)
            b = -It_window

            try:
                # Solve for flow using least-squares
                flow = np.linalg.lstsq(A, b, rcond=None)[0]
                du, dv = flow
            except np.linalg.LinAlgError:
                break  # Singular matrix, skip this point

            # Update displacement
            u += du
            v += dv

            # Check for convergence
            if np.sqrt(du**2 + dv**2) < epsilon:
                break

        # Store the final displacement
        new_x, new_y = x + u, y + v
        if 0 <= new_x < I1.shape[1] and 0 <= new_y < I1.shape[0]:
            tracked_points.append((new_y, new_x))

    return np.array(tracked_points)

def save_frame_with_keypoints(frame, keypoints, output_path, color=(0, 255, 0)):
    """Save a frame with keypoints overlaid."""
    output_frame = frame.copy()
    for y, x in keypoints:
        cv2.circle(output_frame, (int(x), int(y)), 3, color, -1)
    cv2.imwrite(output_path, output_frame)

def plot_and_save_trajectories(frames, initial_keypoints, trajectories, points_out_of_frame, output_path):
    """Plot and save keypoint trajectories."""
    plt.figure(figsize=(10, 10))
    plt.imshow(cv2.cvtColor(frames[0], cv2.COLOR_BGR2RGB))
    plt.title("Keypoint Trajectories")
    plt.axis("off")

    # Plot initial keypoints
    for point in initial_keypoints:
        y, x = point
        plt.plot(x, y, 'go', markersize=5)

    # Plot trajectories for 20 random keypoints
    for traj in trajectories:
        plt.plot(traj[:, 1], traj[:, 0], 'r-')

    # Plot points that moved out of the frame
    for point in points_out_of_frame:
        y, x = point
        plt.plot(x, y, 'bo', markersize=5)

    # Save the figure
    plt.savefig(output_path)
    plt.close()

def save_trajectories(trajectories, output_path):
    """Save trajectories to a CSV file."""
    with open(output_path, 'w') as f:
        f.write("keypoint_id,frame_number,x,y\n")
        for i, traj in enumerate(trajectories):
            for frame_idx, (y, x) in enumerate(traj):
                f.write(f"{i},{frame_idx},{x},{y}\n")

def main():
    # Directory to save results
    results_dir = "tracking_results"
    os.makedirs(results_dir, exist_ok=True)

    # Load the sequence of frames
    num_frames = 50  # Replace with the actual number of frames
    frames = [cv2.imread(f"images/hotel.seq{i}.png") for i in range(num_frames)]

    # Detect corners in the first frame
    keypoints = detect_corners(frames[0])

    # Save the first frame with keypoints
    save_frame_with_keypoints(frames[0], keypoints, os.path.join(results_dir, "keypoints_first_frame.jpg"))

    # Initialize trajectories
    trajectories = [np.array([[y, x]]) for y, x in keypoints]

    # Track features over the sequence
    for i in range(1, len(frames)):
        new_keypoints = lucas_kanade_tracker(frames[i - 1], frames[i], keypoints)

        # Save the frame with tracked keypoints
        save_frame_with_keypoints(frames[i], new_keypoints, os.path.join(results_dir, f"keypoints_frame_{i}.jpg"), color=(0, 0, 255))

        # Update trajectories
        for idx, point in enumerate(new_keypoints):
            trajectories[idx] = np.vstack([trajectories[idx], point])

        # Update keypoints for the next frame
        keypoints = new_keypoints

    # Identify points that moved out of frame
    points_out_of_frame = [traj[-1] for traj in trajectories if traj[-1][0] < 0 or traj[-1][1] < 0]

    # Save trajectory visualization
    plot_and_save_trajectories(frames, trajectories[0], trajectories[:20], points_out_of_frame, os.path.join(results_dir, "trajectories.png"))

    # Save trajectories as a CSV file
    save_trajectories(trajectories, os.path.join(results_dir, "trajectories.csv"))

if __name__ == "__main__":
    main()
