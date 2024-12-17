import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import time
from sklearn.neighbors import NearestNeighbors


def find_edge_points(image):
    """Extract edge points (non-zero pixels) from a binary image."""
    points = np.argwhere(image > 0)
    return points


def match_points(points1, points2):
    """
    Match points from points1 to points2 using nearest neighbors.
    Ensures the two sets have the same number of points.
    """
    nbrs = NearestNeighbors(n_neighbors=1, algorithm='auto').fit(points2)
    distances, indices = nbrs.kneighbors(points1)
    matched_points2 = points2[indices.flatten()]
    return points1, matched_points2


def compute_affine_transformation(points1, points2):
    """
    Compute the affine transformation matrix T that maps points1 to points2.
    """
    num_points = points1.shape[0]
    if num_points < 3:
        raise ValueError("At least 3 points are required for affine transformation.")

    # Augment points1 with a column of ones
    points1_homogeneous = np.hstack((points1, np.ones((num_points, 1))))

    # Solve for T using least squares
    T, _, _, _ = np.linalg.lstsq(points1_homogeneous, points2, rcond=None)
    return T.T


def align_shape(im1, im2):
    """Align shapes in binary images im1 and im2 using an affine transformation."""
    # Find edge points
    points1 = find_edge_points(im1)
    points2 = find_edge_points(im2)

    # Match points
    points1, points2 = match_points(points1, points2)

    # Compute affine transformation
    T = compute_affine_transformation(points1, points2)

    # Transform points1 using T
    points1_homogeneous = np.hstack((points1, np.ones((points1.shape[0], 1))))
    aligned_points1 = (T @ points1_homogeneous.T).T

    return aligned_points1, points1, points2, T


def compute_alignment_error(points1, points2):
    """
    Compute the average alignment error as the mean Euclidean distance
    between corresponding points in points1 and points2.
    """
    error = np.sqrt(np.sum((points1 - points2) ** 2, axis=1))
    return np.mean(error)


def plot_alignment_results(im1, im2, aligned_points1, points2, output_path):
    """Plot and save the alignment results."""
    plt.figure(figsize=(10, 10))
    plt.imshow(im2, cmap='gray')
    plt.scatter(points2[:, 1], points2[:, 0], c='red', label="Target Points")
    plt.scatter(aligned_points1[:, 1], aligned_points1[:, 0], c='green', label="Aligned Points")
    plt.legend()
    plt.title("Shape Alignment")
    plt.axis('off')
    plt.savefig(output_path)
    plt.close()


def main():
    # Directory to save results
    results_dir = "shape_alignment_results"
    os.makedirs(results_dir, exist_ok=True)

    # Dataset files 
    files = [
        "apple_1.png", "apple_2.png", "bat_1.png", "bat_2.png", "bell_1.png", "bell_2.png",
        "bird_1.png", "bird_2.png", "Bone_1.png", "Bone_2.png", "bottle_1.png", "bottle_2.png",
        "brick_1.png", "brick_2.png", "butterfly_1.png", "butterfly_2.png", "camel_1.png", "camel_2.png",
        "car_1.png", "car_2.png", "carriage_1.png", "carriage_2.png", "cattle_1.png", "cattle_2.png",
        "cellular_phone_1.png", "cellular_phone_2.png", "chicken_1.png", "chicken_2.png",
        "children_1.png", "children_2.png", "device7_1.png", "device7_2.png", "dog_1.png", "dog_2.png",
        "elephant_1.png", "elephant_2.png", "face_1.png", "face_2.png", "fork_1.png", "fork_2.png",
        "hammer_1.png", "hammer_2.png", "Heart_1.png", "Heart_2.png", "horse_1.png", "horse_2.png",
        "jar_1.png", "jar_2.png", "turtle_1.png", "turtle_2.png"
    ]

    # Pair images for alignment
    pairs = [(files[i], files[i + 1]) for i in range(0, len(files), 2)]

    # Initialize result tracking
    alignment_errors = []
    runtimes = []

    for pair in pairs:
        img1_path, img2_path = os.path.join("data", pair[0]), os.path.join("data", pair[1])

        # Load images
        im1 = cv2.imread(img1_path, cv2.IMREAD_GRAYSCALE)
        im2 = cv2.imread(img2_path, cv2.IMREAD_GRAYSCALE)

        # Error handling: Skip if images are not loaded
        if im1 is None or im2 is None:
            print(f"Error: Unable to read one or both images: {pair[0]}, {pair[1]}. Skipping...")
            continue

        # Ensure binary images
        _, im1 = cv2.threshold(im1, 127, 255, cv2.THRESH_BINARY)
        _, im2 = cv2.threshold(im2, 127, 255, cv2.THRESH_BINARY)

        # Align shapes
        start_time = time.time()
        aligned_points1, points1, points2, T = align_shape(im1, im2)
        runtime = time.time() - start_time

        # Compute alignment error
        error = compute_alignment_error(aligned_points1, points2)

        # Save results
        output_path = os.path.join(results_dir, f"alignment_{pair[0].split('_')[0]}.png")
        plot_alignment_results(im1, im2, aligned_points1, points2, output_path)

        # Track results
        alignment_errors.append(error)
        runtimes.append(runtime)

        print(f"Processed {pair[0]} and {pair[1]}: Error = {error:.4f}, Runtime = {runtime:.4f} seconds")

    # Save alignment errors and runtimes
    np.savetxt(os.path.join(results_dir, "alignment_errors.csv"), alignment_errors, delimiter=",", header="Alignment Error")
    np.savetxt(os.path.join(results_dir, "runtimes.csv"), runtimes, delimiter=",", header="Runtime")

    print("Alignment results saved in:", results_dir)


if __name__ == "__main__":
    main()
