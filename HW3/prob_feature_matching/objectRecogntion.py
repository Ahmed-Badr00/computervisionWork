import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from scipy.spatial import KDTree
import cv2
import time

def plot_matches(image1, image2, keypoints1, keypoints2, matches, title):
    """Plot matches between two images."""
    img1 = cv2.imread(image1)
    img2 = cv2.imread(image2)
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)

    # Align images horizontally
    height = max(img1.shape[0], img2.shape[0])
    width1, width2 = img1.shape[1], img2.shape[1]
    combined_img = np.zeros((height, width1 + width2, 3), dtype=np.uint8)
    combined_img[:img1.shape[0], :width1, :] = img1
    combined_img[:img2.shape[0], width1:, :] = img2

    plt.figure(figsize=(15, 10))
    plt.imshow(combined_img)

    for match in matches:
        pt1 = keypoints1[match[0]]
        pt2 = keypoints2[match[1]]
        pt2[0] += width1  # Adjust x-coordinate for concatenated image

        plt.plot([pt1[0], pt2[0]], [pt1[1], pt2[1]], 'g-', linewidth=0.8)
        plt.scatter(pt1[0], pt1[1], c='red', s=10)
        plt.scatter(pt2[0], pt2[1], c='blue', s=10)

    plt.axis('off')
    plt.title(f"{title} ({len(matches)} matches)")
    plt.savefig(f"{title.lower().replace(' ', '_')}.png")
    plt.close()

def nearest_neighbor_distance(descriptors1, descriptors2, threshold=200):
    """Method 1: Nearest neighbor matching with distance threshold."""
    matches = []
    tree = KDTree(descriptors2)
    distances, indices = tree.query(descriptors1, k=1)
    
    # Find matches based on distance threshold
    for i, (distance, index) in enumerate(zip(distances, indices)):
        if distance < threshold:
            matches.append((i, index))
    
    return matches

def ratio_test_match(descriptors1, descriptors2, ratio_threshold=0.8):
    """Method 2: Distance ratio test matching."""
    matches = []
    tree = KDTree(descriptors2)
    distances, indices = tree.query(descriptors1, k=2)  # Get 2 nearest neighbors
    
    # Apply ratio test
    for i, (dist, idx) in enumerate(zip(distances, indices)):
        if len(dist) >= 2 and (dist[0] / dist[1] < ratio_threshold):
            matches.append((i, idx[0]))
    
    return matches

def main():
    # Load pre-computed SIFT features
    data = loadmat("SIFT_features.mat")
    
    # Get and transpose the data to correct format
    frame1 = data["Frame1"].T
    frame2 = data["Frame2"].T
    descriptor1 = data["Descriptor1"].T
    descriptor2 = data["Descriptor2"].T
    
    # Extract keypoints
    keypoints1 = frame1[:, :2]
    keypoints2 = frame2[:, :2]
    
    print("Feature Statistics:")
    print(f"Number of keypoints in image 1: {len(keypoints1)}")
    print(f"Number of keypoints in image 2: {len(keypoints2)}")
    
    # Method 1: Nearest Neighbor Distance Matching
    nn_matches = nearest_neighbor_distance(descriptor1, descriptor2)
    print(f"\nNearest Neighbor Distance Matching:")
    print(f"Found {len(nn_matches)} matches")
    plot_matches("stop1.jpg", "stop2.jpg", keypoints1, keypoints2, 
                nn_matches, "Nearest Neighbor Matches")
    
    # Method 2: Ratio Test Matching
    ratio_matches = ratio_test_match(descriptor1, descriptor2)
    print(f"\nRatio Test Matching:")
    print(f"Found {len(ratio_matches)} matches")
    plot_matches("stop1.jpg", "stop2.jpg", keypoints1, keypoints2, 
                ratio_matches, "Ratio Test Matches")
    
    # Comparison of the two methods
    print("\nComparison of Methods:")
    print("1. Nearest Neighbor Distance Matching:")
    print(f"   - Found {len(nn_matches)} matches")
    print("   - Uses absolute distance threshold")
    print("   - More sensitive to the choice of threshold")
    print("   - May include more false matches in complex scenes")
    
    print("\n2. Ratio Test Matching:")
    print(f"   - Found {len(ratio_matches)} matches")
    print("   - Uses relative distance comparison")
    print("   - More robust to different scales and variations")
    print("   - Better at eliminating ambiguous matches")

if __name__ == "__main__":
    main()