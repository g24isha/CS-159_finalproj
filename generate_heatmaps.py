import cv2
import numpy as np
import matplotlib.pyplot as plt

def preprocess_image(img):
    """
    Preprocess image with noise reduction and enhancement
    """
    # Convert to float32 and normalize to 0-1
    img_norm = cv2.normalize(img.astype(np.float32), None, 0, 1, cv2.NORM_MINMAX)
    
    # Apply slight Gaussian blur to reduce noise
    img_blur = cv2.GaussianBlur(img_norm, (3, 3), 0.5)
    
    # Convert to grayscale for structural comparison
    img_gray = cv2.cvtColor((img_blur * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    
    return img_blur, img_gray

def align_and_resize_images(img1, img2):
    """
    Align two images to the same size by resizing the smaller one to match the larger one
    """
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    
    # Get the maximum dimensions
    max_h = max(h1, h2)
    max_w = max(w1, w2)
    
    # Resize both images to the maximum dimensions using Lanczos interpolation
    img1_resized = cv2.resize(img1, (max_w, max_h), interpolation=cv2.INTER_LANCZOS4)
    img2_resized = cv2.resize(img2, (max_w, max_h), interpolation=cv2.INTER_LANCZOS4)
    
    return img1_resized, img2_resized

def generate_difference_heatmap(image_path1, image_path2, threshold=0.3):
    """
    Generate a heatmap showing differences between two images
    Args:
        image_path1: Path to first image
        image_path2: Path to second image
        threshold: Minimum difference threshold (0-1 range, default 0.3)
    """
    # Read the images
    img1 = cv2.imread(image_path1)
    img2 = cv2.imread(image_path2)
    
    if img1 is None or img2 is None:
        raise ValueError("Could not read one or both images")
    
    # Convert to RGB (from BGR)
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
    
    # Align and resize images
    img1_aligned, img2_aligned = align_and_resize_images(img1, img2)
    
    # Preprocess images
    img1_processed, img1_gray = preprocess_image(img1_aligned)
    img2_processed, img2_gray = preprocess_image(img2_aligned)
    
    # Calculate color difference
    color_diff = np.abs(img1_processed - img2_processed)
    color_diff_mean = np.mean(color_diff, axis=2)
    
    # Calculate structural difference using grayscale
    struct_diff = np.abs(img1_gray - img2_gray)
    
    # Combine differences with weights
    combined_diff = 0.7 * color_diff_mean + 0.3 * struct_diff
    
    # Apply threshold with slight smoothing to reduce noise
    diff_smooth = cv2.GaussianBlur(combined_diff, (3, 3), 0)
    diff_smooth[diff_smooth < threshold] = 0
    
    # Enhance differences using gamma correction
    gamma = 0.7  # Increased gamma to reduce sensitivity to small differences
    diff_enhanced = np.power(diff_smooth, gamma)
    
    # Normalize the difference to 0-1 range
    diff_normalized = cv2.normalize(diff_enhanced, None, 0, 1, cv2.NORM_MINMAX)
    
    # Create figure with subplots
    plt.figure(figsize=(20, 5))
    
    # # Plot original images and difference heatmap
    # plt.subplot(141)
    # plt.imshow(img1_aligned)
    # plt.title('Image 1')
    # plt.axis('off')
    
    # plt.subplot(142)
    # plt.imshow(img2_aligned)
    # plt.title('Image 2')
    # plt.axis('off')
    
    # plt.subplot(143)
    # plt.imshow(diff_normalized, cmap='hot')
    # plt.colorbar(label='Difference Intensity')
    # plt.title('Difference Heatmap')
    # plt.axis('off')
    
    # Add a binary threshold visualization
    plt.subplot(144)
    binary_diff = (diff_normalized > 0).astype(np.float32)
    plt.imshow(binary_diff, cmap='gray')
    plt.title('Binary Difference Map')
    plt.axis('off')
    
    # Save the plot
    output_path = 'difference_heatmap.png'
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close()
    
    # Save additional visualization showing only significant differences
    diff2 = img2_aligned.copy()
    mask = diff_normalized > 0
    diff2[mask] = [255, 0, 0]  # Highlight differences in red
    plt.figure(figsize=(10, 5))
    plt.imshow(diff2)
    plt.axis('off')
    plt.title('Differences Highlighted in Red')
    plt.savefig('difference2.png', bbox_inches='tight', dpi=300)
    plt.close()
    
    return output_path

if __name__ == "__main__":
    import sys
    import argparse
    
    image1 = "visprog_new/assets/difflive1.png"
    image2 = "visprog_new/assets/difflive2.png"
    parser = argparse.ArgumentParser(description='Compare two images and generate difference heatmap')
    # parser.add_argument('image1', help='Path to first image')
    # parser.add_argument('image2', help='Path to second image')
    parser.add_argument('--threshold', type=float, default=0.3,
                      help='Minimum difference threshold (0-1 range, default: 0.3)')
    
    args = parser.parse_args()
    
    try:
        output_path = generate_difference_heatmap(image1, image2, args.threshold)
        print(f"Heatmap saved as: {output_path}")
        print("Additional visualization saved as: difference2.png")
    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1) 