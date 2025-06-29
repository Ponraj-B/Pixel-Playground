import streamlit as st
from PIL import Image
import numpy as np

st.set_page_config(page_title="🖼️ NumPy Image Processing Toolkit", layout="centered")

st.title("🖼️ NumPy Image Processing Toolkit")
st.write("Upload an image and explore 15 different NumPy-powered image transformations!")

uploaded_file = st.file_uploader("Upload an image (JPG, JPEG, PNG)", type=["jpg", "jpeg", "png"])

def to_grayscale(img_arr):
    return np.dot(img_arr[..., :3], [0.2989, 0.5870, 0.1140]).astype(np.uint8)

def invert_colors(img_arr):
    return 255 - img_arr

def crop_image(img_arr, left, right, top, bottom):
    return img_arr[top:bottom, left:right]

def adjust_brightness(img_arr, val):
    img = img_arr.astype(np.int16) + val
    img = np.clip(img, 0, 255)
    return img.astype(np.uint8)

def flip_horizontal(img_arr):
    return img_arr[:, ::-1]

def flip_vertical(img_arr):
    return img_arr[::-1, :]

def rotate_90(img_arr):
    return np.rot90(img_arr, k=3)  # rotate clockwise 90

def rotate_180(img_arr):
    return np.rot90(img_arr, k=2)

def rotate_270(img_arr):
    return np.rot90(img_arr, k=1)

def threshold(img_arr, thresh=128):
    gray = to_grayscale(img_arr)
    return ((gray > thresh) * 255).astype(np.uint8)

def add_noise(img_arr, amount=0.02):
    noisy = img_arr.copy()
    total_pixels = img_arr.shape[0] * img_arr.shape[1]
    num_noise = int(amount * total_pixels)
    coords = [np.random.randint(0, i-1, num_noise) for i in img_arr.shape[:2]]

    # Salt noise (white)
    noisy[coords[0], coords[1], :] = 255

    # Pepper noise (black)
    coords2 = [np.random.randint(0, i-1, num_noise) for i in img_arr.shape[:2]]
    noisy[coords2[0], coords2[1], :] = 0

    return noisy

def zoom_image(img_arr, zoom_factor=1.5):
    h, w = img_arr.shape[:2]
    ch, cw = int(h/zoom_factor), int(w/zoom_factor)
    top = (h - ch) // 2
    left = (w - cw) // 2
    zoomed = img_arr[top:top+ch, left:left+cw]
    zoomed_resized = np.array(Image.fromarray(zoomed).resize((w, h)))
    return zoomed_resized

if uploaded_file:
    img = Image.open(uploaded_file).convert('RGB')
    img_array = np.array(img)

    st.subheader("Original Image")
    st.image(img_array, use_column_width=True)

    option = st.selectbox("Select Image Processing Operation", [
        "Original Image",
        "Convert to Grayscale",
        "RGB Channel Split",
        "Invert Colors",
        "Crop Image",
        "Adjust Brightness",
        "Horizontal Flip",
        "Vertical Flip",
        "Rotate 90° Clockwise",
        "Rotate 180°",
        "Rotate 270° Clockwise",
        "Thresholding (Binarize)",
        "Add Noise (Salt & Pepper)",
        "Image Zoom",
        "Image Negative (Manual Invert)",
    ])

    if option == "Original Image":
        st.image(img_array, caption="Original Image", use_column_width=True)

    elif option == "Convert to Grayscale":
        gray = to_grayscale(img_array)
        st.image(gray, caption="Grayscale Image", use_column_width=True, clamp=True)

    elif option == "RGB Channel Split":
        red = img_array.copy()
        red[:, :, 1] = 0
        red[:, :, 2] = 0
        green = img_array.copy()
        green[:, :, 0] = 0
        green[:, :, 2] = 0
        blue = img_array.copy()
        blue[:, :, 0] = 0
        blue[:, :, 1] = 0

        st.write("🔴 Red Channel")
        st.image(red, use_column_width=True)
        st.write("🟢 Green Channel")
        st.image(green, use_column_width=True)
        st.write("🔵 Blue Channel")
        st.image(blue, use_column_width=True)

    elif option == "Invert Colors":
        inverted = invert_colors(img_array)
        st.image(inverted, caption="Inverted Colors", use_column_width=True)

    elif option == "Crop Image":
        h, w = img_array.shape[:2]
        st.write("Adjust cropping region")
        col1, col2 = st.columns(2)
        with col1:
            left = st.slider("Left", 0, w - 1, 0)
            right = st.slider("Right", 1, w, w)
        with col2:
            top = st.slider("Top", 0, h - 1, 0)
            bottom = st.slider("Bottom", 1, h, h)
        if left < right and top < bottom:
            cropped = crop_image(img_array, left, right, top, bottom)
            st.image(cropped, caption="Cropped Image", use_column_width=True)
        else:
            st.warning("Please select valid crop coordinates.")

    elif option == "Adjust Brightness":
        val = st.slider("Brightness (-100 to 100)", -100, 100, 0)
        bright_img = adjust_brightness(img_array, val)
        st.image(bright_img, caption=f"Brightness Adjusted by {val}", use_column_width=True)

    elif option == "Horizontal Flip":
        flipped = flip_horizontal(img_array)
        st.image(flipped, caption="Horizontally Flipped", use_column_width=True)

    elif option == "Vertical Flip":
        flipped = flip_vertical(img_array)
        st.image(flipped, caption="Vertically Flipped", use_column_width=True)

    elif option == "Rotate 90° Clockwise":
        rotated = rotate_90(img_array)
        st.image(rotated, caption="Rotated 90° Clockwise", use_column_width=True)

    elif option == "Rotate 180°":
        rotated = rotate_180(img_array)
        st.image(rotated, caption="Rotated 180°", use_column_width=True)

    elif option == "Rotate 270° Clockwise":
        rotated = rotate_270(img_array)
        st.image(rotated, caption="Rotated 270° Clockwise", use_column_width=True)

    elif option == "Thresholding (Binarize)":
        thresh_val = st.slider("Threshold value", 0, 255, 128)
        thresh_img = threshold(img_array, thresh_val)
        st.image(thresh_img, caption=f"Thresholded at {thresh_val}", use_column_width=True, clamp=True)

    elif option == "Add Noise (Salt & Pepper)":
        amount = st.slider("Noise Amount (0-0.1)", 0.0, 0.1, 0.02, step=0.01)
        noisy = add_noise(img_array, amount)
        st.image(noisy, caption=f"Added Salt & Pepper Noise ({amount*100:.1f}%)", use_column_width=True)

    elif option == "Image Zoom":
        zoom_factor = st.slider("Zoom factor", 1.0, 3.0, 1.5, step=0.1)
        zoomed = zoom_image(img_array, zoom_factor)
        st.image(zoomed, caption=f"Zoomed by factor {zoom_factor}", use_column_width=True)

    elif option == "Image Negative (Manual Invert)":
        # Same as invert but done manually
        negative = 255 - img_array
        st.image(negative, caption="Image Negative (Manual Invert)", use_column_width=True)

else:
    st.info("Please upload an image to get started.")

# Contact section
st.markdown("---")
st.markdown("### 📬 Contact")
st.markdown(
    """
    **Name:** Your Name  
    **Email:** [ponraj322002@gmail.com](mailto:your.email@example.com)  
    **GitHub:** [https://github.com/Ponraj-B](https://github.com/yourusername)  
    **LinkedIn:** [https://in.linkedin.com/in/ponraj-b-96a917264](https://linkedin.com/in/yourprofile)
    """
)
