import streamlit as st
import numpy as np
from skimage.io import imread
from skimage.transform import resize
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import warnings
import cv2
import base64

# Ignore warnings (optional)
warnings.filterwarnings("ignore")

if 'initialized' not in st.session_state:
    with st.spinner("Please wait, the app is initializing..."):
        import time
        time.sleep(2)  # Example: simulate delay for app initialization
        st.session_state['initialized'] = True

# Apply bold style to all text elements globally in the Streamlit app
def apply_global_bold_style():
    global_bold_style = """
    <style>
    .stApp * {
        font-weight: bold !important;
    }
    </style>
    """
    st.markdown(global_bold_style, unsafe_allow_html=True)

apply_global_bold_style()

# Function to blur and convert an image to base64
def blur_and_convert_to_base64(image_path, blur_intensity=(21, 21)):
    image = cv2.imread(image_path)
    blurred_image = cv2.GaussianBlur(image, blur_intensity, 0)
    _, im_arr = cv2.imencode('.png', blurred_image)
    im_bytes = im_arr.tobytes()
    im_b64 = base64.b64encode(im_bytes).decode("utf-8")
    return im_b64

# Function to set a blurred background image
def set_blurred_background_image(image_path):
    base64_image = blur_and_convert_to_base64(image_path)
    background_style = f"""
    <style>
    .stApp {{
        background-image: url("data:image/png;base64,{base64_image}");
        background-size: cover;
        background-position: center;
    }}
    </style>
    """
    st.markdown(background_style, unsafe_allow_html=True)

background_image_path = "C:/New Project/CNN/cloud.png"
set_blurred_background_image(background_image_path)

# Initialize session state variables
if 'segmented_masks' not in st.session_state:
    st.session_state.segmented_masks = []

if 'original_images' not in st.session_state:
    st.session_state.original_images = []

if 'fifth_image_segmented' not in st.session_state:
    st.session_state.fifth_image_segmented = None

if 'combined_segmented_images' not in st.session_state:
    st.session_state.combined_segmented_images = None

if 'display_segmentation' not in st.session_state:
    st.session_state.display_segmentation = False

# Load the trained models
segmentation_model_path = 'C:/New Project/CNN/segmentation_model.h5'
cloud_motion_model_path = "C:/New Project/RNN working/cloud_motion_model.h5"
segmentation_model = load_model(segmentation_model_path)
cloud_motion_model = load_model(cloud_motion_model_path)

# Function to preprocess the uploaded image
def preprocess_image(uploaded_image, target_size):
    if uploaded_image.shape[-1] == 4:
        uploaded_image = uploaded_image[..., :3]
    img = resize(uploaded_image, target_size, preserve_range=True)
    img = img / 255.0
    return img

def display_segmented_mask_with_outline(segmented_mask, fig, axes, img_index, filename, draw_on_original=False, original_image=None):
    """
    Function to display segmented masks with black outlines. Optionally, it can also draw the outlines
    on the original images.

    Parameters:
    - segmented_mask: The binary mask of the segmented area.
    - fig: The matplotlib figure object.
    - axes: The axes array from plt.subplots().
    - img_index: The index of the current image/mask.
    - filename: The filename of the image, used for titles.
    - draw_on_original: Boolean flag to indicate if outlines should be drawn on the original images.
    - original_image: The original image on which outlines will be drawn if draw_on_original is True.
    """
    # Calculate time_step for titles
    time_steps = ['t-3', 't-2', 't-1', 't']
    
    # Find contours on the segmented mask
    contours, _ = cv2.findContours(segmented_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    # Draw black contours on a color version of the segmented mask for visibility
    mask_with_outline = cv2.cvtColor(segmented_mask, cv2.COLOR_GRAY2BGR)  # Convert to BGR to draw colored lines
    cv2.drawContours(mask_with_outline, contours, -1, (0, 0, 0), 3)  # Draw contours in black

    if draw_on_original and original_image is not None:
        # Ensure original_image is in the correct format (BGR if using OpenCV)
        original_with_outline = original_image.copy()
        cv2.drawContours(original_with_outline, contours, -1, (0, 0, 0), 3)  # Draw contours in black on the original image
        # Adapt to axes dimensionality for original images
        if axes.ndim > 1:
            ax_original = axes[0, img_index]
        else:
            ax_original = axes[img_index]
        ax_original.imshow(cv2.cvtColor(original_with_outline, cv2.COLOR_BGR2RGB))  # Convert BGR to RGB for displaying
        ax_original.set_title(f'Original Image at {time_steps[img_index]}', fontsize=10)
        ax_original.axis('off')

    # Adapt to axes dimensionality for segmented masks
    if axes.ndim > 1:
        ax_mask = axes[1, img_index] if draw_on_original else axes[0, img_index]
    else:
        ax_mask = axes[img_index]
    ax_mask.imshow(cv2.cvtColor(mask_with_outline, cv2.COLOR_BGR2RGB))  # Convert BGR to RGB for displaying
    ax_mask.set_title(f'Segmented Mask at {time_steps[img_index]}', fontsize=10)
    ax_mask.axis('off')

# Function to predict and post-process cloud motion
def predict_cloud_motion(segmented_images):
    p = cloud_motion_model.predict(segmented_images)
    op = p[0] * 255
    z = op.reshape(280, 280)
    _, binary_z = cv2.threshold(z, 128, 255, cv2.THRESH_BINARY)
    return binary_z

def add_black_outline_to_image(binary_image):
    # Find contours on the binary image
    contours, _ = cv2.findContours(binary_image.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    # Create an RGB version of the image if it's not already
    if len(binary_image.shape) == 2 or binary_image.shape[2] == 1:
        image_with_outline = cv2.cvtColor(binary_image, cv2.COLOR_GRAY2BGR)
    else:
        image_with_outline = binary_image.copy()
    
    # Draw black contours on the image
    cv2.drawContours(image_with_outline, contours, -1, (0, 0, 0), 1)
    return image_with_outline

# Streamlit UI Code here...
# This includes your title, uploader, buttons, and visualization logic.

st.title('Image Segmentation and Cloud Motion Prediction App')
st.write('Upload 5 images. The app will segment the images and use the first 4 to predict cloud motion. The 5th image will be displayed alongside the predicted cloud motion.')

uploaded_files = st.file_uploader("Choose images...", type=['png', 'jpg', 'jpeg'], accept_multiple_files=True)

if uploaded_files is not None and len(uploaded_files) == 5:
    if st.button('Perform Segmentation'):
        with st.spinner('Performing magic with pixels... 🎩✨'):
            st.session_state.display_segmentation = True
            st.session_state.segmented_masks.clear()
            st.session_state.original_images.clear()
            for i, uploaded_file in enumerate(uploaded_files):
                filename = uploaded_file.name
                uploaded_image = imread(uploaded_file)
                preprocessed_image = preprocess_image(uploaded_image, (960, 960, 3))
                predicted_mask = segmentation_model.predict(np.array([preprocessed_image]))[0]
                thresholded_mask = (predicted_mask > 0.5).astype(np.uint8) * 255
                resized_mask = resize(thresholded_mask, (280, 280), preserve_range=True).astype(np.uint8)
                st.session_state.original_images.append(uploaded_image)
                if i == 4:  # Adjusted index for the 5th image
                    st.session_state.fifth_image_segmented = resized_mask

                else:
                    st.session_state.segmented_masks.append(resized_mask)
            st.session_state.combined_segmented_images = np.array(st.session_state.segmented_masks).reshape(1, 4, 280, 280, 1) / 255
            st.success('Segmentation complete! Behold the segmented images.')

    if st.session_state.display_segmentation and len(st.session_state.segmented_masks) == 4:
        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        t = 4
        for i, (segmented_mask, original_image) in enumerate(zip(st.session_state.segmented_masks, st.session_state.original_images)):
            display_segmented_mask_with_outline(segmented_mask, fig, axes[1], i, uploaded_files[i].name)
            axes[0, i].imshow(original_image)
            if t== 1:
                axes[0, i].set_title(f'Original Image at t', fontsize=10)
            else:
                axes[0, i].set_title(f'Original Image at t - {t-1}', fontsize=10)
            t = t-1
            axes[0, i].axis('off')
        plt.tight_layout()
        st.pyplot(fig)
    
    if st.button('Predict Cloud Motion'):
        with st.spinner('Predicting how clouds dance in the sky... 💃🌦️'):
            if st.session_state.combined_segmented_images is not None and st.session_state.fifth_image_segmented is not None:
                predicted_cloud_motion = predict_cloud_motion(st.session_state.combined_segmented_images)
                predicted_cloud_motion_with_outline = add_black_outline_to_image(predicted_cloud_motion)
                fifth_image_segmented_with_outline = add_black_outline_to_image(st.session_state.fifth_image_segmented)

                fig, axes = plt.subplots(1, 2, figsize=(10, 5))
                axes[0].imshow(cv2.cvtColor(fifth_image_segmented_with_outline, cv2.COLOR_BGR2RGB))
                axes[0].set_title('5th Segmented Image')
                axes[0].set_title('Segmented Image at t+1')

                axes[0].axis('off')

                axes[1].imshow(cv2.cvtColor(predicted_cloud_motion_with_outline, cv2.COLOR_BGR2RGB), cmap='gray')
                axes[1].set_title('Predicted Cloud Motion at t+1')
                axes[1].axis('off')
                st.success('Cloud motion predicted! Comparing it with the actual cloud')
                st.pyplot(fig)
                
            else:
                st.write("Please upload exactly 5 images.")
