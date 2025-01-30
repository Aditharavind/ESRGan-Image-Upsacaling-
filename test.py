import os
import glob
import cv2
import numpy as np
import torch
import streamlit as st
from PIL import Image
import RRDBNet_arch as arch

# Load model
model_path = 'models/RRDB_ESRGAN_x4.pth'  # Path to the model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = arch.RRDBNet(3, 3, 64, 23, gc=32)
model.load_state_dict(torch.load(model_path), strict=True)
model.eval()
model = model.to(device)

# Title and instructions
st.title("Image Super-Resolution with ESRGAN")
st.write("""
    Upload a low-resolution image, and the model will enhance its resolution using the ESRGAN model.
""")

# File uploader for user to upload images
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Read the image
    image = Image.open(uploaded_file)
    image = np.array(image)

    # Display the uploaded image
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Processing the image when the button is clicked
    if st.button("Process Image"):
        # Preprocess the image
        img = image * 1.0 / 255
        img = torch.from_numpy(np.transpose(img[:, :, [2, 1, 0]], (2, 0, 1))).float()
        img_LR = img.unsqueeze(0)
        img_LR = img_LR.to(device)

        # Run the image through the model
        with torch.no_grad():
            output = model(img_LR).data.squeeze().float().cpu().clamp_(0, 1).numpy()
        output = np.transpose(output[[2, 1, 0], :, :], (1, 2, 0))
        output = (output * 255.0).round().astype(np.uint8)

        # Save the processed image
        result_img = Image.fromarray(output)
        result_img.save("results/output_image.png")

        # Show the processed image
        st.image(result_img, caption="Processed Image", use_column_width=True)

        # Provide a download link for the processed image
        with open("results/output_image.png", "rb") as file:
            st.download_button(
                label="Download Processed Image",
                data=file,
                file_name="output_image.png",
                mime="image/png"
            )
