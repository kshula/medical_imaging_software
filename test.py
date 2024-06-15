import streamlit as st
import pydicom
from skimage import measure
import plotly.express as px
import numpy as np

def load_dcm_image(file_path):
    # Load DICOM image
    dcm = pydicom.dcmread(file_path)
    image = dcm.pixel_array
    return image

def analyze_shapes(image):
    # Thresholding the image
    thresh = image > image.mean()

    # Use scikit-image's label function to identify separate shapes
    labels = measure.label(thresh, connectivity=2)

    # Count the number of shapes (organs)
    num_shapes = len(set(labels.flatten())) - 1  # subtract 1 to exclude background (label 0)

    return num_shapes, labels

def main():
    st.title('Medical Image Analysis with Topological Methods')

    # File upload and processing
    uploaded_file = st.file_uploader("Upload DICOM file", type=["dcm"])
    
    if uploaded_file is not None:
        st.subheader('Uploaded DICOM Image')
        image = load_dcm_image(uploaded_file)
        
        # Analyze shapes
        num_shapes, labels = analyze_shapes(image)

        # Create figure using Plotly
        fig = px.imshow(image, binary_string=True)
        
        # Overlay contours on the image to visualize the shapes
        contours = measure.find_contours(labels, 0.5, fully_connected='high')

        for contour in contours:
            fig.add_trace(px.line(x=contour[:, 1], y=contour[:, 0]).data[0])

        fig.update_layout(
            title=f"Detected Shapes: {num_shapes}",
            xaxis_title="X Axis",
            yaxis_title="Y Axis",
            margin=dict(l=40, r=40, t=40, b=40),
        )

        # Display the Plotly figure
        st.plotly_chart(fig)

if __name__ == "__main__":
    main()
