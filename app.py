import streamlit as st
from PIL import Image
import style

st.title("PyTorch Fast Neural Style Transfer")

style_name = st.sidebar.selectbox("Select Style",
    ('candy', 'mosaic', 'rain_princess', 'udnie')
)

# saved model
model = "saved_models/" + style_name + ".pth"

st.write('### Source image:')
image_file = st.file_uploader("Upload an Image", type=["png","jpg","jpeg"])
if image_file is not None:
    img = Image.open(image_file)
    img.save("images/content-images/" + str(image_file.name))   # save image
    st.image(img, width=400)  # View Uploaded Image

    input_image = "images/content-images/" + str(image_file.name)
    output_image = "images/output-images/" + style_name + "-" + str(image_file.name)

if st.button('Stylize'):
    model = style.load_model(model)
    style.stylize(model, input_image, output_image)

    st.write("### Output image:")
    img = Image.open(output_image)
    st.image(img, width=400)