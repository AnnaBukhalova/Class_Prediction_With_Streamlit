from transformers import BeitFeatureExtractor, BeitForImageClassification
import streamlit as st
from PIL import Image


def predict_classes(image):
    feature_extractor = BeitFeatureExtractor.from_pretrained('microsoft/beit-base-patch16-224-pt22k-ft22k')
    model = BeitForImageClassification.from_pretrained('microsoft/beit-base-patch16-224-pt22k-ft22k')
    inputs = feature_extractor(images=image, return_tensors="pt")
    outputs = model(**inputs)
    logits = outputs.logits
    predicted_class_idx = logits.argmax(-1).item()
    predicted_text = model.config.id2label[predicted_class_idx]
    return st.text(predicted_text)


def main():
    st.title("Prediction of ImageNet-22k classes")

    st.write("App based on BEiT base-sized model")
    st.write("Model source: https://huggingface.co/microsoft/beit-base-patch16-224-pt22k-ft22k")

    st.subheader("Choose a file")
    uploaded_file = st.file_uploader(
        "Choose a file", type=["jpg"], label_visibility="collapsed"
    )

    image_placeholder = st.container()
    button_placeholder = st.empty()

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        image_placeholder.subheader("Selected image")
        image_placeholder.image(image, use_column_width=True)
        button = button_placeholder.button("Predict classes", type="primary")

        if button:
            with button_placeholder:
                st.text("Predicted class:")
                predict_classes(image)


if __name__ == 'main':
    main()