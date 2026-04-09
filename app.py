import streamlit as st
import matplotlib.pyplot as plt
import cv2
import numpy as np
import pandas as pd
import time
import pickle
from PIL import Image
from keras.models import load_model

# Load trained model
@st.cache_resource
def load_trained_model():
    return load_model('mnist_cnn.h5')  # Ganti nama file sesuai model kamu

# Load training history
@st.cache_data
def load_training_history():
    try:
        with open('training_history.pkl', 'rb') as f:
            return pickle.load(f)
    except FileNotFoundError:
        return None

# Load test accuracy (jika tersedia, misalnya disimpan sebagai float)
def load_test_accuracy():
    try:
        with open('test_accuracy.pkl', 'rb') as f:
            return pickle.load(f)
    except FileNotFoundError:
        return None

# Streamlit App
def main():
    st.title("🔢 MNIST Digit Classifier")
    st.markdown("""
    Upload an image of a handwritten digit (0–9) and the CNN model will predict it.
    """)

    model = load_trained_model()
    history = load_training_history()
    test_accuracy = load_test_accuracy()

    # Sidebar options
    st.sidebar.header("Options")
    show_model_info = st.sidebar.checkbox("Show Model Architecture")
    show_training_stats = st.sidebar.checkbox("Show Training Statistics")

    if show_model_info:
        st.subheader("Model Architecture")
        from io import StringIO
        import sys
        buffer = StringIO()
        model.summary(print_fn=lambda x: buffer.write(x + '\n'))
        st.text(buffer.getvalue())

    if show_training_stats and history:
        st.subheader("Training Statistics")
        if test_accuracy:
            st.write(f"Final Test Accuracy: *{test_accuracy:.4f}*")

        fig, ax = plt.subplots(1, 2, figsize=(12, 4))
        ax[0].plot(history['accuracy'], label='Train Accuracy')
        ax[0].plot(history['val_accuracy'], label='Validation Accuracy')
        ax[0].set_title('Accuracy')
        ax[0].legend()

        ax[1].plot(history['loss'], label='Train Loss')
        ax[1].plot(history['val_loss'], label='Validation Loss')
        ax[1].set_title('Loss')
        ax[1].legend()

        st.pyplot(fig)
    elif show_training_stats:
        st.warning("⚠️ Training history not found.")

    st.subheader("Digit Recognition")
    uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        img = np.array(image.convert("L"))
        img = cv2.resize(img, (28, 28))
        img = img / 255.0
        img = img.reshape(1, 28, 28, 1)

        st.write("Classifying...")
        progress_bar = st.progress(0)
        for i in range(100):
            time.sleep(0.005)
            progress_bar.progress(i + 1)

        pred = model.predict(img)
        pred_class = np.argmax(pred)
        confidence = np.max(pred)

        st.success(f"Prediction: *{pred_class}* with {confidence*100:.2f}% confidence")

        st.subheader("Prediction Probabilities")
        classes = [str(i) for i in range(10)]
        proba_df = pd.DataFrame({'Digit': classes, 'Probability': pred[0]})
        st.bar_chart(proba_df.set_index('Digit'))

if __name__ == '__main__':
    main()