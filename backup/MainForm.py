import streamlit as st
import numpy as np
from PIL import Image
import cv2
from keras.models import load_model
import tensorflow as tf

def main():  
    trainedModel = load_model('savedTrainedModel')
    #trainedModel = tf.saved_model.load_model('saved model 20k epoch10')
    st.title("Detect Microscopic Motion in Mechanical Equipment using Deep-Learning and smartphones")
    model_loading_state = st.text('Getting things ready! Please wait...')
    model_loading_state.text('The AI is all saved model 20k epoch10set!')
    frameA_Image = st.file_uploader("Upload a input image below for frame A...", type=["jpg", "png", "mp4", "avi"])
    frameB_Image = st.file_uploader("Upload a input image below for frame B...", type=["jpg", "png", "mp4", "avi"])
    input_file1 = st.empty()
    input_file2 = st.empty()
  
    magnificationFactorValue = 5
    predict_button = st.button("Model Predict")
    col1, col2, col3 = st.columns([6, 4, 6])
    if predict_button:
        if frameA_Image is not None:
            if frameB_Image is not None:
                frameA_Image_Array = np.array(
                    [np.array(np.asarray((Image.open(frameA_Image)).resize((384, 384)))).astype(np.uint8)])
                frameB_Image_Array = np.array(
                    [np.array(np.asarray((Image.open(frameB_Image)).resize((384, 384)))).astype(np.uint8)])
                magnification_factor = np.array([np.array(np.array([magnificationFactorValue]))])
                magnification_factor_value = np.array([np.array(np.array([0]))])
                modelPredictImage = trainedModel.predict(
                    [frameA_Image_Array[:1], frameB_Image_Array[:1], magnification_factor_value[:1]])
                minValue = modelPredictImage[0].min()
                maxValue = modelPredictImage[0].max()
                resultImage = (modelPredictImage[0] - minValue) / (maxValue - minValue)
                resultImage = resultImage * 255
                resultImage = resultImage.astype(np.uint8)
                Image.fromarray(resultImage).save('111.png')
                col1.caption('Input Image')
                col1.image([resultImage], use_column_width=True)
                

if __name__ == '__main__':
    main()
