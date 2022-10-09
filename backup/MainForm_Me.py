import streamlit as st
import numpy as np
from PIL import Image
import cv2
import os
from keras.models import load_model
import tensorflow as tf

def main():  
    trainedModel = load_model('saved model 20k epoch10')
    #trainedModel = tf.saved_model.load_model('saved model 20k epoch10')
    st.title("Detect Microscopic Motion in Mechanical Equipment using Deep-Learning and smartphones")
    model_loading_state = st.text('Getting things ready! Please wait...')
    model_loading_state.text('The AI is all saved model 20k epoch10 set!')
    frameA_Image = st.file_uploader("Upload a Input Image below for frame A...", type=["jpg", "png", "mp4", "avi"])
    frameB_Image = st.file_uploader("Upload a Input Image below for frame B...", type=["jpg", "png", "mp4", "avi"])
    input_file1 = st.empty()
    input_file2 = st.empty()
    #img_name_A = ''
    #img_name_B = ''
    #if frameA_Image is not None:
        #img_name_A = frameA_Image.name
        #file_details = {"FileName":img_name_A,"FileType":frameA_Image.type}
        #with open(os.path.join("tempDir",img_name_A),"wb") as f: 
             #f.write(frameA_Image.getbuffer())         
        #st.success("Saved File")
        
    #if frameB_Image is not None:
        #img_name_B = frameB_Image.name
        #file_details = {"FileName":img_name_B,"FileType":frameB_Image.type}
        #with open(os.path.join("tempDir",img_name_B),"wb") as f: 
             #f.write(frameB_Image.getbuffer())         
        #st.success("Saved File")
    
    magnificationFactorValue = 5
    predict_button = st.button("Image Predicted by Model")
    col1, col2, col3 = st.columns([6, 4, 6])
    if predict_button:
        if frameA_Image is not None:
            if frameB_Image is not None:
                frameA_Image_Array = np.array(Image.open(frameA_Image))
                frameB_Image_Array = np.array(Image.open(frameB_Image))
                #frameA_Image_Array = cv2.imread(os.path.join("tempDir",img_name_A))
                #frameB_Image_Array = cv2.imread(os.path.join("tempDir",img_name_B))
                magnification_factor_value = tf.Variable([[[magnificationFactorValue]]])
                magnification_factor_value = tf.expand_dims(magnification_factor_value,axis=0)
                im1= tf.expand_dims(frameA_Image_Array,axis=0)
                im2= tf.expand_dims(frameB_Image_Array,axis=0)
                modelPredictImage = trainedModel.predict(
                    [im1, im2, magnification_factor_value[:1]])               
                image_norm = cv2.normalize(np.squeeze(modelPredictImage[0]), None, alpha=0,beta=1, norm_type=cv2.NORM_MINMAX)
                col1.caption('Predicted Image')
                col1.image([np.squeeze(image_norm)], use_column_width=True,clamp=True, channels='BGR')
                

if __name__ == '__main__':
    main()
