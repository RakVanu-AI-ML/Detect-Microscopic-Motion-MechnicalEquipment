import streamlit as st
import numpy as np
from PIL import Image
import cv2
import os
from keras.models import load_model
import tensorflow as tf
import imageio

def main():  
    trainedModel = load_model('saved model 20k epoch10')
    #trainedModel = tf.saved_model.load_model('saved model 20k epoch10')
    st.title("Detect Microscopic Motion in Mechanical Equipment using Deep-Learning and smartphones")
    model_loading_state = st.text('Getting things ready! Please wait...')
    model_loading_state.text('The AI is all saved model 20k epoch10 set!')
    frameA_Image = st.file_uploader("Please Upload a Input Image below for frame A...", type=["jpg", "png", "mp4", "avi"])
    frameB_Image = st.file_uploader("Please Upload a Input Image below for frame B...", type=["jpg", "png", "mp4", "avi"])
    
    input_file1 = st.empty()
    input_file2 = st.empty()
    img_A = ''
    img_predicted= ''
    #if frameA_Image is not None:
       #img_name_A = frameA_Image.name
        #file_details = {"FileName":img_name_A,"FileType":frameA_Image.type}
        #with open(os.path.join("ModelPredictImages",img_name_A),"wb") as f: 
            # f.write(frameA_Image.getbuffer())         
        #st.success("Saved File")
        
    #if frameB_Image is not None:
        #img_name_B = frameB_Image.name
        #file_details = {"FileName":img_name_B,"FileType":frameB_Image.type}
        #with open(os.path.join("ModelPredictImages",img_name_B),"wb") as f: 
             #f.write(frameB_Image.getbuffer())         
        #st.success("Saved File")
    
    magnificationFactorValue = st.slider("Please select Magnification Factor from slider",1,100)
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
                col1.image([np.squeeze(image_norm)], use_column_width=True,clamp=True, channels='RGB')
                predictedImageByModel = np.squeeze(image_norm)
                SaveImages(frameA_Image, modelPredictImage) #[np.squeeze(image_norm)])
                
#Creating gif using Image list
def CreategifFromImages(pathOfImage1,pathOfImage2,gif_name):
    images_frameAB = []
    pathOfImage1 = os.path.join("ModelPredictImages",pathOfImage1)
    pathOfImage2 = os.path.join("ModelPredictImages",pathOfImage2)
    images_frameAB.append(imageio.imread(pathOfImage1))
    images_frameAB.append(imageio.imread(pathOfImage2))
    imageio.mimsave(os.path.join("ModelPredictImages",gif_name), images_frameAB,'GIF',duration=0.5)
    

#Save Input and Predicted Image for creating gif file
def SaveImages(inputImage, predictedImage):
    if inputImage is not None:
        if predictedImage is not None:
            img_A = "frameAImage.PNG"
            img_Predicted = "predictedImage.PNG"
            inputFileDetails = {"FileName":img_A,"FileType":"PNG"}
            predictedFileDetails = {"FileName":predictedImage,"FileType":"PNG"}
            with open(os.path.join("ModelPredictImages",img_A),"wb") as f: 
                 f.write(inputImage.getbuffer())   
            #imagePredicted = Image.fromarray(predictedImage,'BGR')    
            #imagePredicted.save(os.path.join("ModelPredictImages",img_Predicted))
            im = Image.fromarray(np.squeeze(predictedImage.astype(np.uint8)),'RGB')
            im.save(os.path.join("ModelPredictImages",img_Predicted))
            st.success("Saved File")
            CreategifFromImages(img_A,img_Predicted,"finalGifByModel.gif")
                
hide_menu_style = """
        <style>
        #MainMenu {visibility: hidden; }
        footer {visibility: hidden;}
        </style>
        """
#st.markdown(hide_menu_style, unsafe_allow_html=True)
if __name__ == '__main__':
    main()
