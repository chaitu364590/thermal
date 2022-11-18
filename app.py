import streamlit as st
import cv2
from PIL import Image
import pandas as pd
import numpy as np
import PIL
from PIL import Image,ImageEnhance
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
hide_streamlit_style = """
            <style>
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True) 
uploaded_video = st.file_uploader("Choose video", type=["mp4", "mov"])
frame_skip = 300 # display every 300 frames

df = pd.read_csv("Book2.csv")

y = df['Danger'].values.reshape(-1, 1)
X = df['Temperature'].values.reshape(-1, 1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)
SEED = 42
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = SEED)
regressor = LinearRegression()
regressor.fit(X_train, y_train)
def calc(slope, intercept, Area):
    return slope*Area+intercept

score = calc(regressor.coef_, regressor.intercept_, 9.5)

score = regressor.predict([[9.5]])

y_pred = regressor.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
print(f'Mean absolute error: {mae:.2f}')
print(f'Mean squared error: {mse:.2f}')
print(f'Root mean squared error: {rmse:.2f}')
st.dataframe(y_pred)


if uploaded_video is not None: # run only when user uploads video
    vid = uploaded_video.name
    with open(vid, mode='wb') as f:
        f.write(uploaded_video.read()) # save video to disk

    st.markdown(f"""
    ### Files
    - {vid}
    """,
    unsafe_allow_html=True) # display file name

    vidcap = cv2.VideoCapture(vid) # load video from disk
    cur_frame = 0
    success = True
    

    while success:
        success, frame = vidcap.read() # get next frame from video
        if cur_frame % frame_skip == 0: # only analyze every n=300 frames
            print('frame: {}'.format(cur_frame)) 
            pil_img = Image.fromarray(frame) # convert opencv frame (with type()==numpy) into PIL Image
            st.image(pil_img)
            cur_frame += 1
            our_image_A = pil_img##tview
            st.image(our_image_A)
            our_image_A=np.array(our_image_A.convert('RGB'))
            gray = cv2.cvtColor(our_image_A, cv2.COLOR_BGR2GRAY)
            st.image(gray)

            gray8_image=np.zeros((120, 160), dtype=np.uint8)
            gray8_image=cv2.normalize(gray, gray8_image,0,255,cv2.NORM_MINMAX)
            gray8_image=np.uint8(gray8_image)
            inferno_palette=cv2.applyColorMap(gray8_image, cv2.COLORMAP_INFERNO)
            jet_palette=cv2.applyColorMap(gray8_image, cv2.COLORMAP_JET)
            viridis_palette=cv2.applyColorMap(gray8_image, cv2.COLORMAP_VIRIDIS)
            st.image(gray8_image)
            st.image(inferno_palette)
            st.image(jet_palette)
            st.image(viridis_palette)
  
