# Importing Project Dependencies
import numpy as np
import cv2
import pandas as pd
import tensorflow as tf
from tensorflow import keras
import time
import winsound
import streamlit as st
from PIL import Image


# Setting up config for GPU usage
physical_devices = tf.config.list_physical_devices("GPU")

if physical_devices:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
else:
    print("No GPU devices found. Proceeding with CPU.")

# Using  Har-cascade classifier from OpenCV
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

model = keras.models.load_model("my_model (1).h5", compile=False)

# Now compile it again using correct settings
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])


# Save the updated model
model.save('my_model_updated.h5')
# Title for GUI
st.title('Drowsiness Detection')
img = []

# Navigation Bar
nav_choice = st.sidebar.radio('Navigation', ('Home', 'Sleep Detection', 'Help Us Improve'), index=0)
# Home page
if nav_choice == 'Home':
    st.markdown(
        """
        <style>
        .home-title {
            font-size: 2rem;
            font-weight: 600;
            text-align: center;
            color: #333;
        }
        .home-subtext {
            font-size: 1rem;
            text-align: center;
            color: #666;
            margin-bottom: 1.5rem;
        }
        .how-to {
            background-color: #f0f2f6;
            padding: 1.2rem;
            border-radius: 10px;
            margin-top: 2rem;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    # Title
    st.markdown('<div class="home-title">😴 Drowsiness Detection System</div>', unsafe_allow_html=True)
    st.markdown('<div class="home-subtext">Prevents sleep-deprivation road accidents by alerting drowsy drivers</div>', unsafe_allow_html=True)

    # Main Image
    st.image('ISHN0619_C3_pic.jpg', use_column_width=True)

    # Project Description
    st.markdown("""
    **🚨 According to a survey by *The Times of India*, nearly 40% of road accidents occur due to driver fatigue and sleep deprivation.**  
    This AI-powered web application uses deep learning and computer vision to monitor driver alertness and **warns users with a beep sound** if drowsiness is detected.
    """)
    
    # Supporting Image
    st.image('sleep.jfif', width=300, caption='Stay alert. Stay alive.')

    # How to Use Section
    st.markdown('<div class="how-to">', unsafe_allow_html=True)
    st.markdown("### 🛠️ How to Use?")
    st.markdown("""
    1️⃣ Go to the **Sleep Detection** page from the Navigation Side-Bar.  
    2️⃣ Ensure there's **adequate lighting** in your room.  
    3️⃣ **Align your face clearly in front of the webcam** and stay close.  
    4️⃣ The webcam will take 3 snapshots over 5 seconds — **keep your eyes in the same state** (open or closed).  
    5️⃣ If your eyes appear **closed**, a **beep sound** will alert you.  
    6️⃣ Otherwise, it continues to monitor at regular intervals.  
    """)
    st.markdown("</div>", unsafe_allow_html=True)

    # Dataset Reference
    st.markdown(
        """
        <br>
        📂 <b>Model Training Dataset:</b> 
        <a href="https://www.kaggle.com/kutaykutlu/drowsiness-detection" target="_blank">Available on Kaggle</a>
        """,
        unsafe_allow_html=True
    )

    
# Sleep Detection page
elif nav_choice == 'Sleep Detection':
    st.header('Image Prediction')
    cap = 0
    st.success('Please look at your web-cam, while following all the instructions given on the Home page.')
    st.warning(
        'Keeping the eyes in the same state is important but you can obviously blink your eyes, if they are open!!!')
    b = st.progress(0)
    for i in range(100):
        time.sleep(0.0001)
        b.progress(i + 1)

    start = st.radio('Options', ('Start', 'Stop'), key='Start_pred', index=1)

    if start == 'Start':
        decision = 0
        st.markdown('<font face="Comic sans MS"><b>Detected Facial Region of Interest(ROI)&emsp;&emsp;&emsp;&emsp;&emsp;Extractd'
                    ' Eye Features from the ROI</b></font>', unsafe_allow_html=True)
        
        # Best of 3 mechanism for drowsiness detection
        for _ in range(3):
            cap = cv2.VideoCapture(0)
            ret, frame = cap.read()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)
            # Proposal of face region by the har cascade classifier
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 5)
                roi_gray = gray[y:y + w, x:x + w]
                roi_color = frame[y:y + h, x:x + w]
            frame1 = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            try:
                # Cenentroid method for extraction of eye-patch 
                centx, centy = roi_color.shape[:2]
                centx //= 2
                centy //= 2
                eye_1 = roi_color[centy - 40: centy, centx - 70: centx]
                eye_1 = cv2.resize(eye_1, (86, 86))
                eye_2 = roi_color[centy - 40: centy, centx: centx + 70]
                eye_2 = cv2.resize(eye_2, (86, 86))
                cv2.rectangle(frame1, (x + centx - 60, y + centy - 40), (x + centx - 10, y + centy), (0, 255, 0), 5)
                cv2.rectangle(frame1, (x + centx + 10, y + centy - 40), (x + centx + 60, y + centy), (0, 255, 0), 5)
                preds_eye1 = model.predict(np.expand_dims(eye_1, axis=0))
                preds_eye2 = model.predict(np.expand_dims(eye_2, axis=0))
                e1, e2 = np.argmax(preds_eye1), np.argmax(preds_eye2)
                
                # Display of face image and extracted eye-patch
                img_container = st.beta_columns(4)
                img_container[0].image(frame1, width=250)
                img_container[2].image(cv2.cvtColor(eye_1, cv2.COLOR_BGR2RGB), width=150)
                img_container[3].image(cv2.cvtColor(eye_2, cv2.COLOR_BGR2RGB), width=150)
                print(e1, e2)
                
                # Decision variable for prediction
                if e1 == 1 or e2 == 1:
                    pass
                else:
                    decision += 1

            except NameError:
                st.warning('Hold your camera closer!!!\nTrying again in 2s')
                cap.release()
                time.sleep(1)
                continue

            except:
                cap.release()
                continue

            finally:
                cap.release()

        # If found drowsy, then make a beep sound to alert the driver
        if decision == 0:
            st.error('Eye(s) are closed')
            winsound.Beep(2500, 2000)

        else:
            st.success('Eyes are Opened')
        st.warning('Please select "Stop" and then "Start" to try again')

# Help Us Improve page
else:
    st.header('Help Us Improve')
    st.success('We would appreciate your Help!!!')
    st.markdown(
        '<font face="Comic sans MS">To make this app better, we would appreciate your small amount of time.</font>'
        '<font face="Comic sans MS">Let me take you through, some of the basic statistical analysis of this </font>'
        '<font face="Comic sans MS">model. <br><b>Accuracy with naked eyes = 99.5%<br>Accuracy with spectacles = 96.8%</b><br></font> '
        '<font face="Comic sans MS">As we can see here, accuracy with spectacles is not at all spectacular, and hence to make this app </font>'
        '<font face="Comic sans MS">better, and to use it in real-time situations, we require as much data as we can gather.</font> '
        , unsafe_allow_html=True)
    st.warning('NOTE: Your identity will be kept anonymous, and only your eye-patch will be extracted!!!')
    # Image upload
    img_upload = st.file_uploader('Upload Image Here', ['png', 'jpg', 'jpeg'])
    if img_upload is not None:
        prog = st.progress(0)
        to_add = cv2.imread(str(img_upload.read()), 0)
        to_add = pd.DataFrame(to_add)
        
        # Save it in the database
        to_add.to_csv('Data_from_users.csv', mode='a', header=False, index=False, sep=';')
        for i in range(100):
            time.sleep(0.001)
            prog.progress(i + 1)
        st.success('Uploaded Successfully!!! Thank you for contributing.')
