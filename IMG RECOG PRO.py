#!/usr/bin/env python
# coding: utf-8

# In[5]:


pip install opencv-python


# In[6]:


import cv2

imagePath = 'sid_ds.jpg'


# In[ ]:


img = cv2.imread(imagePath)


# In[ ]:


img.shape


# In[ ]:


gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)



# In[ ]:


gray_image.shape


# In[ ]:


face_classifier = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)


# In[ ]:


face = face_classifier.detectMultiScale(
    gray_image, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40)
)


# In[ ]:


for (x, y, w, h) in face:
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 4)


# In[ ]:


img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


# In[ ]:


import matplotlib.pyplot as plt



plt.figure(figsize=(20,10))
plt.imshow(img_rgb)
plt.axis('off')


# In[ ]:


import cv2

face_classifier = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)


# In[ ]:


video_capture = cv2.VideoCapture(0)


# In[ ]:


def detect_bounding_box(vid):
    gray_image = cv2.cvtColor(vid, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray_image, 1.1, 5, minSize=(40, 40))
    for (x, y, w, h) in faces:
        cv2.rectangle(vid, (x, y), (x + w, y + h), (0, 255, 0), 4)
    return faces



# In[ ]:


while True:

    result, video_frame = video_capture.read()  # read frames from the video
    if result is False:
        break  # terminate the loop if the frame is not read successfully

    faces = detect_bounding_box(
        video_frame
    )  # apply the function we created to the video frame

    cv2.imshow(
        "My Face Detection Project", video_frame
    )  # display the processed frame in a window named "My Face Detection Project"

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

video_capture.release()
cv2.destroyAllWindows()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




