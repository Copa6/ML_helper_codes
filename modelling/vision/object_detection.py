def detect_obj_using_cascade(frame, cascade_xml_file):
    import cv2
    obj_cascade = cv2.CascadeClassifier(cascade_xml_file)
    objects = obj_cascade.detectMultiScale(frame,
                                           scaleFactor=1.10,
                                           minNeighbors=40,
                                           minSize=(24, 24),
                                           flags=cv2.CASCADE_SCALE_IMAGE
                                           )
    return objects
