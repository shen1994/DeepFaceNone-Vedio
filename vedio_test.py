# -*- coding: utf-8 -*-
"""
Created on Wed Jul 25 09:36:44 2018

@author: shen1994
"""

import os
import cv2
import numpy as np
import tensorflow as tf

from ssd_utils import BBoxUtility
from dan_utils import get_meanshape
from dan_utils import fit_to_rect

from dan_utils import crop_resize_rotate

if __name__ == "__main__":
    
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    # face detect model
    detect_graph_def = tf.GraphDef()
    # detect_graph_def.ParseFromString(open("model/pico_face_model.pb", "rb").read())
    detect_graph_def.ParseFromString(open("model/pico_FaceDetect_model.pb", "rb").read())
    detect_tensors = tf.import_graph_def(detect_graph_def, name="")
    detect_sess = tf.Session()
    detect_opt = detect_sess.graph.get_operations()
    detect_x = detect_sess.graph.get_tensor_by_name("input_1:0")
    detect_y = detect_sess.graph.get_tensor_by_name("predictions/concat:0")
    detect_util = BBoxUtility(detect_sess, 2, top_k=8)

    # face proper model
    proper_graph_def = tf.GraphDef()
    proper_graph_def.ParseFromString(open("model/pico_FaceProper_model.pb", "rb").read())
    proper_tensors = tf.import_graph_def(proper_graph_def, name="")
    proper_sess = tf.Session()
    proper_opt = proper_sess.graph.get_operations()
    proper_x = proper_sess.graph.get_tensor_by_name("proper_input:0")
    proper_gender = proper_sess.graph.get_tensor_by_name("pred_gender/Softmax:0")
    proper_age = proper_sess.graph.get_tensor_by_name("pred_age/Softmax:0")

    # face align model
    align_graph_def = tf.GraphDef()
    align_graph_def.ParseFromString(open("model/pico_FaceAlign_model.pb", "rb").read())
    align_tensors = tf.import_graph_def(align_graph_def, name="")
    align_sess = tf.Session()
    align_opt = align_sess.graph.get_operations()
    align_x = align_sess.graph.get_tensor_by_name("align_input:0")
    align_stage1 = align_sess.graph.get_tensor_by_name("align_stage1:0")
    align_stage2 = align_sess.graph.get_tensor_by_name("align_stage2:0")
    align_keepout1 = align_sess.graph.get_tensor_by_name("align_keepout1:0")
    align_keepout2 = align_sess.graph.get_tensor_by_name("align_keepout2:0")
    align_landmark = align_sess.graph.get_tensor_by_name("Stage2/landmark_1:0")

    # camera settins
    vedio_shape = [1920, 1080]
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, vedio_shape[0])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, vedio_shape[1])
    cv2.namedWindow("DeepFace", cv2.WINDOW_NORMAL)
    
    while(True):

        # read one image
        _, o_image = cap.read()
    
        o_image = cv2.cvtColor(o_image, cv2.COLOR_BGR2RGB)
        detect_image = cv2.resize(o_image, (720, 720))
        detect_out = detect_sess.run(detect_y, feed_dict={detect_x: [detect_image]})
        results = detect_util.detection_out(detect_out) 
        det_conf = results[0][:, 1]
        det_xmin = results[0][:, 2]
        det_ymin = results[0][:, 3]
        det_xmax = results[0][:, 4]
        det_ymax = results[0][:, 5]
        top_indices = [i for i, conf in enumerate(det_conf) if conf >= 0.57]
        top_conf = det_conf[top_indices]
        top_xmin = det_xmin[top_indices]
        top_ymin = det_ymin[top_indices]
        top_xmax = det_xmax[top_indices]
        top_ymax = det_ymax[top_indices]
    
        for i in range(top_conf.shape[0]):
            
            # box
            xmin = int(round(top_xmin[i] * vedio_shape[0]))
            ymin = int(round(top_ymin[i] * vedio_shape[1]))
            xmax = int(round(top_xmax[i] * vedio_shape[0]))
            ymax = int(round(top_ymax[i] * vedio_shape[1]))
 
            # proper
            xadd = int((xmax - xmin) * 0.16)
            yadd = int((ymax - ymin) * 0.16)
            new_xmin = xmin - xadd
            new_xmax = xmax + xadd
            new_ymin = ymin - yadd
            new_ymax = ymax + yadd
            if new_xmin < 0:
                new_xmin = 0
            if new_xmax > vedio_shape[0] - 1:
                new_xmax = vedio_shape[0] - 1
            if new_ymin < 0:
                new_ymin = 0
            if new_ymax > vedio_shape[1] - 1:
                new_ymax = vedio_shape[1] - 1  
            rect_image = o_image[new_ymin:new_ymax, new_xmin:new_xmax]
            proper_image = cv2.resize(rect_image, (128, 128))
            gender_list, age_list = proper_sess.run([proper_gender, proper_age], \
                               feed_dict={proper_x: [proper_image]})
            g_gender_list = np.arange(0, 2).reshape(2,)
            g_age_list = np.arange(0, 101).reshape(101,)
            gender = gender_list.dot(g_gender_list)[0]
            age = age_list.dot(g_age_list)[0]

            # align
            xadd = int((xmax - xmin) * 0.16)
            yadd = int((ymax - ymin) * 0.16)
            new_xmin = xmin - xadd
            new_xmax = xmax + xadd
            new_ymin = ymin - yadd
            new_ymax = ymax + yadd
            if new_xmin < 0:
                new_xmin = 0
            if new_xmax > vedio_shape[0] - 1:
                new_xmax = vedio_shape[0] - 1
            if new_ymin < 0:
                new_ymin = 0
            if new_ymax > vedio_shape[1] - 1:
                new_ymax = vedio_shape[1] - 1  
            rect_image = o_image[new_ymin:new_ymax, new_xmin:new_xmax]
    
            align_image = np.mean(rect_image, axis=2)
            meanshape = get_meanshape()
            landmark_value = fit_to_rect(meanshape, [0, 0, align_image.shape[0]-1, align_image.shape[1]-1])
            align_image, transform = crop_resize_rotate(align_image, 112, landmark_value, meanshape)
            landmark = align_sess.run(align_landmark, feed_dict={align_x:[np.resize(align_image, (112, 112, 1))], 
                                                     align_stage1:False, align_stage2:False, 
                                                     align_keepout1:0.0, align_keepout2:0.0})[0]
            landmark = np.resize(landmark, (68, 2))
            landmark = np.dot(landmark - transform[1], np.linalg.inv(transform[0]))
          
            # draw some lines, points, text
            cv2.rectangle(o_image, (xmin, ymin), (xmax, ymax), (0, 0, 255), 2)
            
            age_text = 'Age: %.2f' % age
            gender_text = 'Gender: %.2f' % gender
            cv2.putText(o_image, age_text, (xmin+130, ymin-5), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (0,255,0), 2)
            cv2.putText(o_image, gender_text, (xmin, ymin-5), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (0,255,0), 2)
            
            for cord in range(68):
                cord_x = int(landmark[cord][0]) + new_xmin
                cord_y = int(landmark[cord][1]) + new_ymin
                cv2.circle(o_image, (cord_x, cord_y), 3, (255, 0, 0), -1)

        cv2.imshow("DeepFace", o_image)
            
        if (cv2.waitKey(1) & 0xFF) == ord('q'):
            break
    
    # release all resources
    cv2.destroyAllWindows()
    detect_sess.close()
    proper_sess.close()
    align_sess.close()   
