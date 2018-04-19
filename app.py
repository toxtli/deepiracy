#!/usr/bin/python
# -*- coding: utf-8 -*-
# pylint: disable=C0103
# pylint: disable=E1101

# python app.py 100 video2.mp4 video2.mp4

# Resolution: 1920 x 1080
#
# 2D color
# Operations = 1920 x 1080 x 3
# Time = 2.372
#
# 2D gray
# Operations = 1920 x 1080
# Time = 0.521
#
# SURF
# Time = 0.243
#
# CNN
# Time = 0.087
#
# Tracking
# Time = 0.003
#
# FLANN
# Size 1 = 4542
# Size 2 = 4117
# Time = 0.109
#
# BFMatcher
# Size 1 = 4542
# Size 2 = 4117
# Time = 0.164

import os
import sys
import cv2
import time
import youtube_dl
import numpy as np
import edit_distance
import tensorflow as tf
from imutils.video import FileVideoStream
from imutils.video import FPS
from utils import label_map_util
from utils import visualization_utils_color as vis_util

video_num = 0

def youtube_download_hook(download):
  global video_num
  if download["status"] == "finished":
    video_num += 1
    print(download["filename"])
    print(video_num)
    os.rename(download["filename"], "video%s.mp4" % (video_num))
    if video_num == 2:
      compare_videos("video1.mp4", "video2.mp4")

def compare_videos(path_video_1, path_video_2):
  global detection_graph
  PATH_TO_CKPT = './ssd_inception2.pb'
  PATH_TO_LABELS = './labels.pbtxt'
  thresh = .2
  sequence_sorted = False
  sequence_type = 'char'
  store_output = False
  NUM_CLASSES = 90
  trackers = {}
  positions = {}
  source_frame = 0
  ok = None
  label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
  categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
  category_index = label_map_util.create_category_index(categories)
  detection_graph = tf.Graph()
  with detection_graph.as_default():
      od_graph_def = tf.GraphDef()
      with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
          serialized_graph = fid.read()
          od_graph_def.ParseFromString(serialized_graph)
          tf.import_graph_def(od_graph_def, name='')
  with detection_graph.as_default():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(graph=detection_graph, config=config) as sess:
      fourcc = cv2.VideoWriter_fourcc(*'mp4v')
      #sift = cv2.xfeatures2d.SIFT_create()
      surf = cv2.xfeatures2d.SURF_create()
      #fast = cv2.FastFeatureDetector_create()
      #orb = cv2.ORB_create()
      desc = surf
      descriptor = "surf"
      show_points = 20
      video_1 = cv2.VideoCapture(path_video_1)
      video_2 = cv2.VideoCapture(path_video_2)
      out = None
      _, frame_1 = video_1.read()
      objects_1 = detect_objects(frame_1, thresh, detection_graph, sess, category_index, sequence_sorted=sequence_sorted, sequence_type=sequence_type)
      sequence_1 = objects_1['sequence']
      objects_2 = None
      area_2 = None
      sequence_2 = ''
      desc_kp_1, desc_des_1 = desc.detectAndCompute(frame_1, None)
      print( desc.descriptorSize() )
      print( desc_des_1.shape )
      until_end = False
      frame_num = int(sys.argv[1])
      if frame_num == -1:
        until_end = True
      first_time = True
      use_descriptor = True
      use_detection = False
      use_tracking = False
      matched_area = None
      frames_to_skip = 0
      processed_frames = 0
      while frame_num or until_end:
        frame_num -= 1
        ok, frame_2 = video_2.read()
        if not ok:
          break
        if first_time:
          print(frame_2.shape[0] * frame_2.shape[1])
          first_time = False

        if use_tracking:
          sequence_tmp = ''
          for object_2 in objects_2['objects']:
            start_time = time.time()
            ok, box = trackers[object_2['coords']].update(frame_2)
            box = (int(box[0]), int(box[1]), int(box[2]), int(box[3]))
            elapsed_time = time.time() - start_time
            #print('tracking', elapsed_time)  
            if ok:
              sequence_tmp += object_2['values'][sequence_type]
              cv2.rectangle(frame_2, (box[0], box[2]), (box[1], box[3]), (255, 0, 0))
          num_matches = len(sequence_tmp)
          if num_matches > 0:
            print('eq: %s ref: %s new: %s' % (num_matches, sequence_1, sequence_tmp))
          else:
            source_frame += processed_frames
            print('skipped frames: %s' % (processed_frames))
            video_1.set(cv2.CAP_PROP_POS_FRAMES, source_frame)
            ok, frame_1 = video_1.read()
            if not ok:
              break
            processed_frames = 0
            objects_1 = detect_objects(frame_1, thresh, detection_graph, sess, category_index, sequence_sorted=sequence_sorted, sequence_type=sequence_type)
            sequence_1 = objects_1['sequence']
            desc_kp_1, desc_des_1 = desc.detectAndCompute(frame_1, None)
            use_tracking = False
            use_detection = False
            use_descriptor = True

        if use_detection:
          area_2 = frame_2[matched_area[0]:matched_area[1],matched_area[2]:matched_area[3]]
          objects_2 =  detect_objects(area_2, thresh, detection_graph, sess, category_index, sequence_sorted=sequence_sorted, sequence_type=sequence_type)
          sequence_2 = objects_2['sequence']
          num_matches = get_sequence_matches(sequence_1, sequence_2)
          print('eq: %s ref: %s new: %s' % (num_matches, sequence_1, sequence_2))
          if num_matches > 0:
            #print('ELEMENTS WERE FOUND')
            trackers = {}
            for object_2 in objects_2['objects']:
              trackers[object_2['coords']] = cv2.TrackerKCF_create()
              trackers[object_2['coords']].init(frame_2, object_2['coords'])
            use_tracking = True
            use_detection = False
            use_descriptor = False
          else:
            use_tracking = False
            use_detection = False
            use_descriptor = True

        if use_descriptor:
          matched_area = None
          start_time = time.time()
          desc_kp_2, desc_des_2 = desc.detectAndCompute(frame_2, None)
          elapsed_time = time.time() - start_time
          print(descriptor, elapsed_time)

          if descriptor == "sift" or descriptor == "surf" or descriptor == "fast":
            FLANN_INDEX_KDTREE = 1
            index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
            search_params = dict(checks=50)   # or pass empty dictionary
            start_time = time.time()
            flann = cv2.FlannBasedMatcher(index_params, search_params)
            matches = flann.knnMatch(desc_des_1, desc_des_2, k=2)
            elapsed_time = time.time() - start_time
            print('FLANN', elapsed_time)
            MIN_MATCH_COUNT = 10
            good = []
            for m,n in matches:
              if m.distance < 0.7*n.distance:
                good.append(m)
            area_2 = frame_2
            if len(good) > MIN_MATCH_COUNT:
              src_pts = np.float32([ desc_kp_1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
              dst_pts = np.float32([ desc_kp_2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)
              M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
              matchesMask = mask.ravel().tolist()
              h,w,d = frame_1.shape
              pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
              dst = cv2.perspectiveTransform(pts,M)
              matched_area = get_rect_from_dst(dst)
              frame_2 = cv2.polylines(frame_2,[np.int32(dst)],True,255,3, cv2.LINE_AA)
            else:
              print( "Not enough matches are found - {}/{}".format(len(good), MIN_MATCH_COUNT) )
              matchesMask = None
          elif descriptor == "orb":
            FLANN_INDEX_LSH = 6
            index_params = dict(algorithm = FLANN_INDEX_LSH,
              table_number = 6, # 12
              key_size = 12,     # 20
              multi_probe_level = 1) #2
            # index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
            search_params = dict(checks=50)   # or pass empty dictionary
            flann = cv2.FlannBasedMatcher(index_params, search_params)
            matches = flann.knnMatch(desc_des_1, desc_des_2, k=2)
            good = []
            for m,n in matches:
              if m.distance < 0.7*n.distance:
                good.append(m)
            print(len(good))
            matches_img = cv2.drawMatches(frame_1, desc_kp_1, frame_2, desc_kp_2, good[:show_points], None)
          
          if matched_area:
            area_2 = frame_2[matched_area[0]:matched_area[1],matched_area[2]:matched_area[3]]
            objects_2 =  detect_objects(area_2, thresh, detection_graph, sess, category_index, sequence_sorted=sequence_sorted, sequence_type=sequence_type)
            sequence_2 = objects_2['sequence']
            print('ref_seq', sequence_1)
            print('new_seq', sequence_2)
            num_matches = get_sequence_matches(sequence_1, sequence_2)
            print('num_matches', num_matches)
            if num_matches > 0:
              print('ELEMENTS FOUND')
              use_descriptor = False
              use_detection = True
              use_tracking = False

          #matches_img = cv2.drawMatches(frame_1, desc_kp_1, frame_2, desc_kp_2, good, None, **draw_params)

        processed_frames += 1
        draw_params = dict(
             matchesMask = matchesMask[:show_points], # draw only inliers
             flags = 2)
        #print("%s of %s rate %s" % (len(good), len(matches), len(good)/len(matches)))
        matches_img = cv2.drawMatches(frame_1, desc_kp_1, frame_2, desc_kp_2, good[:show_points], None, **draw_params)
        if store_output:
          if out == None:
            out = cv2.VideoWriter('out.avi', fourcc, 30.0, (matches_img.shape[1], matches_img.shape[0]), True)
          out.write(matches_img)
        cv2.imshow("Matches", matches_img)
        cv2.waitKey(1)
      out.release()



def detect_objects(image, thresh, detection_graph, sess, category_index, sequence_sorted=False, sequence_type='char'):
  image_np = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
  image_np_expanded = np.expand_dims(image_np, axis=0)
  image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
  boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
  scores = detection_graph.get_tensor_by_name('detection_scores:0')
  classes = detection_graph.get_tensor_by_name('detection_classes:0')
  num_detections = detection_graph.get_tensor_by_name('num_detections:0')
  start_time = time.time()
  (boxes, scores, classes, num_detections) = sess.run(
      [boxes, scores, classes, num_detections],
      feed_dict={image_tensor: image_np_expanded})
  elapsed_time = time.time() - start_time
  print('cnn', elapsed_time)
  box = vis_util.visualize_boxes_and_labels_on_image_array(
      image,
      np.squeeze(boxes),
      np.squeeze(classes).astype(np.int32),
      np.squeeze(scores),
      category_index,
      min_score_thresh=thresh,
      use_normalized_coordinates=True,
      line_thickness=4,
      sequence_sorted=sequence_sorted,
      sequence_type=sequence_type)
  return box

def get_sequence_matches(sequence_1, sequence_2):
  sm = edit_distance.SequenceMatcher(a=sequence_1, b=sequence_2)
  sm.get_opcodes()
  sm.ratio()
  sm.get_matching_blocks()
  distance = sm.distance()
  num_matches = sm.matches()
  return num_matches

def get_rect_from_dst(dst):
  top = int(dst[0][0][1]) if dst[0][0][1] < dst[3][0][1] else int(dst[3][0][1])
  bottom = int(dst[1][0][1]) if dst[1][0][1] > dst[2][0][1] else int(dst[2][0][1])
  left = int(dst[0][0][0]) if dst[0][0][0] < dst[1][0][0] else int(dst[1][0][0])
  right = int(dst[2][0][0]) if dst[2][0][0] > dst[3][0][0] else int(dst[3][0][0])
  return (top, bottom, left, right)

def compare_2d_color_images(frame_1, frame_2):
  start_time = time.time()
  matches_num = 0
  height = frame_1.shape[0]
  width = frame_1.shape[1]
  size = width * height
  for i in range(height):
    for j in range(width):
      if frame_1[i][j][0] == frame_2[i][j][0] and frame_1[i][j][1] == frame_2[i][j][1] and frame_1[i][j][2] == frame_2[i][j][2]:
        matches_num += 1
  rate = matches_num / size
  elapsed_time = time.time() - start_time
  print('iterate_2d', elapsed_time)

def compare_2d_gray_images(frame_1, frame_2):
  start_time = time.time()
  matches_num = 0
  gray_1 = cv2.cvtColor(frame_1, cv2.COLOR_BGR2GRAY)
  gray_2 = cv2.cvtColor(frame_2, cv2.COLOR_BGR2GRAY)
  height = gray_1.shape[0]
  width = gray_1.shape[1]
  size = width * height
  for i in range(height):
    for j in range(width):
      if gray_1[i][j] == gray_2[i][j]:
        matches_num += 1
  rate = matches_num / size
  elapsed_time = time.time() - start_time
  print('iterate_2d', elapsed_time)

def compare_1d_gray_images(frame_1, frame_2):
  start_time = time.time()
  matches_num = 0
  gray_1 = cv2.cvtColor(frame_1, cv2.COLOR_BGR2GRAY)
  flat_1 = [j for i in gray_1 for j in i]
  gray_2 = cv2.cvtColor(frame_2, cv2.COLOR_BGR2GRAY)
  flat_2 = [j for i in gray_2 for j in i]
  size = len(flat_1)
  for i in range(size):
      if flat_1[i] == flat_2[i]:
        matches_num += 1
  rate = matches_num / size
  elapsed_time = time.time() - start_time
  print('iterate_1d', elapsed_time)

def load_from_youtube():
  video_1 = "https://www.youtube.com/watch?v=YYcu7R9k3tY"
  video_2 = "https://www.youtube.com/watch?v=AzWEFLaciaU"
  ydl_opts = {"format": "mp4", "progress_hooks": [youtube_download_hook]}
  youtube_dl.YoutubeDL(ydl_opts).download([video_1, video_2])



#load_from_youtube()
compare_videos(sys.argv[2], sys.argv[3])

# 
"""
import sys
import time
import numpy as np
import tensorflow as tf
import cv2
import PIL.Image as Image

sys.path.append("..")

from utils import label_map_util
from utils import visualization_utils_color as vis_util

def find_homography(kp1, des1, kp2, des2):
  bf = cv2.BFMatcher(cv2.NORM_L2)
  # Match descriptors.
  matches = bf.knnMatch(des1,des2,k=2)
  # Apply ratio test
  good = []
  for m,n in matches:
      if m.distance < 0.9*n.distance:
         good.append(m)
  pts1 = []
  pts2 = []
  for elem in good:
    pts1.append(kp1[elem.queryIdx].pt)
    pts2.append(kp2[elem.trainIdx].pt)
  pts1 = np.array(pts1)
  pts2 = np.array(pts2)
  M, mask = cv2.findHomography(pts1, pts2, cv2.RANSAC,5.0)
  count_inliers = np.count_nonzero(mask)
  #print('Number of inliers: ', np.count_nonzero(mask))
  return count_inliers, M

frames = []

def image_alg(image, box):
  border = 0.2
  im_height = len(image)
  im_width = len(image[0])
  (ymin, xmin, ymax, xmax) = (box[0], box[1], box[2], box[3])
  (left, right, top, bottom) = (int(xmin * im_width), int(xmax * im_width), 
                                int(ymin * im_height), int(ymax * im_height))
  #print((left, right, top, bottom))
  border_height = (bottom - top) * border
  top = 0 if (top - border_height) < 0 else (top - border_height)
  bottom = im_height if (bottom + border_height) > im_height else (bottom + border_height)
  scale_y = im_height/(bottom - top)
  output = cv2.resize(image, (0,0), fy=scale_y, fx=scale_y)
  (xleft, xright, xtop, xbottom, xim_width) = (int(left*scale_y), int(right*scale_y),
                                              int(top*scale_y), int(bottom*scale_y),
                                              int(im_width*scale_y))
  extra_width = (im_width - (xright - xleft)) // 2
  new_left = 0 if (xleft - extra_width) < 0 else (xleft - extra_width)
  new_right = xim_width if (xright + extra_width) > xim_width else (xright + extra_width)
  output = output[xtop:xbottom, new_left:new_right]
  #output = image[top:bottom, left:right]
  #output = cv2.resize(output, (0,0), fy=scale_y, fx=scale_y) 
  return output

def image_stab(image):
  global frames
  WINDOW_SIZE = 15 
  skip = 1 # speedup -- set 1 for original speed
  resize = 0.5 #scale video resolution
  frames = []
  mean_homographies = []
  median_homographies = []
  corrected_frames = []
  i = 0

  frames.append(image)
  if len(frames) > 20:
    frames = frames[1:]

  orb = cv2.xfeatures2d.SIFT_create(nfeatures=1000)
  # orb = cv2.FeatureDetector_create("SIFT")
  # orb = cv2.SIFT_create(nfeatures=1000)
  # orb = cv2.SIFT(nfeatures=1000)

  vec_kps = []
  vec_descs = []

  #print('extracting keypoints...')

  for i in range(len(frames)):
    # find the keypoints and descriptors 
    kp1, des1 = orb.detectAndCompute(frames[i],None)

    vec_kps.append(kp1)
    vec_descs.append(des1)

    #print('Frame %d/%d: found %d keypoints'% (i,len(frames),len(kp1)))



  for i in range(len(frames)):
    mean_H = np.zeros((3,3), dtype='float64')
    median_H = []
    mean_C = 0
    median_vals = []
    k =  int(WINDOW_SIZE/2.0)+1
    for j in range(1,k,1): #for each couple neighbor frames iterated by distance
      if i-j >= 0 and i+j < len(frames):
        inliers_c, H = find_homography(vec_kps[i],vec_descs[i], vec_kps[i-j], vec_descs[i-j])
        inliers_c2, H2 = find_homography(vec_kps[i],vec_descs[i], vec_kps[i+j], vec_descs[i+j])
        #print('pair (%d,%d) has %d inliers'% (i,i-j,inliers_c))
        #print('pair (%d,%d) has %d inliers'% (i,i+j,inliers_c2))
        if inliers_c > 80 and inliers_c2 > 80: #ensures that neighbors are equally selected by distance to correctly balance the homography
          mean_H = mean_H + H
          mean_H = mean_H + H2
          mean_C+=2

    if mean_C > 0:
      mean_homographies.append(mean_H/mean_C) # Mean homography
    else:
      mean_homographies.append(np.eye(3, dtype='float64'))
    
    #print mean_H/mean_C
    #print median_vals
    #raw_input()

      #fourcc = cv2.cv.CV_FOURCC('D','I','V','X')
      #fourcc = cv2.cv.CV_FOURCC('R','G','B',' ')
      #fourcc = cv2.cv.CV_FOURCC('Y','U','Y','2')
      #fourcc = cv2.cv.CV_FOURCC('Y','U','Y','U')
      #fourcc = cv2.cv.CV_FOURCC('U','Y','V','Y')
      #fourcc = cv2.cv.CV_FOURCC('I','4','2','0')
      #fourcc = cv2.cv.CV_FOURCC('I','Y','U','V')
      #fourcc = cv2.cv.CV_FOURCC('Y','U','1','2')
      #fourcc = cv2.cv.CV_FOURCC('Y','8','0','0')
      #fourcc = cv2.cv.CV_FOURCC('G','R','E','Y')
      #fourcc = cv2.cv.CV_FOURCC('B','Y','8',' ')
      #fourcc = cv2.cv.CV_FOURCC('Y','1','6',' ')
  
      #fourcc = cv2.cv.CV_FOURCC('M','J','P','G')
  #fourcc = cv2.cv.CV_FOURCC('M','P','E','G')

  crop_x = 80
  crop_y = 60

  size = (frames[0].shape[1]-crop_x*2, frames[0].shape[0]-crop_y*2)
  
  #fourcc = cv2.VideoWriter_fourcc('X','V','I','D')
  #out =  cv2.VideoWriter(file+'__estabilizado.avi',fourcc,30.0,size)#cv2.VideoWriter('stab.mp4',-1, 30.0, (frames[0].shape[0], frames[0].shape[1]))

  #for i in range(len(frames)):
    #corrected = cv2.warpPerspective(frames[i],mean_homographies[i],(0,0))
    #cv2.imshow('video corrected', corrected)
    #cv2.waitKey(1)
    #new_img = corrected[crop_y:frames[0].shape[0]-crop_y, crop_x:frames[0].shape[1]-crop_x]
    #out.write(new_img)
    #out.write(corrected[crop_y:frames[0].shape[0]-crop_y, crop_x:frames[0].shape[1]-crop_x])

  corrected = cv2.warpPerspective(frames[-1],mean_homographies[-1],(0,0))
  new_img = corrected[crop_y:frames[0].shape[0]-crop_y, crop_x:frames[0].shape[1]-crop_x]
  return new_img


# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_CKPT = './model/frozen_inference_graph_face.pb'

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = './protos/face_label_map.pbtxt'

NUM_CLASSES = 2

label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

def load_image_into_numpy_array(image):
  (im_width, im_height) = image.size
  return np.array(image.getdata()).reshape(
      (im_height, im_width, 3)).astype(np.uint8)

#cap = cv2.VideoCapture("./media/test.mp4")
cap = cv2.VideoCapture(0)
cap.open(0)
# time.sleep(2.0)
out = None

detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

with detection_graph.as_default():
  config = tf.ConfigProto()
  config.gpu_options.allow_growth = True
  with tf.Session(graph=detection_graph, config=config) as sess:
    frame_num = 100;
    while frame_num:
      frame_num -= 1
      ret, image = cap.read()
      if ret == 0:
          break

      if out is None:
          [h, w] = image.shape[:2]
          out = cv2.VideoWriter("./media/test_out.avi", cv2.VideoWriter_fourcc(*'H264'), 25.0, (w, h))

      image_np = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

      # the array based representation of the image will be used later in order to prepare the
      # result image with boxes and labels on it.
      # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
      image_np_expanded = np.expand_dims(image_np, axis=0)
      image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
      # Each box represents a part of the image where a particular object was detected.
      boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
      # Each score represent how level of confidence for each of the objects.
      # Score is shown on the result image, together with the class label.
      scores = detection_graph.get_tensor_by_name('detection_scores:0')
      classes = detection_graph.get_tensor_by_name('detection_classes:0')
      num_detections = detection_graph.get_tensor_by_name('num_detections:0')
      # Actual detection.
      start_time = time.time()
      (boxes, scores, classes, num_detections) = sess.run(
          [boxes, scores, classes, num_detections],
          feed_dict={image_tensor: image_np_expanded})
      elapsed_time = time.time() - start_time
      #print('inference time cost: {}'.format(elapsed_time))
      #print(boxes.shape, boxes)
      #print(scores.shape,scores)
      #print(classes.shape,classes)
      #print(num_detections)
      # Visualization of the results of a detection.
      box = vis_util.visualize_boxes_and_labels_on_image_array(
#          image_np,
          image,
          np.squeeze(boxes),
          np.squeeze(classes).astype(np.int32),
          np.squeeze(scores),
          category_index,
          use_normalized_coordinates=True,
          line_thickness=4)
      
      #cv2.imshow('Video', image)
      if len(box) > 0:
        stab = image_alg(image, box[0])
        cv2.imshow('Stab', stab)
        out.write(stab)
        cv2.waitKey(1)


    cap.release()
    out.release()

    #sift_kp_1, sift_des_1 = sift.detectAndCompute(frame_1, None)
    #surf_kp_1, surf_des_1 = surf.detectAndCompute(frame_1, None)
    #orb_kp_1, orb_des_1 = orb.detectAndCompute(frame_1, None)
    #fast_kp_1, fast_des_1 = fast.detectAndCompute(frame_1, None)
    #print( sift.descriptorSize() )
    #print( surf.descriptorSize() )
    #print( orb.descriptorSize() )
    #print( sift_des_1.shape )
    #print( surf_des_1.shape )
    #print( orb_des_1.shape )
    #sift_kp_2, sift_des_2 = sift.detectAndCompute(frame_2, None)
    #surf_kp_2, surf_des_2 = surf.detectAndCompute(frame_2, None)
    #orb_kp_2, orb_des_2 = orb.detectAndCompute(frame_2, None)
    #fast_kp_2, fast_des_2 = orb.detectAndCompute(frame_2, None)
    #print(len(matches))
    #matches_img = cv2.drawMatchesKnn(frame_1, desc_kp_1, frame_2, desc_kp_2, matches[:10], None)

    matchesMask = [[0,0] for i in range(len(matches))] # ratio test as per Lowe's paper
    for i,(m,n) in enumerate(matches):
        if m.distance < 0.7*n.distance:
            matchesMask[i] = [1,0]
    draw_params = dict(matchColor = (0,255,0),
                       singlePointColor = (255,0,0),
                       matchesMask = matchesMask[:10],
                       flags = 0)
    matches_img = cv2.drawMatchesKnn(frame_1, desc_kp_1, frame_2, desc_kp_2, matches[:10], None, **draw_params)

    start_time = time.time()
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True) # create BFMatcher object
    matches = bf.match(desc_des_1, desc_des_2) # Match descriptors.
    matches = sorted(matches, key = lambda x:x.distance) # Sort them in the order of their distance.
    elapsed_time = time.time() - start_time
    print('BFMatcher', elapsed_time)
    print(len(matches))
    matches_img = cv2.drawMatches(frame_1, desc_kp_1, frame_2, desc_kp_2, matches[:10], None, flags=2)    

    #compare_2d_color_images(frame_1, frame_2)
    #compare_2d_gray_images(frame_1, frame_2)

    res = cv2.matchTemplate(gray_1, gray_2, cv2.TM_CCOEFF_NORMED)
    _, max_val, _, max_loc = cv2.minMaxLoc(res)
    print(max_val)
    threshold = 0.8
    if max_val > threshold:
      print("FOUND")

    #cv2.imshow("image 1", gray_1)
    #cv2.imshow("image 2", gray_2)    
"""        