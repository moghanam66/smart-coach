import os
import time

import cv2
import keras_ocr
import torch
import numpy as np
from scipy import signal
from scipy.interpolate import interp1d
from scipy.signal import find_peaks
import pandas as pd
from detection import DetectionModel, center_of_box
from pose import PoseExtractor
from smooth import Smooth
from ball_detection import BallDetector
from statistics import Statistics
from stroke_recognition import ActionRecognition
from utils import get_video_properties, get_dtype, get_stickman_line_connection
from court_detection import CourtDetector
import matplotlib.pyplot as plt
from fore_back import strokes_dict,strokesCrossOrStraight_dict
from bounce import velocity,bounce,create_top_view

def get_stroke_predictions(video_path, stroke_recognition, strokes_frames, player_boxes):
    """
    Get the stroke prediction for all sections where we detected a stroke
    """
    predictions = {}
    cap = cv2.VideoCapture(video_path)
    fps, length, width, height = get_video_properties(cap)
    video_length = 2
    # For each stroke detected trim video part and predict stroke
    for frame_num in strokes_frames:
        # Trim the video (only relevant frames are taken)
        starting_frame = max(0, frame_num - int(video_length * fps * 2 / 3))
        cap.set(1, starting_frame)
        i = 0

        while True:
            ret, frame = cap.read()

            if not ret:
                break

            stroke_recognition.add_frame(frame, player_boxes[starting_frame + i])
            i += 1
            if i == int(video_length * fps):
                break
        # predict the stroke
        probs, stroke = stroke_recognition.predict_saved_seq()
        predictions[frame_num] = {'probs': probs, 'stroke': stroke}
    cap.release()
    return predictions

   
def find_strokes_indices(player_1_boxes, player_2_boxes, ball_positions, skeleton_df, verbose=0):
    """
    Detect strokes frames using location of the ball and players
    """
    ball_x, ball_y = ball_positions[:, 0], ball_positions[:, 1]
    smooth_x = signal.savgol_filter(ball_x, 3, 2 ,mode='mirror')
    smooth_y = signal.savgol_filter(ball_y, 3, 2 ,mode='mirror')

    # Ball position interpolation
    x = np.arange(0, len(smooth_y))
    indices = [i for i, val in enumerate(smooth_y) if np.isnan(val)]
    x = np.delete(x, indices)
    y1 = np.delete(smooth_y, indices)
    y2 = np.delete(smooth_x, indices)
    ball_f2_y = interp1d(x, y1, kind='cubic', fill_value="extrapolate")
    ball_f2_x = interp1d(x, y2, kind='cubic', fill_value="extrapolate")
    xnew = np.linspace(0, len(ball_y), num=len(ball_y), endpoint=True)

    if verbose:
        plt.plot(np.arange(0, len(smooth_y)), smooth_y, 'o', xnew,
                 ball_f2_y(xnew), '-r')
        plt.legend(['data', 'inter'], loc='best')
        plt.show()

    # Player 2 position interpolation
    player_2_centers = np.array([center_of_box(box) for box in player_2_boxes])
    player_2_x, player_2_y = player_2_centers[:, 0], player_2_centers[:, 1]
    player_2_x = signal.savgol_filter(player_2_x, 3, 2 ,mode='mirror')
    player_2_y = signal.savgol_filter(player_2_y, 3, 2 ,mode='mirror')
    x = np.arange(0, len(player_2_y))
    indices = [i for i, val in enumerate(player_2_y) if np.isnan(val)]
    x = np.delete(x, indices)
    y1 = np.delete(player_2_y, indices)
    y2 = np.delete(player_2_x, indices)
    player_2_f_y = interp1d(x, y1, fill_value="extrapolate")

    player_2_f_x = interp1d(x, y2, fill_value="extrapolate")
    xnew = np.linspace(0, len(player_2_y), num=len(player_2_y), endpoint=True)

    if verbose:
        plt.plot(np.arange(0, len(player_2_y)), player_2_y, 'o', xnew, player_2_f_y(xnew), '--g')
        plt.legend(['data', 'inter_cubic', 'inter_lin'], loc='best')
        plt.show()

    coordinates = ball_f2_y(xnew)
    # Find all peaks of the ball y index
    peaks, _ = find_peaks(coordinates)
    if verbose:
        plt.plot(coordinates)
        plt.plot(peaks, coordinates[peaks], "x")
        plt.show()

    neg_peaks, _ = find_peaks(coordinates * -1)
    if verbose:
        plt.plot(coordinates)
        plt.plot(neg_peaks, coordinates[neg_peaks], "x")
        plt.show()

    # Get bottom player wrists positions
    left_wrist_index = 9
    right_wrist_index = 10
    skeleton_df = skeleton_df.fillna(-1)
    left_wrist_pos = skeleton_df.iloc[:, [left_wrist_index, left_wrist_index + 15]].values
    right_wrist_pos = skeleton_df.iloc[:, [right_wrist_index, right_wrist_index + 15]].values

    dists = []
    # Calculate dist between ball and bottom player
    for i, player_box in enumerate(player_1_boxes):
        if player_box[0] is not None:
            player_center = center_of_box(player_box)
            ball_pos = np.array([ball_f2_x(i), ball_f2_y(i)])
            box_dist = np.linalg.norm(player_center - ball_pos)
            right_wrist_dist, left_wrist_dist = np.inf, np.inf
            if right_wrist_pos[i, 0] > 0:
                right_wrist_dist = np.linalg.norm(right_wrist_pos[i, :] - ball_pos)
            if left_wrist_pos[i, 0] > 0:
                left_wrist_dist = np.linalg.norm(left_wrist_pos[i, :] - ball_pos)
            dists.append(min(box_dist, right_wrist_dist, left_wrist_dist))
        else:
            dists.append(None)
    dists = np.array(dists)

    dists2 = []
    # Calculate dist between ball and top player
    for i in range(len(player_2_centers)):
        ball_pos = np.array([ball_f2_x(i), ball_f2_y(i)])
        box_center = np.array([player_2_f_x(i), player_2_f_y(i)])
        box_dist = np.linalg.norm(box_center - ball_pos)
        dists2.append(box_dist)
    dists2 = np.array(dists2)

    strokes_1_indices = []
    # Find stroke for bottom player by thresholding the dists
    for peak in peaks:
        player_box_height = max(player_1_boxes[peak][3] - player_1_boxes[peak][1], 130)
        if dists[peak] < (player_box_height * 4 / 5):
            strokes_1_indices.append(peak)

    strokes_2_indices = []
    # Find stroke for top player by thresholding the dists
    for peak in neg_peaks:
        if dists2[peak] < 100:
            strokes_2_indices.append(peak)

    # Assert the diff between to consecutive strokes is below some threshold
    while True:
        diffs = np.diff(strokes_1_indices)
        to_del = []
        for i, diff in enumerate(diffs):
            if diff < 40:
                max_in = np.argmax([dists[strokes_1_indices[i]], dists[strokes_1_indices[i + 1]]])
                to_del.append(i + max_in)

        strokes_1_indices = np.delete(strokes_1_indices, to_del)
        if len(to_del) == 0:
            break

    # Assert the diff between to consecutive strokes is below some threshold
    while True:
        diffs = np.diff(strokes_2_indices)
        to_del = []
        for i, diff in enumerate(diffs):
            if diff < 40:
                max_in = np.argmax([dists2[strokes_2_indices[i]], dists2[strokes_2_indices[i + 1]]])
                to_del.append(i + max_in)

        strokes_2_indices = np.delete(strokes_2_indices, to_del)
        if len(to_del) == 0:
            break

    # # Assume bounces frames are all the other peaks in the y index graph
    bounces_indices = [x for x in peaks if x not in strokes_1_indices]
    if verbose:
        plt.figure()
        plt.plot(coordinates)
        plt.plot(strokes_1_indices, coordinates[strokes_1_indices], "or")
        plt.plot(strokes_2_indices, coordinates[strokes_2_indices], "og")
        plt.legend(['data', 'player 1 strokes', 'player 2 strokes'], loc='best')
        plt.show()

    return strokes_1_indices, strokes_2_indices,bounces_indices, player_2_f_x, player_2_f_y


def mark_player_box(frame, boxes, frame_num):
    box = boxes[frame_num]
    if box[0] is not None:
        cv2.rectangle(frame, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), [255, 0, 255], 2)
    return frame


def mark_skeleton(skeleton_df, img, img_no_frame, frame_number):
    """
    Mark the skeleton of the bottom player on the frame
    """
    # landmarks colors
    circle_color, line_color = (0, 0, 255), (255, 0, 0)
    stickman_pairs = get_stickman_line_connection()

    skeleton_df = skeleton_df.fillna(-1)
    values = np.array(skeleton_df.values[frame_number], int)
    points = list(zip(values[5:17], values[22:]))
    # draw key points
    for point in points:
        if point[0] >= 0 and point[1] >= 0:
            xy = tuple(np.array([point[0], point[1]], int))
            cv2.circle(img, xy, 2, circle_color, 2)
            cv2.circle(img_no_frame, xy, 2, circle_color, 2)

    # Draw stickman
    for pair in stickman_pairs:
        partA = pair[0] - 5
        partB = pair[1] - 5
        if points[partA][0] >= 0 and points[partA][1] >= 0 and points[partB][0] >= 0 and points[partB][1] >= 0:
            cv2.line(img, points[partA], points[partB], line_color, 1, lineType=cv2.LINE_AA)
            cv2.line(img_no_frame, points[partA], points[partB], line_color, 1, lineType=cv2.LINE_AA)
    return img, img_no_frame


def add_data_to_video(input_video, court_detector, players_detector, ball_detector, strokes_predictions, skeleton_df_p1,skeleton_df_p2,
                      statistics,bounce_indices,
                      show_video, with_frame, output_folder, output_file, p1, p2, f_x, f_y , strokesDictionary,P1CorS,P2CorS):
    """
    Creates new videos with pose stickman, face landmarks and blinks counter
    :param input_video: str, path to the input videos
    :param df: DataFrame, data of the pose stickman positions
    :param show_video: bool, display output videos while processing
    :param with_frame: int, output videos includes the original frame with the landmarks
    (0 - only landmarks, 1 - original frame with landmarks, 2 - original frame with landmarks and only
    landmarks (side by side))
    :param output_folder: str, path to output folder
    :param output_file: str, name of the output file
    :return: None
    """

    player1_boxes = players_detector.player_1_boxes
    player2_boxes = players_detector.player_2_boxes

    player1_dists = statistics.bottomdistarr
    player1_dists.insert(0,0)
    player2_dists = statistics.topdistarr
    player2_dists.insert(0,0)

    if skeleton_df_p1 is not None:
        skeleton_df_p1 = skeleton_df_p1.fillna(-1)
    if skeleton_df_p2 is not None:
        skeleton_df_p2 = skeleton_df_p2.fillna(-1)

    # Read videos file
    cap = cv2.VideoCapture(input_video)

    # videos properties
    fps, length, width, height = get_video_properties(cap)

    final_width = width * 2 if with_frame == 2 else width

    # Video writer
    out = cv2.VideoWriter(os.path.join(output_folder, output_file + '.mp4'),
                          cv2.VideoWriter_fourcc(*'MP4V'), fps, (final_width, height))

    # initialize frame counters
    frame_number = 0
    orig_frame = 0
    framesToWriteOn = 0
    while True:
        orig_frame += 1
        print('Creating new videos frame %d/%d  ' % (orig_frame, length), '\r', end='')
        if not orig_frame % 100:
            print('')
        ret, img = cap.read()

        if not ret:
            break

        # initialize frame for landmarks only
        img_no_frame = np.ones_like(img) * 255

        # add Court location
        img = court_detector.add_court_overlay(img, overlay_color=(0, 0, 255), frame_num=frame_number)
        img_no_frame = court_detector.add_court_overlay(img_no_frame, overlay_color=(0, 0, 255), frame_num=frame_number)

        # add players locations
        img = mark_player_box(img, player1_boxes, frame_number)
        img = mark_player_box(img, player2_boxes, frame_number)
        img_no_frame = mark_player_box(img_no_frame, player1_boxes, frame_number)
        img_no_frame = mark_player_box(img_no_frame, player2_boxes, frame_number)

        # add ball indices
        try:
            if frame_number in bounce_indices:
                xy=ball_detector.xy_coordinates
                center_coordinates = int(xy[frame_number][0]), int(xy[frame_number][1])
                color = (255, 0, 0)
                thickness = -1
                cv2.circle(img, center_coordinates, 10, color, thickness)
        except:
            print("couldn't draw ball bounces")
        # add middle line
        # try:
        #     point1=(court_detector.middle_line[0],court_detector.middle_line[1])
        #     point2=(court_detector.middle_line[2],court_detector.middle_line[3])
        #     cv2.line(img,pt1=point1,pt2=point2,color=(0, 255, 0),thickness=5)
        # except:
        #     print("couldn't draw middle_line")   
        # add ball location
        img = ball_detector.mark_positions(img, frame_num=frame_number)
        img_no_frame = ball_detector.mark_positions(img_no_frame, frame_num=frame_number, ball_color='black')

        # add pose stickman
        if skeleton_df_p1 is not None:
            img, img_no_frame = mark_skeleton(skeleton_df_p1, img, img_no_frame, frame_number)
        if skeleton_df_p2 is not None:
            img, img_no_frame = mark_skeleton(skeleton_df_p2, img, img_no_frame, frame_number)

        print("frame_number" , frame_number)
        #add stroke to video
        if frame_number in strokesDictionary.keys():
            value=strokesDictionary[frame_number]+'('+P1CorS[frame_number]+')'
            framesToWriteOn = 20
            

        if framesToWriteOn > 0 :  
            cv2.putText(img,value ,(int(player1_boxes[frame_number][0]) - 10, int(player1_boxes[frame_number][1]) - 10),cv2.FONT_HERSHEY_SIMPLEX,1, (0, 255, 255), 2, cv2.LINE_4)
            framesToWriteOn-=1

        # Add stroke prediction
        #for i in range(-10, 10):
            
            #if frame_number + i in strokes_predictions.keys():
                #cv2.putText(img, 'STROKE HIT', (200, 200),
                            #cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 255) if i != 0 else (255, 0, 0), 3)

                #probs, stroke = strokes_predictions[frame_number + i]['probs'], strokes_predictions[frame_number + i][
                    #'stroke']
                #cv2.putText(img, 'Forehand - {:.2f}, Backhand - {:.2f}, Service - {:.2f}'.format(*probs),
                            #(70, 400),
                            #cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                #cv2.putText(img, f'Stroke : {stroke}',
                            #(int(player1_boxes[frame_number][0]) - 10, int(player1_boxes[frame_number][1]) - 10),
                            #cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)

                #break
                
        # Add stroke detected
        for i in range(-5, 10):
            # if frame_number + i in p1:
            #     cv2.putText(img, 'Stroke detected', (int(player1_boxes[frame_number][0]) - 10, int(player1_boxes[frame_number][1]) - 10),
            #                 cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255) if i != 0 else (255, 0, 0), 2)

            if frame_number + i in p2:
                cv2.putText(img, 'Stroke detected'+'('+P2CorS[frame_number + i]+')',
                            (int(f_x(frame_number)) - 30, int(f_y(frame_number)) - 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255) if i != 0 else (255, 0, 0), 2)

        cv2.putText(img, 'Distance: {:.2f} m'.format(player1_dists[frame_number] ),
                    (50, 500),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        cv2.putText(img, 'Distance: {:.2f} m'.format(player2_dists[frame_number]),
                    (100, 150),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        # display frame
        if show_video:
            cv2.imshow('Output', img)
            if cv2.waitKey(1) & 0xff == 27:
                cv2.destroyAllWindows()

        # save output videos
        if with_frame == 0:
            final_frame = img_no_frame
        elif with_frame == 1:
            final_frame = img
        else:
            final_frame = np.concatenate([img, img_no_frame], 1)
        out.write(final_frame)
        frame_number += 1
    print('Creating new video frames %d/%d  ' % (length, length), '\n', end='')
    print(f'New videos created, file name - {output_file}.mp4')
    cap.release()
    out.release()
    cv2.destroyAllWindows()


# def create_top_view(court_detector, detection_model):
#     """
#     Creates top view video of the gameplay
#     """
#     court = court_detector.court_reference.court.copy()
#     court = cv2.line(court, *court_detector.court_reference.net, 255, 5)
#     v_width, v_height = court.shape[::-1]
#     court = cv2.cvtColor(court, cv2.COLOR_GRAY2BGR)
#     out = cv2.VideoWriter('output/top_view.avi',
#                           cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 30, (v_width, v_height))
#     # players location on court
#     smoothed_1, smoothed_2 = detection_model.calculate_feet_positions(court_detector)

#     for feet_pos_1, feet_pos_2 in zip(smoothed_1, smoothed_2):
#         frame = court.copy()
#         frame = cv2.circle(frame, (int(feet_pos_1[0]), int(feet_pos_1[1])), 10, (0, 0, 255), 15)
#         if feet_pos_2[0] is not None:
#             frame = cv2.circle(frame, (int(feet_pos_2[0]), int(feet_pos_2[1])), 10, (0, 0, 255), 15)
#         out.write(frame)
#     out.release()
#     cv2.destroyAllWindows()


def video_process(video_path, show_video=False, include_video=True,
                  stickman=True, stickman_box=True, court=True,
                  output_file='output', output_folder='output',
                  smoothing=True, top_view=True):
    """
    Takes videos of one person as input, and calculate the body pose and face landmarks, and saves them as csv files.
    Also, output a result videos with the keypoints marked.
    :param court:
    :param video_path: str, path to the videos
    :param show_video: bool, show processed videos while processing (default = False)
    :param include_video: bool, result output videos will include the original videos as well as the
    keypoints (default = True)
    :param stickman: bool, calculate pose and create stickman using the pose data (default = True)
    :param stickman_box: bool, show person bounding box in the output videos (default = False)
    :param output_file: str, output file name (default = 'output')
    :param output_folder: str, output folder name (default = 'output') will create new folder if it does not exist
    :param smoothing: bool, use smoothing on output data (default = True)
    :return: None
    """
    dtype = get_dtype()

    # initialize extractors
    court_detector = CourtDetector()
    detection_model = DetectionModel(dtype=dtype)
    pose_extractor = PoseExtractor(person_num=1, box=stickman_box, dtype=dtype) if stickman else None
    stroke_recognition = ActionRecognition('storke_classifier_weights.pth')
    ball_detector = BallDetector('saved states/tracknet_weights_2_classes.pth', out_channels=2)

    # Load videos from videos path
    video = cv2.VideoCapture(video_path)
    # get videos properties
    fps, length, v_width, v_height = get_video_properties(video)

    # frame counter
    frame_i = 0
    # time counter
    total_time = 0
    total_firstplayerscores = []
    total_secondplayerscores = []
    # Loop over all frames in the videos
    while True:
        start_time = time.time()
        ret, frame = video.read()
        frame_i += 1
        if ret:
            if frame_i == 1:
                court_detector.detect(frame)
                print(f'Court detection {"Success" if court_detector.success_flag else "Failed"}')
                print(f'Time to detect court :  {time.time() - start_time} seconds')
                start_time = time.time()
                pipeline = keras_ocr.pipeline.Pipeline()
            court_detector.track_court(frame)
            # OCR
            try:
                if frame_i == 1 or frame_i % 60 == 0  :

                    cropped =frame[v_height*3//4:,:v_width*3//9]
                    gray_image = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
                    #img=cv2.Canny(gray_image, 100, 200)
                    images=[cv2.cvtColor(gray_image, cv2.COLOR_GRAY2RGB)]
                    prediction_groups = pipeline.recognize(images)
                    # print("prediction_groups" ,  prediction_groups)
                    listofy = []
                    for  i in prediction_groups[0]:
                        listofy.append(i[1][0][1])
                        
                    firstplayerscore = []
                    secondplayerscore = []
                    listofxPlayerOne=[]
                    listofxPlayerTwo=[]
                    for i in prediction_groups[0]:
                        if i[1][0][1] < np.mean(listofy):
                            firstplayerscore.append(i[0])
                            listofxPlayerOne.append(i[1][0][0])
                        else:
                            secondplayerscore.append(i[0])
                            listofxPlayerTwo.append(i[1][0][0])

                    indexOfScoreP1=listofxPlayerOne.index(max(listofxPlayerOne))
                    largeScoreP1=firstplayerscore[indexOfScoreP1]
                    firstplayerscore.sort(reverse=True,key=len)
                    firstplayerscore.append(largeScoreP1)

                    
                    indexOfScoreP2=listofxPlayerTwo.index(max(listofxPlayerTwo))
                    largeScoreP2=secondplayerscore[indexOfScoreP2]
                    secondplayerscore.sort(reverse=True,key=len)
                    secondplayerscore.append(largeScoreP2)

                    total_firstplayerscores.append(firstplayerscore)
                    total_secondplayerscores.append(secondplayerscore)
                else:
                    total_firstplayerscores.append(total_firstplayerscores[-1])
                    total_secondplayerscores.append(total_secondplayerscores[-1])
            except:
                pass        

            # detect
            detection_model.detect_player_1(frame.copy(), court_detector)
            detection_model.detect_top_persons(frame, court_detector, frame_i)

            # Create stick man figure (pose detection)
            if stickman:
                pose_extractor.extract_pose_p1(frame, detection_model.player_1_boxes,court_detector,frame_i)
                

            ball_detector.detect_ball(court_detector.delete_extra_parts(frame))

            total_time += (time.time() - start_time)
            print('Processing frame %d/%d  FPS %04f ,expected time to finish is %d mins' \
                % (frame_i, length, frame_i / total_time,((length-frame_i)*(total_time/frame_i))/60), '\r', end='')
            if not frame_i % 100:
                print('')
        else:
            break
    print('Processing frame %d/%d  FPS %04f' % (length, length, length / total_time), '\n', end='')
    print('Processing completed')
    video.release()
    cv2.destroyAllWindows()
    detection_model.bfill_player1_boxes()

    # print("total_firstplayerscores" , total_firstplayerscores)
    # print("total_secondplayerscores" , total_secondplayerscores)
    velocities,coords=velocity(ball_detector.xy_coordinates.astype(float),fps)
    
    #OCR: add scores to DF
    try:
        firstplayerscores_df = pd.DataFrame(total_firstplayerscores)
        secondplayerscores_df=pd.DataFrame(total_secondplayerscores)

        #selected columns from the ocr df:[name,score]
        player1_ocr=firstplayerscores_df[[firstplayerscores_df.columns[0],firstplayerscores_df.columns[-1]]]
        player2_ocr=secondplayerscores_df[[secondplayerscores_df.columns[0],secondplayerscores_df.columns[-1]]]
        all_playersScores=[firstplayerscores_df,secondplayerscores_df]
        selected_playersScores=[player1_ocr,player2_ocr]
        selected_df=pd.concat(selected_playersScores,axis=1)
        ocr_all_df=pd.concat(all_playersScores,axis=1)
        try:
            selected_df.rename(columns = {firstplayerscores_df.columns[0]:'player1', firstplayerscores_df.columns[-1]:'player1_score',secondplayerscores_df.columns[0]:'player2',secondplayerscores_df.columns[-1]:'player2_score'}, inplace = True)
        except:
            print('Error renaming ocr columns')

        selected_df.to_csv('players score.csv',index=False)
        
        ocr_all_df.to_csv('ocr_players_all_scores.csv',index=False)
    except:
        print('Error Writing ocr csv')

    detection_model.find_player_2_box()
    video = cv2.VideoCapture(video_path)
    frame_i_p2=0
    while True:
        start_time = time.time()
        ret, frame = video.read()
        frame_i_p2 += 1
        if ret:
            player2_box= detection_model.player_2_boxes[frame_i_p2-1]
            pose_extractor.extract_pose_p2(frame,player2_box,frame_i_p2)
            print('Processing frame for pose estimation palyer two %d/%d  ' \
                % (frame_i_p2, length), '\r', end='')
            if not frame_i_p2 % 100:
                print('')
        else:
            break
    if top_view:
        create_top_view(court_detector, detection_model,ball_detector.xy_coordinates,fps)

    # Save landmarks in csv files
    df = None
    # Save stickman data
    if stickman:
        try:
            df = pose_extractor.save_to_csv(output_folder)
            df2 = pose_extractor.save_to_csv(output_folder,p1=False)
        except:
            print("pose could not be extracted")

    # smooth the output data for better results
    df_smooth = None
    if smoothing:
        smoother = Smooth()
        df_smooth = smoother.smooth(df)
        smoother.save_to_csv(output_folder)
        df_smooth2 = smoother.smooth(df2)
        smoother.save_to_csv(output_folder,p1=False)

    #print(len(detection_model.player_1_boxes) , detection_model.player_1_boxes)
    player1boxes = []
    for x in detection_model.player_1_boxes:
        y=tuple(x)
        player1boxes.append(y)

    player2boxes = []
    for x in detection_model.player_2_boxes:
        y = tuple(x)
        player2boxes.append(y)

    ballpositions = []
    for x in ball_detector.xy_coordinates:
        y = tuple(x)
        ballpositions.append(y)
    
    #player1scoresdf = pd.DataFrame(total_firstplayerscores , columns=["player_name" , "score"])
    #player2scoredf = pd.DataFrame(total_secondplayerscores , columns=["player_name" , "score"])
    #player1scoresdf.to_csv("player1score.csv")
    #player2scoredf.to_csv("player2score.csv")
    print('player1boxes length',len(player1boxes))
    print('player2boxes length',len(player2boxes))
    print('ballpositions length',len(ballpositions))
    print('ballvelocities length',len(velocities))


    

    try:
        dic = {"player1boxes" : player1boxes , "player2boxes" : player2boxes , "ballpositions" :ballpositions,"ballvelocities":velocities}
        df = pd.DataFrame(dic)
        df.to_csv("summary.csv",index=False)
    except:
        print('Error in dic writng to summary')
        pass

    player_1_strokes_indices, player_2_strokes_indices,bounces_indices, f2_x, f2_y = find_strokes_indices(
        detection_model.player_1_boxes,
        detection_model.player_2_boxes,
        ball_detector.xy_coordinates,
        df_smooth)
    # # bounce indicies indicated by time series model
    # bounces_indices= bounce(velocities,coords)
    '''ball_detector.bounces_indices = bounces_indices
    ball_detector.coordinates = (f2_x, f2_y)'''
    predictions = get_stroke_predictions(video_path, stroke_recognition,
                                         player_1_strokes_indices, detection_model.player_1_boxes)

    statistics = Statistics(court_detector, detection_model)
    heatmap = statistics.get_player_position_heatmap()
    statistics.display_heatmap(heatmap, court_detector.court_reference.court, title='Heatmap')
    statistics.get_players_dists()
    statistics.players_distances()
    statistics.player2_distances()

    player1strokesindices = []
    for x in player_1_strokes_indices:
        player1strokesindices.append(x)
    print("player_1_strokes_indices" , player1strokesindices)
    
    player2strokesindices = []
    for x in player_2_strokes_indices:
        player2strokesindices.append(x)
    print( "player_2_strokes_indices" ,player2strokesindices)

    bouncesindices = []
    for x in bounces_indices:
        bouncesindices.append(x)
    print("bounces_indices" , bouncesindices)
    
    ballbouncesdic = {"ball_bounces" : bouncesindices}
    ballbouncesdf = pd.DataFrame(ballbouncesdic)
    ballbouncesdf.to_csv("ballbounces.csv",index=False)
    
    topbaseline = []
    for x in court_detector.baseline_top:
        topbaseline.append(x)
    print("top_base_line" , topbaseline)

    bottombaseline = []
    for x in court_detector.baseline_bottom:
        bottombaseline.append(x)
    print("bottom_base_line" ,bottombaseline)

    leftline = []
    for x in court_detector.left_court_line:
        leftline.append(x)
    print("left_line" , leftline)

    rightline = []
    for x in court_detector.right_court_line:
        rightline.append(x)
    print("right_line" , rightline)

    leftinnerline = []
    for x in court_detector.left_inner_line:
        leftinnerline.append(x)
    print("left_inner_line" , leftinnerline)

    rightinnerline = []
    for x in court_detector.right_inner_line:
        rightinnerline.append(x)
    print("right_inner_line" , rightinnerline)

    topinnerline = []
    for x in court_detector.top_inner_line:
        topinnerline.append(x)
    print("top_inner_line" , topinnerline)

    bottominnerline = []
    for x in court_detector.bottom_inner_line:
        bottominnerline.append(x)
    print("bottom_inner_line" , bottominnerline)

    midline = []
    for x in court_detector.middle_line:
        midline.append(x)
    print("middle_line" , midline)

    print("predictions" , predictions)

    #print("court_detector.bottom_inner_line", len(court_detector.bottom_inner_line), court_detector.bottom_inner_line)
    #print("predictions" , predictions)
    #print(np.ndim(detection_model.player_1_boxes))
    #print(np.ndim(ball_detector.xy_coordinates))
    #dictionary = dict(player1boxesarr = detection_model.player_1_boxes , player2boxesarr =  detection_model.player_2_boxes
                        #,xy_coordinatesarr =  ball_detector.xy_coordinates )
    #print(dictionary)
    #dict_1 = dict([(k,pd.Series(v)) for k,v in dictionary.items()])
    #pddf = pd.DataFrame(dict_1)      
    #pddf.to_csv("summary.csv")
    stickmandf = pd.read_csv("output/stickman_data_smoothed.csv")

    summarydf = pd.read_csv("summary.csv")

    strokesDictionary = strokes_dict(player1strokesindices,player2strokesindices , summarydf , stickmandf,detection_model,court_detector)
    strokesDictionaryStraigthtOrCross = strokesCrossOrStraight_dict(player1strokesindices ,player2strokesindices, summarydf, court_detector)

    print('strokesDictionary: ',strokesDictionary)
    print('strokesDictionaryCorS: ',strokesDictionaryStraigthtOrCross)
    try:
        pd.DataFrame(strokesDictionary.items()).to_csv('player1StrockType.csv',index=False)
    except:
        print("Error writing player1strock to df")

    try:
        pd.DataFrame(strokesDictionaryStraigthtOrCross["player1stroke"].items()).to_csv('player1CorS.csv',index=False)
        pd.DataFrame(strokesDictionaryStraigthtOrCross["player2stroke"].items()).to_csv('player2CorS.csv',index=False)
    except:
        print("Error writing playerstrockCorS to df")

    try:
        add_data_to_video(input_video=video_path, court_detector=court_detector, players_detector=detection_model,
                      ball_detector=ball_detector, strokes_predictions=predictions, skeleton_df_p1=df_smooth,skeleton_df_p2=df_smooth2,
                      statistics=statistics,bounce_indices=bounces_indices,
                      show_video=False, with_frame=1, output_folder=output_folder, output_file=output_file,
                      p1=player_1_strokes_indices, p2=player_2_strokes_indices, f_x=f2_x, f_y=f2_y , strokesDictionary = strokesDictionary,
                      P1CorS=strokesDictionaryStraigthtOrCross["player1stroke"],P2CorS=strokesDictionaryStraigthtOrCross["player2stroke"])
    except:
             print("Error add_data_to_video")             

    # ball_detector.show_y_graph(detection_model.player_1_boxes, detection_model.player_2_boxes)
    player1_dists = statistics.bottomdistarr
    print(player1_dists[-1])
    player2_dists = statistics.topdistarr
    print(player2_dists[-1])

    player1distdic = {"p1dist":player1_dists}
    player1distdf = pd.DataFrame(player1distdic)
    player1distdf.to_csv("player1dist.csv")

    player2distdic = {"p2dist":player2_dists}
    player2distdf = pd.DataFrame(player2distdic)
    player2distdf.to_csv("player2dist.csv")

    

def main():
    torch.cuda.empty_cache()
    s = time.time()
    video_process(video_path='../videos/2_serves.mp4', show_video=True, stickman=True, stickman_box=False, smoothing=True,
                  court=True, top_view=True)
    print(f'Total computation time : {time.time() - s} seconds')


if __name__ == "__main__":
    main()
