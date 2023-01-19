import numpy as np
import pandas as pd
import math
from ast import literal_eval
from court_reference import CourtReference
def center_of_box(box):
        """
        Calculate the center of a box
        """
        if box[0] is None:
            return None, None
        height = box[3] - box[1]
        width = box[2] - box[0]
        return box[0] + width / 2, box[1] + height / 2

def top_of_box_calc(box):
    """
    Calculate the top of the box
    """
    if box[0] is None:
        return None, None
    return box[1]

def strokes_dict(player1strokes,player2strokes , summarydf, stickmandf, detection_model, court_detector):

    for index,row in summarydf.iterrows():
        x = summarydf.at[index,'player1boxes']
        z = literal_eval(x)
        summarydf.at[index,'player1boxes'] = tuple(z)

    for index,row in summarydf.iterrows():
        x = summarydf.at[index,'ballpositions']
        z = literal_eval(x)
        summarydf.at[index,'ballpositions'] = tuple(z)
    
    df3 = summarydf.join(stickmandf)

    for index , row in df3.iterrows():
        player1box = df3.at[index , 'player1boxes']
        x,y = center_of_box(player1box)
        li = [x,y]
        df3.loc[index , "centerofplayer1boxx"] = str(tuple(li))

    for index , row in df3.iterrows():
        player1box = df3.at[index , 'player1boxes']
        y = top_of_box_calc(player1box)
        df3.loc[index , "topofplayer1boxx"] = y

    for index,row in df3.iterrows():
        x = df3.at[index,'centerofplayer1boxx']
        z = literal_eval(x)
        df3.at[index,'centerofplayer1boxx'] = tuple(z)

    
    smoothed_1, _ = detection_model.calculate_feet_positions(court_detector)
    listOfStrokes=player1strokes.copy()
    listOfStrokes.extend(player2strokes)
    listOfStrokes.sort()
    dic1 = {}

    for i,x in enumerate(player1strokes):
        cen_p1_box = df3.loc[x , "centerofplayer1boxx"]
        for j in range(x,0,-1):
            RRx = df3.loc[j , "ballpositions"]
            print(j,"rrx",RRx)
            if (RRx[0] is not None):
                break
        right_wrest_x = df3.loc[x , "right_wrist_x"]
        # left_wrest_y = df3.loc[x , "left_wrist_y"]
        # right_wrest_y = df3.loc[x , "right_wrist_y"]
        top_of_box = (df3.loc[x , "topofplayer1boxx"])
        cort_referance=CourtReference()
        baseline_bottom_y=cort_referance.get_important_lines()[3][1]
        for j in range(x,0,-1):
                left_eye_y=df3.loc[j , "left_eye_y"]
                if ~np.isnan(left_eye_y):
                    break
        print("smooth1",smoothed_1[x])
        print('baseline_bottom_y',baseline_bottom_y)
        print("left_eye_y",left_eye_y)
        print("right_wrest_x",right_wrest_x)
        print("cen_p1_box[0]",cen_p1_box[0])
        print("rrx",RRx)
        print('not np.isnan(right_wrest_x)',not np.isnan(right_wrest_x))
        
        if (RRx[1] is not None) and(RRx[1] + 80 < left_eye_y) and ((smoothed_1[x][1] < baseline_bottom_y+450)and(smoothed_1[x][1] > baseline_bottom_y-65)):
            strokeIndex=listOfStrokes.index(x)
            if strokeIndex == len(listOfStrokes)-1:
                dic1[x] = 'serve'
            else:
                if listOfStrokes[strokeIndex +1] in player2strokes:
                    dic1[x]='serve and returned'
                else:
                    dic1[x]='serve'
        else:
            if not (np.isnan(right_wrest_x)):
                if right_wrest_x >= cen_p1_box[0]:
                    dic1[x] = "forehand"
                else:
                    dic1[x] = "backhand"
            else: 
                if RRx[0] >= cen_p1_box[0]:
                    dic1[x] = "forehand"
                else:
                    dic1[x] = "backhand"   
    return(dic1)

def strokesCrossOrStraight_dict(player1strokes ,player2strokes, summarydf, court_detector):
    df3 = summarydf.copy()
    middleLineX =court_detector.middle_line[0]
    dic1 = {'player1stroke':{},'player2stroke':{}}
    listOfStrokes=player1strokes.copy()
    listOfStrokes.extend(player2strokes)
    listOfStrokes.sort()
    if len(listOfStrokes)>0:
        for i,stroke in enumerate(listOfStrokes):

            for j in range(stroke,0,-1):
                initialBallPosition=df3.loc[j , "ballpositions"]
                if (initialBallPosition[0] is not None) and (initialBallPosition[1] is not None):
                    break

            if i !=len(listOfStrokes)-1:
                if stroke in player1strokes :
                    dic1["player1stroke"][stroke]='straight'
                    frames_ditected=0
                    for indexBallPosition in range(stroke,listOfStrokes[i+1]):
                        RRx = df3.loc[indexBallPosition , "ballpositions"]
                        if (RRx[0] is not None):
                            if ((RRx[0] > middleLineX and initialBallPosition[0]< middleLineX) or \
                            (RRx[0] < middleLineX and initialBallPosition[0] > middleLineX)):
                                frames_ditected+=1
                                if frames_ditected>3:
                                    dic1["player1stroke"][stroke]='cross'
                                    break
                            else:
                                frames_ditected=0
                else:
                    dic1["player2stroke"][stroke]='straight'
                    frames_ditected=0
                    for indexBallPosition in range(stroke,listOfStrokes[i+1]):
                        RRx = df3.loc[indexBallPosition , "ballpositions"]
                        if (RRx[0] is not None):
                            if ((RRx[0] > middleLineX and initialBallPosition[0]< middleLineX) or \
                            (RRx[0] < middleLineX and initialBallPosition[0]> middleLineX)):
                                frames_ditected+=1
                                if frames_ditected>3:
                                    dic1["player2stroke"][stroke]='cross'
                                    break
                            else:
                                frames_ditected=0
            else:
                if stroke in player1strokes :
                    dic1["player1stroke"][stroke]='straight'
                    frames_ditected=0
                    for indexBallPosition in range(stroke,summarydf.shape[0]):
                        RRx = df3.loc[indexBallPosition , "ballpositions"]
                        if (RRx[0] is not None):
                            if ((RRx[0] > middleLineX and initialBallPosition[0]< middleLineX) or \
                            (RRx[0] < middleLineX and initialBallPosition[0]> middleLineX)):
                                frames_ditected+=1
                                if frames_ditected>5:
                                    dic1["player1stroke"][stroke]='cross'
                                    break
                        else:
                            frames_ditected=0
                else:
                    dic1["player2stroke"][stroke]='straight'
                    frames_ditected=0
                    for indexBallPosition in range(stroke,summarydf.shape[0]):
                        RRx = df3.loc[indexBallPosition , "ballpositions"]
                        if (RRx[0] is not None):
                            if ((RRx[0] > middleLineX and initialBallPosition[0]< middleLineX) or \
                            (RRx[0] < middleLineX and initialBallPosition[0]> middleLineX)):
                                frames_ditected+=1
                                if frames_ditected>5:
                                    dic1["player2stroke"][stroke]='cross'
                                    break
                        else:
                            frames_ditected=0

        return(dic1)
#player_1_strokes_indices = [62, 125, 198, 286]
#stickmandf = pd.read_csv("output/stickman_data_smoothed.csv")
#summarydf = pd.read_csv("summary.csv")

#strokes_dict(player_1_strokes_indices , summarydf , stickmandf)