import pandas as pd 
import cv2 
import numpy as np
from sktime.datatypes._panel._convert import from_2d_array_to_nested
from pickle import load,dump
from sklearn.pipeline import Pipeline
from sktime.classification.interval_based import TimeSeriesForestClassifier
from sktime.transformations.panel.compose import ColumnConcatenator

def draw_ball_position(frame, court_detector, xy, i):
        """
        Calculate the ball position of both players using the inverse transformation of the court and the x, y positions
        """
        inv_mats = court_detector.game_warp_matrix[i]
        coord = xy
        img = frame.copy()
        # Ball locations
        if coord is not None:
          p = np.array(coord,dtype='float64')
          ball_pos = np.array([p[0].item(), p[1].item()]).reshape((1, 1, 2))
          transformed = cv2.perspectiveTransform(ball_pos, inv_mats)[0][0].astype('int64')
          cv2.circle(frame, (transformed[0], transformed[1]), 35, (0,255,255), -1)
        else:
          pass
        return img 

def create_top_view(court_detector, detection_model, xy, fps):
    """
    Creates top view video of the gameplay
    """
    x, y = diff_xy(xy)
    coords=remove_outliers(x, y, xy)
    # Interpolation
    coords = interpolation(coords)
    coords = coords[:]
    court = court_detector.court_reference.court.copy()
    court = cv2.line(court, *court_detector.court_reference.net, 255, 5)
    v_width, v_height = court.shape[::-1]
    court = cv2.cvtColor(court, cv2.COLOR_GRAY2BGR)
    out = cv2.VideoWriter('output/minimap.mp4',cv2.VideoWriter_fourcc('X', 'V', 'I', 'D'), fps, (v_width, v_height))
    # players location on court
    smoothed_1, smoothed_2 = detection_model.calculate_feet_positions(court_detector)
    i = 0 
    for feet_pos_1, feet_pos_2 in zip(smoothed_1, smoothed_2):
        frame = court.copy()
        frame = cv2.circle(frame, (int(feet_pos_1[0]), int(feet_pos_1[1])), 45, (255, 0, 0), -1)
        if feet_pos_2[0] is not None:
            frame = cv2.circle(frame, (int(feet_pos_2[0]), int(feet_pos_2[1])), 45, (255, 0, 0), -1)
        draw_ball_position(frame, court_detector, coords[i], i)
        i += 1
        out.write(frame)
    out.release()
    cv2.destroyAllWindows()

def interpolation(coords):
  coords =coords.copy().astype(float)
  x, y = [x[0] if x is not None else np.nan for x in coords], [x[1] if x is not None else np.nan for x in coords]

  xxx = np.array(x) # x coords
  yyy = np.array(y) # y coords

  nons, yy = nan_helper(xxx)
  xxx[nons]= np.interp(yy(nons), yy(~nons), xxx[~nons])
  nans, xx = nan_helper(yyy)
  yyy[nans]= np.interp(xx(nans), xx(~nans), yyy[~nans])

  newCoords = [*zip(xxx,yyy)]

  return newCoords

def nan_helper(y):
    """Helper to handle indices and logical indices of NaNs.

    Input:
        - y, 1d numpy array with possible NaNs
    Output:
        - nans, logical indices of NaNs
        - index, a function, with signature indices= index(logical_indices),
          to convert logical indices of NaNs to 'equivalent' indices
    Example:
        >>> # linear interpolation of NaNs
        >>> nans, x= nan_helper(y)
        >>> y[nans]= np.interp(x(nans), x(~nans), y[~nans])
    """

    return np.isnan(y), lambda z: z.nonzero()[0]

def diff_xy(coords):
  coords = coords.copy().astype(float)
  diff_list = []
  for i in range(0, len(coords)-1):
    if (coords[i][0] is not np.NaN) and \
        (coords[i+1][0] is not np.NaN) and \
        (coords[i][1] is not np.NaN) and\
        (coords[i+1][1] is not np.NaN):
      point1 = coords[i]
      point2 = coords[i+1]

      diff = [abs(point2[0] - point1[0]), abs(point2[1] - point1[1])]
      diff_list.append(diff)
    else:
      diff_list.append(np.NaN)
  
  xx, yy = np.array([x[0] if x is not None else np.nan for x in diff_list]), np.array([x[1] if x is not None else np.nan for x in diff_list])
  
  return xx, yy

def remove_outliers(x, y, coords):
    coords = coords.copy().astype(float)
    ids = set(np.where(x > 50)[0]) & set(np.where(y > 50)[0])
    for id in ids:
        left, middle, right = coords[id-1], coords[id], coords[id+1]
        if left is np.NaN :
            left = [0]
        if  right is np.NaN:
            right = [0]
        if middle is np.NaN:
            middle = [0]
        MAX = max(map(list, (left, middle, right)))
        if MAX == [0]:
            pass
        else:
            try:
                coords[np.where(coords == tuple(MAX))] = None
            except ValueError:
                coords[np.where(coords == MAX)] = None
    return coords

def velocity(xy:list,fps:float):
    coords=xy.copy()
    for _ in range(3):
        x, y = diff_xy(coords)
        coords=remove_outliers(x, y, coords)

    # interpolation
    coords = interpolation(coords)
    #time
    t=1/fps

    # velocty 
    Vx = []
    Vy = []
    V = []
    frames = [*range(len(coords))]

    for i in range(len(coords)-1):
        p1 = coords[i]
        p2 = coords[i+1]
        
        x = (p1[0]-p2[0])/t
        y = (p1[1]-p2[1])/t
        Vx.append(x)
        Vy.append(y)

    for i in range(len(Vx)):
        vx = Vx[i]
        vy = Vy[i]
        v = (vx**2+vy**2)**0.5
        V.append(v)
    Vx.insert(0,0)
    Vy.insert(0,0)
    V.insert(0, 0)
    return (V,coords)
def train_bounce_model():
    data_train=pd.read_csv('bigDF.csv',index_col=0)
    y=data_train['bounce']

    # df.shift
    for i in range(20, 0, -1): 
        data_train[f'lagX_{i}'] = data_train['x'].shift(i, fill_value=0)
    for i in range(20, 0, -1): 
        data_train[f'lagY_{i}'] = data_train['y'].shift(i, fill_value=0)
    for i in range(20, 0, -1): 
        data_train[f'lagV_{i}'] = data_train['V'].shift(i, fill_value=0)

    data_train.drop(['x', 'y', 'V','bounce'], 1, inplace=True)

    Xs = data_train[['lagX_20', 'lagX_19', 'lagX_18', 'lagX_17', 'lagX_16',
        'lagX_15', 'lagX_14', 'lagX_13', 'lagX_12', 'lagX_11', 'lagX_10',
        'lagX_9', 'lagX_8', 'lagX_7', 'lagX_6', 'lagX_5', 'lagX_4', 'lagX_3',
        'lagX_2', 'lagX_1']]
    Xs = from_2d_array_to_nested(Xs.to_numpy())

    Ys = data_train[['lagY_20', 'lagY_19', 'lagY_18', 'lagY_17',
        'lagY_16', 'lagY_15', 'lagY_14', 'lagY_13', 'lagY_12', 'lagY_11',
        'lagY_10', 'lagY_9', 'lagY_8', 'lagY_7', 'lagY_6', 'lagY_5', 'lagY_4',
        'lagY_3', 'lagY_2', 'lagY_1']]
    Ys = from_2d_array_to_nested(Ys.to_numpy())

    Vs = data_train[['lagV_20', 'lagV_19', 'lagV_18',
        'lagV_17', 'lagV_16', 'lagV_15', 'lagV_14', 'lagV_13', 'lagV_12',
        'lagV_11', 'lagV_10', 'lagV_9', 'lagV_8', 'lagV_7', 'lagV_6', 'lagV_5',
        'lagV_4', 'lagV_3', 'lagV_2', 'lagV_1']]
    Vs = from_2d_array_to_nested(Vs.to_numpy())
    X=pd.DataFrame()
    X['Xs']=Xs
    X['Ys']=Ys
    X['Vs']=Vs
    steps = [
        ("concatenate", ColumnConcatenator()),
        ("classify", TimeSeriesForestClassifier(n_estimators=100)),
    ]
    clf = Pipeline(steps)
    clf.fit(X, y)
    filename = 'saved states/clf.pkl'
    dump(clf, open(filename, 'wb'))



def bounce(V:list,coords:list):
    xy = coords[:]

    # Predicting Bounces 
    test_df = pd.DataFrame({'x': [coord[0] for coord in xy[:-1]], 'y':[coord[1] for coord in xy[:-1]], 'V': V[1:]})
    # df.shift
    test_df.to_csv('test_df.csv',index=False)
    for i in range(20, 0, -1): 
        test_df[f'lagX_{i}'] = test_df['x'].shift(i, fill_value=0)
    for i in range(20, 0, -1): 
        test_df[f'lagY_{i}'] = test_df['y'].shift(i, fill_value=0)
    for i in range(20, 0, -1): 
        test_df[f'lagV_{i}'] = test_df['V'].shift(i, fill_value=0)



    Xs = test_df[['lagX_20', 'lagX_19', 'lagX_18', 'lagX_17', 'lagX_16',
        'lagX_15', 'lagX_14', 'lagX_13', 'lagX_12', 'lagX_11', 'lagX_10',
        'lagX_9', 'lagX_8', 'lagX_7', 'lagX_6', 'lagX_5', 'lagX_4', 'lagX_3',
        'lagX_2', 'lagX_1']]
    Xs = from_2d_array_to_nested(Xs.astype('float').to_numpy(dtype='float'))

    Ys = test_df[['lagY_20', 'lagY_19', 'lagY_18', 'lagY_17',
        'lagY_16', 'lagY_15', 'lagY_14', 'lagY_13', 'lagY_12', 'lagY_11',
        'lagY_10', 'lagY_9', 'lagY_8', 'lagY_7', 'lagY_6', 'lagY_5', 'lagY_4',
        'lagY_3', 'lagY_2', 'lagY_1']]
    Ys = from_2d_array_to_nested(Ys.astype('float').to_numpy(dtype='float'))

    Vs = test_df[['lagV_20', 'lagV_19', 'lagV_18',
        'lagV_17', 'lagV_16', 'lagV_15', 'lagV_14', 'lagV_13', 'lagV_12',
        'lagV_11', 'lagV_10', 'lagV_9', 'lagV_8', 'lagV_7', 'lagV_6', 'lagV_5',
        'lagV_4', 'lagV_3', 'lagV_2', 'lagV_1']]
    Vs = from_2d_array_to_nested(Vs.astype('float').to_numpy(dtype='float'))
    X_test=pd.DataFrame()
    X_test['Xs']=Xs
    X_test['Ys']=Ys
    X_test['Vs']=Vs
    # load the pre-trained classifier  
    clf = load(open('saved states/clf.pkl', 'rb'))
    predcted = clf.predict(X_test)
    idx = list(np.where(predcted == 1)[0])
    idx = np.array(idx)

    return idx

