import sys
sys.path.append(".")

import cv2
import numpy as np
from tensorflow import keras
from Solver.sudokuBoard import SudokuBoard
import time
import configparser


CELL_DIM=50


def preprocess_image(image):
    new_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    new_image = cv2.GaussianBlur(new_image, (5, 5), 3)
    new_image = cv2.adaptiveThreshold(
        new_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 13, 2)
    new_image = cv2.morphologyEx(new_image, cv2.MORPH_CLOSE, (3, 3))
    new_image = cv2.morphologyEx(new_image, cv2.MORPH_OPEN, (3, 3))

    return new_image


def find_board_corners(image):
    contours, _ = cv2.findContours(
        image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    max_area = 0
    max_conture = None
    
    for conture in contours:
        peri = cv2.arcLength(conture, True)
        conture_approx = cv2.approxPolyDP(conture, 0.05 * peri, True)
        area = cv2.contourArea(conture_approx)
        if(area > max_area and len(conture_approx) == 4):
            max_area = area
            max_conture = conture_approx

    return max_conture, max_area

def orient_points_in_conture_clockwise(conture):
    ordered_index=np.argpartition(conture[:,1], 2)
    top2_index=ordered_index[:2]
    bottom2_index=ordered_index[2:]

    if(conture[top2_index[0]][0]>conture[top2_index[1]][0]):
        top_left=conture[top2_index[1]]
        top_right=conture[top2_index[0]]
    else:
        top_left=conture[top2_index[0]]
        top_right=conture[top2_index[1]]

    if(conture[bottom2_index[0]][0]>conture[bottom2_index[1]][0]):
        bottom_left=conture[bottom2_index[1]]
        bottom_right=conture[bottom2_index[0]]
    else:
        bottom_left=conture[bottom2_index[0]]
        bottom_right=conture[bottom2_index[1]]
    
    
    return np.array([top_left,top_right,bottom_right,bottom_left])


def find_distance(x, y):
    return np.sqrt((x[0]-y[0])**2+(x[1]-y[1])**2)


def get_perspective_transformation_matrix(conture):
    source_points = np.reshape(conture, (4, 2)).astype(np.float32)
    source_points=orient_points_in_conture_clockwise(source_points)

    width = max(int(find_distance(source_points[0], source_points[1])), int(
        find_distance(source_points[3], source_points[2])))
    height = max(int(find_distance(source_points[0], source_points[3])), int(
        find_distance(source_points[1], source_points[2])))

    dest_points = np.array(
        [[0, 0], [width-1, 0], [width-1, height-1], [0, height-1]], dtype=np.float32)
    
    perspective_matrix = cv2.getPerspectiveTransform(
        source_points, dest_points)

    return perspective_matrix, width, height

def get_cells_from_image(image,width,height,margin_size=0.1, number_of_cells=9):
    size_x=width/number_of_cells
    size_y=height/number_of_cells
    margin_x=size_x*margin_size
    margin_y=size_y*margin_size
    cells=[]
    position_x=0
    position_y=0
    while(position_y<=height-size_y):
        while(position_x<=width-size_x):
            cell=image[int(position_y+margin_y):int(position_y+size_y-margin_y),int(position_x+margin_x):int(position_x+size_x-margin_y)]
            cell=cv2.resize(cell,(CELL_DIM,CELL_DIM))
            cells.append(cell)
            position_x+=int(size_x)
        position_y+=int(size_y)
        position_x=0    
    return cells

def filter_cells_for_classification(cells):
    filtered_cells=[]
    values=[]
    for cell in cells:
        contours, _ = cv2.findContours(cell, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if(len(contours)>0):
            c = max(contours, key = cv2.contourArea)
            mask = np.zeros(cell.shape, dtype="uint8")
            cv2.drawContours(mask, [c], -1, 255, -1)
            percentFilled = cv2.countNonZero(mask) / float(cell.shape[0] *cell.shape[1])

            if(percentFilled<0.1):
                values.append(0)
            else:
                digit = cv2.bitwise_and(cell, cell, mask=mask)
                if(np.count_nonzero(digit[int(CELL_DIM*0.3):-int(CELL_DIM*0.3),int(CELL_DIM*0.3):-int(CELL_DIM*0.3)])>0):
                    filtered_cells.append(np.reshape(digit,(1,CELL_DIM,CELL_DIM,1)))
                    values.append(-1)
                else:
                    values.append(0)
        else:
            values.append(0)
    
    return filtered_cells,values


def draw_solution_mask(shape,width,height,matrix,filled_cells,size_up_dim=50):
    image=np.full(shape,255,dtype='uint8')
    fontScale = 1
    color = (0, 0, 0)
    thickness = 2
    font = cv2.FONT_HERSHEY_SCRIPT_COMPLEX
    dim_x=int(width/9)
    dim_y=int(height/9)
    if(dim_x>size_up_dim and dim_y>size_up_dim):
        fontScale=2
    index=0
    for row in range(0,9):
        for column in range(0,9):
            if(filled_cells[index]==0):
                position_x=int(column*dim_x+dim_x*0.3)
                position_y=int(row*dim_y+dim_y*0.78)
                image=cv2.putText(
                    img=image,
                    text=str(int(matrix[row,column])),
                    org=(position_x,position_y), 
                    bottomLeftOrigin=False,
                    fontScale=fontScale,
                    thickness=thickness,
                    color=color,
                    fontFace=font,
                    lineType=cv2.LINE_AA)
            index+=1

    return image


def check_border(image, threshold=0.2, border_width=5):
    width=image.shape[1]
    height=image.shape[0]
    
    top_border=image[0:border_width,:]
    bottom_border=image[-border_width:,:]
    left_border=image[:,0:border_width]
    right_border=image[:,-border_width:]

    top_border_pct_filed=np.count_nonzero(top_border)/(width*border_width)
    bottom_border_pct_filed=np.count_nonzero(bottom_border)/(width*border_width)
    left_border_pct_filed=np.count_nonzero(left_border)/(height*border_width)
    right_border_pct_filed=np.count_nonzero(right_border)/(height*border_width)
    #print(top_border_pct_filed,bottom_border_pct_filed,left_border_pct_filed,right_border_pct_filed)
    if(top_border_pct_filed>threshold and bottom_border_pct_filed>threshold and left_border_pct_filed>threshold and right_border_pct_filed>threshold):
        return True
    else:
        return False


if __name__ == "__main__":
    config = configparser.ConfigParser()
    config.read('config.ini')

    model = keras.models.load_model(config['Resources']['model'])
    image = cv2.imread(config['Resources']['image'])
 

    image_area=image.shape[0]*image.shape[1]
    start_time=time.time()

    image_pre = preprocess_image(image)
    image_area=image_pre.shape[0]*image_pre.shape[1]

    conture,conture_area = find_board_corners(image_pre)
    img_contures = cv2.drawContours(image, [conture], -1, (100, 100, 255), 2)
    perspective_matrix, width, height = get_perspective_transformation_matrix(conture)
    #invers_matrix=np.linalg.inv(perspective_matrix)
    img_warped = cv2.warpPerspective(image_pre, perspective_matrix, (width, height))
    #img_warped_color=cv2.warpPerspective(image, perspective_matrix, (width, height))
    #cv2.imshow("warped", img_warped)
    #cv2.imshow('conture',img_contures)
    #print(time.time()-start_time)  
    #print(image_area,conture_area)
    if(check_border(img_warped) and image_area*0.2<conture_area):
    
        cells=get_cells_from_image(img_warped,width,height)

        filtered_cells,values=filter_cells_for_classification(cells)

        # for index,cell in enumerate(filtered_cells):
        #     cv2.imwrite(f"F:\\ML Projects\\CUBIC Praksa\\SudokuSolver\\Data\\TestCells\\{index}.jpg",np.reshape(cell,(50,50)))

        if(len(filtered_cells)>0):
            batch=np.concatenate(filtered_cells)

            prediction = model.predict_on_batch(batch)
            predicted_values=np.argmax(prediction,axis=1)+1

        predicted_index=0
        board_input_matrix=np.copy(values)
        for index,value in enumerate(values):
            if(value==-1):
                board_input_matrix[index]=int(predicted_values[predicted_index])
                predicted_index+=1

        game_board=SudokuBoard(np.reshape(board_input_matrix,(9,9)),input_type='array')
                
                
        if(game_board.solve()):
            print(time.time()-start_time)  
            solved=draw_solution_mask(img_warped.shape,width,height,game_board.board,values)
            solved=cv2.warpPerspective(solved,perspective_matrix,(image.shape[1],image.shape[0]),borderValue=255,flags=cv2.WARP_INVERSE_MAP)

            #cv2.imshow('solved',solved)
            #cv2.imshow('org',image)
            image[:,:,1]=np.bitwise_and(image[:,:,1],solved)
        else:
            print('NO SOLUTION FOUND')
            print(game_board.board)
    else:
        print('BAD CONTURE')

    cv2.imshow("final_image", image)
    print(time.time()-start_time)    
    cv2.waitKey(0)