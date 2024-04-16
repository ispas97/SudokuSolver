import sys
sys.path.append(".")

import cv2
import numpy as np
from tensorflow import keras
from Solver.sudokuBoard import SudokuBoard
import processImage
import time
import configparser

if __name__ == "__main__":
    config = configparser.ConfigParser()
    config.read('config.ini')

    model = keras.models.load_model(config['Resources']['model'])
    vid = cv2.VideoCapture(config['Resources']['video'])

    if (vid.isOpened()== False):
        print("Error opening video")

    board_is_solved=False
    solution=None
    sol_values=None

    fps = vid.get(cv2.CAP_PROP_FPS)
    frame_time=1000/fps

    while(vid.isOpened()):
        start_time=time.time()
        ret, frame = vid.read()
        if(ret==True):
            frame_size=frame.shape[0]*frame.shape[1]
            image_pre = processImage.preprocess_image(frame)
            conture, area = processImage.find_board_corners(image_pre)
            #img_contures = cv2.drawContours(image_pre, [conture], -1, (100, 100, 255), 2)
            #cv2.imwrite("F:\\ML Projects\\CUBIC Praksa\\SudokuSolver\\Data\\warped\\org_frame.jpg",image_pre)
            perspective_matrix, width, height = processImage.get_perspective_transformation_matrix(conture)
            img_warped = cv2.warpPerspective(image_pre, perspective_matrix, (width, height))
            if(area>0.05*frame_size and processImage.check_border(img_warped)):
                #cv2.imwrite("F:\\ML Projects\\CUBIC Praksa\\SudokuSolver\\Data\\warped\\img_warped2.jpg",img_warped)
                #cv2.imwrite("F:\\ML Projects\\CUBIC Praksa\\SudokuSolver\\Data\\warped\\frame2.jpg",frame)
                cells=processImage.get_cells_from_image(img_warped,width,height)
                filtered_cells,values=processImage.filter_cells_for_classification(cells)
                if(not board_is_solved):
                    batch=np.concatenate(filtered_cells)
                    prediction = model.predict_on_batch(batch)
                    predicted_values=np.argmax(prediction,axis=1)+1
                    board_input_matrix=np.copy(values)
                    predicted_index=0
                    for index,value in enumerate(values):
                        if(value==-1):
                            board_input_matrix[index]=int(predicted_values[predicted_index])
                            predicted_index+=1
                    game_board=SudokuBoard(np.reshape(board_input_matrix,(9,9)),input_type='array')
                    if(game_board.solve()):
                        solution=game_board.board
                        sol_values=values
                        board_is_solved=True
                    else:
                        board_is_solved=False
                elif(board_is_solved):
                    solved=processImage.draw_solution_mask(img_warped.shape,width,height,solution,sol_values)
                    solved=cv2.warpPerspective(solved,perspective_matrix,(frame.shape[1],frame.shape[0]),borderValue=255,flags=cv2.WARP_INVERSE_MAP)
                    frame[:,:,1]=np.bitwise_and(frame[:,:,1],solved)
            else:
                cv2.putText(frame,'NO SUDOKU',(0,100),fontScale=2,color=(0,255,0),thickness=2,fontFace=cv2.FONT_HERSHEY_PLAIN)
                

            time_difference=time.time()-start_time

            cv2.putText(frame,str(int(1/time_difference)),(0,50),fontScale=2,color=(0,255,0),thickness=2,fontFace=cv2.FONT_HERSHEY_PLAIN)
            cv2.imshow('frame', frame)
        else:
            break

        remaining_frame_time=int(frame_time-time_difference*1000)
        if cv2.waitKey(max(remaining_frame_time,1)) & 0xFF == ord('q'):
            break

    vid.release()
    cv2.destroyAllWindows()


    