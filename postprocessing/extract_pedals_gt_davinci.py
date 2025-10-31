import numpy as np
import cv2 as cv
import csv
import glob
import os
# import comparePlot

# onePixel = (600,1020)
# twoPixel = (725,1020)
# threePixel = (1230,1020)
# fourPixel = (1580,1020)


#  udpated for davinci pedal locations at 720p left view
onePixel = (133, 1209)
twoPixel = (253, 1209)
threePixel = (370, 1209)
fourPixel = (490, 1209)
threshold = 150


# def guiPedalExtraction(path):
#     frameNum = 0
#     vidpath = glob.glob(f"{path}/obs/*.mkv")
#     cap = cv.VideoCapture(vidpath[0])
#     print(vidpath[0])
#     cap.set(cv.CAP_PROP_POS_FRAMES, frameNum)
#     if os.path.exists(f"{path}/obs/pedal_detection.csv"):
#         print(f"{path}/obs/pedal_detection.csv exists already!")
#         return        
#     csvFile = open(f"{path}/obs/pedal_detection.csv", 'w', newline='')
#     out = csv.writer(csvFile)
#     out.writerow(["Frame Num","Upper Left", "Lower Left", "Upper Right", "Lower Right"])
#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             break
#         # cv.imshow("frame",frame)
#         frameNum += 1
#         if frameNum % 5000 == 0:
#             print(frameNum)
#         oneColor = frame[onePixel[1],onePixel[0]]
#         twoColor = frame[twoPixel[1],twoPixel[0]]
#         threeColor = frame[threePixel[1], threePixel[0]]
#         fourColor = frame[fourPixel[1],fourPixel[0]]
#         # print(threeColor,fourColor)
#         if (int(oneColor[1]) + int(oneColor[2]))/2 > threshold:
#             # cv.imwrite(f'{path}/img{frameNum}.jpg', frame)
#             upperLeft = 1
#         else:
#             upperLeft = 0
#         if int(oneColor[0]) > threshold:
#             # cv.imwrite(f'{path}/img{frameNum}.jpg', frame)
#             lowerLeft = 1
#         else:
#             lowerLeft = 0
#         if (int(fourColor[1]) + int(fourColor[2]))/2 > threshold or (int(threeColor[1]) + int(threeColor[2]))/2 > threshold:
#             # cv.imwrite(f'{path}/img{frameNum}.jpg', frame)
#             upperRight = 1
#         else:
#             upperRight = 0
#         if int(fourColor[0]) > threshold or int(threeColor[0]) > threshold:
#             # cv.imwrite(f'{path}/img{frameNum}.jpg', frame)
#             lowerRight = 1
#         else:
#             lowerRight = 0
#         out.writerow([frameNum,upperLeft, lowerLeft, upperRight, lowerRight])
#         csvFile.flush()
#         if cv.waitKey(1) & 0xFF == ord('q'): 
#             break
#     print("Saved csv")
#     cap.release()


if __name__ == "__main__":
    dataset_dir ="/standard/UVA-DSA/MIDAS/Organized/Bootcamp/SuturingV2/Processed/*"
    matchingPath = glob.glob(dataset_dir)
    # matchingPath.pop(0)
    for path in matchingPath:
        if "Gesture_Clips" in path:
            continue
        frameNum = 0
        vidpath = glob.glob(f"{path}/synched_data/*.mp4")
        if len(vidpath) > 0:
            print(f"Processing {path} with {vidpath[0]}")
        else:
            print(f"No video found for {path}, skipping...")
            continue

        cap = cv.VideoCapture(vidpath[0])
        cap.set(cv.CAP_PROP_POS_FRAMES, frameNum)
        # csvFile = open(f"{path}/obs/pedal_detection.csv", 'w', newline='')
        # out = csv.writer(csvFile)
        # out.writerow(["Frame Num","Upper Left", "Lower Left", "Upper Right", "Lower Right"])

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            # cv.imshow("frame",frame)
            frameNum += 1

            # print("Processing frame ", frameNum, "with shape ", frame.shape)
            # # visualize pixel pedal locations on frame with circles
            # cv.circle(frame, onePixel, 10, (0, 255, 0), -1)
            # cv.circle(frame, twoPixel, 10, (0, 255, 0), -1)
            # cv.circle(frame, threePixel, 10, (0, 255, 0), -1)
            # cv.circle(frame, fourPixel, 10, (0, 255, 0), -1)
            # # save frame with circles for debugging
            # cv.imwrite(f"{path}/pedal_debug_frame_{frameNum}.jpg", frame)


            oneColor = frame[onePixel[0],onePixel[1]]
            twoColor = frame[twoPixel[0],twoPixel[1]]
            threeColor = frame[threePixel[0], threePixel[1]]
            fourColor = frame[fourPixel[0],fourPixel[1]]
            # print(threeColor,fourColor)

            if (int(oneColor[1]) + int(oneColor[2]))/2 > threshold:
                # cv.imwrite(f'{path}/img{frameNum}.jpg', frame)
                upperLeft = 1
            else:
                upperLeft = 0
            if int(oneColor[0]) > threshold:
                # cv.imwrite(f'{path}/img{frameNum}.jpg', frame)
                lowerLeft = 1
            else:
                lowerLeft = 0

            if (int(fourColor[1]) + int(fourColor[2]))/2 > threshold or (int(threeColor[1]) + int(threeColor[2]))/2 > threshold:
                # cv.imwrite(f'{path}/img{frameNum}.jpg', frame)
                upperRight = 1
            else:
                upperRight = 0
            if int(fourColor[0]) > threshold or int(threeColor[0]) > threshold:
                # cv.imwrite(f'{path}/img{frameNum}.jpg', frame)
                lowerRight = 1
            else:
                lowerRight = 0

            if frameNum % 10 == 0:
                print(frameNum)
                print(oneColor, twoColor, threeColor, fourColor)
                print(upperLeft, lowerLeft, upperRight, lowerRight)

            # out.writerow([frameNum,upperLeft, lowerLeft, upperRight, lowerRight])
            # csvFile.flush()
            if cv.waitKey(1) & 0xFF == ord('q'): 
                break
        print("Saved csv")
        cap.release()
        # comparePlot.plot(path)
        # print("Saved plot")