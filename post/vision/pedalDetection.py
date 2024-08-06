import numpy as np
import cv2 as cv
import csv

path = "post/data/Bowel_S216_T1_2024-07-18/obs/2024-07-18 15-48-29"
frameNum = 0
onePixel = (600,1020)
twoPixel = (725,1020)
threePixel = (1230,1020)
fourPixel = (1580,1020)
threshold = 150

if __name__ == "__main__":
    cap = cv.VideoCapture(f"{path}.mkv")
    cap.set(cv.CAP_PROP_POS_FRAMES, frameNum)
    cv.namedWindow('frame',cv.WINDOW_NORMAL)
    csvFile = open(f"{path}.csv", 'w', newline='')
    out = csv.writer(csvFile)
    out.writerow(["Frame Num","Upper Left", "Lower Left", "Upper Right", "Lower Right"])

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # cv.imshow("frame",frame)
        frameNum += 1
        oneColor = frame[onePixel[1],onePixel[0]]
        twoColor = frame[twoPixel[1],twoPixel[0]]
        threeColor = frame[threePixel[1], threePixel[0]]
        fourColor = frame[fourPixel[1],fourPixel[0]]
        # print(threeColor,fourColor)

        if (int(oneColor[1]) + int(oneColor[2]))/2 > threshold:
            cv.imwrite(f'{path}/img{frameNum}.jpg', frame)
            upperLeft = 1
        else:
            upperLeft = 0
        if int(oneColor[0]) > threshold:
            cv.imwrite(f'{path}/img{frameNum}.jpg', frame)
            lowerLeft = 1
        else:
            lowerLeft = 0

        if (int(fourColor[1]) + int(fourColor[2]))/2 > threshold or (int(threeColor[1]) + int(threeColor[2]))/2 > threshold:
            cv.imwrite(f'{path}/img{frameNum}.jpg', frame)
            upperRight = 1
        else:
            upperRight = 0
        if int(fourColor[0]) > threshold or int(threeColor[0]) > threshold:
            cv.imwrite(f'{path}/img{frameNum}.jpg', frame)
            lowerRight = 1
        else:
            lowerRight = 0

        out.writerow([frameNum,upperLeft, lowerLeft, upperRight, lowerRight])
        # prevColor = rightColor
        if cv.waitKey(1) & 0xFF == ord('q'): 
            break
    print("Done")
    cap.release()