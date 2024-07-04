# import the opencv library 
import cv2 
import time

# define a video capture object 0
vid = cv2.VideoCapture(3, apiPreference=cv2.CAP_ANY, params=[
    cv2.CAP_PROP_FRAME_WIDTH, 3840,
    cv2.CAP_PROP_FRAME_HEIGHT, 1080]) 
num = 0

while(True): 
	
	# Capture the video frame 
	# by frame 
	ret, frame = vid.read() 

	if(ret == True):
	# Display the resulting frame 
		# frame = cv2.resize(frame, (0,0),fx=0.25,fy=0.25)
		start_t = time.time()
		cv2.imshow('frame', frame) 
		end_t = time.time()
		print("Time taken to show image: ", end_t-start_t)
		#reads first image from queue, will wait 0.5 seconds
		#  before declaring the queue empty
		#creates unique output name for image
		start_t = time.time()
		name = "./images/output_" + str(num) + ".png"
		#writes image with 'name' name to src dir as a png
		cv2.imwrite(name, frame, [cv2.IMWRITE_PNG_COMPRESSION, 1])
		end_t = time.time()
		print("Time taken to write image: ", end_t-start_t)
		#moves image with 'name' to a folder called images
		num+=1
		
		# the 'q' button is set as the 
		# quitting button you may use any 
		# desired button of your choice 
		if cv2.waitKey(1) & 0xFF == ord('q'): 
			break

# After the loop release the cap object 
vid.release() 
# Destroy all the windows 
cv2.destroyAllWindows()