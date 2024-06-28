# import the opencv library 
import cv2 


# define a video capture object 0
vid = cv2.VideoCapture(1) 
num = 0

while(True): 
	
	# Capture the video frame 
	# by frame 
	ret, frame = vid.read() 

	if(ret == True):
	# Display the resulting frame 
		# frame = cv2.resize(frame, (640, 480))
		cv2.imshow('frame', frame) 

		#reads first image from queue, will wait 0.5 seconds before declaring the queue empty
		#creates unique output name for image
		name = "./images/output_" + str(num) + ".png"
		#writes image with 'name' name to src dir as a png
		cv2.imwrite(name, frame, [cv2.IMWRITE_PNG_COMPRESSION, 1])
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
