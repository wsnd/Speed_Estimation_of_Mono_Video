import cv2
import dlib
import math
from imutils.object_detection import non_max_suppression
from enum import Enum

class Detection_type(Enum):
    Car = 0
    Pedestrian = 1

# Car Classifier
carCascade = cv2.CascadeClassifier('myhaar.xml')
# Perdestrian Classifier
hog = cv2.HOGDescriptor()  
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

video = cv2.VideoCapture('cars.mp4')

# WIDTH = 1280
# HEIGHT = 720
WIDTH = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
HEIGHT = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = video.get(cv2.CAP_PROP_FPS)
#OBJ_WIDTH, OBJ_HEIGHT = 0, 0
# initial distance to get focal length
initial_distance = 25.0
#focal_length = 0.0

measure_person_height = 83 # the pixels of person in video
measure_car_height = 100 # the pixels of car in video


def get_focal_length(measure_height, obj_height):
	return measure_height * initial_distance / obj_height

def get_distance(obj_width, obj_height, box_width, box_height, focal_length):
	distance_1 = obj_width * focal_length / box_width
	distance_2 = obj_height * focal_length / box_height
	return distance_1 * 0.95 + distance_2 * 0.05

def estimateSpeed(location1, location2, ppm):
	""" calculate speed of object """
	d_pixels = math.sqrt(math.pow(location2[0] - location1[0], 2) + math.pow(location2[1] - location1[1], 2))
	# ppm = location2[2] / carWidth
	d_meters = d_pixels / ppm
	#print("d_pixels=" + str(d_pixels), "d_meters=" + str(d_meters))
	# fps = 18
	speed = d_meters * fps * 3.6
	return speed
	

def trackMultipleObjects(detection_type):
	rectangleColor = (0, 255, 0)
	frameCounter = 0
	currentObjectID = 0
	fps = 0
	
	if detection_type == Detection_type.Car:
		OBJ_WIDTH, OBJ_HEIGHT = 1.8, 1.8
		focal_length = get_focal_length(measure_car_height, OBJ_HEIGHT)
	elif detection_type == Detection_type.Pedestrian:
		OBJ_WIDTH, OBJ_HEIGHT = 0.8, 1.75
		focal_length = get_focal_length(measure_person_height, OBJ_HEIGHT)
	objectTracker = {} # dict for {carID : tracker}
	# carNumbers = {}
	objectLocation1 = {} # car location of previous frame
	objectLocation2 = {} # car location of current frame
	distance = [None] * 100
	speed = [None] * 100
	
	# Write output to video file
	out = cv2.VideoWriter('outpy.avi', cv2.VideoWriter_fourcc('M','J','P','G'), 10, (WIDTH, HEIGHT))

	while True:
		# start_time = time.time()
		rc, image = video.read()
		if type(image) == type(None):
			break
		
		image = cv2.resize(image, (WIDTH, HEIGHT))
		resultImage = image.copy()
		
		frameCounter = frameCounter + 1
		
		objectIDtoDelete = [] # the carID need to be removed

		for objectID in objectTracker.keys():
			trackingQuality = objectTracker[objectID].update(image)
			# remove the low quality traker 
			if trackingQuality < 7:
				objectIDtoDelete.append(objectID)

		# remove from carTraker, carLocation1, carLocation2		
		for objectID in objectIDtoDelete:
			print ('Removing carID ' + str(objectID) + ' from list of trackers.')
			print ('Removing carID ' + str(objectID) + ' previous location.')
			print ('Removing carID ' + str(objectID) + ' current location.')
			objectTracker.pop(objectID, None)
			objectLocation1.pop(objectID, None)
			objectLocation2.pop(objectID, None)
			
		if not (frameCounter % 30):
			if detection_type == Detection_type.Car:
				OBJ_WIDTH, OBJ_HEIGHT = 1.8, 1.8
				gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
				objects = carCascade.detectMultiScale(gray, 1.1, 13, 18, (64, 64))
			elif detection_type == Detection_type.Pedestrian:
				OBJ_WIDTH, 
				objects, w = hog.detectMultiScale(image, 2, (4, 4), (32, 32), 1.05, 2)
			# use nms to reduce overleap objects
			objects = non_max_suppression(objects, probs=1, overlapThresh=0.15)
			for (_x, _y, _w, _h) in objects:
				x = int(_x)
				y = int(_y)
				w = int(_w)
				h = int(_h)
			
				x_bar = x + 0.5 * w
				y_bar = y + 0.5 * h
				
				matchCarID = None

                # check for if detected car is matched with the car in the objectTracker
				for objectID in objectTracker.keys():
					trackedPosition = objectTracker[objectID].get_position()
					
					t_x = int(trackedPosition.left())
					t_y = int(trackedPosition.top())
					t_w = int(trackedPosition.width())
					t_h = int(trackedPosition.height())
					
					t_x_bar = t_x + 0.5 * t_w
					t_y_bar = t_y + 0.5 * t_h
				
					if ((t_x <= x_bar <= (t_x + t_w)) and (t_y <= y_bar <= (t_y + t_h)) and (x <= t_x_bar <= (x + w)) and (y <= t_y_bar <= (y + h))):
						matchCarID = objectID
				# if there is no matched car, add to objectTracker
				if matchCarID is None:
					print ('Creating new tracker ' + str(currentObjectID))
					
					tracker = dlib.correlation_tracker()
					tracker.start_track(image, dlib.rectangle(x, y, x + w, y + h))
					
					objectTracker[currentObjectID] = tracker
					objectLocation1[currentObjectID] = [x, y, w, h]

					currentObjectID = currentObjectID + 1
		
		#cv2.line(resultImage,(0,480),(1280,480),(255,0,0),5)

        # plot the rectangle for objectTracker
		for objectID in objectTracker.keys():
			trackedPosition = objectTracker[objectID].get_position()
					
			t_x = int(trackedPosition.left())
			t_y = int(trackedPosition.top())
			t_w = int(trackedPosition.width())
			t_h = int(trackedPosition.height())
			
			cv2.rectangle(resultImage, (t_x, t_y), (t_x + t_w, t_y + t_h), rectangleColor, 4)
			
			# speed estimation
			objectLocation2[objectID] = [t_x, t_y, t_w, t_h]
		
		# end_time = time.time()
		
		#if not (end_time == start_time):
		# 	fps = 1.0/(end_time - start_time)
		
		#cv2.putText(resultImage, 'FPS: ' + str(int(fps)), (620, 30),cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)


		for i in objectLocation1.keys():	
			if frameCounter % 1 == 0:
				[x1, y1, w1, h1] = objectLocation1[i]
				[x2, y2, w2, h2] = objectLocation2[i]
		
				# print 'previous location: ' + str(carLocation1[i]) + ', current location: ' + str(carLocation2[i])
				objectLocation1[i] = [x2, y2, w2, h2]
		
				# print 'new previous location: ' + str(carLocation1[i])
				if [x1, y1, w1, h1] != [x2, y2, w2, h2]:
					#if (speed[i] == None or speed[i] == 0) and y1 >= 275 and y1 <= 285:
					speed[i] = estimateSpeed([x1, y1, w1, h1], [x2, y2, w2, h2], 0.95 * w2 / OBJ_WIDTH + 0.05 * h2 / OBJ_HEIGHT)
					distance[i] = get_distance(OBJ_WIDTH, OBJ_HEIGHT, w2, h2, focal_length)

					#if y1 > 275 and y1 < 285:
					# if speed[i] != None and y1 >= 180:
					cv2.putText(resultImage, 
					"Distance:" + str(round(distance[i], 1)) +  "m Speed:" + str(round(speed[i], 2)) + "km/h", 
					(int(x1 + w1/2), int(y1-5)),cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
					#print ('CarID ' + str(i) + ': speed is ' + str("%.2f" % round(speed[i], 0)) + ' km/h.\n')

					#else:
					#	cv2.putText(resultImage, "Far Object", (int(x1 + w1/2), int(y1)),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

						#print ('CarID ' + str(i) + ' Location1: ' + str(carLocation1[i]) + ' Location2: ' + str(carLocation2[i]) + ' speed is ' + str("%.2f" % round(speed[i], 0)) + ' km/h.\n')
		cv2.imshow('result', resultImage)
		# Write the frame into the file 'output.avi'
		out.write(resultImage)


		if cv2.waitKey(33) == 27:
			break
	
	cv2.destroyAllWindows()

if __name__ == '__main__':
	trackMultipleObjects(Detection_type.Car)

