import cv2
import numpy as np
import math
import random
from scipy.integrate import solve_ivp

global clicked, click_coords
clicked = False
click_coords = (0,0)

SCALE_FACTOR = 0.5


def click_event(event, x, y, flags, param):
    global clicked, click_coords
    if event == cv2.EVENT_LBUTTONDOWN:
        clicked = True
        orig_x = int(x / SCALE_FACTOR)
        orig_y = int(y / SCALE_FACTOR)
        click_coords = (orig_x, orig_y)
        print(click_coords)
    elif event == cv2.EVENT_LBUTTONUP:
        clicked = False


def calculate_focal_length_horizontal(width_px, hfov_degrees):
    horizontal_px = width_px
    hfov_radians = math.radians(hfov_degrees)
    f_pixel = horizontal_px / (2 * math.tan(hfov_radians / 2))
    return f_pixel


def distance_to_camera(screen_size, physical_size, focal_length):
    return (focal_length * physical_size) / screen_size


def calculate_speed(inches, fps, frames):
    seconds = frames / fps
    speed_ips = inches / seconds

    return speed_ips / 17.6


def find_launch_angle(ball1, ball2):
    x1, y1, z1 = ball1[0]
    x2, y2, z2 = ball2[0]
    theta = math.atan((y2-y1)/math.sqrt((x2-x1)**2 + (z2-z1)**2))
    launch_angle = math.degrees(theta)
    return launch_angle


def pixel_to_3d(x, y, depth, width, height, hfov_degrees):
    hfov = math.radians(hfov_degrees)

    # Horizontal focal length
    fx = width / (2 * math.tan(hfov / 2))

    # Calculate vertical FOV from aspect ratio
    vfov = 2 * math.atan(
        math.tan(hfov / 2) * height / width
    )

    fy = height / (2 * math.tan(vfov / 2))

    cx = width / 2
    cy = height / 2

    X = (x - cx) * depth / fx
    Y = (y - cy) * depth / fy
    Z = depth

    return X, Y, Z


def calculate_baseball_distance(exit_velo_mph, launch_angle_deg):
    # abbreviations used for constants to simplify equations g = Gravity, m = Mass, 
    g, m, rho, Cd, r = 9.81, 0.145, 1.2, 0.3, 0.037
    A = np.pi * r**2 
    B = 0.5 * rho * Cd * A / m  

    v0_mps = exit_velo_mph * 0.44704       
    angle_rad = np.radians(launch_angle_deg) 

    y0 = [0, v0_mps * np.cos(angle_rad), 3, v0_mps * np.sin(angle_rad)]

    def baseball_motion(t, y):
        x, vx, z, vz = y
        v = np.sqrt(vx**2 + vz**2)
        ax = -B * v * vx
        az = -g - B * v * vz
        return [vx, ax, vz, az]

    def hit_ground(t, y): return y[2]
    hit_ground.terminal = True
    hit_ground.direction = -1

    sol = solve_ivp(baseball_motion, [0, 10], y0, events=hit_ground, max_step=0.01)
    distance_meters = sol.y[0][-1]
    distance_feet = distance_meters * 3.28084
    return distance_feet


def fit_circle_3_points(p1, p2, p3):
    """Calculates center and radius from 3 points."""
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3
    A = np.array([[x1 - x2, y1 - y2], [x2 - x3, y2 - y3]])
    B = np.array([[(x1**2 - x2**2 + y1**2 - y2**2) / 2], 
                  [(x2**2 - x3**2 + y2**2 - y3**2) / 2]])
    try:
        center = np.linalg.solve(A, B)
        xc, yc = float(center[0][0]), float(center[1][0])
        r = np.sqrt((x1 - xc)**2 + (y1 - yc)**2)
        return xc, yc, r
    except np.linalg.LinAlgError:
        return None


def calculate_distance_3d(ball1, ball2, width, height, hfov):
    x1, y1, r1 = ball1[0]
    x2, y2, r2 = ball2[0]

    diameter1 = 2 * r1
    diameter2 = 2 * r2

    focal_length = calculate_focal_length_horizontal(width, hfov)

    d1 = distance_to_camera(diameter1, 2.9, focal_length)
    d2 = distance_to_camera(diameter2, 2.9, focal_length)

    p1 = pixel_to_3d(x1, y1, d1, width, height, hfov)
    p2 = pixel_to_3d(x2, y2, d2, width, height, hfov)

    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    dz = p2[2] - p1[2]

    distance = math.sqrt(dx**2 + dy**2 + dz**2)

    return distance, d1, d2, p1, p2


# Using RANSAC circle fitting to fit circles to ball instead of countouring with bounding rectangle or min enclosing circle
def find_ball_from_mask(mask_image, draw_image, max_expected_radius=300, iterations=500, tolerance=2.0):
    global finished_balls

    # Find all shapes on the mask screen
    contours, _ = cv2.findContours(mask_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        finished_balls = balls.copy()
        balls.clear()
        return None
        
    # Identifies and saves the largest contour
    largest_contour = max(contours, key=cv2.contourArea)
    contour_area = cv2.contourArea(largest_contour)
    
    # Convex hull is taken because convex shapes can not be a part of the ball's perimiter
    hull = cv2.convexHull(largest_contour).reshape(-1, 2)
    num_hull_pts = len(hull)

    cv2.drawContours(draw_image, [hull], -1, 100 , -1)
    
    if num_hull_pts < 3:
        print("The largest shape is too small to fit a circle.")
        return None
        
    best_circle = None
    max_inliers = -1
    
    # RANSAC loop
    for _ in range(iterations):
        p1, p2, p3 = random.sample(list(hull), 3)
        
        circle = fit_circle_3_points(p1, p2, p3)
        if circle is None:
            continue
        xc, yc, r = circle
        
        # Rejects situations with a radius larger than the maximum expected value
        if r > max_expected_radius:
            continue
            
        # Ensures the entire hull fits inside this circle
        hull_distances = np.sqrt((hull[:, 0] - xc)**2 + (hull[:, 1] - yc)**2)
        if np.any(hull_distances > (r + tolerance)):
            continue
            
        # Circle is scored based on how many hull points sit on the circumference
        residual = np.abs(hull_distances - r)
        inliers = np.sum(residual < tolerance)
        
        if inliers > max_inliers:
            max_inliers = inliers
            best_circle = (xc, yc, r)
    
    return best_circle, contour_area 


def draw_completed_ball(image, circle_data):
    if circle_data is None or circle_data[0] is None:
        return image
    
    xc, yc, r = circle_data[0]
    
    center_coordinates = (int(round(xc)), int(round(yc)))
    radius = int(round(r))
    
    # Draws fitted RANSAC circle
    cv2.circle(image, center_coordinates, radius, color=(0, 255, 0), thickness=3)
    
    # Draws Red dot at center
    cv2.circle(image, center_coordinates, radius=5, color=(0, 0, 255), thickness=-1)
    
    return image

 
def add_to_balls(current_circle_data):
    if current_circle_data is None:
        return None
    elif current_circle_data[0] is None:
        return None
    
    xc1, yc1, r1 = current_circle_data[0]

    if(top_left[0] < xc1 < bottom_right[0] and top_left[1] < yc1 < bottom_right[1] ):
        balls.append(current_circle_data)


video_file = 'demo_clip.mp4'


cap2 = cv2.VideoCapture(video_file)
fps = cap2.get(cv2.CAP_PROP_FPS)

window_name2 = "click the center of the ball"
cv2.namedWindow(window_name2, cv2.WINDOW_NORMAL) 

video_width = int(cap2.get(cv2.CAP_PROP_FRAME_WIDTH))
video_height = int(cap2.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(f"Video resolution: {video_width}x{video_height}")

cv2.resizeWindow(window_name2, int(video_width * SCALE_FACTOR), int(video_height * SCALE_FACTOR))
cv2.setMouseCallback(window_name2, click_event)

LAPLACIAN_THRESH = 30
NO_BALL_TEXT = "no ball detected"
DETECTED_TEXT = "ball detected"
FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 2
COLOR = (255, 0, 0)
TEXT_THICKNESS = 5
text1 = NO_BALL_TEXT
text2 = text3 = " "

HFOV = 69.3
detected_speed = None
launch_angle = None
detected_speeds = []
launch_angles = []
balls = []
finished_balls = []
frames = 0

paused = True
ret2, frame2 = cap2.read()
while cap2.isOpened():
    if not paused:
        ret2, frame2 = cap2.read()

    if not ret2:
            break

    hsv2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2HSV)
    cv2.putText(frame2, "Click approximately the center of the ball", (10, 80), FONT, FONT_SCALE, COLOR, TEXT_THICKNESS)
    cv2.putText(frame2, "Use the space bar to pause and play the video", (10, 160), FONT, FONT_SCALE, COLOR, TEXT_THICKNESS)
    cv2.putText(frame2, "Avoid clicking any seams", (10, 240), FONT, FONT_SCALE, COLOR, TEXT_THICKNESS)
    cv2.putText(frame2, "Click q to move on to the next screen", (10, 320), FONT, FONT_SCALE, COLOR, TEXT_THICKNESS)

    cv2.imshow(window_name2, cv2.resize(frame2, (0,0), fx=SCALE_FACTOR, fy=SCALE_FACTOR))
    if clicked:
        upper_mask = np.array([180, min(hsv2[click_coords[1], click_coords[0]][1] + 20, 255), min(255 , hsv2[click_coords[1], click_coords[0]][2] + 50)])
        lower_mask = np.array([0, 0, max(hsv2[click_coords[1], click_coords[0]][2] - 70, 0)])

    key = cv2.waitKey(30) & 0xFF
    if key == 32:
        paused = not paused
    if key == ord('q'):
        break

    
    

cap2.release()
cv2.destroyAllWindows()



cap = cv2.VideoCapture(video_file)

window_name = "current frame"
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
cv2.resizeWindow(window_name, int(video_width * SCALE_FACTOR), int(video_height * SCALE_FACTOR))
cv2.setMouseCallback(window_name, click_event)

if not cap.isOpened():
    print('Error opening video file')
    exit()
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    frames += 1

    if frames == 1:
        while not clicked:
            display_frame = cv2.resize(frame, (0,0), fx=SCALE_FACTOR, fy=SCALE_FACTOR)
            cv2.putText(display_frame, "Click top left of ball tracking area...", (10, 80), FONT, FONT_SCALE, COLOR, TEXT_THICKNESS)
            cv2.imshow(window_name, display_frame)  
            if cv2.waitKey(1) & 0xFF == 27:
                exit()
        top_left = click_coords
        clicked = False
        cv2.waitKey(300) 

        while not clicked:
            display_frame = cv2.resize(frame, (0,0), fx=SCALE_FACTOR, fy=SCALE_FACTOR)
            cv2.putText(display_frame, "Click bottom right of ball tracking area...", (10, 80), FONT, FONT_SCALE, COLOR, TEXT_THICKNESS)
            cv2.imshow(window_name, display_frame)  
            if cv2.waitKey(1) & 0xFF == 27:
                exit()
        bottom_right = click_coords
        clicked = False
        
        delayed_cap = cv2.VideoCapture(video_file)

    blurred = cv2.GaussianBlur(frame, (1, 1), 0)
    gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)


    laplacian_raw = cv2.Laplacian(gray, cv2.CV_64F, ksize=3)
    laplacian_abs = np.uint8(np.absolute(laplacian_raw))
    _, mask = cv2.threshold(laplacian_abs, LAPLACIAN_THRESH, 255, cv2.THRESH_BINARY)

    hsv_mask = cv2.inRange(hsv, lower_mask, upper_mask)

    if frames >= 2:
        ret, delayed_frame = delayed_cap.read()
        if not ret:
            break

        cleanframe = frame.copy()


        delayed_blurred = cv2.GaussianBlur(delayed_frame, (1, 1), 0)
        delayed_gray = cv2.cvtColor(delayed_blurred, cv2.COLOR_BGR2GRAY)
        delayed_hsv = cv2.cvtColor(delayed_frame, cv2.COLOR_BGR2HSV)
        
        delayed_laplacian_raw = cv2.Laplacian(delayed_gray, cv2.CV_64F, ksize=3)
        delayed_laplacian_abs = np.uint8(np.absolute(delayed_laplacian_raw))
        _, delayed_mask = cv2.threshold(delayed_laplacian_abs, LAPLACIAN_THRESH, 255, cv2.THRESH_BINARY)
        delayed_hsv_mask = cv2.inRange(delayed_hsv, lower_mask, upper_mask)


        gray_difference = cv2.absdiff(mask, delayed_mask)
        hsv_difference = cv2.absdiff(hsv_mask, delayed_hsv_mask)

        difference = cv2.bitwise_or(gray_difference, hsv_difference)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (100, 5))
        gray_current_ball = cv2.bitwise_and(mask, gray_difference)
        gray_current_ball_closed = cv2.morphologyEx(gray_current_ball, cv2.MORPH_CLOSE, kernel)
        gray_delayed_ball = cv2.bitwise_and(delayed_mask, gray_difference) 
        gray_delayed_ball_closed = cv2.morphologyEx(gray_delayed_ball, cv2.MORPH_CLOSE, kernel)

        hsv_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        hsv_current_ball = cv2.bitwise_and(hsv_mask, hsv_difference)
        hsv_current_ball_open = cv2.morphologyEx(hsv_current_ball, cv2.MORPH_OPEN, hsv_kernel)
        hsv_delayed_ball = cv2.bitwise_and(delayed_hsv_mask, hsv_difference)
        hsv_delayed_ball_open = cv2.morphologyEx(hsv_delayed_ball, cv2.MORPH_OPEN, hsv_kernel)

        '''
        9/3 use the laplacian mask and the hsv mask to identify the whole ball. 
        The HSV mask identifies the inner, whiter parts of the ball but struggles to identify the outer more shadowed parts. 
        The laplacian edge detection finds the edges of the ball which, in grayscale, are much brighter than the background tarp. 
        ''' 


        current_ball = cv2.bitwise_and(gray_current_ball_closed, hsv_current_ball_open)
        current_ball_open = cv2.morphologyEx(current_ball, cv2.MORPH_OPEN, hsv_kernel)
        delayed_ball = cv2.bitwise_and(gray_delayed_ball_closed, hsv_delayed_ball_open)
        delayed_ball_open = cv2.morphologyEx(delayed_ball, cv2.MORPH_OPEN, hsv_kernel)

        
        
        current_circle = find_ball_from_mask(current_ball_open, None)
        

        draw_completed_ball(frame, current_circle)
        add_to_balls(current_circle)

        display_current = cv2.resize(frame, (0,0), fx=SCALE_FACTOR, fy=SCALE_FACTOR)
        cv2.putText(display_current, text1, (10, 80), FONT, FONT_SCALE, COLOR, TEXT_THICKNESS)
        cv2.putText(display_current, text2, (10, 160), FONT, FONT_SCALE, COLOR, TEXT_THICKNESS)
        cv2.putText(display_current, text3, (10, 240), FONT, FONT_SCALE, COLOR, TEXT_THICKNESS)
        cv2.imshow(window_name, display_current)

        if(not(text1 == NO_BALL_TEXT) and not(text1 == DETECTED_TEXT)):
            cv2.waitKey(2000)

        if(current_circle is None):
            text1 = NO_BALL_TEXT
        elif(len(balls) >= 2):
            text1 = DETECTED_TEXT
        text2 = text3 = " "
        if(len(balls) == 0 and len(finished_balls) >= 2):
            j = 0
            ball1 = ((0,0,0),0,0)
            ball2 = ((0,0,0),0,0)
            while(True):
                if(j >= len(finished_balls)):
                    break
                finished_balls[j] = finished_balls[j][0], finished_balls[j][1], j
                j += 1

            j = 0
            while(True):
                if(j >= len(finished_balls)):
                    break
                if(finished_balls[j][1] > ball1[1] or finished_balls[j][1] > ball2[1]):
                    if(ball2[1] > ball1[1]):
                        ball1 = ball2
                    ball2 = finished_balls[j]
                j += 1
            
            total_distance = calculate_distance_3d(ball2, ball1, video_width, video_height, HFOV)

            
            detected_speed = calculate_speed(total_distance[0], fps, ball2[2]-ball1[2])
            launch_angle = find_launch_angle(ball2, ball1)
            distance = calculate_baseball_distance(detected_speed, launch_angle)
            text1 = f"MPH: {detected_speed}"
            text2 = f"Launch Angle: {launch_angle}"
            text3 = f"Distance: {distance}"
        
        if cv2.waitKey(1) & 0xFF == ord('w'):
            break
        
cap.release()
delayed_cap.release()
cv2.destroyAllWindows()
