import numpy as np
import cv2
import time
from WebcamVideoStream import WebcamVideoStream
from FramesPerSec import FramesPerSec
from Request import Request
import threading
import sys
from VideoShow import VideoShow


class VideoCamera(threading.Thread):
    # Frame Props
    frame_width = 10

    # Display colors
    red = (0, 0, 255)
    green = (0, 255, 0)
    white = (255, 255, 255)
    cyan = (255, 255, 0)
    black = (0, 0, 0)
    dark_gray = (75, 75, 75)

    threaded_input = False
    threaded_output = False
    font = cv2.FONT_HERSHEY_DUPLEX

    # img_res = (320, 240)
    # img_res = (640, 480)
    # img_res = (800, 600)
    # img_res = (1280, 720)
    img_res = (1920, 1080)

    display_scale = 2

    req_interval = 0.5

    def __init__(self, src=0, display=True):
        threading.Thread.__init__(self)

        self.display = display

        # Using OpenCV to capture from device 0
        if (self.threaded_input == True):
            self.video = WebcamVideoStream(src=src,
                                           img_res=self.img_res,
                                           playback=True).start()
        else:
            self.video = cv2.VideoCapture(src)
            ####################################
            ### Ubuntu Hack - to improve FPS ###
            ####################################
            self.video.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
            self.video.set(cv2.CAP_PROP_FRAME_WIDTH, self.img_res[0])
            self.video.set(cv2.CAP_PROP_FRAME_HEIGHT, self.img_res[1])

        # read the first frame
        self.current_frame = self.frame = self.read_frame()

        if self.threaded_output == True:
            # Initialize Video Display thread
            frame = self.read_frame()
            self.video_shower = VideoShow(frame, scale_factor=self.display_scale,
                                          shape=frame.shape, display_name="FaceRec").start()

        try:
            # Face API request
            self.request = Request(frame=self.frame, interval=self.req_interval).start()
        except:
            self.request = None

        # Start FPS counter
        self.fps = FramesPerSec().start()

    def __del__(self):
        if (self.video is not None):
            if (self.threaded_input == True):
                self.video.stop()
            else:
                self.video.release()
            if (cv2 is not None):
                cv2.destroyAllWindows()

    def get_json(self):
        return self.face_names

    def read_frame(self):
        success = False
        frame = None
        # try to get the input frame
        if (self.threaded_input == True):
            frame = self.video.read()
            if (frame is not None):
                success = True
        else:
            success, frame = self.video.read()
            if (success == False):  # loop playback
                self.video.set(cv2.CAP_PROP_POS_FRAMES, 0)

        if success != True:
            # set the blank image to camera resolution
            frame = np.zeros((320, 240, 3), np.uint8)

        return frame

    def get_frame(self):
        # We are using Motion JPEG, but OpenCV defaults to capture raw images,
        # so we must encode it into JPEG in order to correctly display the
        # video stream.
        ret, jpeg = cv2.imencode('.jpg', self.current_frame)
        return jpeg.tobytes()

    def run(self):
        # get the first frame
        self.frame = self.read_frame()

        while True:
            self.frame = self.read_frame()

            # request detections in frame
            self.request.frame = self.frame

            # draw the video resolution
            cv2.putText(self.frame, "{}x{}".format(self.img_res[0], self.img_res[1]),
                        (0, 30), self.font, 1.0,
                        self.cyan, 2)

            # draw the FPS
            cv2.putText(self.frame, "{:.0f} fps".format(self.fps.fps()), (0, 60),
                        self.font, 1.0, self.cyan, 2)

            # increment FPS counter
            self.fps.increment()

            # draw the face metrics
            self.draw_metrics()

            # keep track of current frame
            self.current_frame = cv2.resize(self.frame, (0, 0),
                                            fx=float(1.0 / self.display_scale),
                                            fy=float(1.0 / self.display_scale))

            if(self.display == True):
                # display the resulting image using OpenCV
                cv2.imshow("Video", self.current_frame)
                if cv2.waitKey(1) == ord("q"):
                    sys.exit(1)
                    break

            if self.threaded_output == True:
                self.video_shower.frame = self.current_frame

    def simple_display(self):
        while True:
            self.frame = self.read_frame()

            # draw the video resolution
            cv2.putText(self.frame, "{}x{}".format(self.img_res[0], self.img_res[1]),
                        (0, 30), self.font, 1.0,
                        self.cyan, 2)

            # draw the FPS
            cv2.putText(self.frame, "{:.0f} fps".format(self.fps.fps()), (0, 60),
                        self.font, 1.0, self.cyan, 2)

            # increment FPS counter
            self.fps.increment()

            # keep track of current frame
            self.current_frame = self.frame

            if(self.display == True):
                # assign the frame to current_frame
                self.current_frame = cv2.resize(self.frame, (0, 0),
                                                fx=float(1.0 / self.display_scale),
                                                fy=float(1.0 / self.display_scale))

                # display the resulting image using OpenCV
                cv2.imshow("Video", self.current_frame)
                if cv2.waitKey(1) == ord("q"):
                    sys.exit(1)
                    break
            self.video_shower.frame = self.current_frame

    def current_milli_time(self):
        return int(round(time.time() * 1000))

    def drawFPS(self):
        cv2.putText(self.frame, "{:.0f} fps".format(self.fps.fps()), (0, 30), self.font, 1.0, self.cyan, 2)
        return

    def draw_frame(self, top, right, bottom, left, box_color, line_thickness, outer):
        x1, y1 = (left, top)
        x2, y2 = (left, bottom)
        x3, y3 = (right, top)
        x4, y4 = (right, bottom)

        h_line_length = int(abs((right - left) / 4))
        v_line_length = int(abs((bottom - top) / 4))

        if (outer == True):
            h_line_length += 2 * self.frame_width
            v_line_length += 2 * self.frame_width

        cv2.line(self.frame, (x1, y1), (x1, y1 + v_line_length), box_color, line_thickness)  # -- top-left
        cv2.line(self.frame, (x1, y1), (x1 + h_line_length, y1), box_color, line_thickness)

        cv2.line(self.frame, (x2, y2), (x2, y2 - v_line_length), box_color, line_thickness)  # -- bottom-left
        cv2.line(self.frame, (x2, y2), (x2 + h_line_length, y2), box_color, line_thickness)

        cv2.line(self.frame, (x3, y3), (x3 - h_line_length, y3), box_color, line_thickness)  # -- top-right
        cv2.line(self.frame, (x3, y3), (x3, y3 + v_line_length), box_color, line_thickness)

        cv2.line(self.frame, (x4, y4), (x4, y4 - v_line_length), box_color, line_thickness)  # -- bottom-right
        cv2.line(self.frame, (x4, y4), (x4 - h_line_length, y4), box_color, line_thickness)

    def draw_lines(self, top, right, bottom, left, box_color):
        x = int((left + right) / 2)
        y = int((top + bottom) / 2)
        a = abs((right - left) / 2)
        x1, y1 = (left, top)
        x2, y2 = (left, bottom)
        x3, y3 = (right, top)
        x4, y4 = (right, bottom)
        line_length = int(abs((right - left) / 8))

        cv2.circle(self.frame, (int(x + a), y), 2, box_color, -1)  # -- right
        cv2.circle(self.frame, (x, int(y + a)), 2, box_color, -1)  # -- top
        cv2.circle(self.frame, (int(x - a), y), 2, box_color, -1)  # -- left
        cv2.circle(self.frame, (x, int(y - a)), 2, box_color, -1)  # -- bottom

        cv2.line(self.frame, (int(x + a - line_length / 2), y), (int(x + a + line_length / 2), y), box_color,
                 2)  # -- right
        cv2.line(self.frame, (x, int(y + a - line_length / 2)), (x, int(y + a + line_length / 2)), box_color,
                 2)  # -- top
        cv2.line(self.frame, (int(x - a - line_length / 2), y), (int(x - a + line_length / 2), y), box_color,
                 2)  # -- left
        cv2.line(self.frame, (x, int(y - a - line_length / 2)), (x, int(y - a + line_length / 2)), box_color,
                 2)  # -- bottom

    def draw_metrics(self):
        # draw the metrics
        if (self.request.faces is None):
            return

        # draw the number of faces
        num_faces = 0
        if (self.request.faces is not None):
            num_faces = len(self.request.faces)
        cv2.putText(self.frame, "{:.0f} faces".format(num_faces), (0, 90), self.font, 1.0, self.cyan, 2)


        for detection in self.request.faces:
            face = detection['face']
            confidence = float(detection['confidence'])
            left = int(detection['left'])
            top = int(detection['top'])
            right = int(detection['right'])
            bottom = int(detection['bottom'])

            box_color = self.cyan
            text_color = self.cyan
            # Draw a box around the face
            self.draw_frame(top, right, bottom, left, box_color, 2, False)
            self.draw_frame(top - self.frame_width, right + self.frame_width,
                            bottom + self.frame_width, left - self.frame_width,
                            box_color, 6, True)
            cv2.putText(self.frame, "Name: " + face["name"],
                        (right + 2 * self.frame_width, top + 2 * self.frame_width + 5),
                        self.font, 0.75, text_color, 2)
            # If the result is a recognized face, add details from DB
            if "Unknown" not in face["name"]:
                cv2.putText(self.frame, "Username: " + face["username"],
                            (right + 2 * self.frame_width, top + 2 * self.frame_width + 35),
                            self.font, 0.75,
                            text_color, 2)
                cv2.putText(self.frame, face["gender"] + ", Age: " + face["age"],
                            (right + 2 * self.frame_width, top + 2 * self.frame_width + 65),
                            self.font, 0.75,
                            text_color, 2)

            # draw the bounding box of the face along with the associated probability
            # text = "Confidence: {:.2f}%".format(confidence * 100)
            # y = top - 10 if top - 10 > 10 else top + 10
            # cv2.putText(self.frame, text, (left, y), self.font, 0.75, self.green, 2)

        '''
        for face in self.request.faces:
            top = face['top']
            left = face['left']
            bottom = top + height
            right = left + width

            # Draw bounding box around the face
            self.draw_frame(top, right, bottom, left, box_color, 2, False)
            self.draw_frame(top - self.frame_width, right + self.frame_width,
                            bottom + self.frame_width, left - self.frame_width, box_color, 6, True)

            # text with background
            font_scale = 1.0
            thickness = 2

            # Age
            age = "Age: {:.0f}".format(face['faceAttributes']['age'])
            # get the width and height of the text box
            (text_width, text_height) = cv2.getTextSize(age, self.font, fontScale=font_scale, thickness=thickness)[0]
            cv2.rectangle(self.frame,
                          (right + 2 * self.frame_width, top + 2 * self.frame_width - int(text_height)),
                          (
                              right + 2 * self.frame_width + text_width,
                              top + 2 * self.frame_width + int(text_height / 2)),
                          self.dark_gray, cv2.FILLED)
            cv2.putText(self.frame, age,
                        (right + 2 * self.frame_width, top + 2 * self.frame_width),
                        self.font, font_scale, self.white, thickness)

            # Gender
            gender = "Gender: {}".format(face['faceAttributes']['gender'])
            # get the width and height of the text box
            (text_width, text_height) = cv2.getTextSize(gender, self.font, fontScale=font_scale, thickness=thickness)[0]
            cv2.rectangle(self.frame,
                          (right + 2 * self.frame_width, top + 2 * self.frame_width + 35 - int(text_height)),
                          (right + 2 * self.frame_width + text_width,
                           top + 2 * self.frame_width + 35 + int(text_height / 2)),
                          self.dark_gray, cv2.FILLED)
            cv2.putText(self.frame, gender,
                        (right + 2 * self.frame_width, top + 2 * self.frame_width + 35),
                        self.font, font_scale, self.white, thickness)

            # Emotion
            emotions = face['faceAttributes']['emotion']
            emotion = "Emotion: {}".format(self.get_emotion(emotions))
            # get the width and height of the text box
            (text_width, text_height) = cv2.getTextSize(emotion, self.font, fontScale=font_scale, thickness=thickness)[
                0]
            cv2.rectangle(self.frame,
                          (right + 2 * self.frame_width, top + 2 * self.frame_width + 70 - int(text_height)),
                          (right + 2 * self.frame_width + text_width,
                           top + 2 * self.frame_width + 70 + int(text_height / 2)),
                          self.dark_gray, cv2.FILLED)
            cv2.putText(self.frame, emotion,
                        (right + 2 * self.frame_width, top + 2 * self.frame_width + 70),
                        self.font, font_scale, self.white, thickness)
        '''


if __name__ == "__main__":
    device = 0
    vc = VideoCamera(src=device, display=True)
    vc.run()
    # vc.simple_display()
    # vc.join()
