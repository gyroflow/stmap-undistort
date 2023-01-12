import numpy as np
import imageio
import cv2
import json
from tkinter import Tk
from tkinter.filedialog import askopenfilename, asksaveasfilename
import os

imageio.plugins.freeimage.download()



class FisheyeCalibrator:
    """Class for calculating camera matrix and distortion coefficients
       from images or videoframes
       Mostly based on https://stackoverflow.com/a/50876130
       9x6 chessboard by default: https://raw.githubusercontent.com/opencv/opencv/master/doc/pattern.png
    """
    def __init__(self, chessboard_size=(9,6)):
        pass


    def get_stmap(self, fov_scale = 1.0):

        img_dim = self.calib_dimension

        new_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(self.K, self.D,
                img_dim, np.eye(3), fov_scale=fov_scale)

        map1, map2 = cv2.fisheye.initUndistortRectifyMap(self.K, self.D, np.eye(3), new_K, img_dim, cv2.CV_32FC1)

        return map1/self.calib_dimension[0], map2/self.calib_dimension[1]


    def load_calibration_json(self, filename):

        with open(filename, "r") as infile:
            cal_data = json.load(infile)

        try:
            print("Preset name: {}".format(cal_data["name"]))
            print("Made with {} frames using calibrator version {} on date {}"
                .format(cal_data["num_images"],
                        cal_data["calibrator_version"],
                        cal_data["date"]))

            cal_width = cal_data["calib_dimension"]["w"]
            cal_height = cal_data["calib_dimension"]["h"]

            self.calib_dimension = (cal_width, cal_height)

            if "orig_dimension" in cal_data:
                orig_w = cal_data["orig_dimension"]["w"]
                orig_h = cal_data["orig_dimension"]["h"]
                self.input_horizontal_stretch = cal_data["input_horizontal_stretch"]
                self.orig_dimension = (orig_w, orig_h)
            else:
                self.input_horizontal_stretch = 1
                self.orig_dimension = self.calib_dimension


            self.num_images = self.num_images_used = cal_data["num_images"]

            self.RMS_error = cal_data["fisheye_params"]["RMS_error"]
            self.K = np.array(cal_data["fisheye_params"]["camera_matrix"])
            self.D = np.array(cal_data["fisheye_params"]["distortion_coeffs"])

        except ZeroDivisionError:
            raise KeyError("Error loading preset file")
        

    def load_calibration_prompt(self, printinfo = False):
        """Trigger file browser to load calibration preset
        Args:
            printinfo (bool, optional): Print extra info from preset file. Defaults to False.
        """

        # file browser prompt
        self.filename = askopenfilename(title = "Select calibration preset file",
                                   filetypes = (("JSON files","*.json"),))

        self.load_calibration_json(self.filename)



if __name__ == "__main__":
    Tk().withdraw() # hide root window
    undistort = FisheyeCalibrator()
    undistort.load_calibration_prompt()

    save_name = os.path.splitext(os.path.split(undistort.filename)[-1])[0] + ".exr"

    undistort_fov_scale = 1
    map1, map2 = undistort.get_stmap(undistort_fov_scale)

    red, green, blue = map1, map2, map1*0.0

    rgb = np.dstack((red, green, blue))

    rgb = np.flip(rgb, 0)

    save_path = asksaveasfilename(title = "Select calibration preset file", initialfile = save_name,
                                   filetypes = ((".exr","*.exr"),(".png", "*.png"),) )

    imageio.imwrite(save_path, rgb)

    print(f"stmap saved at {save_path}")


