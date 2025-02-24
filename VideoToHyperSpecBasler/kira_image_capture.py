# KIRa Hyperspectral Image Capture Module
# Author: Fabian Erichsmeier
# Mail: fabian.erichsmeier@fh-bielefeld.de
# From: 30.03.2023
# Python version: 3.11.0

import numpy as np  # version 1.24.1
from pypylon import pylon  # version 1.9
from pypylon import genicam
from dataclasses import dataclass
from spectral import envi  # version 0.23.1
import cv2 as cv #version 4.5.562


@dataclass
class PikaL:
    # Pika L Parameters, from table on the Camera Setup and Windowing Page
    ROI_WIDTH = 900
    ROI_HEIGHT = 600
    Y_BINNING = 2

    # Device-specific parameters obtained from Resonon Camera Configuration Report
    X_OFFSET = 516
    Y_OFFSET = 336
    A = 0.00010489999840501696
    B = 0.933646023273468
    C = 51.4379997253418
    SERIAL_NUMBER = "23511926"


@dataclass
class Basler:
    # Pika L Parameters, from table on the Camera Setup and Windowing Page
    ROI_WIDTH = 1296
    ROI_HEIGHT = 1032
    Y_BINNING = 1

    # Device-specific parameters obtained from Resonon Camera Configuration Report
    X_OFFSET = 0
    Y_OFFSET = 0
    A = 0.00013639214204672206
    B = 1.0280932959428337
    C = 404.99999999999994
    SERIAL_NUMBER = "40484617"


class CameraManager():

    def __init__(self):
        self.cameras = []

    def add_cameras(self):
        # Set according to application necessities
        max_cameras_to_use = 2

        # Get the transport layer factory.
        tl_factory = pylon.TlFactory.GetInstance()

        # Get all attached devices and exit application if no device is found.
        devices = tl_factory.EnumerateDevices()
        if len(devices) == 0:
            raise pylon.RuntimeException("No camera present.")

        # Create an array of instant cameras for the found devices and avoid exceeding a maximum number of devices.
        self.cameras = pylon.InstantCameraArray(min(len(devices), max_cameras_to_use))

        try:
            # Create and attach all Pylon Devices.
            for i, camera in enumerate(self.cameras):
                camera.Attach(tl_factory.CreateDevice(devices[i]))
                camera.Open()
                # Print the model name of the camera.
                print(
                    f"Using device {camera.GetDeviceInfo().GetModelName()} with SN {camera.GetDeviceInfo().GetSerialNumber()}")
        except genicam.RuntimeException:
            raise Exception(f"Devices seem to be occupied by another process.")

    def remove_camera(self, serial_number: str):
        # Removes camera from the list of managed devices.
        for i, camera in enumerate(self.cameras):
            if camera.GetDeviceInfo().GetSerialNumber() == serial_number:
                camera.Close()
                del self.cameras[i]
                break

    def set_exposure(self, serial_number: str, exposure_time: int):
        # Sets the exposure time of the camera with the specified serial number.
        for camera in self.cameras:
            if camera.GetDeviceInfo().GetSerialNumber() == serial_number:
                camera.ExposureTime.SetValue(exposure_time)
                return
        raise Exception(f"Camera with serial number {serial_number} not found")

    def set_framerate(self, serial_number: str, framerate: float):
        # Set the framerate of the camera with the given serial number.
        for camera in self.cameras:
            if camera.GetDeviceInfo().GetSerialNumber() == serial_number:
                camera.AcquisitionFrameRateEnable.SetValue(True)
                camera.AcquisitionFrameRate.SetValue(framerate)
                return
        raise Exception(f"Camera with serial number {serial_number} not found")

    def set_gain(self, serial_number: str, gain: float):
        # Set the gain of the camera with the given serial number.
        for camera in self.cameras:
            if camera.GetDeviceInfo().GetSerialNumber() == serial_number:
                camera.Gain.SetValue(gain)
                return
        raise Exception(f"Camera with serial number {serial_number} not found")

    def set_camera_window(self, serial_number: str, width: int, height: int, x_offset: int, y_offset: int,
                          y_binning: int):
        # Make sure the camera is not binned before configuring the window, then add desired binning
        for camera in self.cameras:
            if camera.GetDeviceInfo().GetSerialNumber() == serial_number:
                camera.BinningHorizontal.SetValue(1)
                camera.BinningVertical.SetValue(1)
                camera.Width.SetValue(width)
                camera.Height.SetValue(height)
                camera.OffsetX.SetValue(x_offset)
                camera.OffsetY.SetValue(y_offset)
                camera.BinningVertical.SetValue(y_binning)
                return
        raise Exception(f"Camera with serial number {serial_number} not found")

    def capture_frame(self, serial_number: str):
        # Capture a frame and return it as a numpy-array.
        for camera in self.cameras:
            if camera.GetDeviceInfo().GetSerialNumber() == serial_number:
                grab_result = camera.GrabOne(5000, pylon.TimeoutHandling_ThrowException)
                if grab_result.GrabSucceeded():
                    if HyperspecUtility.pixel_correction is True:
                        # with pixel error correction
                        return np.transpose(HyperspecUtility.correct_pixel_errors(grab_result.Array))
                    else:
                        # without pixel error correction
                        return np.transpose(grab_result.Array)                   
                else:
                    raise Exception(f"Error capturing frame from camera with serial number {serial_number}")
        raise Exception(f"Camera with serial number {serial_number} not found")

    def grab_hyperspec(self, serial_number: int, line_count: int, path: str, should_correct: bool, ref: int):
        # Grab hyperspectral image cube with or without flat-field correction
        for camera in self.cameras:
            if camera.GetDeviceInfo().GetSerialNumber() == serial_number:
                camera.StartGrabbingMax(line_count)
                cube = np.empty(shape=(line_count, camera.Width.GetValue(), camera.Height.GetValue()))
                i = 0

                if should_correct is True:
                    dark_cube = np.array(HyperspecUtility.read_cube(path, "dark_cube", should_scale=False))
                    reflectance_cube = np.array(HyperspecUtility.read_cube(path, "white_cube", should_scale=False))
                    # print(reflectance_cube)
                    while camera.IsGrabbing():
                        grab_result = camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)
                        if grab_result.GrabSucceeded():
                            if HyperspecUtility.pixel_correction is True:
                                # with pixel error correction
                                corrected_frame = HyperspecUtility.correct_frame(np.transpose(HyperspecUtility.correct_pixel_errors(grab_result.Array)), dark_cube, reflectance_cube, ref)
                            else:
                                # without pixel error correction
                                corrected_frame = HyperspecUtility.correct_frame(np.transpose(grab_result.Array), dark_cube, reflectance_cube, ref)
                            # this data has been flat-field corrected and is ready for further processing
                            cube[i, :, :] = corrected_frame
                            i += 1
                else:
                    while camera.IsGrabbing():
                        grab_result = camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)
                        if grab_result.GrabSucceeded():
                            if HyperspecUtility.pixel_correction is True:
                                # with pixel error correction
                                cube[i, :, :] = np.transpose(HyperspecUtility.correct_pixel_errors(grab_result.Array))
                            else:
                                # without pixel error correction
                                cube[i, :, :] = np.transpose(grab_result.Array)
                            i += 1

                return cube
        raise Exception(f"Camera with serial number {serial_number} not found")

    def grab_dark_cube(self, serial_number: str):
        # Grab an average dark reference
        line_count = 30
        for camera in self.cameras:
            if camera.GetDeviceInfo().GetSerialNumber() == serial_number:
                # record a reference dark frame
                # print("Ready to record dark frame. Place a lens cap on the imager.")
                # input("Press Enter to begin recording.")  # wait for user to be ready
                dark_cube = HyperspecUtility.capture_average_frame(camera, line_count)
                return dark_cube
        raise Exception(f"Camera with serial number {serial_number} not found")

    def grab_white_cube(self, serial_number: str):
        # Grab white reference
        line_count = 60
        for camera in self.cameras:
            if camera.GetDeviceInfo().GetSerialNumber() == serial_number:
                # record a reference white frame
                # print("Ready to record white frame. Place reference material in the field of view of imager.")
                # input("Press Enter to begin recording.")  # wait for user to be ready
                white_cube = HyperspecUtility.capture_average_frame(camera, line_count)
                return white_cube
        raise Exception(f"Camera with serial number {serial_number} not found")
    
    def grab_shading_image(self, serial_number: str):
        # Grab white reference
        line_count = 60
        for camera in self.cameras:
            if camera.GetDeviceInfo().GetSerialNumber() == serial_number:
                # record a reference white frame
                # print("Ready to record white frame. Place reference material in the field of view of imager.")
                # input("Press Enter to begin recording.")  # wait for user to be ready
                shading_image = HyperspecUtility.capture_average_frame(camera, line_count)
                return shading_image[0,:,:]
        raise Exception(f"Camera with serial number {serial_number} not found")


class HyperspecUtility():
    pixel_correction = False
    
    def __init__(self) -> None:
        pass

    @staticmethod
    def get_wavelength_for_channel(band_number: int,
                                   y_offset: int,
                                   y_binning: int,
                                   A: float,
                                   B: float,
                                   C: float) -> float:
        # The wavelength calibration equation is defined in terms of un-binned and un-windowed sensor coordinates.
        # here, we convert the band number of the binned and windowed region to its equivalent un-binned location
        # and apply the calibration equation.
        #
        # The term Y_BINNING / 2.0 - 0.5 is a correction to ensure the wavelength is calculated from the center of the
        # binned region, instead of the edge (using 0-based, C style indexing)
        camera_pixel = y_offset + band_number * y_binning + y_binning / 2.0 - 0.5
        return A * camera_pixel ** 2 + B * camera_pixel + C
    
    @staticmethod
    def correct_raw_image(raw_image: np.ndarray,
                      white_image: np.ndarray,
                      reference_reflectance: float) -> np.ndarray:
        
        # shading correction
        numerator = raw_image.astype(np.float64)
        denominator = np.squeeze(white_image, axis=0)
        result = reference_reflectance * np.divide(numerator, denominator, out=np.copy(numerator), where=denominator != 0)

        return result.astype(np.uint16)

    @staticmethod
    def correct_frame(raw_frame: np.ndarray,
                      dark_frame: np.ndarray,
                      white_frame: np.ndarray,
                      reference_reflectance: float) -> np.ndarray:
        # Flat-field correction of a given frame against dark and white references
        # res = (raw - dark) / (white - dark) * scaling_factor
        
        # dark_frame + white_frame
        #numerator = raw_frame.astype(np.float64) - np.squeeze(dark_frame, axis=0)
        #denominator = np.squeeze(white_frame, axis=0) - np.squeeze(dark_frame, axis=0)
        #result = reference_reflectance * np.divide(numerator, denominator, out=np.copy(numerator), where=denominator != 0)
        
        # only white_frame
        numerator = raw_frame.astype(np.float64)
        denominator = np.squeeze(white_frame, axis=0)
        result = reference_reflectance * np.divide(numerator, denominator, out=np.copy(numerator), where=denominator != 0)
        
        # logarithmic characteristic curve
        #result = reference_reflectance + (raw_frame.astype(np.float64) - np.squeeze(white_frame, axis=0))

        return result.astype(np.uint16)

    # define a function to collect several frames from a camera and average them element-wise
    @staticmethod
    def capture_average_frame(camera: pylon.InstantCamera, count: int) -> np.ndarray:
        # Captures multiple frames and averages them in line dimension
        # Array is kept 3-dimensional to comply with Spectral standards
        buffer = np.empty(shape=(count, camera.Width.GetValue(), camera.Height.GetValue()))
        i = 0
        camera.StartGrabbing()
        while i < count:
            grab_result = camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)
            if grab_result.GrabSucceeded():
                if HyperspecUtility.pixel_correction is True:
                    # with pixel error correction
                    buffer[i, :, :] = np.transpose(HyperspecUtility.correct_pixel_errors(grab_result.Array))
                else:
                    # without pixel error correction
                    buffer[i, :, :] = np.transpose(grab_result.Array)
                i += 1
            grab_result.Release()
        camera.StopGrabbing()
        return buffer.mean(axis=0, keepdims=True)

    @staticmethod
    def write_cube(buffer: np.array,
                   meta: dict,
                   path: str,
                   filename: str):
        # Writes cube to BIL-file with corresponding header including given metadata
        envi.save_image(f"{path}\\{filename}", buffer, metadata=meta, force=True, ext="bil", interleave="bil",
                        dtype=np.uint16)

    @staticmethod
    def read_cube(path: str,
                  filename: str,
                  should_scale: bool) -> np.ndarray:
        # Reads data from BIL-file and returns them as np.ndarray
        hdr_file_path = f"{path}\{filename}.hdr"
        bil_file_path = f"{path}\{filename}.bil"

        bil_file = envi.open(hdr_file_path, bil_file_path)
        bil_data = bil_file.load(dtype=np.uint16, scale=should_scale)

        return bil_data

    @staticmethod
    def generate_metadata(camera: pylon.InstantCamera,
                          line_count: int,
                          y_offset: int,
                          y_binning: int,
                          A: float,
                          B: float,
                          C: float,
                          ref: int) -> dict:
        # Creates and returns metadata dictionary for ENVI-header files
        metadata = {}
        metadata["description"] = "KIRa"
        metadata["samples"] = camera.Height.GetValue()
        metadata["lines"] = line_count
        metadata["bands"] = camera.Width.GetValue()
        metadata["spectral binning"] = y_binning
        metadata["interleave"] = "bil"
        metadata["bit depth"] = 12
        metadata["data type"] = 12
        metadata["header offset"] = 0
        metadata["framerate"] = camera.AcquisitionFrameRate.GetValue()
        metadata["shutter"] = camera.ExposureTime.GetValue()
        metadata["gain"] = camera.Gain.GetValue()
        metadata["imager type"] = camera.GetDeviceInfo().GetModelName()
        metadata["imager serial number"] = camera.GetDeviceInfo().GetSerialNumber()
        metadata["wavelength units"] = "nm"
        metadata[
            "wavelength"] = f"{{ {', '.join(str(HyperspecUtility.get_wavelength_for_channel(band_number, y_offset, y_binning, A, B, C)) for band_number in range(0, camera.Height.GetValue()))} }}"
        metadata["reflectance scale factor"] = ref
        return metadata
    
    @staticmethod
    def correct_pixel_errors(image):
        # width:832, height:1032, yoff:0, xoff:220
        #x = [24, 285, 228, 291, 291, 414, 557, 611, 575, 647, 544, 497, 568, 599, 749, 357, 291]
        #y = [116, 612, 204, 281, 58, 106, 123, 145, 333, 358, 473, 562, 622, 1007, 55, 76, 58]
        # width:340, height:1032, yoff:0, xoff:420
        x = [91, 157, 128, 214, 28, 90, 297, 85]
        y = [58, 76, 87, 106, 204, 281, 562, 612]
        image_out = image.copy()
        for i in range(len(x)):
            image_out = HyperspecUtility.edit_pixel_error(image_out, y[i], x[i])
        return image_out
    
    @staticmethod
    def edit_pixel_error(image, i, j):
        # edge-pixel
        #val1 = ((image[i-2, j-1].astype(np.float32) + image[i-2, j-2] + image[i-1, j-2])/3.0).astype(np.uint16)
        image[i-1, j-1] = ((image[i-2, j-1].astype(np.float32) + image[i-2, j-2] + image[i-1, j-2])/3.0).astype(np.uint16)
        #val2 = ((image[i+1, j-2].astype(np.float32) + image[i+2, j-2] + image[i+2, j-1])/3.0).astype(np.uint16)
        image[i+1, j-1] = ((image[i+1, j-2].astype(np.float32) + image[i+2, j-2] + image[i+2, j-1])/3.0).astype(np.uint16)
        #val3 = ((image[i-2, j+1].astype(np.float32) + image[i-2, j+2] + image[i-1, j+2])/3.0).astype(np.uint16)
        image[i-1, j+1] = ((image[i-2, j+1].astype(np.float32) + image[i-2, j+2] + image[i-1, j+2])/3.0).astype(np.uint16)
        #val4 = ((image[i+1, j+2].astype(np.float32) + image[i+2, j+2] + image[i+2, j+1])/3.0).astype(np.uint16)
        image[i+1, j+1] = ((image[i+1, j+2].astype(np.float32) + image[i+2, j+2] + image[i+2, j+1])/3.0).astype(np.uint16)

        # cross pixel
        #val5 = ((image[i-1, j-1].astype(np.float32) + image[i+1, j-1] + image[i, j-2])/3.0).astype(np.uint16)
        image[i, j-1] = ((image[i-1, j-1].astype(np.float32) + image[i+1, j-1] + image[i, j-2])/3.0).astype(np.uint16)
        #val6 = ((image[i-1, j-1].astype(np.float32) + image[i-1, j+1] + image[i-2, j])/3.0).astype(np.uint16)
        image[i-1, j] = ((image[i-1, j-1].astype(np.float32) + image[i-1, j+1] + image[i-2, j])/3.0).astype(np.uint16)
        #val7 = ((image[i-1, j+1].astype(np.float32) + image[i+1, j+1] + image[i, j+2])/3.0).astype(np.uint16)
        image[i, j+1] = ((image[i-1, j+1].astype(np.float32) + image[i+1, j+1] + image[i, j+2])/3.0).astype(np.uint16)
        #val8 = ((image[i+1, j-1].astype(np.float32) + image[i+1, j+1] + image[i+2, j])/3.0).astype(np.uint16)
        image[i+1, j] =((image[i+1, j-1].astype(np.float32) + image[i+1, j+1] + image[i+2, j])/3.0).astype(np.uint16)

        # middle pixel
        #val9 = ((image[i, j-1].astype(np.float32) + image[i-1, j] + image[i, j+1] + image[i+1, j])/4.0).astype(np.uint16)
        image[i, j] = ((image[i, j-1].astype(np.float32) + image[i-1, j] + image[i, j+1] + image[i+1, j])/4.0).astype(np.uint16)

        return image
    


