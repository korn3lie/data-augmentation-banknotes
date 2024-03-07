"""
MIT License

Copyright (c) 2023 Sparkfish LLC

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
"""
import random
import cv2
import numpy as np
from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont
from skimage.morphology import thin
from sklearn.datasets import make_blobs
from augraphy.augmentations.brightness import Brightness


class InkGenerator:
    """Core object to generate different inks effect.

    :param ink_draw_method: Content of ink generation, select from "lines" or "text".
    :param ink_draw_method: string
    :param ink_draw_iterations: Tuple of ints determining the drawing iterations
    :param ink_draw_iterations: int
    :param ink_location: Tuple of ints determining location of ink drawings.
            Or use "random: for random line location.
    :type ink_location: tuple or string
    :param ink_background: Background of ink generation.
    :param ink_background: numpy array
    :param ink_background_size: Tuple of ints (height, width) or (height, width, channels)
        determining the size of new background for ink generation.
        A new background will be created only if ink_background is not provided.
    :param ink_background_size: tuple
    :param ink_background_color: Tuple of ints (BGR) determining the color of background.
    :type ink_background_color: tuple
    :param ink_color: Tuple of ints (BGR) determining the color of ink.
    :type ink_color: tuple
    :param ink_min_brightness: Flag to enable min brightness in the generated ink.
    :type ink_min_brightness: int
    :param ink_min_brightness_value_range: Pair of ints determining the range for min brightness value in the generated ink.
    :type ink_min_brightness_value_range: tuple
    :param ink_draw_size_range: Pair of floats determining the range for
           the size of the ink drawing.
    :type ink_draw_size_range: tuple
    :param ink_thickness_range: Pair of floats determining the range for the thickness of the generated ink.
    :type scribbles_thickness_range: tuple
    :param ink_brightness_change: A list of value change for the brightness of the ink.
           If more than one value is provided, the final value will be randomly selected.
    :type ink_brightness_change: list
    :param ink_skeletonize: Flag to enable skeletonization in the generated drawings.
    :type ink_skeletonize: int
    :param ink_skeletonize_iterations_range: Pair of ints determining the number of iterations in skeletonization process.
    :type ink_skeletonize_iterations_range: int
    :param ink_text: Text value of ink generation, valid only if ink_draw_method is "text".
    :param ink_text: string
    :param ink_text_font: List contain paths to font types. Valid only if ink content is "text".
    :type ink_text_font: list
    :param ink_text_rotate_range: Tuple of ints to determine rotation angle of "text" based drawings.
    :type ink_text_rotate_range: tuple
    :param ink_lines_coordinates: A list contain coordinates of line.
    :type ink_lines_coordinates: list
    :param ink_lines_stroke_count_range: Pair of floats determining the range for
           the number of created lines.
    :type ink_lines_stroke_count_range: tuple
    """

    def __init__(
        self,
        ink_type,
        ink_draw_method,
        ink_draw_iterations,
        ink_location,
        ink_background,
        ink_background_size,
        ink_background_color,
        ink_color,
        ink_min_brightness,
        ink_min_brightness_value_range,
        ink_draw_size_range,
        ink_thickness_range,
        ink_brightness_change,
        ink_skeletonize,
        ink_skeletonize_iterations_range,
        ink_text,
        ink_text_font,
        ink_text_rotate_range,
        ink_lines_coordinates,
        ink_lines_stroke_count_range,
    ):
        self.ink_type = ink_type
        self.ink_draw_method = ink_draw_method
        self.ink_draw_iterations = ink_draw_iterations
        self.ink_location = ink_location
        self.ink_background = ink_background
        self.ink_background_size = ink_background_size
        self.ink_background_color = ink_background_color
        self.ink_color = ink_color
        self.ink_min_brightness = ink_min_brightness
        self.ink_min_brightness_value_range = ink_min_brightness_value_range
        self.ink_draw_size_range = ink_draw_size_range
        self.ink_thickness_range = ink_thickness_range
        self.ink_brightness_change = ink_brightness_change
        self.ink_skeletonize = ink_skeletonize
        self.ink_skeletonize_iterations_range = ink_skeletonize_iterations_range
        self.ink_text = ink_text
        self.ink_text_font = ink_text_font
        self.ink_text_rotate_range = ink_text_rotate_range
        self.ink_lines_coordinates = ink_lines_coordinates
        self.ink_lines_stroke_count_range = ink_lines_stroke_count_range

    def apply_brightness(self, image):
        """Brighten image based on the minimum brightness value by using Brightness augmentation.

        :param image: The image to be brighten.
        :type image: numpy.array (numpy.uint8)
        """

        # get location of intensity < min intensity
        min_intensity = random.randint(self.ink_min_brightness_value_range[0], self.ink_min_brightness_value_range[1])
        image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        y_location, x_location = np.where(image_hsv[:, :, 2] < min_intensity)

        # if there's location where intensity < min intensity, apply brightness
        if len(y_location) > 0:
            image_min_intensity = min(image_hsv[:, :, 2][y_location, x_location])
            if image_min_intensity > 0 and image_min_intensity < min_intensity:
                brighten_ratio = abs(image_min_intensity - min_intensity) / image_min_intensity
                brighten_min = 1 + brighten_ratio
                brighten_max = 1 + brighten_ratio + 0.5
                brightness = Brightness(brightness_range=(brighten_min, brighten_max))
                image = brightness(image)

        return image

    def rotate_image(self, mat, angle, white_background=1):
        """Rotates an image (angle in degrees) and expands image to avoid
        cropping.
        """

        if white_background:
            mat = cv2.bitwise_not(mat)
        height, width = mat.shape[:2]  # image shape has 3 dimensions
        image_center = (
            width / 2,
            height / 2,
        )  # getRotationMatrix2D needs coordinates in reverse order (width, height) compared to shape

        rotation_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)

        # rotation calculates the cos and sin, taking absolutes of those.
        abs_cos = abs(rotation_mat[0, 0])
        abs_sin = abs(rotation_mat[0, 1])

        # find the new width and height bounds
        bound_w = int(height * abs_sin + width * abs_cos)
        bound_h = int(height * abs_cos + width * abs_sin)

        # subtract old image center (bringing image back to origo) and adding the new image center coordinates
        rotation_mat[0, 2] += bound_w / 2 - image_center[0]
        rotation_mat[1, 2] += bound_h / 2 - image_center[1]

        # rotate image with the new bounds and translated rotation matrix
        rotated_mat = cv2.warpAffine(mat, rotation_mat, (bound_w, bound_h))

        if white_background:
            rotated_mat = cv2.bitwise_not(rotated_mat)

        return rotated_mat

    def binary_threshold(self,image,threshold_method,threshold_arguments):

        # convert image to grascale
        if len(image.shape) > 2:
            grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            grayscale = image

        # if return grayscale image if threshold method is grayscale
        if threshold_method == "grayscale":
            return grayscale

        if threshold_arguments:
            # get input arguments for threshold function
            input_arguments = ""
            for input_argument in threshold_arguments:
                # for string argument value
                if isinstance(threshold_arguments[input_argument], str):
                    input_value = threshold_arguments[input_argument]
                # for non-string argument value
                else:
                    input_value = str(threshold_arguments[input_argument])
                # merge argument name and their value
                input_arguments += "," + input_argument + "=" + input_value

            # apply binary function and get threshold
            binary_threshold = eval(threshold_method + "(grayscale" + input_arguments + ")")
        else:
            # apply binary function and get threshold
            binary_threshold = eval(threshold_method + "(grayscale)")

        # apply binary threshold
        if threshold_method == "cv2.threshold":
            binary_threshold, image_binary = binary_threshold
        else:
            image_binary = np.uint8((grayscale > binary_threshold) * 255)

        return image_binary
    

    def apply_pen_effect(self, ink_image, ink_background):
        """Apply foreground image with pen effect to background image.

        :param ink_image: Image with pen drawings.
        :type ink_image: numpy.array (numpy.uint8)
        :param image: The background image.
        :type image: numpy.array (numpy.uint8)
        """
        return cv2.multiply(ink_image, ink_background, scale=1 / 255)


    def generate_lines(self, ink_background):
        """Generated lines drawing in background image.

        :param ink_backgrounde: The background image.
        :type ink_background: numpy.array (numpy.uint8)
        """

        # ink background is the max size
        max_height, max_width = ink_background.shape[:2]

        ink_draw_iterations = random.randint(
            self.ink_draw_iterations[0],
            self.ink_draw_iterations[1],
        )

        # background across all iterations
        combined_background = np.full((max_height, max_width, 3), fill_value=255, dtype="uint8")

        for i in range(ink_draw_iterations):

            # background of lines image
            lines_background = np.full((max_height, max_width, 3), fill_value=255, dtype="uint8")

            # each stroke count
            ink_lines_stroke_count = random.randint(
                self.ink_lines_stroke_count_range[0],
                self.ink_lines_stroke_count_range[1],
            )

            if self.ink_lines_coordinates == "random":
                # get size of foreground
                size = random.randint(max(self.ink_draw_size_range[0], 30), max(40, self.ink_draw_size_range[1]))
                xsize = ysize = min([size, max_height, max_width])
            else:
                # if coordinates are provided, all lines will be drew at one time
                ink_lines_stroke_count = 1

                xpoint_min = max_width
                ypoint_min = max_height
                xpoint_max = 0
                ypoint_max = 0
                for points in self.ink_lines_coordinates:
                    xpoints = points[:, 0]
                    ypoints = points[:, 1]
                    xpoint_min = min(xpoint_min, min(xpoints))
                    ypoint_min = min(ypoint_min, min(ypoints))
                    xpoint_max = max(xpoint_max, max(xpoints))
                    ypoint_max = max(ypoint_max, max(ypoints))

                # add offset, to prevent cut off of thicken drawing at the edges of image
                offset = 50
                xsize = xpoint_max - xpoint_min + (offset * 2)
                ysize = ypoint_max - ypoint_min + (offset * 2)

                # reset coordinates so that it starts at min coordinates
                ink_lines_coordinates = []
                for points in self.ink_lines_coordinates:
                    points_new = []
                    xpoints = points[:, 0]
                    ypoints = points[:, 1]
                    for xpoint, ypoint in zip(xpoints, ypoints):
                        points_new.append([xpoint - xpoint_min + offset, ypoint - ypoint_min + offset])
                    ink_lines_coordinates.append(np.array(points_new))

                # fixed ink location if lines coordinates are provided
                self.ink_location = (xpoint_min, ypoint_min)

            if self.ink_location == "random":
                # random paste location
                xstart = random.randint(0, max(1, max_width - xsize - 1))
                ystart = random.randint(0, max(1, max_height - ysize - 1))
            else:
                xstart, ystart = self.ink_location
                xstart = max(0, xstart - offset)
                ystart = max(0, ystart - offset)
                if xstart < 0:
                    xstart = 0
                elif xstart + xsize >= max_width:
                    xsize = max_width - xstart
                if ystart < 0:
                    ystart = 0
                elif ystart + ysize >= max_height:
                    ysize = max_height - ystart

            # create each stroke of lines
            for i in range(ink_lines_stroke_count):

                # generate lines thickness
                ink_thickness = random.randint(
                    self.ink_thickness_range[0],
                    self.ink_thickness_range[1],
                )

                # foreground of line image
                line_image = np.full((ysize, xsize, 3), fill_value=255, dtype="uint8")

                if self.ink_lines_coordinates == "random":
                    x = np.array(
                        [
                            random.randint(5, xsize - 25),
                            random.randint(5, xsize - 25),
                            random.randint(5, xsize - 25),
                            random.randint(5, xsize - 25),
                            random.randint(5, xsize - 25),
                        ],
                    )
                    y = np.array(
                        [
                            random.randint(5, ysize - 25),
                            random.randint(5, ysize - 25),
                            random.randint(5, ysize - 25),
                            random.randint(5, ysize - 25),
                            random.randint(5, ysize - 25),
                        ],
                    )

                    start_stop = [
                        random.randint(5, ysize // 2),
                        random.randint(ysize // 2, ysize - 5),
                    ]

                    # Initilaize y axis
                    lspace = np.linspace(min(start_stop), max(start_stop))

                    # calculate the coefficients.
                    z = np.polyfit(x, y, 2)

                    # calculate x axis
                    line_fitx = z[0] * lspace**2 + z[1] * lspace + z[2]
                    verts = np.array(list(zip(line_fitx.astype(int), lspace.astype(int))))
                    ink_lines_coordinates = [verts]

                # get a patch of background
                line_background = lines_background[ystart : ystart + ysize, xstart : xstart + xsize]

                # draw lines
                cv2.polylines(
                    line_image,
                    ink_lines_coordinates,
                    False,
                    self.ink_color,
                    thickness=ink_thickness,
                )

                # apply line image with ink effect to background
                line_background = self.apply_ink_effect(line_image, line_background)

                # reassign background patch to background
                lines_background[ystart : ystart + ysize, xstart : xstart + xsize] = line_background

            # combine backgrounds in each iteration
            combined_background = cv2.multiply(lines_background, combined_background, scale=1 / 255)

        # skeletonize image (optional)
        if self.ink_skeletonize:
            binary_image = cv2.cvtColor(255 - combined_background, cv2.COLOR_BGR2GRAY)
            binary_image[binary_image > 0] = 1
            max_iter = random.randint(
                self.ink_skeletonize_iterations_range[0],
                self.ink_skeletonize_iterations_range[1],
            )
            thin_mask = thin(binary_image, max_iter=max_iter) * 1
            for i in range(3):
                combined_background[:, :, i][thin_mask == 0] = 255

        # brighten image to reach minimum brightness (optional)
        if self.ink_min_brightness:
            combined_background = self.apply_brightness(combined_background)

        # merge image with lines with ink background
        image_output = cv2.multiply(combined_background, ink_background, scale=1 / 255)

        return image_output, combined_background

    def apply_ink_effect(self, foreground_image, background_image):
        """Function to apply various ink effect.

        :param foreground_image: Foreground image with lines or text.
        :type foreground_image: numpy.array (numpy.uint8)
        :param background_image: The background image.
        :type background_image: numpy.array (numpy.uint8)
        """
        # make sure both images are in a same size
        bysize, bxsize = background_image.shape[:2]
        if foreground_image.shape[0] != bysize or foreground_image.shape[1] != bxsize:
            foreground_image = cv2.resize(foreground_image, (bxsize, bysize), interpolation=cv2.INTER_AREA)

        image_merged = self.apply_pen_effect(foreground_image, background_image)
        
        return image_merged

    def generate_text(self, ink_background):
        """Generated texts drawing in background image.

        :param ink_backgrounde: The background image.
        :type ink_background: numpy.array (numpy.uint8)
        """
        # ink background is the max size
        max_height, max_width = ink_background.shape[:2]

        ink_draw_iterations = random.randint(
            self.ink_draw_iterations[0],
            self.ink_draw_iterations[1],
        )

        # background across all iterations
        combined_background = np.full((max_height, max_width, 3), fill_value=255, dtype="uint8")

        for i in range(ink_draw_iterations):

            # foreground and background of text image
            texts_image = np.full((max_height, max_width, 3), fill_value=255, dtype="uint8")
            texts_background = np.full((max_height, max_width, 3), fill_value=255, dtype="uint8")

            # convert image to PIL
            texts_image_PIL = Image.fromarray(texts_image)
            draw = ImageDraw.Draw(texts_image_PIL)
            # set font and size
            font = ImageFont.truetype(
                random.choice(self.ink_text_font),
                size=int(random.randint(self.ink_draw_size_range[0], self.ink_draw_size_range[1]) / 4),
            )
            if self.ink_text == "random":
                text = random.choice(["DEMO", "APPROVED", "CHECKED", "ORIGINAL", "COPY", "CONFIDENTIAL"])
            else:
                text = self.ink_text

            # thickness of text
            ink_thickness = random.randint(
                self.ink_thickness_range[0],
                self.ink_thickness_range[1],
            )

            # draw text
            draw.text(
                (int(max_width / 2), int(max_height / 2)),
                text,
                font=font,
                stroke_width=ink_thickness,
                fill=self.ink_color,
            )

            # convert it back to numpy array
            texts_image = np.array(texts_image_PIL)

            # rotate image
            texts_image = self.rotate_image(
                texts_image,
                random.randint(self.ink_text_rotate_range[0], self.ink_text_rotate_range[1]),
            )

            # resize to make sure rotated image size is consistent
            texts_image = cv2.resize(texts_image, (max_width, max_height), interpolation=cv2.INTER_AREA)

            # remove additional blank area
            binary_image = self.binary_threshold(texts_image, threshold_method="threshold_otsu", threshold_arguments={})

            coordinates = cv2.findNonZero(255 - binary_image)
            x, y, xsize, ysize = cv2.boundingRect(coordinates)
            # minimum size
            xsize = max(5, xsize)
            ysize = max(5, ysize)
            texts_image = texts_image[y : y + ysize, x : x + xsize]

            if self.ink_location == "random":
                # random paste location
                xstart = random.randint(0, max(0, max_width - xsize - 1))
                ystart = random.randint(0, max(0, max_height - ysize - 1))
            else:
                xstart, ystart = self.ink_location
                if xstart < 0:
                    xstart = 0
                elif xstart + xsize >= max_width:
                    xstart = max_width - xsize - 1
                if ystart < 0:
                    ystart = 0
                elif ystart + ysize >= max_height:
                    ystart = max_height - ysize - 1

            text_background = texts_background[ystart : ystart + ysize, xstart : xstart + xsize]

            # apply foreground image to background
            text_background = self.apply_ink_effect(texts_image, text_background)

            texts_background[ystart : ystart + ysize, xstart : xstart + xsize] = text_background
            
            # combine backgrounds in each iteration
            combined_background = cv2.multiply(texts_background, combined_background, scale=1 / 255)

        # skeletonize image (optional)
        if self.ink_skeletonize:
            binary_image = cv2.cvtColor(255 - combined_background, cv2.COLOR_BGR2GRAY)
            binary_image[binary_image > 0] = 1
            max_iter = random.randint(
                self.ink_skeletonize_iterations_range[0],
                self.ink_skeletonize_iterations_range[1],
            )
            thin_mask = thin(binary_image, max_iter=max_iter) * 1
            for i in range(3):
                combined_background[:, :, i][thin_mask == 0] = 255

        # brighten image to reach minimum brightness (optional)
        if self.ink_min_brightness:
            combined_background = self.apply_brightness(combined_background)

        # merge image with texts with ink background
        image_output = cv2.multiply(combined_background, ink_background, scale=1 / 255)

        return image_output, combined_background

    def generate_ink(
        self,
        ink_type=None,
        ink_draw_method=None,
        ink_draw_iterations=None,
        ink_location=None,
        ink_background=None,
        ink_background_size=None,
        ink_background_color=None,
        ink_color=None,
        ink_min_brightness=None,
        ink_min_brightness_value_range=None,
        ink_draw_size_range=None,
        ink_thickness_range=None,
        ink_brightness_change=None,
        ink_skeletonize=None,
        ink_text=None,
        ink_text_font=None,
        ink_text_rotate_range=None,
        ink_lines_coordinates=None,
        ink_lines_curvy=None,
        ink_lines_stroke_count_range=None,
    ):
    
        """
        Main function to print ink into the background.

        Parameters:
            ink_type (str): Types of ink, select from "pen", "marker" or "highlighter".
            ink_draw_method (str): Content of ink generation, select from "lines" or "text".
            ink_draw_iterations (int): Number of drawing iterations.
            ink_location (tuple): Location of ink drawings as a tuple of coordinates.
                                  Use "random" for random line location.
            ink_background (np.ndarray): Background of ink generation as a numpy array.
            ink_background_size (tuple): Size of the new background for ink generation as a tuple.
                                         A new background will be created if ink_background is not provided.
            ink_background_color (tuple): Color of the background as a tuple of BGR values.
            ink_color (tuple): Color of the ink as a tuple of BGR values.
            ink_min_brightness (int): Flag to enable minimum brightness in the generated ink.
            ink_min_brightness_value_range (tuple): Range for minimum brightness value in the generated ink.
            ink_draw_size_range (tuple): Range for the size of the ink drawing.
            ink_thickness_range (tuple): Range for the thickness of the created ink.
            ink_brightness_change (list): List of value changes for the brightness of the ink.
                                          The final value will be randomly selected if more than one value is provided.
            ink_skeletonize (int): Flag to enable skeletonization in the generated drawings.
            ink_skeletonize_iterations_range (tuple): Range for the number of iterations in the skeletonization process.
            ink_text (str): Text value of ink generation, valid only if ink_draw_method is "text".
            ink_text_font (list): List containing paths to font types. Valid only if ink content is "text".
            ink_text_rotate_range (tuple): Range for the rotation angle of "text" based drawings.
            ink_lines_coordinates (list): List containing coordinates of lines.
            ink_lines_stroke_count_range (tuple): Range for the number of created lines.
        """

        # If input is not None, replace self parameters

        if ink_type is not None:
            self.ink_type = ink_type
        if ink_draw_method is not None:
            self.ink_draw_method = ink_draw_method
        if ink_draw_iterations is not None:
            self.ink_draw_iterations = ink_draw_iterations
        if ink_location is not None:
            self.ink_location = ink_location
        if ink_background is not None:
            self.ink_background = ink_background
        if ink_background_size is not None:
            self.ink_background_size = ink_background_size
        if ink_background_color is not None:
            self.ink_background_color = ink_background_color
        if ink_color is not None:
            self.ink_color = ink_color
        if ink_min_brightness is not None:
            self.ink_min_brightness = ink_min_brightness
        if ink_min_brightness_value_range is not None:
            self.ink_min_brightness_value_range = ink_min_brightness_value_range
        if ink_draw_size_range is not None:
            self.ink_draw_size_range = ink_draw_size_range
        if ink_thickness_range is not None:
            self.ink_thickness_range = ink_thickness_range
        if ink_brightness_change is not None:
            self.ink_brightness_change = ink_brightness_change
        if ink_skeletonize is not None:
            self.ink_skeletonize = ink_skeletonize
        if ink_text is not None:
            self.ink_text = ink_text
        if ink_text_font is not None:
            self.ink_text_font = ink_text_font
        if ink_text_rotate_range is not None:
            self.ink_text_rotate_range = ink_text_rotate_range
        if ink_lines_coordinates is not None:
            self.ink_lines_coordinates = ink_lines_coordinates
        if ink_lines_curvy is not None:
            self.ink_lines_curvy = ink_lines_curvy
        if ink_lines_stroke_count_range is not None:
            self.ink_lines_stroke_count_range = ink_lines_stroke_count_range

        # retrieve or create background
        if isinstance(self.ink_background, np.ndarray):
            ink_background = self.ink_background.copy()
        else:
            ink_background = np.full(self.ink_background_size, fill_value=self.ink_background_color, dtype="uint8")


        # generate ink effect
        if self.ink_draw_method == "lines":
            image_output, mask = self.generate_lines(ink_background)
        else:
            image_output, mask = self.generate_text(ink_background)


        return image_output, mask
