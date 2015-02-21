package com.dominio.fabio.balldetectordemo;

import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;

import java.util.ArrayList;
import java.util.List;

import static org.opencv.core.Core.bitwise_and;
import static org.opencv.core.Core.circle;
import static org.opencv.core.Core.countNonZero;
import static org.opencv.core.Core.inRange;
import static org.opencv.core.Core.rectangle;
import static org.opencv.core.Core.split;
import static org.opencv.imgproc.Imgproc.GaussianBlur;
import static org.opencv.imgproc.Imgproc.HoughCircles;
import static org.opencv.imgproc.Imgproc.cvtColor;
import static org.opencv.imgproc.Imgproc.equalizeHist;

import android.util.Log;

/**
 * Created by fabio on 12/02/15.
 */
public class BallDetector {

    private static final String TAG = "BallDetector";
    // Define detection result
    //class DetectionResult{
    //    DetectionResult()

    //}

    // Define constants

    // Define instance variables
    // Frame size [pxl]
    private int width = 0;
    private int height = 0;
    // Size of the erosion and dilation kernels (circle)
    private int erosion_size = 2;
    private int dilation_size = 2;
    // Circle hough accumulator minimum threshold
    private int circle_threshold = 50;
    private int max_radius = 0;
    // Processing area size [pxl]
    private int square_size = 200;
    // Number of retained pixels from the color thresholding  within the detected maximum circle
    private int retained_pxl_nr = 0;
    // Minimum density [%] of the retained pixels from the color thresholding  within the detected maximum circle for detected object acceptance
    private double MINIMUM_DENSITY = 0.5;
    // Initialize openCV data structures and windows
    Mat frame, data, data_gray, data_lab, data_thresholded_red, data_thresholded_green, data_thresholded_blue, data_thresholded_yellow, element, mask, calibration_frame, data_xyz;
    //Mat controls(1, 400, CV_8U);
    Mat circles;
    float circ[] = new float[3];
    float circT[] = new float[3];
    Point square_position = new Point();

    Scalar redMin = new Scalar(0, 150, 128);
    Scalar redMax = new Scalar(255, 255, 255);
    Scalar greenMin = new Scalar(0, 0, 0);
    Scalar greenMax = new Scalar(255, 128, 150);
    Scalar blueMin = new Scalar(0, 128, 0);
    Scalar blueMax = new Scalar(255, 255, 128);
    Scalar yellowMin = new Scalar(0, 0, 128);
    Scalar yellowMax = new Scalar(255, 150, 255);

    Scalar blue = new Scalar(0, 0, 255);
    Scalar green = new Scalar(0, 255, 0);
    Scalar red = new Scalar(255, 0, 0);
    Scalar yellow = new Scalar(255, 255, 0);
    Scalar white = new Scalar(255, 255, 255);
    Scalar purple = new Scalar(255, 0, 255);

    /*
        Constructor
     */
    BallDetector() {
        // Initialize variables
        try {
            data_xyz = new Mat();
            data = new Mat();
            data_gray = new Mat();
            data_lab = new Mat();
            data_thresholded_red = new Mat();
            data_thresholded_green = new Mat();
            data_thresholded_blue = new Mat();
            data_thresholded_yellow = new Mat();
            element = new Mat();
            mask = new Mat();
            calibration_frame = new Mat();
            circles = new Mat();

        } catch (Exception e) {
            e.printStackTrace();
        }

    }


    //DetectionResult findBall(Mat frame){
    int findBall(Mat frame) {

//        Log.e(TAG, "in findBall");
        int color = 0;
        /*
            Declare needed variables and data structures
	    */
        // Get image size
        width = frame.cols();
        height = frame.rows();
        if (width == 0 || height == 0) {
            //return null;
            return color;
        }

        // Set minimum square size
        square_size = square_size < 10 ? 10 : square_size;

        // Compute square position
        square_position.x = width / 2 - square_size / 2;
        square_position.y = height / 2 - square_size / 2;

        // Extract subframe
        data = frame.submat(new Rect((int) square_position.x, (int) square_position.y, square_size, square_size));

        // Perform white balancing
        Scalar avg = Core.mean(data);
        cvtColor(data, data_xyz, Imgproc.COLOR_RGB2XYZ);
        Core.multiply(data_xyz, avg, data_xyz);
        List<Mat> planes = new ArrayList<Mat>();
        split(data_xyz, planes);
        Mat y = new Mat(data.rows(), data.cols(), CvType.CV_8UC1, new Scalar(100));
        planes.set(1, y);
        Core.merge(planes, data_xyz);

//        x = xy(:,1);
//        y = xy(:,2);
//        iy = 1 ./ (y + 1e-6);
//
//        X = x .* Y .* iy;
//        Z = Y .* (1-x-y) .* iy;
//
//        Y = ones(X) * Y;
//
//        xyz = [X Y Z];



		/*
		Max circle discovery
		*/
        // Convert image to grayscale
        cvtColor(data, data_gray, Imgproc.COLOR_RGB2GRAY);

        // Apply Histogram Equalization
        equalizeHist(data_gray, data_gray);

        // Reduce the noise to avoid false circle detection
        GaussianBlur(data_gray, data_gray, new Size(9, 9), 2, 2);

        // Apply the Hough Transform to find the circles
        circles = new Mat();
        HoughCircles(data_gray, circles, Imgproc.CV_HOUGH_GRADIENT, 1, data_gray.rows() / 8, 100, circle_threshold, 0, square_size / 2);

        // Get the circle with the highest radius
        max_radius = 0;

        for (int i = 0; i < circles.cols(); i++) {
            circles.get(0, i, circT);
            if (circT[2] > max_radius) {
                circ[0] = circT[0];
                circ[1] = circT[1];
                circ[2] = circT[2];
                max_radius = (int) circT[2];
            }
        }

		 /*
		    Perform filtering on color
		 */
        //Convert the captured frame from BGR to Lab
        cvtColor(data, data_lab, Imgproc.COLOR_RGB2Lab);

        // Perform color range thresholding to test each color
        inRange(data_lab, redMin, redMax, data_thresholded_red);
        inRange(data_lab, greenMin, greenMax, data_thresholded_green);
        inRange(data_lab, blueMin, blueMax, data_thresholded_blue);
        inRange(data_lab, yellowMin, yellowMax, data_thresholded_yellow);

		 /*
		    // Apply erosion and dilation to the obtained binary mask
		 if (erosion_size > 0){
			element = getStructuringElement(MORPH_ELLIPSE, Size(erosion_size, erosion_size));
			erode(data_thresholded, data_thresholded, element);
		 }
		 if (dilation_size > 0){
			element = getStructuringElement(MORPH_ELLIPSE, Size(dilation_size, dilation_size));
			dilate(data_thresholded, data_thresholded, element);
		 }
		 */

		 /*
		    Determine whether the detected maximum circle is referred to an object with the selected color
		 */
        if (circles.cols() > 0) {
            // Create a binary mask from the detected maximum circle area
            mask = Mat.zeros(square_size, square_size, CvType.CV_8U);
            circle(mask, new Point(circ[0], circ[1]), (int) circ[2], white, -1, 8, 0);
            // Compute circle area (note how the circle can partly lie outside the box)
            double circle_area = countNonZero(mask);

            calibration_frame = Mat.zeros(height, width, CvType.CV_8U);
            Mat area = calibration_frame.submat(new Rect((int) square_position.x, (int) square_position.y, square_size, square_size));
            data_thresholded_red.copyTo(area);
            // Apply the obtained mask to the thresholded image
            bitwise_and(mask, data_thresholded_red, data_thresholded_red);
            // Compute the density of retained pixels within the mask
            double density_red = countNonZero(data_thresholded_red) / circle_area;

            calibration_frame = Mat.zeros(height, width, CvType.CV_8U);
            area = calibration_frame.submat(new Rect((int) square_position.x, (int) square_position.y, square_size, square_size));
            data_thresholded_green.copyTo(area);
            // Apply the obtained mask to the thresholded image
            bitwise_and(mask, data_thresholded_green, data_thresholded_green);
            // Compute the density of retained pixels within the mask
            double density_green = countNonZero(data_thresholded_green) / circle_area;

            calibration_frame = Mat.zeros(height, width, CvType.CV_8U);
            area = area = calibration_frame.submat(new Rect((int) square_position.x, (int) square_position.y, square_size, square_size));
            data_thresholded_blue.copyTo(area);
            // Apply the obtained mask to the thresholded image
            bitwise_and(mask, data_thresholded_blue, data_thresholded_blue);
            // Compute the density of retained pixels within the mask
            double density_blue = countNonZero(data_thresholded_blue) / circle_area;

            calibration_frame = Mat.zeros(height, width, CvType.CV_8U);
            area = area = calibration_frame.submat(new Rect((int) square_position.x, (int) square_position.y, square_size, square_size));
            data_thresholded_yellow.copyTo(area);
            // Apply the obtained mask to the thresholded image
            bitwise_and(mask, data_thresholded_yellow, data_thresholded_yellow);
            // Compute the density of retained pixels within the mask
            double density_yellow = countNonZero(data_thresholded_yellow) / circle_area;

            // Compute max density
            double max_density = 0;

            if (density_red > max_density) {
                max_density = density_red;
                color = 1;
            }

            if (density_green > max_density) {
                max_density = density_green;
                color = 2;
            }

            if (density_blue > max_density) {
                max_density = density_blue;
                color = 3;
            }

            if (density_yellow > max_density) {
                max_density = density_yellow;
                color = 4;
            }

            Scalar circle_color;
            switch (color) {
                case 1:
                    circle_color = red;
                    break;
                case 2:
                    circle_color = green;
                    break;
                case 3:
                    circle_color = blue;
                    break;
                case 4:
                    circle_color = yellow;
                    break;
                default:
                    circle_color = white;
                    break;
            }

            if (max_density > MINIMUM_DENSITY) {
                Log.e(TAG, "found something: color = " + color);
                // Draw the max circle
                circle(frame, new Point(square_position.x + circ[0], square_position.y + circ[1]), 3, purple, -1, 8, 0);
                // circle outline
                circle(frame, new Point(square_position.x + circ[0], square_position.y + circ[1]), (int) circ[2], circle_color, 3, 8, 0);

                color |= 0x10;
//                Log.e(TAG, "find ball will return color = " + color);
            }
        }

        // Draw square
        rectangle(frame, new Point(square_position.x, square_position.y), new Point(square_position.x + square_size, square_position.y + square_size), green);


        return color;
    }
}
