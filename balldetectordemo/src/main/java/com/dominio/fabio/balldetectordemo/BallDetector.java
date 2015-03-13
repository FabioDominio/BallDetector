package com.dominio.fabio.balldetectordemo;

import android.util.Log;

import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfPoint;
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
import static org.opencv.imgproc.Imgproc.Canny;
import static org.opencv.imgproc.Imgproc.GaussianBlur;
import static org.opencv.imgproc.Imgproc.HoughCircles;
import static org.opencv.imgproc.Imgproc.cvtColor;
import static org.opencv.imgproc.Imgproc.equalizeHist;

/**
 * Created by fabio on 12/02/15.
 */
public class BallDetector {
    /*
    Enumerator of considered circle colors
     */
    public enum CircleColor {COLOR_RED, COLOR_GREEN, COLOR_BLUE, COLOR_YELLOW, COLOR_OTHER};

    /*
    Enumerator of the detection status
     */
    public enum DetectionStatus{DETECTED, NOT_DETECTED, IN_PROGRESS};

    /*
    Class defining a detection result
     */
    public class DetectionResult{
        // Define inner variables
        int center_x;
        int center_y;
        int radius;
        CircleColor color;

        /*
        Detection result parametric constructor
         */
        private DetectionResult(int center_x, int center_y, int radius, CircleColor color){
            this.center_x = center_x;
            this.center_y = center_y;
            this.radius = radius;
            this.color = color;
        }

        /*
        Detection result non parametric constructor
         */
        private DetectionResult(){
            new DetectionResult(0, 0, 0, CircleColor.COLOR_OTHER);
        }
    }

    /*
        Non parametric constructor
     */
    BallDetector (){
        // Initialize variables and data structures
        detectionResult = new DetectionResult();
        squarePosition = new Point();
    }

    /**
     * Method detecting the maximum radius sphere and its color within a given frame
     * @param frame to analyze
     * @return the detected sphere data or null
     */
    DetectionResult findBall(Mat frame) {

        int frameWidth = 0;
        int frameHeight = 0;

        // Check frame size and return null if frame is void
        frameWidth = frame.cols();
        frameHeight = frame.rows();
        if (frameWidth == 0 || frameHeight == 0) {
            Log.e(TAG, "Ball detection error: passed an empty frame!");
            return null;
        }

        // Check detection area dimensions
        if (frameWidth / 2 < squareSize / 2 || frameHeight / 2 < squareSize / 2) {
            Log.e(TAG, "Ball detection error: detection area exceeds frame size!");
            return null;
        }

        // Compute square position
        squarePosition.x = frameWidth / 2 - squareSize / 2;
        squarePosition.y = frameHeight / 2 - squareSize / 2;

        // Extract detection area (only index)
        Mat detectionArea = frame.submat(new Rect((int) squarePosition.x, (int) squarePosition.y, squareSize, squareSize));

        // TODO: perform white balancing if needed, and evaluate whether it should be performed on the whole frame or only within the detection area

       	//
		//Max circle discovery
		//
        // Convert image to grayscale
        Mat detectionAreaGray = new Mat();
        cvtColor(detectionArea, detectionAreaGray, Imgproc.COLOR_RGB2GRAY);

        // Apply Histogram Equalization if desired
        if (equalizeHistogram)
            equalizeHist(detectionAreaGray, detectionAreaGray);

        // Reduce the noise to avoid false circle detection
        GaussianBlur(detectionAreaGray, detectionAreaGray, gaussianFilterSize, 2, 2);

        // Apply the Hough Transform to find the circles with a radius between 1/8 and 1/2 of the detection area size
        Mat detectedCircles = new Mat();
        HoughCircles(detectionAreaGray, detectedCircles, Imgproc.CV_HOUGH_GRADIENT, 1, detectionAreaGray.rows() / 8, houghCannyThreshold, circleThreshold , squareSize / 8, squareSize / 2);

        // Return null if no circle has been detected
        if (detectedCircles.empty()) {
            if (debug){
                // Show edges in place of color frame
                Canny(detectionAreaGray, detectionAreaGray, houghCannyThreshold, houghCannyThreshold /3);
                cvtColor(detectionAreaGray, detectionArea, Imgproc.COLOR_GRAY2RGBA);
            }
            return null;
        }

        // Else get the circle with the highest radius, assuming it is referred to the ball on the foreground
        int maxRadius = 0;
        float tempCircle[] = new float[3];
        for (int i = 0; i < detectedCircles.cols(); i++) {
            detectedCircles.get(0, i, tempCircle);
            if (tempCircle[2] > maxRadius) {
                detectionResult.center_x = (int) tempCircle[0];
                detectionResult.center_y = (int) tempCircle[1];
                detectionResult.radius = (int) tempCircle[2];
                maxRadius = (int) tempCircle[2];
            }
        }

        //
		//   Perform filtering on color
		//
        //Convert the captured frame from RGB to Lab
        Mat detectionAreaLab = new Mat();
        cvtColor(detectionArea, detectionAreaLab, Imgproc.COLOR_RGB2Lab);

        if (debug){
            // Show edges in place of color frame
            Canny(detectionAreaGray, detectionAreaGray, houghCannyThreshold, houghCannyThreshold /3);
            cvtColor(detectionAreaGray, detectionArea, Imgproc.COLOR_GRAY2RGBA);
        }

        // Perform color range thresholding to test each color
        Mat detectionAreaLabThresholdedRed = new Mat();
        Mat detectionAreaLabThresholdedGreen = new Mat();
        Mat detectionAreaLabThresholdedBlue = new Mat();
        Mat detectionAreaLabThresholdedYellow = new Mat();
        inRange(detectionAreaLab, redMin, redMax, detectionAreaLabThresholdedRed);
        inRange(detectionAreaLab, greenMin, greenMax, detectionAreaLabThresholdedGreen);
        inRange(detectionAreaLab, blueMin, blueMax, detectionAreaLabThresholdedBlue);
        inRange(detectionAreaLab, yellowMin, yellowMax, detectionAreaLabThresholdedYellow);

		// Apply erosion and dilation to the obtained binary masks, if desired
        if (morphologicalClosure){
            Mat element = Imgproc.getStructuringElement(Imgproc.MORPH_ELLIPSE, new Size(erosionSize, erosionSize));
            Imgproc.erode(detectionAreaLabThresholdedRed, detectionAreaLabThresholdedRed, element);
            Imgproc.erode(detectionAreaLabThresholdedGreen, detectionAreaLabThresholdedGreen, element);
            Imgproc.erode(detectionAreaLabThresholdedBlue, detectionAreaLabThresholdedBlue, element);
            Imgproc.erode(detectionAreaLabThresholdedYellow, detectionAreaLabThresholdedYellow, element);
            element = Imgproc.getStructuringElement(Imgproc.MORPH_ELLIPSE, new Size(dilationSize, dilationSize));
            Imgproc.dilate(detectionAreaLabThresholdedRed, detectionAreaLabThresholdedRed, element);
            Imgproc.dilate(detectionAreaLabThresholdedGreen, detectionAreaLabThresholdedGreen, element);
            Imgproc.dilate(detectionAreaLabThresholdedBlue, detectionAreaLabThresholdedBlue, element);
            Imgproc.dilate(detectionAreaLabThresholdedYellow, detectionAreaLabThresholdedYellow, element);
        }

        //
		//Determine whether the detected maximum circle is referred to an object with the selected color
		//
        // Create a binary mask from the detected maximum circle area
        Mat circleMask = Mat.zeros(squareSize, squareSize, CvType.CV_8U);
        circle(circleMask, new Point(detectionResult.center_x, detectionResult.center_y), detectionResult.radius, WHITE, -1, 8, 0);
        // Compute circle area (note how the circle can partly lie outside the box)
        double circleArea = countNonZero(circleMask);

        // Apply the obtained mask to the thresholded image
        bitwise_and(circleMask, detectionAreaLabThresholdedRed, detectionAreaLabThresholdedRed);
        bitwise_and(circleMask, detectionAreaLabThresholdedGreen, detectionAreaLabThresholdedGreen);
        bitwise_and(circleMask, detectionAreaLabThresholdedBlue, detectionAreaLabThresholdedBlue);
        bitwise_and(circleMask, detectionAreaLabThresholdedYellow, detectionAreaLabThresholdedYellow);

        // Close holes, if required
        if (fillHoles){
            List<MatOfPoint> contoursRed = new ArrayList<>();
            List<MatOfPoint> contoursGreen = new ArrayList<>();
            List<MatOfPoint> contoursBlue = new ArrayList<>();
            List<MatOfPoint> contoursYellow = new ArrayList<>();
            Imgproc.findContours(detectionAreaLabThresholdedRed, contoursRed, new Mat(), Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_NONE);
            Imgproc.findContours(detectionAreaLabThresholdedGreen, contoursGreen, new Mat(), Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_NONE);
            Imgproc.findContours(detectionAreaLabThresholdedBlue, contoursBlue, new Mat(), Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_NONE);
            Imgproc.findContours(detectionAreaLabThresholdedYellow, contoursYellow, new Mat(), Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_NONE);
            for (int i = 0; i< contoursRed.size(); ++i)
                Imgproc.drawContours(detectionAreaLabThresholdedRed, contoursRed,i, WHITE, -1);
            for (int i = 0; i< contoursGreen.size(); ++i)
                Imgproc.drawContours(detectionAreaLabThresholdedGreen, contoursGreen,i, WHITE, -1);
            for (int i = 0; i< contoursBlue.size(); ++i)
                Imgproc.drawContours(detectionAreaLabThresholdedBlue, contoursBlue,i, WHITE, -1);
            for (int i = 0; i< contoursYellow.size(); ++i)
                Imgproc.drawContours(detectionAreaLabThresholdedYellow, contoursYellow,i, WHITE, -1);
        }

        // Compute the density of retained pixels within the mask
        double densityRed = countNonZero(detectionAreaLabThresholdedRed) / circleArea;
        double densityGreen = countNonZero(detectionAreaLabThresholdedGreen) / circleArea;
        double densityBlue = countNonZero(detectionAreaLabThresholdedBlue) / circleArea;
        double densityYellow = countNonZero(detectionAreaLabThresholdedYellow) / circleArea;

        // Compute max density
        double maxDensity = 0;
        if (densityRed > maxDensity) {
            maxDensity = densityRed;
            detectionResult.color = CircleColor.COLOR_RED;
        }

        if (densityGreen > maxDensity) {
            maxDensity = densityGreen;
            detectionResult.color = CircleColor.COLOR_GREEN;
        }

        if (densityBlue > maxDensity) {
            maxDensity = densityBlue;
            detectionResult.color = CircleColor.COLOR_BLUE;
        }

        if (densityYellow > maxDensity) {
            maxDensity = densityYellow;
            detectionResult.color = CircleColor.COLOR_YELLOW;
        }

        // Only accept circle if the density is higher than the given threshold
        if (maxDensity > minimumAcceptedDensity) {
            // Set global circle position
            detectionResult.center_x += squarePosition.x;
            detectionResult.center_y += squarePosition.y;
            return detectionResult;
        } else
            return null;

    }// end findBall

    // Define utility methods
    /*
    Find ball of selected color
     */
    public DetectionResult findBall(Mat frame, CircleColor color){
        DetectionResult res = findBall(frame);
        if (res == null || res.color != color)
            // No ball detected or detected ball with a different color
            return null;
        else
            return res;
    }

    public DetectionResult findBallRefined(Mat frame){
        // Detect ball
        DetectionResult res = findBall(frame);
        if (res == null)
            return null;
        else{
            switch (res.color){
                case COLOR_RED: detectedRedFrames++; break;
                case COLOR_GREEN: detectedGreenFrames++; break;
                case COLOR_BLUE: detectedBlueFrames++; break;
                case COLOR_YELLOW: detectedYellowFrames++; break;
            }
            detectionFrameNumber++;
            res.color = CircleColor.COLOR_OTHER;

            if (detectionFrameNumber > detectionFrames) {
                // Make a decision
                CircleColor color = CircleColor.COLOR_OTHER;
                int maxFrames = 0;
                if (detectedRedFrames > maxFrames){
                    color = CircleColor.COLOR_RED;
                    maxFrames = detectedRedFrames;
                }
                if (detectedGreenFrames > maxFrames){
                    color = CircleColor.COLOR_GREEN;
                    maxFrames = detectedGreenFrames;
                }
                if (detectedBlueFrames > maxFrames){
                    color = CircleColor.COLOR_BLUE;
                    maxFrames = detectedBlueFrames;
                }
                if (detectedYellowFrames > maxFrames){
                    color = CircleColor.COLOR_YELLOW;
                    maxFrames = detectedYellowFrames;
                }
                res.color = color;
                // reset all
                detectedRedFrames = 0;
                detectedGreenFrames = 0;
                detectedBlueFrames = 0;
                detectedYellowFrames = 0;
                detectionFrameNumber = 0;
            }
            return res;
        }
    }

    /*
    Set new detection area size
    @param new detection area size
     */
    public void setDetectionAreaSize(int newSize){
        squareSize = newSize;
    }

    /*
    Set new gaussian filter size
    @param new gaussian filter size
     */
    public void setGaussianFilterSize(int newSize){
        gaussianFilterSize = new Size(newSize, newSize);
    }

    /*
    Set new circle detection accumulator threshold
    @param new accumulator threshold
     */
    public void setHoughAccumulatorThreshold(int newThreshold){
        circleThreshold = newThreshold;
    }

    /*
    Set new circle detection Canny filter threshold
    @param new Canny threshold
     */
    public void setCannyThreshold(int newThreshold){
        houghCannyThreshold = newThreshold;
    }

    /*
    Set new morphological closure thresholds
    @param new erosion threshold
    @param new dilation threshold
     */
    public void seMorphologicalThresholds(int newErosion, int newDilation){
        erosionSize = newErosion;
        dilationSize = newDilation;
    }

    /*
    Set new circle acceptance threshold
    @param new acceptance threshold
   */
    public void setCircleAcceptanceThreshold(double newThreshold){
        minimumAcceptedDensity = newThreshold;
    }

    /*
    Set morphological filter status
    @param new filter status
    */
    public void enableMorphologicalFilter(boolean flag){
        morphologicalClosure = flag;
    }

    /*
    Set histogram equalization status
    @param TRUE to enable histogram equalization, FALSE otherwise
    */
    public void enableHistogramEqualization(boolean flag){
        equalizeHistogram = flag;
    }

    /*
    Set hole filling status
    @param TRUE to enable hole filling, FALSE otherwise
    */
    public void enableHoleFilling(boolean flag){
        fillHoles = flag;
    }

    /*
    Set new Lab red range
    @param new minimum
    @param new maximum
    */
    public void setRangeRed(Scalar newMin, Scalar newMax){
        redMin = newMin;
        redMax = newMax;
    }

    /*
    Set new Lab green range
    @param new minimum
    @param new maximum
    */
    public void setRangeGreen(Scalar newMin, Scalar newMax){
        greenMin = newMin;
        greenMax = newMax;
    }

    /*
    Set new Lab blue range
    @param new minimum
    @param new maximum
    */
    public void setRangeBlue(Scalar newMin, Scalar newMax){
        blueMin = newMin;
        blueMax = newMax;
    }

    /*
    Set new Lab yellow range
    @param new minimum
    @param new maximum
    */
    public void setRangeYellow(Scalar newMin, Scalar newMax){
        yellowMin = newMin;
        yellowMax = newMax;
    }

    /*
    Enable/disable debug mode
     */
    public void toggleDebug(){
        debug = !debug;
    }

    /*
    Define constants
     */
    private static final String TAG = "BallDetector";
    private static final int DEFAULT_HOUGH_ACCUMULATOR = 50;
    private static final int DEFAULT_EROSION_SIZE = 2;
    private static final int DEFAULT_DILATION_SIZE = 2;
    private static final int DEFAULT_SQUARE_SIZE = 200;
    private static final int DEFAULT_HOUGH_CANNY_THRESHOLD = 150;
    private static final Size DEFAULT_GAUSSIAN_SIZE = new Size(9,9);
    private static final double DEFAULT_MINIMUM_ACCEPTED_CIRCLE_DENSITY = 0.5;
    private static final int DEFAULT_SKIPPED_FRAMES = 5;
    private static final int DEFAULT_DETECTION_FRAMES = 5; // Number of frames to collect before accepting the result
    private static final Scalar BLUE = new Scalar(0, 0, 255);
    private static final Scalar GREEN = new Scalar(0, 255, 0);
    private static final Scalar RED = new Scalar(255, 0, 0);
    private static final Scalar YELLOW = new Scalar(255, 255, 0);
    private static final Scalar WHITE = new Scalar(255, 255, 255);
    private static final Scalar PURPLE = new Scalar(255, 0, 255);
    private static final Scalar MIN_RED = new Scalar(0, 150, 128);
    private static final Scalar MAX_RED = new Scalar(255, 255, 255);
    private static final Scalar MIN_GREEN = new Scalar(0, 0, 0);
    private static final Scalar MAX_GREEN = new Scalar(255, 128, 150);
    private static final Scalar MIN_BLUE = new Scalar(0, 128, 0);
    private static final Scalar MAX_BLUE = new Scalar(255, 255, 128);
    private static final Scalar MIN_YELLOW = new Scalar(0, 0, 128);
    private static final Scalar MAX_YELLOW = new Scalar(255, 150, 255);

    /*
     Define instance variables and data structures
    */
    // Detection result
    private DetectionResult detectionResult;
    // Size of the erosion and dilation kernels (circle)
    private int erosionSize = DEFAULT_EROSION_SIZE;
    private int dilationSize = DEFAULT_DILATION_SIZE;
    private boolean morphologicalClosure = false;
    // Histogram equalization
    private boolean equalizeHistogram = false;
    // Hole filling
    private boolean fillHoles = false;
    // Gaussian filter
    private Size gaussianFilterSize = DEFAULT_GAUSSIAN_SIZE;
    // Circle hough accumulator minimum threshold
    private int circleThreshold = DEFAULT_HOUGH_ACCUMULATOR;
    private int houghCannyThreshold = DEFAULT_HOUGH_CANNY_THRESHOLD;
    // Processing area size and position [pxl]
    private int squareSize = DEFAULT_SQUARE_SIZE;
    private Point squarePosition;
    // Minimum density [%] of the retained pixels from the color thresholding  within the detected maximum circle for detected object acceptance
    private double minimumAcceptedDensity = DEFAULT_MINIMUM_ACCEPTED_CIRCLE_DENSITY;
    // Lab color thresholding ranges
    private Scalar redMin = MIN_RED;
    private Scalar redMax = MAX_RED;
    private Scalar greenMin = MIN_GREEN;
    private Scalar greenMax = MAX_GREEN;
    private Scalar blueMin = MIN_BLUE;
    private Scalar blueMax = MAX_BLUE;
    private Scalar yellowMin = MIN_YELLOW;
    private Scalar yellowMax = MAX_YELLOW;
    private int detectionFrameNumber = 0;
    private int detectionFrames = DEFAULT_DETECTION_FRAMES;
    private int detectedRedFrames = 0;
    private int detectedGreenFrames = 0;
    private int detectedBlueFrames = 0;
    private int detectedYellowFrames = 0;
    private boolean debug = false;
}