package com.dominio.fabio.balldetectordemo;

import android.util.Log;

import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfPoint;
import org.opencv.core.MatOfPoint2f;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.RotatedRect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;

import java.util.ArrayList;
import java.util.List;

import static org.opencv.core.Core.bitwise_and;
import static org.opencv.core.Core.circle;
import static org.opencv.core.Core.countNonZero;
import static org.opencv.core.Core.inRange;
import static org.opencv.core.Core.line;
import static org.opencv.imgproc.Imgproc.Canny;
import static org.opencv.imgproc.Imgproc.GaussianBlur;
import static org.opencv.imgproc.Imgproc.cvtColor;
import static org.opencv.imgproc.Imgproc.drawContours;
import static org.opencv.imgproc.Imgproc.equalizeHist;
import static org.opencv.imgproc.Imgproc.minAreaRect;

/**
 * Created by fabio on 12/02/15.
 */
public class BlockDetector {
    /*
    Enumerator of considered block colors
     */
    public enum BlockColor {COLOR_RED, COLOR_GREEN, COLOR_BLUE, COLOR_YELLOW, COLOR_OTHER};

    /*
    Enumerator of the detection status
     */
    public enum DetectionStatus{DETECTED, NOT_DETECTED, IN_PROGRESS};

    /*
    Class defining a detection result
     */
    public class DetectionResult{
        // Define inner variables
        RotatedRect block;
        BlockColor color;

        /*
        Detection result parametric constructor
         */
        private DetectionResult(RotatedRect block, BlockColor color){
            this.block = block;
            this.color = color;
        }

        /*
        Detection result non parametric constructor
         */
        private DetectionResult(){
            new DetectionResult(null, BlockColor.COLOR_OTHER);
        }
    }

    /*
        Non parametric constructor
     */
    BlockDetector (){
        // Initialize variables and data structures
        detectionResult = new DetectionResult();
        squarePosition = new Point();
    }

    /**
     * Method detecting the maximum area block and its color within a given frame
     * @param frame to analyze
     * @return the detected block data or null
     */
    DetectionResult findBlock(Mat frame) {

        int frameWidth = 0;
        int frameHeight = 0;

        // Check frame size and return null if frame is void
        frameWidth = frame.cols();
        frameHeight = frame.rows();
        if (frameWidth == 0 || frameHeight == 0) {
            Log.e(TAG, "Block detection error: passed an empty frame!");
            return null;
        }

        // Check detection area dimensions
        if (frameWidth / 2 < squareSize / 2 || frameHeight / 2 < squareSize / 2) {
            Log.e(TAG, "Block detection error: detection area exceeds frame size!");
            return null;
        }

        // Compute square position
        squarePosition.x = frameWidth / 2 - squareSize / 2;
        squarePosition.y = frameHeight / 2 - squareSize / 2;

        // Extract detection area (only index)
        Mat detectionArea = frame.submat(new Rect((int) squarePosition.x, (int) squarePosition.y, squareSize, squareSize));

        // TODO: perform white balancing if needed, and evaluate whether it should be performed on the whole frame or only within the detection area

        //
        //Max block discovery
        // TODO: check if it's a good criteria for assessing the block correctness
        //
        // Convert image to grayscale
        Mat detectionAreaGray = new Mat();
        cvtColor(detectionArea, detectionAreaGray, Imgproc.COLOR_RGB2GRAY);

        // Apply Histogram Equalization if desired
        if (equalizeHistogram)
            equalizeHist(detectionAreaGray, detectionAreaGray);

        // Reduce the noise to avoid false block detection
        GaussianBlur(detectionAreaGray, detectionAreaGray, gaussianFilterSize, 2, 2);

        // Detect edges
        Canny(detectionAreaGray, detectionAreaGray, cannyThresholdMin, cannyThresholdMax);

        // Apply the Hough Transform to find lines in the image
        Mat detectedLines = new Mat();
        Imgproc.HoughLines(detectionAreaGray, detectedLines, houghRho, houghTheta, houghThreshold);

        // Return null if no line has been detected
        if (detectedLines.empty()){
            if (debug){
                // Show edges in place of color frame
                cvtColor(detectionAreaGray, detectionArea, Imgproc.COLOR_GRAY2RGBA);
            }
            return null;
        }

        for (int i = 0; i < detectedLines.cols(); ++i){
            Point p1 = new Point();
            Point p2 = new Point();
            double x = 0;
            double y = 0;
            boolean p1set = false;
            boolean p2set = false;
            float tempLine[] = new float[2];
            // compute intersection with line x=0
            detectedLines.get(0, i, tempLine);
            x = 0;
            y = (tempLine[0] - x*Math.cos(tempLine[1])) / Math.sin(tempLine[1]);
            if (y >= 0 && y < squareSize){
                p1.x = x;
                p1.y = y;
                p1set = true;
            }

            // compute intersection with line x=square_size
            x = squareSize;
            y = (tempLine[0] - x*Math.cos(tempLine[1])) / Math.sin(tempLine[1]);
            if (y >= 0 && y < squareSize){
                if (!p1set){
                    p1.x = x;
                    p1.y = y;
                    p1set = true;
                }
                else{
                    p2.x = x;
                    p2.y = y;
                    p2set = true;
                }
            }

            // compute intersection with line y=0
            y = 0;
            x = (tempLine[0] - y*Math.sin(tempLine[1])) / Math.cos(tempLine[1]);
            if (x >= 0 && x < squareSize){
                if (!p1set){
                    p1.x = x;
                    p1.y = y;
                    p1set = true;
                }
                else{
                    p2.x = x;
                    p2.y = y;
                    p2set = true;
                }
            }

            // compute intersection with line y=0
            y = squareSize;
            x = (tempLine[0] - y*Math.sin(tempLine[1])) / Math.cos(tempLine[1]);
            if (x >= 0 && x < squareSize){
                if (!p1set){
                    p1.x = x;
                    p1.y = y;
                    p1set = true;
                }
                else{
                    p2.x = x;
                    p2.y = y;
                    p2set = true;
                }
            }

            p1.x = p1.x+squarePosition.x;
            p1.y = p1.y+squarePosition.y;
            p2.x = p2.x+squarePosition.x;
            p2.y = p2.y+squarePosition.y;

            //Point p1 = Point(square_position.x+0,square_position.y+y);
            //Point p2 = Point(square_position.x+x,square_position.y+0);
            line(frame, p1, p2, GREEN);
            //cv::line(frame, Point(square_position.x+lines[i][0],square_position.y+lines[i][1]), Point(square_position.x+lines[i][2],square_position.y+lines[i][3]), green, 3, CV_AA);
        }


        // Else compute all the intersection points between pair of lines forming an angle around 90°
        Point[][] adjMat = new Point[detectedLines.cols()][detectedLines.cols()];
        float tempLine[] = new float[2];
        for (int i = 0; i < adjMat.length; ++i)
            // Check intersection with other lines
            for (int j = i + 1; j < adjMat.length; ++j){
                detectedLines.get(0, i, tempLine);
                double r1 = tempLine[0];
                double cos1 = Math.cos(tempLine[1]);
                double sin1 = Math.sin(tempLine[1]);
                detectedLines.get(0, j, tempLine);
                double r2 = tempLine[0];
                double cos2 = Math.cos(tempLine[1]);
                double sin2 = Math.sin(tempLine[1]);
                double m1 = -cos1/sin1;
                double m2 = -cos2/sin2;
                double angle = Math.atan(Math.abs((m1-m2)/(1+m2*m1)));
                //
                // Manage special cases
                //
                // Orthogonal lines
                if (m1*m2 == -1)
                    angle = 1.57;

                if (m1 == 0)
                    // Line 1 parallel to the x axis
                    if (m2 == 0)
                        // Line 2 parallel to the x axis
                        angle = 0;
                    else
                        angle = Math.atan(Math.abs(m2));

                if (m2 == 0)
                    // Line 2 parallel to the x axis
                    if (m1 == 0)
                        // Line 1 parallel to the x axis
                        angle = 0;
                    else
                        angle = Math.atan(Math.abs(m1));

                if (Double.isInfinite(m1))
                    // Line 1 parallel to the y axis
                    if (Double.isInfinite(m2))
                        // Line 2 parallel to the y axis
                        angle = 0;
                    else
                        angle = 1.57 - Math.atan(Math.abs(m2));

                if (Double.isInfinite(m2))
                    // Line 2 parallel to the y axis
                    if (Double.isInfinite(m1))
                        // Line 1 parallel to the y axis
                        angle = 0;
                    else
                        angle = 1.57 - Math.atan(Math.abs(m1));

                // Compute intersection point between the two lines in the whole frame
                double x = (r2*sin1 - r1*sin2) / (cos2*sin1 -cos1*sin2);
                double y = 0;
                if (sin1 != 0)
                    //y = (r1-x*cos1) /sin1 + squarePosition.y;
                    y = (r1-x*cos1) /sin1;
                else
                    //y = (r2-x*cos2) /sin2 + squarePosition.y;
                    y = (r2-x*cos2) /sin2;
                //x += squarePosition.x;
                /*
				// Compute intersection
                // P(x,y) = (r_2sin(theta_1) - r_1sin(theta_2))/(cos(theta_2)sin(theta_1) -cos(theta_1)sin(theta_2)), -cos(theta_2)/sin(theta_2)x + r_2/sin(theta_2)
				double m1 = (lines[i][3] - lines[i][1]) / (double)(lines[i][2] - lines[i][0]);
				double m2 = (lines[j][3] - lines[j][1]) / (double)(lines[j][2] - lines[j][0]);
				double x1 = lines[i][0];
				double x2 = lines[j][0];
				double y1 = lines[i][1];
				double y2 = lines[j][1];
				double x = (-y2+y1+m2*x2-m1*x1)/(m2-m1);
				double y = m2*(x-x2)+y2;
				double cosine = ((x-x1)*(x-x2)+(y-y1)*(y-y2))/(sqrt((x-x1)*(x-x1)+(y-y1)*(y-y1))*sqrt((x-x2)*(x-x2)+(y-y2)*(y-y2)));
				if(cosine >= -0.17 && cosine <= 0.17 && x >= 0 && x< data.cols && y >= 0 && y < data.rows){
				*/

                // Register intersection if the angle is around 90° and the intersection is within the detection area
                //if (angle >= 1.48 && angle <= 1.66 && x > squarePosition.x && x < squareSize + squarePosition.x && y > squarePosition.y && y < squareSize + squarePosition.y){
                if (angle >= 1.48 && angle <= 1.66 && x > 0 && x < squareSize && y > 0 && y < squareSize){
                    // Angle around 90°
                    adjMat[i][j] = new Point(x,y);
                    adjMat[j][i] = new Point(x,y);
                }
            }

        // draw intersection points
        for (int i = 0; i < adjMat.length; ++i)
            // Check intersection with other lines
            for (int j = i + 1; j < adjMat.length; ++j){
                if (adjMat[i][j] != null)
                    circle(frame, new Point(adjMat[i][j].x + squarePosition.x, adjMat[i][j].y + squarePosition.y), 3, PURPLE, 3);
            }


        // Compute all the line quadruples forming a rectangle
        // TODO: avoid creating the same rectangles starting from a different line
        //List<MatOfInt4> rectangles = new ArrayList<>();
        List<RotatedRect> rectangles = new ArrayList<>();

        for (int i = 0; i < adjMat.length; ++i){
            int line1 = i;
            int line2 = -1;
            int line3 = -1;
            int line4 = -1;
            for (int j = 0; j < adjMat.length; ++j)
                if (adjMat[line1][j] != null && j != line1){
                    line2 = j;
                    for (int k = 0; k < adjMat.length; ++k)
                        if (adjMat[line2][k] != null && k != line1 && k != line2){
                            line3 = k;
                            for (int l = 0; l < adjMat.length; ++l)
                                if (adjMat[line3][l] != null && l != line1 && l != line2 && l != line3){
                                    line4 = l;
                                    // Look for adj[l][i] existence
                                    for (int m = 0; m < adjMat.length; ++m)
                                        if (adjMat[line4][m] != null && m == i) {
                                            // Find rotated min area rect given int corners
                                            MatOfPoint2f vertices = new MatOfPoint2f(adjMat[line1][line2], adjMat[line2][line3], adjMat[line3][line4], adjMat[line4][line1]);
                                            RotatedRect rectangle = minAreaRect(vertices);
                                            //rectangles.add(rectangles.size(), new MatOfInt4(line1, line2, line3, line4));
                                            // Evaluate rectangle size
                                            if (rectangle.size.width > minBlockAcceptedLength && rectangle.size.height > minBlockAcceptedLength)
                                                rectangles.add(rectangles.size(), rectangle);
                                        }
                                }
                        }
                } // end select line2
        }

        // Select rectangle
        // TODO: define rectangle selection method
        // Return null if no valid rectangles have been found
        if (rectangles.isEmpty()) {
            if (debug){
                // Show edges in place of color frame
                cvtColor(detectionAreaGray, detectionArea, Imgproc.COLOR_GRAY2RGBA);
            }
            return null;
        }

        // For the moment, select the rectangle with the maximum area
        int maxArea = 0;
        RotatedRect selectedRectangle = null;
        for (RotatedRect rectangle:rectangles){
            if (rectangle.size.width * rectangle.size.height > maxArea){
                maxArea = (int) (rectangle.size.width * rectangle.size.height);
                selectedRectangle = rectangle;
            }
        }

        //
        //   Perform filtering on color
        //
        //Convert the captured frame from RGB to Lab
        Mat detectionAreaLab = new Mat();
        cvtColor(detectionArea, detectionAreaLab, Imgproc.COLOR_RGB2Lab);

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

        //
        //Determine whether the detected maximum circle is referred to an object with the selected color
        //
        // Create a binary mask from the detected maximum rectangle area
        Mat rectangleMask = Mat.zeros(squareSize, squareSize, CvType.CV_8U);
        Point[] vertices = new Point[4];
        selectedRectangle.points(vertices);
        List<MatOfPoint> contours = new ArrayList<>();
        contours.add(0, new MatOfPoint(vertices[0], vertices[1], vertices[2], vertices[3], vertices[0]));
        drawContours(rectangleMask, contours, 0, WHITE, -1);
        // Compute circle area (note how the circle can partly lie outside the box)
        double circleArea = countNonZero(rectangleMask);

        // Apply the obtained mask to the thresholded image
        bitwise_and(rectangleMask, detectionAreaLabThresholdedRed, detectionAreaLabThresholdedRed);
        // Compute the density of retained pixels within the mask
        double densityRed = countNonZero(detectionAreaLabThresholdedRed) / circleArea;

        bitwise_and(rectangleMask, detectionAreaLabThresholdedGreen, detectionAreaLabThresholdedGreen);
        // Compute the density of retained pixels within the mask
        double densityGreen = countNonZero(detectionAreaLabThresholdedGreen) / circleArea;

        bitwise_and(rectangleMask, detectionAreaLabThresholdedBlue, detectionAreaLabThresholdedBlue);
        // Compute the density of retained pixels within the mask
        double densityBlue = countNonZero(detectionAreaLabThresholdedBlue) / circleArea;

        bitwise_and(rectangleMask, detectionAreaLabThresholdedYellow, detectionAreaLabThresholdedYellow);
        // Compute the density of retained pixels within the mask
        double densityYellow = countNonZero(detectionAreaLabThresholdedYellow) / circleArea;

        // Compute max density
        double maxDensity = 0;
        if (densityRed > maxDensity) {
            maxDensity = densityRed;
            detectionResult.color = BlockColor.COLOR_RED;
        }

        if (densityGreen > maxDensity) {
            maxDensity = densityGreen;
            detectionResult.color = BlockColor.COLOR_GREEN;
        }

        if (densityBlue > maxDensity) {
            maxDensity = densityBlue;
            detectionResult.color = BlockColor.COLOR_BLUE;
        }

        if (densityYellow > maxDensity) {
            maxDensity = densityYellow;
            detectionResult.color = BlockColor.COLOR_YELLOW;
        }

        if (debug){
            // Show edges in place of color frame
            cvtColor(detectionAreaGray, detectionArea, Imgproc.COLOR_GRAY2RGBA);
        }

        Point[] pts = new Point[4];
        selectedRectangle.points(pts);
        pts[0].x += squarePosition.x;
        pts[0].y += squarePosition.y;
        pts[1].x += squarePosition.x;
        pts[1].y += squarePosition.y;
        pts[2].x += squarePosition.x;
        pts[2].y += squarePosition.y;
        pts[3].x += squarePosition.x;
        pts[3].y += squarePosition.y;
        MatOfPoint2f vert = new MatOfPoint2f(pts[0], pts[1], pts[2], pts[3]);
        selectedRectangle = minAreaRect(vert);

        //selectedRectangle.boundingRect().y += -selectedRectangle.center.y + squarePosition.y;

        // Only accept block if the density is higher than the given threshold
        if (maxDensity > minimumAcceptedDensity) {
            detectionResult.block = selectedRectangle;
            return detectionResult;
        } else
            return null;

    }// end findBlock

    // Define utility methods
    /*
    Find ball of selected color
     */
    public DetectionResult findBlock(Mat frame, BlockColor color){
        DetectionResult res = findBlock(frame);
        if (res == null || res.color != color)
            // No ball detected or detected ball with a different color
            return null;
        else
            return res;
    }

    public DetectionResult findBlockRefined(Mat frame){
        // Detect ball
        DetectionResult res = findBlock(frame);
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
            res.color = BlockColor.COLOR_OTHER;

            if (detectionFrameNumber > detectionFrames) {
                // Make a decision
                BlockColor color = BlockColor.COLOR_OTHER;
                int maxFrames = 0;
                if (detectedRedFrames > maxFrames){
                    color = BlockColor.COLOR_RED;
                    maxFrames = detectedRedFrames;
                }
                if (detectedGreenFrames > maxFrames){
                    color = BlockColor.COLOR_GREEN;
                    maxFrames = detectedGreenFrames;
                }
                if (detectedBlueFrames > maxFrames){
                    color = BlockColor.COLOR_BLUE;
                    maxFrames = detectedBlueFrames;
                }
                if (detectedYellowFrames > maxFrames){
                    color = BlockColor.COLOR_YELLOW;
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
        minBlockAcceptedLength = squareSize / 3;
    }

    /*
    Set new gaussian filter size
    @param new gaussian filter size
     */
    public void setGaussianFilterSize(int newSize){
        gaussianFilterSize = new Size(newSize, newSize);
    }

    /*
    Set new morphological closure thresholds
    @param new erosion threshold
    @param new dilation threshold
     */
    public void setMorphologicalThresholds(int newErosion, int newDilation){
        erosionSize = newErosion;
        dilationSize = newDilation;
    }

    /*
    Set new Canny edge detector thresholds
    @param new minimum threshold
    @param new maximum threshold
     */
    public void setCannyThresholds(int newMin, int newMax){
        cannyThresholdMin = newMin;
        cannyThresholdMax = newMax;
    }

    /*
    Set new hough line detector thresholds
    @param new minimum threshold
    @param new maximum threshold
     */
    public void setHoughThresholds(int rho, double theta, int minVotes){
        houghRho = rho;
        houghTheta = theta;
        houghThreshold = minVotes;
    }

    /*
    Set new circle acceptance threshold
    @param new acceptance threshold
   */
    public void setBlockAcceptanceThreshold(double newThreshold){
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
    private static final String TAG = "BlockDetector";
    private static final int DEFAULT_HOUGH_THRESHOLD = 50;
    private static final int DEFAULT_HOUGH_RHO = 1;
    private static final double DEFAULT_HOUGH_THETA = 10 * Math.PI / 180;
    private static final int DEFAULT_EROSION_SIZE = 2;
    private static final int DEFAULT_DILATION_SIZE = 2;
    private static final int DEFAULT_SQUARE_SIZE = 200;
    private static final int DEFAULT_CANNY_MIN = 50;
    private static final int DEFAULT_CANNY_MAX = 150;
    private static final Size DEFAULT_GAUSSIAN_SIZE = new Size(9,9);
    private static final double DEFAULT_MINIMUM_ACCEPTED_BLOCK_DENSITY = 0.5;
    private static final int DEFAULT_MINIMUM_ACCEPTED_BLOCK_LENGTH = DEFAULT_SQUARE_SIZE / 3;
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
    // Canny thresholds
    private int cannyThresholdMin = DEFAULT_CANNY_MIN;
    private int cannyThresholdMax = DEFAULT_CANNY_MAX;
    // Hough thresholds
    private int houghRho = DEFAULT_HOUGH_RHO;
    private double houghTheta = DEFAULT_HOUGH_THETA;
    private int houghThreshold = DEFAULT_HOUGH_THRESHOLD;
    // Processing area size and position [pxl]
    private int squareSize = DEFAULT_SQUARE_SIZE;
    private Point squarePosition;
    // Minimum density [%] of the retained pixels from the color thresholding  within the detected maximum circle for detected object acceptance
    private double minimumAcceptedDensity = DEFAULT_MINIMUM_ACCEPTED_BLOCK_DENSITY;
    // Minimum block shortest side length accepted
    private int minBlockAcceptedLength = DEFAULT_MINIMUM_ACCEPTED_BLOCK_LENGTH;
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