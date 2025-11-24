#include <opencv2/opencv.hpp>
#include <vector>
using namespace cv;
using namespace std;

// ---------- Helper: Safe image loading ----------
static Mat loadImage(const string &filename, int flags = IMREAD_COLOR)
{
    Mat im = imread(filename, flags);
    if (!im.empty())
        return im;
    vector<string> prefixes = {"", "kepek/", "../kepek/"};
    for (auto &p : prefixes)
    {
        im = imread(p + filename, flags);
        if (!im.empty())
            return im;
    }
    cerr << "Error: Cannot find " << filename << endl;
    return Mat();
}

// ---------- Helper: Wait for space or Q ----------
static bool waitForSpace()
{
    while (true)
    {
        int key = waitKey(0);
        if (key == 'q' || key == 'Q')
        {
            exit(0);
        }
        else if (key == ' ')
        {
            return true;
        }
    }
}

// ---------- Task A: Manual Hough Circle Detection ----------
void houghCircleManual()
{
    // Step 1: Load the image in color
    Mat imColor = loadImage("hod.jpg", IMREAD_COLOR);
    if (imColor.empty())
    {
        cerr << "Error: Cannot load hod.jpg. Please create a test image with circles." << endl;
        return;
    }

    // Step 2: Define the radius we're searching for
    const int R = 89;

    // Step 3: Split into channels and display
    vector<Mat> channels;
    split(imColor, channels);

    Mat grid1;
    hconcat(channels[0], channels[1], grid1);
    Mat temp;
    hconcat(grid1, channels[2], temp);
    grid1 = temp.clone();

    putText(grid1, "Blue Channel", Point(10, 30), FONT_HERSHEY_SIMPLEX, 0.8, Scalar(255), 2);
    putText(grid1, "Green Channel", Point(channels[0].cols + 10, 30), FONT_HERSHEY_SIMPLEX, 0.8, Scalar(255), 2);
    putText(grid1, "Red Channel", Point(2 * channels[0].cols + 10, 30), FONT_HERSHEY_SIMPLEX, 0.8, Scalar(255), 2);

    imshow("Task A - Hough Circles Manual (SPACE=next, Q=quit)", grid1);
    waitForSpace();

    // For this example, let's work with the blue channel (usually good for circles)
    Mat im = channels[0].clone();

    // Step 4: Run Canny edge detector
    Mat edges;
    GaussianBlur(im, im, Size(5, 5), 1.5);
    Canny(im, edges, 50, 150);

    Mat grid2;
    cvtColor(im, temp, COLOR_GRAY2BGR);
    Mat edgesColor;
    cvtColor(edges, edgesColor, COLOR_GRAY2BGR);
    hconcat(temp, edgesColor, grid2);

    putText(grid2, "Blurred Image", Point(10, 30), FONT_HERSHEY_SIMPLEX, 0.8, Scalar(0, 255, 0), 2);
    putText(grid2, "Canny Edges", Point(im.cols + 10, 30), FONT_HERSHEY_SIMPLEX, 0.8, Scalar(0, 255, 0), 2);

    imshow("Task A - Hough Circles Manual (SPACE=next, Q=quit)", grid2);
    waitForSpace();

    // Step 5: Create a black image and draw a white circle in the center
    Mat imP = Mat::zeros(2 * R + 10, 2 * R + 10, CV_8UC1);
    circle(imP, Point(imP.cols / 2, imP.rows / 2), R, Scalar(255), 1);

    Mat imPDisplay;
    cvtColor(imP, imPDisplay, COLOR_GRAY2BGR);
    resize(imPDisplay, imPDisplay, Size(400, 400));
    putText(imPDisplay, "Template Circle", Point(10, 30), FONT_HERSHEY_SIMPLEX, 0.8, Scalar(0, 255, 0), 2);

    imshow("Task A - Hough Circles Manual (SPACE=next, Q=quit)", imPDisplay);
    waitForSpace();

    // Step 6 & 7: Extract circle points from template
    vector<Point> circPoint;
    for (int y = 0; y < imP.rows; y++)
    {
        for (int x = 0; x < imP.cols; x++)
        {
            if (imP.at<uchar>(y, x) > 0)
            {
                circPoint.push_back(Point(x - imP.cols / 2, y - imP.rows / 2));
                imP.at<uchar>(y, x) = 0;
            }
        }
    }

    // Step 8: Create accumulator (Hough space)
    imP = Mat::zeros(edges.rows, edges.cols, CV_8UC1);

    for (int y = 0; y < edges.rows; y++)
    {
        for (int x = 0; x < edges.cols; x++)
        {
            if (edges.at<uchar>(y, x) > 0)
            {
                // For each edge point, vote for all possible circle centers
                for (size_t i = 0; i < circPoint.size(); i++)
                {
                    int cx = x + circPoint[i].x;
                    int cy = y + circPoint[i].y;

                    if (cx >= 0 && cx < imP.cols && cy >= 0 && cy < imP.rows)
                    {
                        uchar val = imP.at<uchar>(cy, cx);
                        if (val < 255)
                            imP.at<uchar>(cy, cx) = val + 1;
                    }
                }
            }
        }
    }

    // Normalize for display
    Mat imPNorm;
    normalize(imP, imPNorm, 0, 255, NORM_MINMAX);

    Mat grid3;
    cvtColor(edges, edgesColor, COLOR_GRAY2BGR);
    Mat accumColor;
    applyColorMap(imPNorm, accumColor, COLORMAP_JET);
    hconcat(edgesColor, accumColor, grid3);

    putText(grid3, "Edge Image", Point(10, 30), FONT_HERSHEY_SIMPLEX, 0.8, Scalar(0, 255, 0), 2);
    putText(grid3, "Hough Accumulator", Point(edges.cols + 10, 30), FONT_HERSHEY_SIMPLEX, 0.8, Scalar(0, 255, 0), 2);

    imshow("Task A - Hough Circles Manual (SPACE=next, Q=quit)", grid3);
    waitForSpace();

    // Step 9-11: Find and draw circles iteratively
    Mat result = imColor.clone();
    Scalar colors[] = {Scalar(0, 0, 255), Scalar(0, 255, 0), Scalar(255, 0, 0),
                       Scalar(255, 255, 0), Scalar(255, 0, 255), Scalar(0, 255, 255)};
    int colorIdx = 0;

    while (true)
    {
        // Step 9: Find the maximum in the accumulator
        Point pmax;
        minMaxLoc(imP, NULL, NULL, NULL, &pmax);

        double maxVal;
        minMaxLoc(imP, NULL, &maxVal, NULL, NULL);

        // If the maximum is too low, stop
        if (maxVal < 20)
            break;

        // Step 10: Draw the circle on the result image
        circle(result, pmax, R, colors[colorIdx % 6], 3);
        circle(result, pmax, 3, colors[colorIdx % 6], -1);
        colorIdx++;

        // Step 11: Black out the found circle in the accumulator
        circle(imP, pmax, 20, Scalar(0), -1);

        // Display result
        Mat grid4;
        Mat accumDisplay;
        normalize(imP, accumDisplay, 0, 255, NORM_MINMAX);
        applyColorMap(accumDisplay, accumDisplay, COLORMAP_JET);

        Mat resultResized = result.clone();
        if (result.cols != accumDisplay.cols || result.rows != accumDisplay.rows)
        {
            resize(result, resultResized, accumDisplay.size());
        }

        hconcat(resultResized, accumDisplay, grid4);

        putText(grid4, "Detected Circles", Point(10, 30), FONT_HERSHEY_SIMPLEX, 0.8, Scalar(0, 255, 0), 2);
        putText(grid4, "Updated Accumulator", Point(resultResized.cols + 10, 30), FONT_HERSHEY_SIMPLEX, 0.8, Scalar(0, 255, 0), 2);

        imshow("Task A - Hough Circles Manual (SPACE=next, Q=quit)", grid4);

        int key = waitKey(0);
        if (key == 'q' || key == 'Q')
        {
            exit(0);
        }
        else if (key == 27) // ESC
        {
            break;
        }
        else if (key == ' ')
        {
            continue;
        }
    }

    // Final result
    Mat finalGrid;
    cvtColor(edges, edgesColor, COLOR_GRAY2BGR);
    Mat resultResized = result.clone();
    if (result.cols != edgesColor.cols || result.rows != edgesColor.rows)
    {
        resize(result, resultResized, edgesColor.size());
    }
    hconcat(edgesColor, resultResized, finalGrid);

    putText(finalGrid, "Edges", Point(10, 30), FONT_HERSHEY_SIMPLEX, 0.8, Scalar(0, 255, 0), 2);
    putText(finalGrid, "Final Result", Point(edgesColor.cols + 10, 30), FONT_HERSHEY_SIMPLEX, 0.8, Scalar(0, 255, 0), 2);

    imshow("Task A - Hough Circles Manual (SPACE=next, Q=quit)", finalGrid);
    waitForSpace();
}

// ---------- Task B: Built-in HoughCircles Function ----------
void houghCircleBuiltin()
{
    // Load the image
    Mat imColor = loadImage("hod.jpg", IMREAD_COLOR);
    if (imColor.empty())
    {
        cerr << "Error: Cannot load hod.jpg" << endl;
        return;
    }

    // Convert to grayscale
    Mat gray;
    cvtColor(imColor, gray, COLOR_BGR2GRAY);

    // Apply Gaussian blur
    GaussianBlur(gray, gray, Size(9, 9), 2);

    Mat grid1;
    Mat grayColor;
    cvtColor(gray, grayColor, COLOR_GRAY2BGR);
    hconcat(imColor, grayColor, grid1);

    putText(grid1, "Original Image", Point(10, 30), FONT_HERSHEY_SIMPLEX, 0.8, Scalar(0, 255, 0), 2);
    putText(grid1, "Blurred Grayscale", Point(imColor.cols + 10, 30), FONT_HERSHEY_SIMPLEX, 0.8, Scalar(0, 255, 0), 2);

    imshow("Task B - Built-in HoughCircles (SPACE=next, Q=quit)", grid1);
    waitForSpace();

    // Detect circles using HoughCircles
    vector<Vec3f> circles;
    HoughCircles(gray, circles, HOUGH_GRADIENT, 1,
                 gray.rows / 8, // min distance between circles
                 100, 30,       // Canny thresholds
                 10, 150);      // min and max radius

    // Draw the detected circles
    Mat result = imColor.clone();
    for (size_t i = 0; i < circles.size(); i++)
    {
        Point center(cvRound(circles[i][0]), cvRound(circles[i][1]));
        int radius = cvRound(circles[i][2]);

        // Draw circle center
        circle(result, center, 3, Scalar(0, 255, 0), -1);
        // Draw circle outline
        circle(result, center, radius, Scalar(0, 0, 255), 3);

        // Add text with radius
        string text = "r=" + to_string(radius);
        putText(result, text, Point(center.x - 20, center.y - radius - 10),
                FONT_HERSHEY_SIMPLEX, 0.5, Scalar(255, 0, 0), 2);
    }

    Mat grid2;
    hconcat(imColor, result, grid2);

    putText(grid2, "Original", Point(10, 30), FONT_HERSHEY_SIMPLEX, 0.8, Scalar(0, 255, 0), 2);
    putText(grid2, "HoughCircles Result", Point(imColor.cols + 10, 30), FONT_HERSHEY_SIMPLEX, 0.8, Scalar(0, 255, 0), 2);

    string info = "Found " + to_string(circles.size()) + " circles";
    putText(grid2, info, Point(imColor.cols + 10, 60), FONT_HERSHEY_SIMPLEX, 0.6, Scalar(255, 255, 0), 2);

    imshow("Task B - Built-in HoughCircles (SPACE=next, Q=quit)", grid2);
    waitForSpace();
}

// ---------- Main ----------
int main()
{
    cout << "Hough Transform Circle Detection" << endl;
    cout << "Controls: SPACE=next slide, Q=quit, ESC=skip to next task" << endl;
    cout << "Note: Please ensure 'hod.jpg' exists with circles drawn on it." << endl;
    cout << endl;

    houghCircleBuiltin();
    houghCircleManual();

    cout << "All tasks completed!" << endl;
    return 0;
}