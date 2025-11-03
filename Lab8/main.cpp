
#include <opencv2/opencv.hpp>
#include <iostream>
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

// ---------- Helper: show image and wait for SPACE/Q ----------
static bool waitSpaceOrQuit(const string &winname)
{
    while (true)
    {
        int key = waitKey(0);
        if (key == 'q' || key == 'Q') // quit entire program
            exit(0);
        else if (key == ' ') // proceed
            return true;
    }
}

// ---------- Task 1: LUT (brightness, contrast, gamma) ----------
void taskLUT(const string &filename)
{
    Mat in = loadImage(filename, IMREAD_COLOR);
    if (in.empty())
        return;

    // Initial parameters from PDF example
    float brightness = 0.0f; // -1..1 (we will add)
    float contrast = 2.0f;   // factor >0
    float gamma = 2.0f;      // >0

    // Build LUT: 1x256 CV_8UC3, each entry same for B,G,R
    Mat lut(1, 256, CV_8UC3);
    for (int i = 0; i < 256; ++i)
    {
        // normalize to [0,1]
        float v = i / 255.0f;

        // According to PDF: apply gamma first, then contrast, then brightness
        // gamma transform: value = pow(value, 1. / gamma)
        float vg = pow(max(v, 0.0f), 1.0f / max(gamma, 1e-6f));

        // contrast: contrast * (value - .5) + .5
        float vc = contrast * (vg - 0.5f) + 0.5f;

        // brightness: add brightness (range approx -1..1)
        float vb = vc + brightness;

        // clamp
        vb = min(max(vb, 0.0f), 1.0f);

        uchar outv = static_cast<uchar>(round(vb * 255.0f));
        lut.at<Vec3b>(0, i) = Vec3b(outv, outv, outv);
    }

    Mat outLUT;
    LUT(in, lut, outLUT);

    // Display original and LUT result side-by-side in grid with labels
    Mat left = in.clone();
    Mat right = outLUT.clone();

    Mat grid;
    hconcat(left, right, grid);

    putText(grid, "Original", Point(10, 30), FONT_HERSHEY_SIMPLEX, 0.8, Scalar(255, 255, 255), 2);
    putText(grid, "LUT (brightness/contrast/gamma)", Point(left.cols + 10, 30), FONT_HERSHEY_SIMPLEX, 0.7, Scalar(255, 255, 255), 2);

    namedWindow("Task 1 - LUT (SPACE=next, Q=quit)", WINDOW_AUTOSIZE);
    imshow("Task 1 - LUT (SPACE=next, Q=quit)", grid);

    // Wait for user
    waitSpaceOrQuit("Task 1 - LUT (SPACE=next, Q=quit)");

    // Additionally show the LUT as a small visual (grayscale ramp)
    Mat lutVis(50, 256, CV_8UC3);
    for (int x = 0; x < 256; ++x)
        rectangle(lutVis, Point(x, 0), Point(x, 50), Scalar(lut.at<Vec3b>(0, x)[0], lut.at<Vec3b>(0, x)[0], lut.at<Vec3b>(0, x)[0]), FILLED);

    Mat combined;
    vconcat(grid, Mat(100, grid.cols, CV_8UC3, Scalar(0, 0, 0)), combined); // spacer
    Mat small;
    resize(lutVis, small, Size(grid.cols / 2, 50), 0, 0, INTER_NEAREST);

    // place lutVis under the right image
    Mat finalGrid = Mat::zeros(grid.rows + 60, grid.cols, grid.type());
    grid.copyTo(finalGrid(Rect(0, 0, grid.cols, grid.rows)));
    Mat roi = finalGrid(Rect(left.cols + 10, grid.rows + 5, small.cols, small.rows));
    small.copyTo(roi);

    putText(finalGrid, "LUT Visual (grayscale ramp)", Point(left.cols + 10, grid.rows + 10 + small.rows), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(255, 255, 255), 1);

    imshow("Task 1 - LUT (SPACE=next, Q=quit)", finalGrid);
    waitSpaceOrQuit("Task 1 - LUT (SPACE=next, Q=quit)");
    destroyWindow("Task 1 - LUT (SPACE=next, Q=quit)");
}

// ---------- Task 2: Color matrix (brightness, contrast, saturation) ----------
void taskColorMatrix(const string &filename)
{
    Mat in = loadImage(filename, IMREAD_COLOR);
    if (in.empty())
        return;

    // initial parameters from PDF
    float b = 0.0f; // brightness
    float c = 2.0f; // contrast
    float s = 2.0f; // saturation
    float t = (1.0f - c) / 2.0f;

    // Per PDF use these luminance weights (the PDF suggests two possible sets;
    // we use the first pair: 0.3086,0.6094,0.0820)
    float sr = (1.0f - s) * 0.3086f;
    float sg = (1.0f - s) * 0.6094f;
    float sb = (1.0f - s) * 0.0820f;

    // Build 4x4 custom matrix as CV_32F
    // Rows correspond to output channels R',G',B',A'; last row handles translation (t+b)
    // The PDF writes matrix as:
    // {c * (sr + s), c * (sr),     c * (sr),     0},
    // {c * (sg),     c * (sg + s), c * (sg),     0},
    // {c * (sb),     c * (sb),     c * (sb + s), 0},
    // {t + b,        t + b,        t + b,        1}
    float customMatrixData[16] = {
        c * (sr + s), c * (sr), c * (sr), 0.0f,
        c * (sg), c * (sg + s), c * (sg), 0.0f,
        c * (sb), c * (sb), c * (sb + s), 0.0f,
        t + b, t + b, t + b, 1.0f};
    Mat M(4, 4, CV_32F, customMatrixData);

    // Convert input to float RGBA 0..1
    Mat working;
    // Convert BGR -> RGBA first
    cvtColor(in, working, COLOR_BGR2RGBA);
    working.convertTo(working, CV_32F, 1.0 / 255.0);

    // Apply 4x4 matrix per-pixel using cv::transform
    Mat transformed;
    // cv::transform expects channels = M.cols (4) -> source has 4 channels
    transform(working, transformed, M);

    // After transform, convert back: clamp to 0..1, convert RGBA->BGR 0..255
    // clamp
    cv::min(transformed, 1.0f, transformed);
    cv::max(transformed, 0.0f, transformed);

    Mat outFloat = transformed.clone();
    // convert back to 0..255
    outFloat.convertTo(outFloat, CV_8UC4, 255.0);
    Mat outBGR;
    cvtColor(outFloat, outBGR, COLOR_RGBA2BGR);

    // Display original and transformed side-by-side
    Mat grid;
    hconcat(in, outBGR, grid);
    putText(grid, "Original", Point(10, 30), FONT_HERSHEY_SIMPLEX, 0.8, Scalar(255, 255, 255), 2);
    putText(grid, "ColorMatrix (brightness/contrast/saturation)", Point(in.cols + 10, 30), FONT_HERSHEY_SIMPLEX, 0.6, Scalar(255, 255, 255), 2);

    namedWindow("Task 2 - ColorMatrix (SPACE=next, Q=quit)", WINDOW_AUTOSIZE);
    imshow("Task 2 - ColorMatrix (SPACE=next, Q=quit)", grid);

    waitSpaceOrQuit("Task 2 - ColorMatrix (SPACE=next, Q=quit)");

    // Also show the matrix numerically in a small image for visualization
    // Create a simple image with matrix text
    Mat textImg(200, grid.cols, CV_8UC3, Scalar(0, 0, 0));
    string lines[4];
    for (int r = 0; r < 4; ++r)
    {
        char buf[200];
        snprintf(buf, sizeof(buf), "[% .4f, % .4f, % .4f, % .4f]",
                 customMatrixData[r * 4 + 0], customMatrixData[r * 4 + 1], customMatrixData[r * 4 + 2], customMatrixData[r * 4 + 3]);
        lines[r] = buf;
        putText(textImg, lines[r], Point(10, 40 + r * 40), FONT_HERSHEY_PLAIN, 1.2, Scalar(255, 255, 255), 1);
    }

    Mat final;
    vconcat(grid, textImg, final);
    imshow("Task 2 - ColorMatrix (SPACE=next, Q=quit)", final);
    waitSpaceOrQuit("Task 2 - ColorMatrix (SPACE=next, Q=quit)");
    destroyWindow("Task 2 - ColorMatrix (SPACE=next, Q=quit)");
}

// ---------- Main ----------
int main(int argc, char **argv)
{
    // filenames - you can replace these with any images you have
    string fname1 = "kep.png";        // for LUT task - as in the PDF instructions
    string fname2 = "color_test.jpg"; // for color matrix task

    // allow user to pass filenames on command line
    if (argc >= 2)
        fname1 = argv[1];
    if (argc >= 3)
        fname2 = argv[2];

    // Task 1: LUT (brightness/contrast/gamma)
    taskLUT(fname1);

    // Task 2: Color matrix (brightness/contrast/saturation)
    taskColorMatrix(fname2);

    return 0;
}
