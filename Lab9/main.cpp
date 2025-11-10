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

    float brightness = 0.0f; // -1..1
    float contrast = 2.0f;
    float gamma = 2.0f;

    Mat lut(1, 256, CV_8UC3);
    for (int i = 0; i < 256; ++i)
    {
        float v = i / 255.0f;
        float vg = pow(max(v, 0.0f), 1.0f / max(gamma, 1e-6f));
        float vc = contrast * (vg - 0.5f) + 0.5f;
        float vb = vc + brightness;
        vb = min(max(vb, 0.0f), 1.0f);
        uchar outv = static_cast<uchar>(round(vb * 255.0f));
        lut.at<Vec3b>(0, i) = Vec3b(outv, outv, outv);
    }

    Mat outLUT;
    LUT(in, lut, outLUT);

    Mat grid;
    hconcat(in, outLUT, grid);

    putText(grid, "Original", Point(10, 30), FONT_HERSHEY_SIMPLEX, 0.8, Scalar(255, 255, 255), 2);
    putText(grid, "LUT (brightness/contrast/gamma)", Point(in.cols + 10, 30),
            FONT_HERSHEY_SIMPLEX, 0.7, Scalar(255, 255, 255), 2);

    namedWindow("Task 1 - LUT (SPACE=next, Q=quit)", WINDOW_AUTOSIZE);
    imshow("Task 1 - LUT (SPACE=next, Q=quit)", grid);
    waitSpaceOrQuit("Task 1 - LUT (SPACE=next, Q=quit)");

    // --- SAFER LUT VISUALIZATION BLOCK ---
    Mat lutVis(50, 256, CV_8UC3);
    for (int x = 0; x < 256; ++x)
        rectangle(lutVis, Point(x, 0), Point(x, 50),
                  Scalar(lut.at<Vec3b>(0, x)[0], lut.at<Vec3b>(0, x)[0], lut.at<Vec3b>(0, x)[0]), FILLED);

    Mat small;
    resize(lutVis, small, Size(grid.cols / 2, 50), 0, 0, INTER_NEAREST);

    // Create final grid with enough margin
    int finalH = grid.rows + 80;
    Mat finalGrid(finalH, grid.cols, CV_8UC3, Scalar(0, 0, 0));
    grid.copyTo(finalGrid(Rect(0, 0, grid.cols, grid.rows)));

    // Compute safe ROI placement
    int roiX = in.cols + 10;
    int roiY = grid.rows + 10;
    int roiW = min(small.cols, finalGrid.cols - roiX);
    int roiH = min(small.rows, finalGrid.rows - roiY);

    if (roiW > 0 && roiH > 0)
    {
        Rect roi(roiX, roiY, roiW, roiH);
        small(Rect(0, 0, roiW, roiH)).copyTo(finalGrid(roi));
    }

    putText(finalGrid, "LUT Visual (grayscale ramp)",
            Point(in.cols + 10, grid.rows + 10 + small.rows + 10),
            FONT_HERSHEY_SIMPLEX, 0.5, Scalar(255, 255, 255), 1);

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

    float b = 0.0f, c = 2.0f, s = 2.0f;
    float t = (1.0f - c) / 2.0f;

    float sr = (1.0f - s) * 0.3086f;
    float sg = (1.0f - s) * 0.6094f;
    float sb = (1.0f - s) * 0.0820f;

    float customMatrixData[16] = {
        c * (sr + s), c * (sr), c * (sr), 0.0f,
        c * (sg), c * (sg + s), c * (sg), 0.0f,
        c * (sb), c * (sb), c * (sb + s), 0.0f,
        t + b, t + b, t + b, 1.0f};
    Mat M(4, 4, CV_32F, customMatrixData);

    Mat working;
    cvtColor(in, working, COLOR_BGR2RGBA);
    working.convertTo(working, CV_32F, 1.0 / 255.0);

    Mat transformed;
    transform(working, transformed, M);
    cv::min(transformed, 1.0f, transformed);
    cv::max(transformed, 0.0f, transformed);

    Mat outFloat;
    transformed.convertTo(outFloat, CV_8UC4, 255.0);
    Mat outBGR;
    cvtColor(outFloat, outBGR, COLOR_RGBA2BGR);

    Mat grid;
    hconcat(in, outBGR, grid);
    putText(grid, "Original", Point(10, 30), FONT_HERSHEY_SIMPLEX, 0.8, Scalar(255, 255, 255), 2);
    putText(grid, "ColorMatrix (brightness/contrast/saturation)",
            Point(in.cols + 10, 30), FONT_HERSHEY_SIMPLEX, 0.6, Scalar(255, 255, 255), 2);

    namedWindow("Task 2 - ColorMatrix (SPACE=next, Q=quit)", WINDOW_AUTOSIZE);
    imshow("Task 2 - ColorMatrix (SPACE=next, Q=quit)", grid);
    waitSpaceOrQuit("Task 2 - ColorMatrix (SPACE=next, Q=quit)");

    Mat textImg(200, grid.cols, CV_8UC3, Scalar(0, 0, 0));
    for (int r = 0; r < 4; ++r)
    {
        char buf[200];
        snprintf(buf, sizeof(buf), "[% .4f, % .4f, % .4f, % .4f]",
                 customMatrixData[r * 4 + 0], customMatrixData[r * 4 + 1],
                 customMatrixData[r * 4 + 2], customMatrixData[r * 4 + 3]);
        putText(textImg, buf, Point(10, 40 + r * 40),
                FONT_HERSHEY_PLAIN, 1.2, Scalar(255, 255, 255), 1);
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
    string fname1 = "forest.png";
    string fname2 = "forest.png";

    if (argc >= 2)
        fname1 = argv[1];
    if (argc >= 3)
        fname2 = argv[2];

    taskLUT(fname1);
    taskColorMatrix(fname2);
    return 0;
}
