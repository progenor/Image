#include <opencv2/opencv.hpp>
using namespace cv;
using namespace std;

// Helper: show and wait for q/Q/ESC
bool showAndWait(const string &win, const Mat &img)
{
    imshow(win, img);
    while (true)
    {
        int key = waitKey(0);
        if (key == 'q' || key == 'Q')
            break;
        if (key == 27)
            return false;
    }
    destroyWindow(win);
    return true;
}

int main()
{
    // --- A. Feladat ---
    Mat bin = imread("pityoka.png", IMREAD_GRAYSCALE);
    if (bin.empty())
    {
        cerr << "Nem találom a pityoka.png képet!" << endl;
        return 1;
    }
    threshold(bin, bin, 128, 255, THRESH_BINARY);

    Mat eroded, dilated, open, close;
    Mat element = getStructuringElement(MORPH_RECT, Size(3, 3));

    // 10x erode, then 10x dilate (opening)
    eroded = bin.clone();
    for (int i = 0; i < 10; ++i)
        erode(eroded, eroded, element);
    dilated = eroded.clone();
    for (int i = 0; i < 10; ++i)
        dilate(dilated, dilated, element);
    Mat openResult = dilated;
    if (!showAndWait("A: 10x erode, 10x dilate (Opening)", openResult))
        return 0;

    // 10x dilate, then 10x erode (closing)
    dilated = bin.clone();
    for (int i = 0; i < 10; ++i)
        dilate(dilated, dilated, element);
    eroded = dilated.clone();
    for (int i = 0; i < 10; ++i)
        erode(eroded, eroded, element);
    Mat closeResult = eroded;
    if (!showAndWait("A: 10x dilate, 10x erode (Closing)", closeResult))
        return 0;

    // --- B. Feladat ---
    for (int r = 1; r <= 11; r += 2)
    {
        Mat ell = getStructuringElement(MORPH_ELLIPSE, Size(r, r));
        Mat er, di;
        erode(bin, er, ell);
        dilate(bin, di, ell);
        Mat grid(bin.rows, bin.cols * 2, bin.type());
        er.copyTo(grid(Rect(0, 0, bin.cols, bin.rows)));
        di.copyTo(grid(Rect(bin.cols, 0, bin.cols, bin.rows)));
        putText(grid, "Erode r=" + to_string(r), Point(10, 30), FONT_HERSHEY_SIMPLEX, 1, 255, 2);
        putText(grid, "Dilate r=" + to_string(r), Point(bin.cols + 10, 30), FONT_HERSHEY_SIMPLEX, 1, 255, 2);
        if (!showAndWait("B: Erode/Dilate ellipse r=" + to_string(r), grid))
            return 0;
    }

    // --- C. Feladat ---
    Mat color = imread("bond.jpg");
    if (color.empty())
    {
        cerr << "Nem találom a bond.jpg képet!" << endl;
        return 1;
    }
    Mat scribble_black = color.clone();
    line(scribble_black, Point(20, 20), Point(200, 200), Scalar(0, 0, 0), 10);
    line(scribble_black, Point(100, 50), Point(300, 250), Scalar(0, 0, 0), 8);
    line(scribble_black, Point(50, 250), Point(250, 50), Scalar(0, 0, 0), 6);

    for (int r = 3; r <= 21; r += 6)
    {
        Mat elem = getStructuringElement(MORPH_RECT, Size(r, r));
        Mat dil;
        dilate(scribble_black, dil, elem);
        Mat grid(color.rows, color.cols * 2, color.type());
        scribble_black.copyTo(grid(Rect(0, 0, color.cols, color.rows)));
        dil.copyTo(grid(Rect(color.cols, 0, color.cols, color.rows)));
        putText(grid, "Original (black lines)", Point(10, 30), FONT_HERSHEY_SIMPLEX, 1, Scalar(0, 255, 255), 2);
        putText(grid, "Dilated r=" + to_string(r), Point(color.cols + 10, 30), FONT_HERSHEY_SIMPLEX, 1, Scalar(0, 255, 255), 2);
        if (!showAndWait("C: Dilate black lines r=" + to_string(r), grid))
            return 0;
    }

    Mat scribble_white = color.clone();
    line(scribble_white, Point(20, 20), Point(200, 200), Scalar(255, 255, 255), 10);
    line(scribble_white, Point(100, 50), Point(300, 250), Scalar(255, 255, 255), 8);
    line(scribble_white, Point(50, 250), Point(250, 50), Scalar(255, 255, 255), 6);

    for (int r = 3; r <= 21; r += 6)
    {
        Mat elem = getStructuringElement(MORPH_RECT, Size(r, r));
        Mat ero;
        erode(scribble_white, ero, elem);
        Mat grid(color.rows, color.cols * 2, color.type());
        scribble_white.copyTo(grid(Rect(0, 0, color.cols, color.rows)));
        ero.copyTo(grid(Rect(color.cols, 0, color.cols, color.rows)));
        putText(grid, "Original (white lines)", Point(10, 30), FONT_HERSHEY_SIMPLEX, 1, Scalar(0, 0, 255), 2);
        putText(grid, "Eroded r=" + to_string(r), Point(color.cols + 10, 30), FONT_HERSHEY_SIMPLEX, 1, Scalar(0, 0, 255), 2);
        if (!showAndWait("C: Erode white lines r=" + to_string(r), grid))
            return 0;
    }

    // --- D. Feladat ---
    for (int r = 3; r <= 21; r += 6)
    {
        Mat elem = getStructuringElement(MORPH_RECT, Size(r, r));
        Mat grad;
        morphologyEx(color, grad, MORPH_GRADIENT, elem);
        Mat grid(color.rows, color.cols * 2, color.type());
        color.copyTo(grid(Rect(0, 0, color.cols, color.rows)));
        grad.copyTo(grid(Rect(color.cols, 0, color.cols, color.rows)));
        putText(grid, "Original", Point(10, 30), FONT_HERSHEY_SIMPLEX, 1, Scalar(255, 255, 255), 2);
        putText(grid, "Gradient r=" + to_string(r), Point(color.cols + 10, 30), FONT_HERSHEY_SIMPLEX, 1, Scalar(255, 255, 255), 2);
        if (!showAndWait("D: Morph gradient r=" + to_string(r), grid))
            return 0;
    }

    // --- E. Feladat ---
    Mat kukac = imread("kukac.png", IMREAD_GRAYSCALE);
    if (kukac.empty())
    {
        cerr << "Nem találom a kukac.png képet!" << endl;
        return 1;
    }
    Mat imH(kukac.size(), CV_8UC1);
    for (int x = 0; x < imH.cols; ++x)
        imH(Rect(x, 0, 1, imH.rows)).setTo(Scalar(x * 256 / imH.cols));

    // TopHat: kukacok világosabbak a háttérnél
    Mat inputTopHat;
    addWeighted(imH, 0.9, kukac, 0.1, 0, inputTopHat);

    for (int r = 3; r <= 21; r += 6)
    {
        Mat elem = getStructuringElement(MORPH_RECT, Size(r, r));
        Mat tophat;
        morphologyEx(inputTopHat, tophat, MORPH_TOPHAT, elem);
        Mat grid(kukac.rows, kukac.cols * 2, kukac.type());
        inputTopHat.copyTo(grid(Rect(0, 0, kukac.cols, kukac.rows)));
        tophat.copyTo(grid(Rect(kukac.cols, 0, kukac.cols, kukac.rows)));
        putText(grid, "Input TopHat", Point(10, 30), FONT_HERSHEY_SIMPLEX, 1, 255, 2);
        putText(grid, "TopHat r=" + to_string(r), Point(kukac.cols + 10, 30), FONT_HERSHEY_SIMPLEX, 1, 255, 2);
        if (!showAndWait("E: TopHat r=" + to_string(r), grid))
            return 0;
    }

    // BlackHat: kukacok sötétebbek a háttérnél
    Mat inputBlackHat;
    addWeighted(imH, 0.9, kukac, -0.1, 25, inputBlackHat);

    for (int r = 3; r <= 21; r += 6)
    {
        Mat elem = getStructuringElement(MORPH_RECT, Size(r, r));
        Mat blackhat;
        morphologyEx(inputBlackHat, blackhat, MORPH_BLACKHAT, elem);
        Mat grid(kukac.rows, kukac.cols * 2, kukac.type());
        inputBlackHat.copyTo(grid(Rect(0, 0, kukac.cols, kukac.rows)));
        blackhat.copyTo(grid(Rect(kukac.cols, 0, kukac.cols, kukac.rows)));
        putText(grid, "Input BlackHat", Point(10, 30), FONT_HERSHEY_SIMPLEX, 1, 255, 2);
        putText(grid, "BlackHat r=" + to_string(r), Point(kukac.cols + 10, 30), FONT_HERSHEY_SIMPLEX, 1, 255, 2);
        if (!showAndWait("E: BlackHat r=" + to_string(r), grid))
            return 0;
    }

    return 0;
}