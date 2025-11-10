#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <cmath>
#include <string>

using namespace cv;
using namespace std;

// ---------- Helper: Safe image loading ----------
static Mat loadImage(const string &filename, int flags = IMREAD_GRAYSCALE)
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

// ---------- Task: Accelerated Fuzzy C-Means (histogram-based) ----------
void taskFCM(const string &filename, int c = 3, float m = 1.5f, int iterations = 20)
{
    // Load grayscale image
    Mat imGray = loadImage(filename, IMREAD_GRAYSCALE);
    if (imGray.empty())
        return;

    const int Ng = 256;
    if (c < 2)
        c = 2;
    if (c > 10)
        c = 10; // safety cap

    const float eps = 1e-8f;
    const float mm = -1.0f / (m - 1.0f);

    // Histogram H[l]
    vector<int> H(Ng, 0);
    for (int y = 0; y < imGray.rows; ++y)
    {
        const uchar *row = imGray.ptr<uchar>(y);
        for (int x = 0; x < imGray.cols; ++x)
            H[row[x]]++;
    }

    // allocate arrays
    vector<vector<float>> u(c, vector<float>(Ng, 0.0f)); // partition matrix u[i][l]
    vector<float> v(c, 0.0f);                            // prototypes
    vector<vector<float>> d2(c, vector<float>(Ng, 0.0f));

    // initialize prototypes (as suggested in PDF)
    for (int i = 0; i < c; ++i)
        v[i] = 255.0f * (i + 1.0f) / (2.0f * c); // spaced initial values

    // iterate
    for (int iter = 0; iter < iterations; ++iter)
    {
        // compute d2
        for (int l = 0; l < Ng; ++l)
        {
            for (int i = 0; i < c; ++i)
            {
                float diff = l - v[i];
                d2[i][l] = diff * diff;
            }

            // find smallest d2
            int winner = 0;
            for (int i = 1; i < c; ++i)
                if (d2[i][l] < d2[winner][l])
                    winner = i;

            if (d2[winner][l] < eps) // handle zero-distance
            {
                for (int i = 0; i < c; ++i)
                    u[i][l] = 0.0f;
                u[winner][l] = 1.0f;
            }
            else
            {
                float sum = 0.0f;
                for (int i = 0; i < c; ++i)
                {
                    // d2[i][l] > 0 here
                    u[i][l] = pow(d2[i][l], mm);
                    sum += u[i][l];
                }
                if (sum == 0.0f)
                {
                    // numeric guard: if sum is zero (shouldn't), distribute uniformly
                    for (int i = 0; i < c; ++i)
                        u[i][l] = 1.0f / c;
                }
                else
                {
                    for (int i = 0; i < c; ++i)
                        u[i][l] /= sum;
                }
            }
        } // end l loop

        // update prototypes v[i]
        for (int i = 0; i < c; ++i)
        {
            double sumUp = 0.0;
            double sumDn = 0.0;
            for (int l = 0; l < Ng; ++l)
            {
                double um = pow(u[i][l], m);
                sumUp += H[l] * um * l;
                sumDn += H[l] * um;
            }
            if (sumDn == 0.0)
            {
                // keep previous v[i] if denominator is zero
                // (should be rare unless H is zero everywhere)
            }
            else
            {
                v[i] = static_cast<float>(sumUp / sumDn);
            }
        }
    } // end iterations

    // Build LUT: assign each gray level to winner's prototype (rounded)
    Mat lut(1, Ng, CV_8U);
    for (int l = 0; l < Ng; ++l)
    {
        int winner = 0;
        for (int i = 1; i < c; ++i)
            if (u[i][l] > u[winner][l])
                winner = i;
        int val = cvRound(v[winner]);
        val = std::min(255, std::max(0, val));
        lut.at<uchar>(0, l) = static_cast<uchar>(val);
    }

    Mat imSegmented;
    LUT(imGray, lut, imSegmented); // single-channel result with only c intensities

    // For visualization, convert to BGR
    Mat imGrayBGR, imSegmentedBGR;
    cvtColor(imGray, imGrayBGR, COLOR_GRAY2BGR);
    cvtColor(imSegmented, imSegmentedBGR, COLOR_GRAY2BGR);

    // Create legend text for prototypes
    Mat legend(60, imGrayBGR.cols, CV_8UC3, Scalar(0, 0, 0));
    for (int i = 0; i < c; ++i)
    {
        string txt = format("v[%d]=%.2f", i, v[i]);
        putText(legend, txt, Point(10 + i * 160, 35), FONT_HERSHEY_SIMPLEX, 0.6, Scalar(200, 200, 200), 1);
        // draw a small rectangle colored with corresponding grayscale prototype
        int xrect = 10 + i * 160;
        rectangle(legend, Point(xrect, 8), Point(xrect + 30, 30), Scalar((int)v[i], (int)v[i], (int)v[i]), FILLED);
    }

    // Grid: original | segmented
    Mat topGrid;
    hconcat(imGrayBGR, imSegmentedBGR, topGrid);
    putText(topGrid, "Original", Point(10, 30), FONT_HERSHEY_SIMPLEX, 0.8, Scalar(255, 255, 255), 2);
    putText(topGrid, "Segmented (FCM, c=" + to_string(c) + ", m=" + to_string(m) + ")", Point(imGrayBGR.cols + 10, 30), FONT_HERSHEY_SIMPLEX, 0.7, Scalar(255, 255, 255), 2);

    // We will show membership functions if c <= 3
    Mat memPlot;
    if (c <= 3)
    {
        // Create plot canvas: width = 3*Ng so each l spaced, but nicer is 3*Ng? pdf used 3*l spacing width 768 for Ng=256 -> 768 = 3*256
        int scaleX = 3;
        int plotW = scaleX * Ng;
        int plotH = 400;
        memPlot = Mat(plotH, plotW, CV_8UC3, Scalar(0, 0, 0));

        // colors for up to 3 classes
        Scalar cols[3] = {Scalar(0, 0, 255), Scalar(0, 255, 0), Scalar(255, 0, 0)}; // BGR: red, green, blue
        for (int i = 0; i < c; ++i)
        {
            for (int l = 0; l < Ng; ++l)
            {
                int px = 1 + scaleX * l;
                int py = cvRound(plotH * (1.0f - u[i][l])); // u in [0,1], invert for plotting top=0
                circle(memPlot, Point(px, py), 2, cols[i % 3], FILLED, LINE_AA);
            }
        }
        // axis labels
        putText(memPlot, "Fuzzy membership functions u_i(l)", Point(10, 20), FONT_HERSHEY_SIMPLEX, 0.6, Scalar(200, 200, 200), 1);
        putText(memPlot, "l (intensity 0..255)", Point(plotW - 180, plotH - 10), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(200, 200, 200), 1);
    }
    else
    {
        // if c>3, create small text panel describing that membership plots are not shown
        memPlot = Mat(200, topGrid.cols, CV_8UC3, Scalar(0, 0, 0));
        string msg = "Membership plots not shown for c > 3";
        putText(memPlot, msg, Point(10, 100), FONT_HERSHEY_SIMPLEX, 0.8, Scalar(200, 200, 200), 1);
    }

    // assemble final display: topGrid (side-by-side) then legend then memPlot
    Mat finalImg;
    {
        // make widths compatible for vertical concat: topGrid width = 2*cols, memPlot width maybe different
        int finalW = max(topGrid.cols, memPlot.cols);
        // create canvas tall enough
        int finalH = topGrid.rows + legend.rows + memPlot.rows + 20;
        finalImg = Mat(finalH, finalW, CV_8UC3, Scalar(0, 0, 0));

        // copy topGrid
        topGrid.copyTo(finalImg(Rect(0, 0, topGrid.cols, topGrid.rows)));
        // copy legend below topGrid
        legend.copyTo(finalImg(Rect(0, topGrid.rows, legend.cols, legend.rows)));
        // copy memPlot below legend (aligned left)
        memPlot.copyTo(finalImg(Rect(0, topGrid.rows + legend.rows + 10, memPlot.cols, memPlot.rows)));
    }

    // show final
    string winname = "FCM Segmentation (SPACE=next, Q=quit)";
    namedWindow(winname, WINDOW_AUTOSIZE);
    imshow(winname, finalImg);
    waitSpaceOrQuit(winname);
    destroyWindow(winname);

    // Additionally: show histogram + membership curves overlayed (optional small visualization)
    {
        int histW = 512, histH = 200;
        Mat histImg(histH, histW, CV_8UC3, Scalar(0, 0, 0));
        // compute normalized histogram
        vector<float> histf(Ng, 0.0f);
        int total = imGray.rows * imGray.cols;
        for (int l = 0; l < Ng; ++l)
            histf[l] = (float)H[l] / (float)total;
        // scale hist to height
        float maxh = 0.0f;
        for (int l = 0; l < Ng; ++l)
            if (histf[l] > maxh)
                maxh = histf[l];
        if (maxh <= 0.0f)
            maxh = 1.0f;

        // draw histogram as vertical lines
        for (int l = 0; l < Ng; ++l)
        {
            int x = cvRound((l / 255.0f) * (histW - 1));
            int h = cvRound((histf[l] / maxh) * (histH - 20));
            line(histImg, Point(x, histH - 1), Point(x, histH - 1 - h), Scalar(120, 120, 120));
        }

        // overlay membership functions (scaled into histogram area)
        Scalar cols2[3] = {Scalar(0, 0, 255), Scalar(0, 255, 0), Scalar(255, 0, 0)};
        for (int i = 0; i < min(c, 3); ++i)
        {
            Point prev(-1, -1);
            for (int l = 0; l < Ng; ++l)
            {
                int x = cvRound((l / 255.0f) * (histW - 1));
                int y = cvRound((1.0f - u[i][l]) * (histH - 1));
                if (prev.x >= 0)
                    line(histImg, prev, Point(x, y), cols2[i], 1, LINE_AA);
                prev = Point(x, y);
            }
        }
        namedWindow("Histogram + memberships (SPACE=next, Q=quit)", WINDOW_AUTOSIZE);
        imshow("Histogram + memberships (SPACE=next, Q=quit)", histImg);
        waitSpaceOrQuit("Histogram + memberships (SPACE=next, Q=quit)");
        destroyWindow("Histogram + memberships (SPACE=next, Q=quit)");
    }

    // finally show just the segmented image larger and wait
    {
        namedWindow("Segmented result (SPACE=next, Q=quit)", WINDOW_AUTOSIZE);
        imshow("Segmented result (SPACE=next, Q=quit)", imSegmentedBGR);
        waitSpaceOrQuit("Segmented result (SPACE=next, Q=quit)");
        destroyWindow("Segmented result (SPACE=next, Q=quit)");
    }
}

// ---------- Main ----------
int main(int argc, char **argv)
{
    string fname = "agy.bmp";
    int c = 3;
    float m = 2.0f;
    int iter = 20;

    // allow CLI: program [image] [c] [m] [iterations]
    if (argc >= 2)
        fname = argv[1];
    if (argc >= 3)
        c = atoi(argv[2]);
    if (argc >= 4)
        m = (float)atof(argv[3]);
    if (argc >= 5)
        iter = atoi(argv[4]);

    cout << "FCM segmentation\n";
    cout << "Image: " << fname << "\nClasses c=" << c << " m=" << m << " iterations=" << iter << "\n";
    cout << "Controls: SPACE = next, Q = quit\n";

    taskFCM(fname, c, m, iter);
    return 0;
}
