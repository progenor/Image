#include <opencv2/opencv.hpp>
#include <vector>
#include <cmath>
using namespace cv;
using namespace std;

// ---------- Helper Functions ----------

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

void getColor(const Mat &im, int x, int y, uchar &blue, uchar &green, uchar &red)
{
    Vec3b color = im.at<Vec3b>(y, x);
    blue = color[0];
    green = color[1];
    red = color[2];
}

void setColor(Mat &im, int x, int y, uchar blue, uchar green, uchar red)
{
    im.at<Vec3b>(y, x) = Vec3b(blue, green, red);
}

uchar getGray(const Mat &im, int x, int y)
{
    return im.at<uchar>(y, x);
}

void setGray(Mat &im, int x, int y, uchar v)
{
    im.at<uchar>(y, x) = v;
}

int compare(const void *p1, const void *p2)
{
    uint v1 = *(uint *)p1;
    uint v2 = *(uint *)p2;

    // Extract luminance (stored in highest byte)
    uint lum1 = v1 / 0x1000000;
    uint lum2 = v2 / 0x1000000;

    if (lum1 > lum2)
        return 1;
    else if (lum1 < lum2)
        return -1;
    else
        return 0;
}

// ---------- Watershed Algorithm ----------

void watershedSegmentation()
{
    const uchar bits[8] = {1, 2, 4, 8, 16, 32, 64, 128};
    const int dx[8] = {1, 1, 0, -1, -1, -1, 0, 1};
    const int dy[8] = {0, -1, -1, -1, 0, 1, 1, 1};

    // Load color image
    Mat imColor = loadImage("3.jpg", IMREAD_COLOR);
    if (imColor.empty())
    {
        cerr << "Error: Cannot load 3.jpg. Please provide a test image." << endl;
        return;
    }

    // Convert to grayscale
    Mat imO;
    cvtColor(imColor, imO, COLOR_BGR2GRAY);

    // Display original images
    Mat grid1;
    Mat grayColor;
    cvtColor(imO, grayColor, COLOR_GRAY2BGR);
    hconcat(imColor, grayColor, grid1);
    putText(grid1, "Color Image", Point(10, 30), FONT_HERSHEY_SIMPLEX, 0.8, Scalar(0, 255, 0), 2);
    putText(grid1, "Grayscale", Point(imColor.cols + 10, 30), FONT_HERSHEY_SIMPLEX, 0.8, Scalar(0, 255, 0), 2);

    imshow("Watershed Segmentation (SPACE=next, Q=quit)", grid1);
    waitKey(800);

    // Initialize images
    Mat imG = imO.clone();
    Mat imE = imO.clone();
    Mat imKi = imO.clone();
    Mat imBe = imO.clone();
    Mat imSegm = imColor.clone();
    Mat imSegmMed = imColor.clone();
    Mat imMap = imO.clone();
    Mat imL(imO.rows, imO.cols, CV_16SC1);

    // Split color channels
    vector<Mat> imColors;
    split(imColor, imColors);
    Mat imBlue = imColors[0];
    Mat imGreen = imColors[1];
    Mat imRed = imColors[2];

    // Compute gradients for all channels
    Mat imSum = Mat::zeros(imO.size(), CV_8UC1);

    // Blue channel gradients
    Sobel(imBlue, imL, CV_16S, 1, 0);
    convertScaleAbs(imL, imE);
    Sobel(imBlue, imL, CV_16S, 0, 1);
    convertScaleAbs(imL, imG);
    add(imE, imG, imG);
    addWeighted(imSum, 1, imG, 0.33333, 0, imSum);

    // Green channel gradients
    Sobel(imGreen, imL, CV_16S, 1, 0);
    convertScaleAbs(imL, imE);
    Sobel(imGreen, imL, CV_16S, 0, 1);
    convertScaleAbs(imL, imG);
    add(imE, imG, imG);
    addWeighted(imSum, 1, imG, 0.33333, 0, imSum);

    // Red channel gradients
    Sobel(imRed, imL, CV_16S, 1, 0);
    convertScaleAbs(imL, imE);
    Sobel(imRed, imL, CV_16S, 0, 1);
    convertScaleAbs(imL, imG);
    add(imE, imG, imG);
    addWeighted(imSum, 1, imG, 0.33333, 0, imG);

    // Preprocessing - Gaussian blur
    GaussianBlur(imG, imG, Size(9, 9), 0);

    // Display gradient
    Mat grid2;
    Mat gradColor;
    applyColorMap(imG, gradColor, COLORMAP_JET);
    hconcat(grayColor, gradColor, grid2);
    putText(grid2, "Grayscale", Point(10, 30), FONT_HERSHEY_SIMPLEX, 0.8, Scalar(0, 255, 0), 2);
    putText(grid2, "Gradient Map", Point(imO.cols + 10, 30), FONT_HERSHEY_SIMPLEX, 0.8, Scalar(0, 255, 0), 2);

    imshow("Watershed Segmentation (SPACE=next, Q=quit)", grid2);
    waitKey(800);

    // Step 0 - Initialization
    erode(imG, imE, getStructuringElement(MORPH_RECT, Size(3, 3)));
    imSegm.setTo(Scalar(50, 50, 50));
    imSegmMed.setTo(Scalar(150, 150, 150));
    imBe.setTo(Scalar(0));
    imKi.setTo(Scalar(8));
    imMap.setTo(Scalar(0));

    // Step 1 - Handle steep slopes
    for (int x = 0; x < imBe.cols; ++x)
    {
        for (int y = 0; y < imBe.rows; ++y)
        {
            int fp = getGray(imG, x, y);
            int q = getGray(imE, x, y);

            if (q < fp)
            {
                for (uchar irany = 0; irany < 8; ++irany)
                {
                    if (x + dx[irany] >= 0 && x + dx[irany] < imBe.cols &&
                        y + dy[irany] >= 0 && y + dy[irany] < imBe.rows)
                    {
                        int fpv = getGray(imG, x + dx[irany], y + dy[irany]);
                        if (fpv == q)
                        {
                            setGray(imKi, x, y, irany);
                            setGray(imMap, x, y, 255);
                            uchar volt = getGray(imBe, x + dx[irany], y + dy[irany]);
                            uchar lesz = volt | bits[irany];
                            setGray(imBe, x + dx[irany], y + dy[irany], lesz);
                            break;
                        }
                    }
                }
            }
        }
    }

    // Display Step 1 result
    Mat grid3;
    Mat mapColor;
    cvtColor(imMap, mapColor, COLOR_GRAY2BGR);
    hconcat(gradColor, mapColor, grid3);
    putText(grid3, "Gradient", Point(10, 30), FONT_HERSHEY_SIMPLEX, 0.8, Scalar(0, 255, 0), 2);
    putText(grid3, "Step 1: Steep Slopes", Point(gradColor.cols + 10, 30), FONT_HERSHEY_SIMPLEX, 0.8, Scalar(0, 255, 0), 2);

    imshow("Watershed Segmentation (SPACE=next, Q=quit)", grid3);
    waitKey(800);

    // Step 2 - Handle plateaus using FIFO
    Point *fifo = new Point[imBe.cols * imBe.rows];
    int nextIn = 0;
    int nextOut = 0;

    for (int x = 0; x < imBe.cols; ++x)
    {
        for (int y = 0; y < imBe.rows; ++y)
        {
            int fp = getGray(imG, x, y);
            int pout = getGray(imKi, x, y);
            if (pout == 8)
                continue;

            int added = 0;
            for (uchar irany = 0; irany < 8; ++irany)
            {
                if (x + dx[irany] >= 0 && x + dx[irany] < imBe.cols &&
                    y + dy[irany] >= 0 && y + dy[irany] < imBe.rows)
                {
                    int fpv = getGray(imG, x + dx[irany], y + dy[irany]);
                    int pvout = getGray(imKi, x + dx[irany], y + dy[irany]);
                    if (fpv == fp && pvout == 8)
                    {
                        if (added == 0)
                            fifo[nextIn++] = Point(x, y);
                        added++;
                    }
                }
            }
        }
    }

    while (nextOut < nextIn)
    {
        Point p = fifo[nextOut++];
        int fp = getGray(imG, p.x, p.y);

        for (uchar irany = 0; irany < 8; ++irany)
        {
            if (p.x + dx[irany] >= 0 && p.x + dx[irany] < imBe.cols &&
                p.y + dy[irany] >= 0 && p.y + dy[irany] < imBe.rows)
            {
                int fpv = getGray(imG, p.x + dx[irany], p.y + dy[irany]);
                int pvout = getGray(imKi, p.x + dx[irany], p.y + dy[irany]);
                if (fp == fpv && pvout == 8)
                {
                    setGray(imKi, p.x + dx[irany], p.y + dy[irany], (irany + 4) % 8);
                    setGray(imMap, p.x + dx[irany], p.y + dy[irany], 255);
                    setGray(imBe, p.x, p.y, bits[(irany + 4) % 8] | getGray(imBe, p.x, p.y));
                    fifo[nextIn++] = Point(p.x + dx[irany], p.y + dy[irany]);
                }
            }
        }
    }

    // Display Step 2 result
    Mat grid4;
    cvtColor(imMap, mapColor, COLOR_GRAY2BGR);
    hconcat(gradColor, mapColor, grid4);
    putText(grid4, "Gradient", Point(10, 30), FONT_HERSHEY_SIMPLEX, 0.8, Scalar(0, 255, 0), 2);
    putText(grid4, "Step 2: Plateaus", Point(gradColor.cols + 10, 30), FONT_HERSHEY_SIMPLEX, 0.8, Scalar(0, 255, 0), 2);

    imshow("Watershed Segmentation (SPACE=next, Q=quit)", grid4);
    waitKey(800);

    // Step 3 - Find local minima using stack
    Point *stack = new Point[imBe.cols * imBe.rows];
    int nrStack = 0;

    for (int x = 0; x < imBe.cols; ++x)
    {
        for (int y = 0; y < imBe.rows; ++y)
        {
            int fp = getGray(imG, x, y);
            int pout = getGray(imKi, x, y);
            if (pout != 8)
                continue;

            for (uchar irany = 0; irany < 8; ++irany)
            {
                if (x + dx[irany] >= 0 && x + dx[irany] < imBe.cols &&
                    y + dy[irany] >= 0 && y + dy[irany] < imBe.rows)
                {
                    int fpv = getGray(imG, x + dx[irany], y + dy[irany]);
                    int pvout = getGray(imKi, x + dx[irany], y + dy[irany]);
                    if (pvout == 8 && fp == fpv)
                    {
                        setGray(imKi, x + dx[irany], y + dy[irany], (irany + 4) % 8);
                        setGray(imMap, x + dx[irany], y + dy[irany], 255);
                        setGray(imBe, x, y, bits[(irany + 4) % 8] | getGray(imBe, x, y));
                        stack[nrStack++] = Point(x + dx[irany], y + dy[irany]);
                    }
                }
            }

            while (nrStack > 0)
            {
                Point pv = stack[--nrStack];
                int fpv = getGray(imG, pv.x, pv.y);

                for (uchar irany = 0; irany < 8; ++irany)
                {
                    if (pv.x + dx[irany] >= 0 && pv.x + dx[irany] < imBe.cols &&
                        pv.y + dy[irany] >= 0 && pv.y + dy[irany] < imBe.rows)
                    {
                        int fpvv = getGray(imG, pv.x + dx[irany], pv.y + dy[irany]);
                        int pvvout = getGray(imKi, pv.x + dx[irany], pv.y + dy[irany]);
                        if (fpv == fpvv && pvvout == 8 && (!(pv.x + dx[irany] == x && pv.y + dy[irany] == y)))
                        {
                            setGray(imMap, pv.x + dx[irany], pv.y + dy[irany], 255);
                            setGray(imKi, pv.x + dx[irany], pv.y + dy[irany], (irany + 4) % 8);
                            setGray(imBe, pv.x, pv.y, bits[(irany + 4) % 8] | getGray(imBe, pv.x, pv.y));
                            stack[nrStack++] = Point(pv.x + dx[irany], pv.y + dy[irany]);
                        }
                    }
                }
            }
        }
    }

    // Display Step 3 result (local minima)
    Mat grid5;
    cvtColor(imMap, mapColor, COLOR_GRAY2BGR);
    hconcat(gradColor, mapColor, grid5);
    putText(grid5, "Gradient", Point(10, 30), FONT_HERSHEY_SIMPLEX, 0.8, Scalar(0, 255, 0), 2);
    putText(grid5, "Step 3: Local Minima", Point(gradColor.cols + 10, 30), FONT_HERSHEY_SIMPLEX, 0.8, Scalar(0, 255, 0), 2);

    imshow("Watershed Segmentation (SPACE=next, Q=quit)", grid5);
    waitKey(800);

    // Step 4 - Build watershed basins
    uint *medbuff = new uint[imBe.cols * imBe.rows];
    int label = 0;
    nextIn = 0;
    int spotSum[3];

    for (int x = 0; x < imBe.cols; ++x)
    {
        for (int y = 0; y < imBe.rows; ++y)
        {
            int pout = getGray(imKi, x, y);
            if (pout != 8)
                continue;

            stack[nrStack++] = Point(x, y);
            for (int i = 0; i < 3; ++i)
                spotSum[i] = 0;

            while (nrStack > 0)
            {
                Point pv = stack[--nrStack];
                fifo[nextIn++] = pv;

                uchar r, g, b;
                getColor(imColor, pv.x, pv.y, b, g, r);
                spotSum[0] += (int)b;
                spotSum[1] += (int)g;
                spotSum[2] += (int)r;

                uint o = (int)r * 0x10000 + (int)g * 0x100 + (int)b;
                o += (uint)(round((float)r * 0.299 + (float)g * 0.587 + (float)b * 0.114) * 0x1000000);
                medbuff[nextIn - 1] = o;

                int pvin = getGray(imBe, pv.x, pv.y);
                for (uchar irany = 0; irany < 8; ++irany)
                {
                    if ((bits[irany] & pvin) > 0)
                    {
                        stack[nrStack++] = Point(pv.x + dx[(irany + 4) % 8], pv.y + dy[(irany + 4) % 8]);
                    }
                }
            }

            label++;

            for (int i = 0; i < 3; ++i)
            {
                spotSum[i] = round((float)spotSum[i] / nextIn);
            }

            qsort(medbuff, nextIn, sizeof(uint), compare);
            int medR = (medbuff[nextIn / 2] % 0x1000000) / 0x10000;
            int medG = (medbuff[nextIn / 2] % 0x10000) / 0x100;
            int medB = (medbuff[nextIn / 2] % 0x100);

            for (int i = 0; i < nextIn; ++i)
            {
                setColor(imSegm, fifo[i].x, fifo[i].y, (uchar)spotSum[2], (uchar)spotSum[1], (uchar)spotSum[0]);
                setColor(imSegmMed, fifo[i].x, fifo[i].y, (uchar)medR, (uchar)medG, (uchar)medB);
            }
            nextIn = 0;
        }
    }

    // Display final results
    Mat grid6;
    hconcat(imSegm, imSegmMed, grid6);
    putText(grid6, "Average Color Segmentation", Point(10, 30), FONT_HERSHEY_SIMPLEX, 0.7, Scalar(0, 255, 0), 2);
    putText(grid6, "Median Color Segmentation", Point(imSegm.cols + 10, 30), FONT_HERSHEY_SIMPLEX, 0.7, Scalar(0, 255, 0), 2);

    string info = "Regions found: " + to_string(label);
    putText(grid6, info, Point(10, 60), FONT_HERSHEY_SIMPLEX, 0.6, Scalar(255, 255, 0), 2);

    imshow("Watershed Segmentation (SPACE=next, Q=quit)", grid6);
    waitForSpace();

    // Final comparison
    Mat grid7;
    hconcat(imColor, imSegm, grid7);
    Mat temp;
    hconcat(grid7, imSegmMed, temp);
    grid7 = temp.clone();

    putText(grid7, "Original", Point(10, 30), FONT_HERSHEY_SIMPLEX, 0.8, Scalar(0, 255, 0), 2);
    putText(grid7, "Average Colors", Point(imColor.cols + 10, 30), FONT_HERSHEY_SIMPLEX, 0.8, Scalar(0, 255, 0), 2);
    putText(grid7, "Median Colors", Point(2 * imColor.cols + 10, 30), FONT_HERSHEY_SIMPLEX, 0.8, Scalar(0, 255, 0), 2);

    imshow("Watershed Segmentation (SPACE=next, Q=quit)", grid7);
    waitForSpace();

    // Cleanup
    delete[] fifo;
    delete[] stack;
    delete[] medbuff;

    cout << "Watershed segmentation complete. Regions: " << label << endl;
}

// ---------- Main ----------

int main()
{
    cout << "Watershed Segmentation Algorithm" << endl;
    cout << "Controls: SPACE=next slide, Q=quit" << endl;
    cout << "Note: Please ensure '3.jpg' exists for segmentation." << endl;
    cout << endl;

    watershedSegmentation();

    cout << "All tasks completed!" << endl;
    return 0;
}