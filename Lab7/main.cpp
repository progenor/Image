#include <opencv2/opencv.hpp>
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

// ---------- Helper: get and set gray pixel ----------
static inline uchar getGray(const Mat &im, int x, int y)
{
    return im.at<uchar>(y, x);
}
static inline void setBlack(Mat &im, int x, int y)
{
    im.at<uchar>(y, x) = 0;
}

// ---------- Task 1: Distance Transform ----------
void distanceTransformTask()
{
    Mat im = loadImage("kep.png", IMREAD_GRAYSCALE);
    if (im.empty())
        return;

    threshold(im, im, 128, 255, THRESH_BINARY);

    Mat dist;
    distanceTransform(im, dist, DIST_L2, 3, CV_32F);
    dist.convertTo(dist, CV_8U, 5, 0);

    Mat se = getStructuringElement(MORPH_ELLIPSE, Size(5, 5));
    Mat dilated = dist.clone();

    Mat grid;
    hconcat(im, dist, grid);
    putText(grid, "Original", Point(10, 30), FONT_HERSHEY_SIMPLEX, 0.8, Scalar(255), 2);
    putText(grid, "Distance Map", Point(im.cols + 10, 30), FONT_HERSHEY_SIMPLEX, 0.8, Scalar(255), 2);

    imshow("Task 1 - Distance Transform (SPACE=next, Q=quit)", grid);

    while (true)
    {
        int key = waitKey(0);
        if (key == 'q' || key == 'Q')
            exit(0);
        else if (key == ' ')
            break;
    }

    // Show iterative dilation
    for (int i = 0; i < 10; ++i)
    {
        Mat tmp;
        dilate(dilated, tmp, se);
        tmp.copyTo(dilated, dilated);

        hconcat(dist, dilated, grid);
        putText(grid, "Distance", Point(10, 30), FONT_HERSHEY_SIMPLEX, 0.8, Scalar(255), 2);
        putText(grid, "Dilated Map", Point(dist.cols + 10, 30), FONT_HERSHEY_SIMPLEX, 0.8, Scalar(255), 2);

        imshow("Task 1 - Distance Transform (SPACE=next, Q=quit)", grid);

        int key = waitKey(0);
        if (key == 'q' || key == 'Q')
            exit(0);
        else if (key == ' ')
            continue;
    }
}

// ---------- Task 2: Skeletonization using Golay Masks ----------
void golaySkeletonTask()
{
    static const int Golay[72] = {
        0, 0, 0, -1, 1, -1, 1, 1, 1,
        -1, 0, 0, 1, 1, 0, -1, 1, -1,
        1, -1, 0, 1, 1, 0, 1, -1, 0,
        -1, 1, -1, 1, 1, 0, -1, 0, 0,
        1, 1, 1, -1, 1, -1, 0, 0, 0,
        -1, 1, -1, 0, 1, 1, 0, 0, -1,
        0, -1, 1, 0, 1, 1, 0, -1, 1,
        0, 0, -1, 0, 1, 1, -1, 1, -1};

    Mat imO = loadImage("pityoka.png", IMREAD_GRAYSCALE);
    if (imO.empty())
        return;

    threshold(imO, imO, 128, 255, THRESH_BINARY);
    Mat imP = imO.clone();

    imshow("Task 2 - Golay Skeleton (SPACE=next, Q=quit)", imO);
    while (true)
    {
        int key = waitKey(0);
        if (key == 'q' || key == 'Q')
            exit(0);
        else if (key == ' ')
            break;
    }

    int count;
    do
    {
        count = 0;
        for (int l = 0; l < 8; l++)
        {
            for (int y = 1; y < imO.rows - 1; y++)
            {
                for (int x = 1; x < imO.cols - 1; x++)
                {
                    if (getGray(imO, x, y) > 0)
                    {
                        bool erase = true;
                        int index = 9 * l;
                        for (int j = y - 1; j <= y + 1; j++)
                        {
                            for (int i = x - 1; i <= x + 1; i++)
                            {
                                int maskVal = Golay[index++];
                                if ((maskVal == 1 && getGray(imO, i, j) == 0) ||
                                    (maskVal == 0 && getGray(imO, i, j) > 0))
                                {
                                    erase = false;
                                }
                            }
                        }
                        if (erase)
                        {
                            setBlack(imP, x, y);
                            count++;
                        }
                    }
                }
            }
            imO = imP.clone();
        }

        Mat grid;
        hconcat(imP, imO, grid);
        putText(grid, "Current", Point(10, 30), FONT_HERSHEY_SIMPLEX, 0.8, Scalar(255), 2);
        putText(grid, "Updated", Point(imP.cols + 10, 30), FONT_HERSHEY_SIMPLEX, 0.8, Scalar(255), 2);

        imshow("Task 2 - Golay Skeleton (SPACE=next, Q=quit)", grid);

        int key = waitKey(100);
        if (key == 'q' || key == 'Q')
            exit(0);
    } while (count > 0);

    waitKey(0);
}

// ---------- Main ----------
int main()
{
    distanceTransformTask();
    golaySkeletonTask();
    return 0;
}
