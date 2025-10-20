#include <opencv2/opencv.hpp>
using namespace cv;
using namespace std;

// Helper: try multiple likely paths/extensions to load an image from the workspace
static Mat loadImage(const string &filename)
{
    Mat im = imread(filename);
    if (!im.empty())
        return im;
    size_t pos = filename.find_last_of('.');
    if (pos != string::npos)
    {
        string base = filename.substr(0, pos);
        string ext = filename.substr(pos);
        if (ext == ".jpg")
            im = imread(base + ".JPG");
        else if (ext == ".JPG")
            im = imread(base + ".jpg");
        if (!im.empty())
            return im;
    }
    vector<string> prefixes = {"kepek/", "../kepek/", "/home/progenor/Documents/code/Sch/Image/kepek/"};
    for (const auto &p : prefixes)
    {
        im = imread(p + filename);
        if (!im.empty())
            return im;
        if (pos != string::npos)
        {
            string base = filename.substr(0, pos);
            string ext = filename.substr(pos);
            if (ext == ".jpg")
                im = imread(p + base + ".JPG");
            else if (ext == ".JPG")
                im = imread(p + base + ".jpg");
            if (!im.empty())
                return im;
        }
    }
    return Mat();
}

// Equalize image in Y channel (works for color and grayscale)
static void equalizeHistogram(Mat &im)
{
    if (im.empty())
        return;
    if (im.channels() == 1)
    {
        equalizeHist(im, im);
        return;
    }
    Mat ycrcb;
    cvtColor(im, ycrcb, COLOR_BGR2YCrCb);
    vector<Mat> channels;
    split(ycrcb, channels);
    equalizeHist(channels[0], channels[0]);
    merge(channels, ycrcb);
    cvtColor(ycrcb, im, COLOR_YCrCb2BGR);
}

static Mat drawColorHistImage(const Mat &image)
{
    vector<Mat> bgr_planes;
    split(image, bgr_planes);

    int histSize = 256;
    float range[] = {0, 256};
    const float *histRange = {range};
    bool uniform = true, accumulate = false;

    vector<Mat> hists(3);
    for (int i = 0; i < 3; ++i)
    {
        calcHist(&bgr_planes[i], 1, nullptr, Mat(), hists[i], 1, &histSize, &histRange, uniform, accumulate);
        normalize(hists[i], hists[i], 0, 100, NORM_MINMAX);
    }

    Mat histImage(300, 256, CV_8UC3, Scalar(255, 255, 255));
    vector<Scalar> colors = {Scalar(255, 0, 0), Scalar(0, 255, 0), Scalar(0, 0, 255)};
    for (int ch = 0; ch < 3; ++ch)
    {
        rectangle(histImage, Rect(0, ch * 100, 256, 100), colors[ch], -1);
        for (int i = 0; i < 256; ++i)
        {
            line(histImage,
                 Point(i, (ch + 1) * 100),
                 Point(i, (ch + 1) * 100 - cvRound(hists[ch].at<float>(i))),
                 Scalar(0, 0, 0), 1);
        }
    }
    return histImage;
}

int main()
{
    vector<string> imageFiles = {"cheguevara.jpg", "japan.jpg", "muzeum.jpg", "oroszlan.jpg"};

    for (const auto &imageFile : imageFiles)
    {
        Mat im = loadImage(imageFile);
        if (im.empty())
        {
            cerr << "Hiba: Nem találom a " << imageFile << " fájlt!" << endl;
            continue;
        }

        // Original image histogram (colored)
        Mat origHistDisplay = drawColorHistImage(im);

        // Histogram equalization
        Mat imEqualized = im.clone();
        equalizeHistogram(imEqualized);

        // Equalized histogram (colored)
        Mat eqHistDisplay = drawColorHistImage(imEqualized);

        // Create image display
        Mat imageDisplay(im.rows, im.cols * 2, im.type());
        im.copyTo(imageDisplay(Rect(0, 0, im.cols, im.rows)));
        imEqualized.copyTo(imageDisplay(Rect(im.cols, 0, im.cols, im.rows)));

        // Create histogram display
        Mat histDisplay(300, 256 * 2, CV_8UC3, Scalar(255, 255, 255));
        origHistDisplay.copyTo(histDisplay(Rect(0, 0, 256, 300)));
        eqHistDisplay.copyTo(histDisplay(Rect(256, 0, 256, 300)));

        // Add labels to images
        putText(imageDisplay, "Eredeti kep", Point(10, 30), FONT_HERSHEY_SIMPLEX, 1, Scalar(255, 255, 255), 2);
        putText(imageDisplay, "Kiegyenlitett kep", Point(im.cols + 10, 30), FONT_HERSHEY_SIMPLEX, 1, Scalar(255, 255, 255), 2);

        putText(histDisplay, "Eredeti hiszt.", Point(10, 30), FONT_HERSHEY_SIMPLEX, 0.8, Scalar(0, 0, 0), 2);
        putText(histDisplay, "Kiegy. hiszt.", Point(256 + 10, 30), FONT_HERSHEY_SIMPLEX, 0.8, Scalar(0, 0, 0), 2);

        imshow("Kepek", imageDisplay);
        imshow("Hisztogramok", histDisplay);

        // Wait for 'q' or 'Q' to go to next image, ESC to quit all
        while (true)
        {
            int key = waitKey(0);
            if (key == 'q' || key == 'Q')
                break;
            if (key == 27) // ESC
                return 0;
        }
    }
    return 0;
}