#include "opencv2/opencv.hpp"

using namespace std;
using namespace cv;

void showMyImage(Mat &imBig, Mat &im, int &index)
{
    im.copyTo(imBig(Rect((index % 6) * (im.cols), (index / 6) * (im.rows),
                         im.cols, im.rows)));
    imshow("Ablak", imBig);
    index = (index + 1) % 18;
    waitKey();
}

int main()
{
    Mat im = imread("eper.jpg");
    if (im.empty())
    {
        cout << "Could not open or find the image" << endl;
        return -1;
    }

    Mat imBig = Mat(im.rows * 3, im.cols * 6, im.type());
    imBig.setTo(Scalar(128, 128, 255, 0));

    int index = 0;

    Mat result = im.clone();
    showMyImage(imBig, result, index);

    Mat bgr[3];
    split(im, bgr);
    Mat &b = bgr[0], &g = bgr[1], &r = bgr[2];

    Mat z(im.rows, im.cols, CV_8UC1, Scalar(0));

    // 1.
    merge(vector<Mat>{z, g, r}, result);
    showMyImage(imBig, result, index);

    merge(vector<Mat>{b, z, r}, result);
    showMyImage(imBig, result, index);

    merge(vector<Mat>{b, g, z}, result);
    showMyImage(imBig, result, index);

    // 2.
    merge(vector<Mat>{b, z, z}, result);
    showMyImage(imBig, result, index);

    merge(vector<Mat>{z, g, z}, result);
    showMyImage(imBig, result, index);

    merge(vector<Mat>{z, z, r}, result);
    showMyImage(imBig, result, index);

    // 3.
    merge(vector<Mat>{b, g, r}, result);
    showMyImage(imBig, result, index);

    merge(vector<Mat>{b, r, g}, result);
    showMyImage(imBig, result, index);

    merge(vector<Mat>{g, b, r}, result);
    showMyImage(imBig, result, index);

    merge(vector<Mat>{g, r, b}, result);
    showMyImage(imBig, result, index);

    merge(vector<Mat>{r, b, g}, result);
    showMyImage(imBig, result, index);

    merge(vector<Mat>{r, g, b}, result);
    showMyImage(imBig, result, index);

    // 4. Replace one color component with its negative (3 images)
    Mat nb = ~b;
    merge(vector<Mat>{nb, g, r}, result);
    showMyImage(imBig, result, index);

    Mat ng = ~g;
    merge(vector<Mat>{b, ng, r}, result);
    showMyImage(imBig, result, index);

    Mat nr = ~r;
    merge(vector<Mat>{b, g, nr}, result);
    showMyImage(imBig, result, index);

    // 5.
    Mat ycrcb;
    cvtColor(im, ycrcb, COLOR_BGR2YCrCb);

    Mat ycrCbChannels[3];
    split(ycrcb, ycrCbChannels);
    Mat &y = ycrCbChannels[0], &cr = ycrCbChannels[1], &cb = ycrCbChannels[2];

    Mat ny = ~y;
    merge(vector<Mat>{ny, cr, cb}, ycrcb);
    cvtColor(ycrcb, result, COLOR_YCrCb2BGR);
    showMyImage(imBig, result, index);

    // 6.
    result = ~im;
    showMyImage(imBig, result, index);

    Mat a = imread("plafon.jpg");

    Mat gray;
    cvtColor(a, gray, COLOR_BGR2GRAY);
    Mat monochrome;
    cvtColor(gray, monochrome, COLOR_GRAY2BGR);

    imshow("Ablak", gray);
    waitKey();
    imshow("Asblak", monochrome);
    waitKey();

    return 0;
}