#include "opencv2/opencv.hpp"

using namespace std;
using namespace cv;
void intro()
{
    Mat im = imread("../kepek/esik.jpg", 1);
    imshow("Ez itt egy alma", im);
    waitKey(0);
}

void lab01()
{
    Mat im1 = imread("../kepek/3.JPG", 1);
    Mat im2 = imread("../kepek/5.JPG", 1);
    imshow("Film", im1);
    waitKey(0);
    Mat im3 = im2.clone();
    for (float q = 0; q < 1.01; q += 0.02f)
    {
        addWeighted(im1, 1.0f - q, im2, q, 0, im3);
        imshow("Film", im3);
        waitKey(100);
        if (q < 0.67 && q > 0.65)
            imwrite("keverek.bmp", im3);
    }
    waitKey(0);
}

int main()
{
    intro();
    lab01();
}