#include <opencv2/opencv.hpp>
using namespace cv;
using namespace std;

int main()
{
    Mat img = imread("plafon.jpg", IMREAD_GRAYSCALE);
    if (img.empty())
    {
        cout << "Image not found!" << endl;
        return -1;
    }

    float data1[9] = {-1, 0, 1, -1, 0, 1, -1, 0, 1};
    float data2[9] = {1, 0, -1, 1, 0, -1, 1, 0, -1};
    float data3[9] = {-1, -1, -1, 0, 0, 0, 1, 1, 1};
    float data4[9] = {1, 1, 1, 0, 0, 0, -1, -1, -1};

    Mat k1(3, 3, CV_32F, data1), k2(3, 3, CV_32F, data2), k3(3, 3, CV_32F, data3), k4(3, 3, CV_32F, data4);
    Mat g1, g2, g3, g4;
    filter2D(img, g1, -1, k1);
    filter2D(img, g2, -1, k2);
    filter2D(img, g3, -1, k3);
    filter2D(img, g4, -1, k4);

    Mat vert = g1 + g2;
    Mat hori = g3 + g4;
    Mat all = vert + hori;
    Mat thin;
    threshold(all, thin, 110, 255, THRESH_BINARY);
    Mat canny;
    Canny(img, canny, 100, 200);

    vector<Mat> imgs = {img, g1, g2, g3, g4, vert, hori, all, thin, canny};
    for (auto &m : imgs)
        if (m.type() != CV_8U)
            normalize(m, m, 0, 255, NORM_MINMAX, CV_8U);

    int w = img.cols, h = img.rows;
    Mat grid(h * 3, w * 3, CV_8U, Scalar(0));

    imgs[0].copyTo(grid(Rect(0, 0, w, h)));
    imgs[1].copyTo(grid(Rect(w, 0, w, h)));
    imgs[2].copyTo(grid(Rect(2 * w, 0, w, h)));
    imgs[3].copyTo(grid(Rect(0, h, w, h)));
    imgs[4].copyTo(grid(Rect(w, h, w, h)));
    imgs[5].copyTo(grid(Rect(2 * w, h, w, h)));
    imgs[6].copyTo(grid(Rect(0, 2 * h, w, h)));
    imgs[7].copyTo(grid(Rect(w, 2 * h, w, h)));
    imgs[8].copyTo(grid(Rect(2 * w, 2 * h, w, h)));

    putText(grid, "1", Point(10, 30), FONT_HERSHEY_SIMPLEX, 1, 255, 2);
    putText(grid, "2", Point(w + 10, 30), FONT_HERSHEY_SIMPLEX, 1, 255, 2);
    putText(grid, "2", Point(2 * w + 10, 30), FONT_HERSHEY_SIMPLEX, 1, 255, 2);
    putText(grid, "2", Point(10, h + 30), FONT_HERSHEY_SIMPLEX, 1, 255, 2);
    putText(grid, "2", Point(w + 10, h + 30), FONT_HERSHEY_SIMPLEX, 1, 255, 2);
    putText(grid, "2", Point(2 * w + 10, h + 30), FONT_HERSHEY_SIMPLEX, 1, 255, 2);
    putText(grid, "2", Point(10, 2 * h + 30), FONT_HERSHEY_SIMPLEX, 1, 255, 2);
    putText(grid, "2", Point(w + 10, 2 * h + 30), FONT_HERSHEY_SIMPLEX, 1, 255, 2);
    putText(grid, "2", Point(2 * w + 10, 2 * h + 30), FONT_HERSHEY_SIMPLEX, 1, 255, 2);

    imshow("All Results Grid", grid);
    imshow("3", imgs[9]);

    int key;
    while (true)
    {
        key = waitKey(0);
        if (key == 27 || key == 'q' || key == 'Q')
            break;
        if (key == 91 || key == 92)
            continue;
    }
    return 0;
}