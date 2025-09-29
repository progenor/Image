#include "opencv2/opencv.hpp"
#include <iostream>
#include <string>

using namespace std;
using namespace cv;

const string WINDOW_NAME = "Filter Results";

void exercise1_ShiftFilter(const Mat &originalImage);
void exercise2_LowPassFilter(const Mat &originalImage);
void exercise3_HighPassFilter(const Mat &originalImage);
void exercise4_GaussAndBlurFilters(const Mat &originalImage);
void exercise5_MedianFilterNoise(const Mat &originalImage);
void exercise6_MedianFilterBinary();

int main()
{
    Mat image = imread("./plafon.jpg", IMREAD_COLOR);
    if (image.empty())
    {
        cout << "Could not open or find the image" << endl;
        return -1;
    }

    namedWindow(WINDOW_NAME, WINDOW_NORMAL);

    int choice;
    cout << "Choose an exercise to run (1-6): ";
    cin >> choice;

    switch (choice)
    {
    case 1:
        exercise1_ShiftFilter(image);
        break;
    case 2:
        exercise2_LowPassFilter(image);
        break;
    case 3:
        exercise3_HighPassFilter(image);
        break;
    case 4:
        exercise4_GaussAndBlurFilters(image);
        break;
    case 5:
        exercise5_MedianFilterNoise(image);
        break;
    case 6:
        exercise6_MedianFilterBinary();
        break;
    default:
        cout << "Invalid choice!" << endl;
        break;
    }

    destroyWindow(WINDOW_NAME);
    return 0;
}

void exercise1_ShiftFilter(const Mat &originalImage)
{
    Mat image = originalImage.clone();
    Mat result;

    // Show original image first
    putText(image, "Original Image", Point(20, 30), FONT_HERSHEY_SIMPLEX, 1, Scalar(0, 0, 255), 2);
    imshow(WINDOW_NAME, image);
    waitKey(1000);

    float values[9] = {0, 0, 0,
                       1, 0, 0,
                       0, 0, 0};

    Mat shift_kernel = Mat(3, 3, CV_32FC1, values);

    for (int i = 0; i < 10; i++)
    {
        filter2D(image, result, -1, shift_kernel);

        // Add iteration number to image
        putText(result, "Shift Filter - Step " + to_string(i + 1), Point(20, 30), FONT_HERSHEY_SIMPLEX, 1, Scalar(0, 0, 255), 2);
        imshow(WINDOW_NAME, result);
        waitKey(500);

        result.copyTo(image);
    }
}

void exercise2_LowPassFilter(const Mat &originalImage)
{
    Mat image = originalImage.clone();
    Mat result;

    // Show original image first
    putText(image, "Original Image", Point(20, 30), FONT_HERSHEY_SIMPLEX, 1, Scalar(0, 0, 255), 2);
    imshow(WINDOW_NAME, image);
    waitKey(1000);

    float values[9] = {0.1, 0.1, 0.1,
                       0.1, 0.2, 0.1,
                       0.1, 0.1, 0.1};

    Mat lpf_kernel = Mat(3, 3, CV_32FC1, values);

    for (int i = 0; i < 10; i++)
    {
        filter2D(image, result, -1, lpf_kernel);

        putText(result, "Low-pass Filter - Step " + to_string(i + 1), Point(20, 30), FONT_HERSHEY_SIMPLEX, 1, Scalar(0, 0, 255), 2);
        imshow(WINDOW_NAME, result);
        waitKey(500);

        result.copyTo(image);
    }
}

void exercise3_HighPassFilter(const Mat &originalImage)
{
    Mat result;
    Mat image = originalImage.clone();

    // Show original image first
    putText(image, "Original Image", Point(20, 30), FONT_HERSHEY_SIMPLEX, 1, Scalar(0, 0, 255), 2);
    imshow(WINDOW_NAME, image);
    waitKey(1000);

    for (float k = 0.2; k <= 2.0; k += 0.2)
    {
        float centerVal = 1 + k;
        float sideVal = -k / 4;

        float values[9] = {0, sideVal, 0,
                           sideVal, centerVal, sideVal,
                           0, sideVal, 0};

        Mat hpf_kernel = Mat(3, 3, CV_32FC1, values);

        filter2D(originalImage, result, -1, hpf_kernel);

        putText(result, "High-pass Filter - k=" + to_string(k), Point(20, 30), FONT_HERSHEY_SIMPLEX, 1, Scalar(0, 0, 255), 2);
        imshow(WINDOW_NAME, result);
        waitKey(500);
    }
}

void exercise4_GaussAndBlurFilters(const Mat &originalImage)
{
    Mat resultBlur, resultGauss;
    Mat image = originalImage.clone();

    // Show original image first
    putText(image, "Original Image", Point(20, 30), FONT_HERSHEY_SIMPLEX, 1, Scalar(0, 0, 255), 2);
    imshow(WINDOW_NAME, image);
    waitKey(1000);

    for (int k = 3; k <= 21; k += 2)
    {
        // Blur filter
        blur(originalImage, resultBlur, Size(k, k));
        putText(resultBlur, "Blur Filter - k=" + to_string(k), Point(20, 30), FONT_HERSHEY_SIMPLEX, 1, Scalar(0, 0, 255), 2);
        imshow(WINDOW_NAME, resultBlur);
        waitKey(500);

        // Gaussian filter
        GaussianBlur(originalImage, resultGauss, Size(k, k), 1);
        putText(resultGauss, "Gaussian Filter - k=" + to_string(k), Point(20, 30), FONT_HERSHEY_SIMPLEX, 1, Scalar(0, 0, 255), 2);
        imshow(WINDOW_NAME, resultGauss);
        waitKey(500);
    }
}

void exercise5_MedianFilterNoise(const Mat &originalImage)
{
    Mat image = originalImage.clone();
    Mat result;

    // Add noise
    for (int db = 0; db < 20; ++db)
    {
        line(image,
             Point(rand() % image.cols, rand() % image.rows),
             Point(rand() % image.cols, rand() % image.rows),
             Scalar(0, 0, 0, 0),
             1 + db % 2);
    }

    // Show noisy image
    putText(image, "Image with Noise", Point(20, 30), FONT_HERSHEY_SIMPLEX, 1, Scalar(0, 0, 255), 2);
    imshow(WINDOW_NAME, image);
    waitKey(1000);

    for (int size = 3; size <= 21; size += 2)
    {
        medianBlur(image, result, size);

        putText(result, "Median Filter - size=" + to_string(size), Point(20, 30), FONT_HERSHEY_SIMPLEX, 1, Scalar(0, 0, 255), 2);
        imshow(WINDOW_NAME, result);
        waitKey(500);
    }
}

void exercise6_MedianFilterBinary()
{
    Mat binaryImage = Mat::zeros(400, 400, CV_8UC1);

    // Create an "amoeba" shape
    vector<Point> points;
    for (int i = 0; i < 10; i++)
    {
        points.push_back(Point(200 + rand() % 100 - 50, 200 + rand() % 100 - 50));
    }
    fillConvexPoly(binaryImage, points.data(), points.size(), Scalar(255));

    for (int i = 0; i < 5; i++)
    {
        int x = 200 + rand() % 140 - 70;
        int y = 200 + rand() % 140 - 70;
        circle(binaryImage, Point(x, y), 20 + rand() % 20, Scalar(255), -1);
    }

    // Convert to 3-channel for text display
    Mat displayImage;
    cvtColor(binaryImage, displayImage, COLOR_GRAY2BGR);
    putText(displayImage, "Original Binary Image", Point(20, 30), FONT_HERSHEY_SIMPLEX, 1, Scalar(0, 0, 255), 2);
    imshow(WINDOW_NAME, displayImage);
    waitKey(1000);

    Mat result = binaryImage.clone();
    Mat resultDisplay;

    for (int i = 0; i < 200; i++)
    {
        medianBlur(result, result, 21);

        if (i % 10 == 0)
        {
            cvtColor(result, resultDisplay, COLOR_GRAY2BGR);
            putText(resultDisplay, "Median Filtered Binary - Step " + to_string(i),
                    Point(20, 30), FONT_HERSHEY_SIMPLEX, 1, Scalar(0, 0, 255), 2);
            imshow(WINDOW_NAME, resultDisplay);
            waitKey(100);
        }
    }
}