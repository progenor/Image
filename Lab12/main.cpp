#include <opencv2/opencv.hpp>
#include <vector>
#include <iostream>
using namespace cv;
using namespace std;

// ---------- Helper Functions ----------

uchar getGray(const Mat &im, int x, int y)
{
    return im.at<uchar>(y, x);
}

void setGray(Mat &im, int x, int y, uchar v)
{
    im.at<uchar>(y, x) = v;
}

int regionGrowing(Mat im, Point p0, Point &pbf, Point &pja)
{
    int count = 0;
    Point *fifo = new Point[0x100000];
    int nextIn = 0;
    int nextOut = 0;
    pbf = p0;
    pja = p0;

    if (getGray(im, p0.x, p0.y) < 128)
    {
        delete[] fifo;
        return 0;
    }

    fifo[nextIn++] = p0;
    setGray(im, p0.x, p0.y, 100);

    while (nextIn > nextOut)
    {
        Point p = fifo[nextOut++];
        ++count;

        if (p.x > 0 && getGray(im, p.x - 1, p.y) > 128)
        {
            fifo[nextIn++] = Point(p.x - 1, p.y);
            setGray(im, p.x - 1, p.y, 100);
            if (pbf.x > p.x - 1)
                pbf.x = p.x - 1;
        }

        if (p.x < im.cols - 1 && getGray(im, p.x + 1, p.y) > 128)
        {
            fifo[nextIn++] = Point(p.x + 1, p.y);
            setGray(im, p.x + 1, p.y, 100);
            if (pja.x < p.x + 1)
                pja.x = p.x + 1;
        }

        if (p.y > 0 && getGray(im, p.x, p.y - 1) > 128)
        {
            fifo[nextIn++] = Point(p.x, p.y - 1);
            setGray(im, p.x, p.y - 1, 100);
            if (pbf.y > p.y - 1)
                pbf.y = p.y - 1;
        }

        if (p.y < im.rows - 1 && getGray(im, p.x, p.y + 1) > 128)
        {
            fifo[nextIn++] = Point(p.x, p.y + 1);
            setGray(im, p.x, p.y + 1, 100);
            if (pja.y < p.y + 1)
                pja.y = p.y + 1;
        }
    }

    delete[] fifo;
    return count;
}

// ---------- Task 1: Play video in color and grayscale ----------

void task1_PlayVideo(const string &videoFile)
{
    cout << "\n=== Task 1: Play video in color and grayscale ===" << endl;

    VideoCapture cap(videoFile);
    if (!cap.isOpened())
    {
        cerr << "Error opening video file: " << videoFile << endl;
        return;
    }

    while (true)
    {
        Mat frame;
        cap >> frame;

        if (frame.empty())
            break;

        Mat grayFrame;
        cvtColor(frame, grayFrame, COLOR_BGR2GRAY);

        imshow("Task 1: Color Video", frame);
        imshow("Task 1: Grayscale Video", grayFrame);

        if (waitKey(30) == 'q')
            break;
    }

    cap.release();
    destroyWindow("Task 1: Color Video");
    destroyWindow("Task 1: Grayscale Video");
    cout << "Task 1 completed." << endl;
}

// ---------- Task 2: Edge detection ----------

void task2_EdgeDetection(const string &videoFile)
{
    cout << "\n=== Task 2: Edge detection ===" << endl;

    VideoCapture cap(videoFile);
    if (!cap.isOpened())
    {
        cerr << "Error opening video file: " << videoFile << endl;
        return;
    }

    while (true)
    {
        Mat frame;
        cap >> frame;

        if (frame.empty())
            break;

        Mat gray, edges;
        cvtColor(frame, gray, COLOR_BGR2GRAY);
        Canny(gray, edges, 50, 150);

        imshow("Task 2: Original", frame);
        imshow("Task 2: Edge Detection", edges);

        if (waitKey(30) == 'q')
            break;
    }

    cap.release();
    destroyWindow("Task 2: Original");
    destroyWindow("Task 2: Edge Detection");
    cout << "Task 2 completed." << endl;
}

// ---------- Task 3: Median/Gaussian Blur ----------

void task3_BlurFilters(const string &videoFile)
{
    cout << "\n=== Task 3: Median and Gaussian Blur ===" << endl;

    VideoCapture cap(videoFile);
    if (!cap.isOpened())
    {
        cerr << "Error opening video file: " << videoFile << endl;
        return;
    }

    while (true)
    {
        Mat frame;
        cap >> frame;

        if (frame.empty())
            break;

        Mat medianBlurred, gaussianBlurred;
        medianBlur(frame, medianBlurred, 5);
        GaussianBlur(frame, gaussianBlurred, Size(5, 5), 0);

        imshow("Task 3: Original", frame);
        imshow("Task 3: Median Blur", medianBlurred);
        imshow("Task 3: Gaussian Blur", gaussianBlurred);

        if (waitKey(30) == 'q')
            break;
    }

    cap.release();
    destroyWindow("Task 3: Original");
    destroyWindow("Task 3: Median Blur");
    destroyWindow("Task 3: Gaussian Blur");
    cout << "Task 3 completed." << endl;
}

// ---------- Task 4: Low-pass filter ----------

void task4_LowPassFilter(const string &videoFile)
{
    cout << "\n=== Task 4: Low-pass filter ===" << endl;

    VideoCapture cap(videoFile);
    if (!cap.isOpened())
    {
        cerr << "Error opening video file: " << videoFile << endl;
        return;
    }

    while (true)
    {
        Mat frame;
        cap >> frame;

        if (frame.empty())
            break;

        Mat lowPass;
        blur(frame, lowPass, Size(7, 7));

        imshow("Task 4: Original", frame);
        imshow("Task 4: Low-pass Filter", lowPass);

        if (waitKey(30) == 'q')
            break;
    }

    cap.release();
    destroyWindow("Task 4: Original");
    destroyWindow("Task 4: Low-pass Filter");
    cout << "Task 4 completed." << endl;
}

// ---------- Task 5: High-pass filter (Laplace) ----------

void task5_HighPassFilter(const string &videoFile)
{
    cout << "\n=== Task 5: High-pass filter (Laplace) ===" << endl;

    VideoCapture cap(videoFile);
    if (!cap.isOpened())
    {
        cerr << "Error opening video file: " << videoFile << endl;
        return;
    }

    while (true)
    {
        Mat frame;
        cap >> frame;

        if (frame.empty())
            break;

        Mat gray, laplace, highPass;
        cvtColor(frame, gray, COLOR_BGR2GRAY);
        Laplacian(gray, laplace, CV_16S, 3);
        convertScaleAbs(laplace, highPass);

        imshow("Task 5: Original", frame);
        imshow("Task 5: Laplace (High-pass)", highPass);

        if (waitKey(30) == 'q')
            break;
    }

    cap.release();
    destroyWindow("Task 5: Original");
    destroyWindow("Task 5: Laplace (High-pass)");
    cout << "Task 5 completed." << endl;
}

// ---------- Task 6: Histogram equalization ----------

void task6_HistogramEqualization(const string &videoFile)
{
    cout << "\n=== Task 6: Histogram equalization ===" << endl;

    VideoCapture cap(videoFile);
    if (!cap.isOpened())
    {
        cerr << "Error opening video file: " << videoFile << endl;
        return;
    }

    while (true)
    {
        Mat frame;
        cap >> frame;

        if (frame.empty())
            break;

        Mat ycrcb, equalized;
        cvtColor(frame, ycrcb, COLOR_BGR2YCrCb);

        vector<Mat> channels;
        split(ycrcb, channels);
        equalizeHist(channels[0], channels[0]);
        merge(channels, ycrcb);

        cvtColor(ycrcb, equalized, COLOR_YCrCb2BGR);

        imshow("Task 6: Original", frame);
        imshow("Task 6: Histogram Equalized", equalized);

        if (waitKey(30) == 'q')
            break;
    }

    cap.release();
    destroyWindow("Task 6: Original");
    destroyWindow("Task 6: Histogram Equalized");
    cout << "Task 6 completed." << endl;
}

// ---------- Task 7: Resize (shrink/enlarge) ----------

void task7_Resize(const string &videoFile)
{
    cout << "\n=== Task 7: Resize video ===" << endl;

    VideoCapture cap(videoFile);
    if (!cap.isOpened())
    {
        cerr << "Error opening video file: " << videoFile << endl;
        return;
    }

    while (true)
    {
        Mat frame;
        cap >> frame;

        if (frame.empty())
            break;

        Mat shrunk, enlarged;
        resize(frame, shrunk, Size(), 0.5, 0.5, INTER_LINEAR);
        resize(frame, enlarged, Size(), 1.5, 1.5, INTER_LINEAR);

        imshow("Task 7: Original", frame);
        imshow("Task 7: Shrunk (0.5x)", shrunk);
        imshow("Task 7: Enlarged (1.5x)", enlarged);

        if (waitKey(30) == 'q')
            break;
    }

    cap.release();
    destroyWindow("Task 7: Original");
    destroyWindow("Task 7: Shrunk (0.5x)");
    destroyWindow("Task 7: Enlarged (1.5x)");
    cout << "Task 7 completed." << endl;
}

// ---------- Task 8: Motion detection ----------

void task8_MotionDetection(const string &videoFile)
{
    cout << "\n=== Task 8: Motion detection ===" << endl;

    VideoCapture cap(videoFile);
    if (!cap.isOpened())
    {
        cerr << "Error opening video file: " << videoFile << endl;
        return;
    }

    // a) Read first 10 frames and select one as background
    Mat background;
    for (int i = 0; i < 10; i++)
    {
        Mat frame;
        cap >> frame;
        if (frame.empty())
        {
            cerr << "Video has fewer than 10 frames" << endl;
            return;
        }

        if (i == 5) // Use frame 5 as background
        {
            resize(frame, background, Size(), 0.25, 0.25, INTER_LINEAR);
        }
    }

    cout << "Background frame selected (frame 5)" << endl;

    // b-g) Process remaining frames
    while (true)
    {
        Mat frame;
        cap >> frame;

        if (frame.empty())
            break;

        // Resize to quarter size
        Mat frameSmall;
        resize(frame, frameSmall, Size(), 0.25, 0.25, INTER_LINEAR);

        // c) Calculate absolute difference
        Mat diff1, diff2, diff;
        subtract(frameSmall, background, diff1);
        subtract(background, frameSmall, diff2);
        add(diff1, diff2, diff);

        // d) Split into channels and sum, then threshold
        vector<Mat> channels;
        split(diff, channels);
        Mat sumChannels = channels[0] + channels[1] + channels[2];

        Mat binary;
        compare(sumChannels, 170, binary, CMP_GE);

        // e) Erosion to remove small white spots
        Mat eroded;
        erode(binary, eroded, getStructuringElement(MORPH_RECT, Size(3, 3)));

        // f) Region growing to find largest white blob
        Mat imR = eroded.clone();
        Point pbf, pja;
        Rect roi;
        int roiSize = 0;
        int nrRect = 0;

        for (int y = 0; y < imR.rows; y++)
        {
            for (int x = 0; x < imR.cols; x++)
            {
                if (getGray(imR, x, y) > 128)
                {
                    int res = regionGrowing(imR, Point(x, y), pbf, pja);
                    if (res > 500)
                    {
                        if (nrRect == 0 || res > roiSize)
                        {
                            roi.x = pbf.x;
                            roi.y = pbf.y;
                            roi.width = pja.x - pbf.x + 1;
                            roi.height = pja.y - pbf.y + 1;
                            roiSize = res;
                        }
                        ++nrRect;
                    }
                }
            }
        }

        // g) Draw rectangle on original frame
        Mat result = frameSmall.clone();
        if (nrRect > 0)
        {
            rectangle(result, Point(roi.x, roi.y),
                      Point(roi.x + roi.width, roi.y + roi.height),
                      Scalar(0, 255, 255), 2);
        }

        imshow("Task 8: Original Small", frameSmall);
        imshow("Task 8: Difference", sumChannels);
        imshow("Task 8: Binary", binary);
        imshow("Task 8: After Erosion", eroded);
        imshow("Task 8: Motion Detection Result", result);

        if (waitKey(30) == 'q')
            break;
    }

    cap.release();
    destroyWindow("Task 8: Original Small");
    destroyWindow("Task 8: Difference");
    destroyWindow("Task 8: Binary");
    destroyWindow("Task 8: After Erosion");
    destroyWindow("Task 8: Motion Detection Result");
    cout << "Task 8 completed." << endl;
}

// ---------- Main ----------

int main()
{
    cout << "====================================================" << endl;
    cout << "Video Processing Tasks - Complete Solution" << endl;
    cout << "====================================================" << endl;
    cout << "Controls: Press 'q' to exit current task" << endl;
    cout << endl;

    // Change these filenames to match your video files
    string videoFile1 = "felho.avi";
    string videoFile2 = "IMG_6909.MOV";

    cout << "Which task would you like to run?" << endl;
    cout << "1. Play video (color and grayscale)" << endl;
    cout << "2. Edge detection" << endl;
    cout << "3. Blur filters (Median/Gaussian)" << endl;
    cout << "4. Low-pass filter" << endl;
    cout << "5. High-pass filter (Laplace)" << endl;
    cout << "6. Histogram equalization" << endl;
    cout << "7. Resize video" << endl;
    cout << "8. Motion detection" << endl;
    cout << "9. Run all tasks sequentially" << endl;
    cout << "0. Exit" << endl;
    cout << "\nEnter choice: ";

    int choice;
    cin >> choice;

    switch (choice)
    {
    case 1:
        task1_PlayVideo(videoFile1);
        break;
    case 2:
        task2_EdgeDetection(videoFile1);
        break;
    case 3:
        task3_BlurFilters(videoFile1);
        break;
    case 4:
        task4_LowPassFilter(videoFile1);
        break;
    case 5:
        task5_HighPassFilter(videoFile1);
        break;
    case 6:
        task6_HistogramEqualization(videoFile1);
        break;
    case 7:
        task7_Resize(videoFile1);
        break;
    case 8:
        task8_MotionDetection(videoFile2);
        break;
    case 9:
        cout << "\nRunning all tasks..." << endl;
        task1_PlayVideo(videoFile1);
        task2_EdgeDetection(videoFile1);
        task3_BlurFilters(videoFile1);
        task4_LowPassFilter(videoFile1);
        task5_HighPassFilter(videoFile1);
        task6_HistogramEqualization(videoFile1);
        task7_Resize(videoFile1);
        task8_MotionDetection(videoFile2);
        break;
    case 0:
        cout << "Exiting..." << endl;
        return 0;
    default:
        cout << "Invalid choice!" << endl;
        return 1;
    }

    cout << "\n====================================================" << endl;
    cout << "All selected tasks completed successfully!" << endl;
    cout << "====================================================" << endl;

    waitKey(0);
    destroyAllWindows();

    return 0;
}