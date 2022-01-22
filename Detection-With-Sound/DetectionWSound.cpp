#include <iostream>
#include <fstream>
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/dnn/all_layers.hpp>
#include <espeak-ng/speak_lib.h>
//////////// Visp Libraries ////////////
#include <visp3/sensor/vpRealSense2.h>
#include <visp3/core/vpMutex.h>
#include <visp3/core/vpThread.h>
#include <visp3/gui/vpDisplayGDI.h>
#include <visp3/gui/vpDisplayX.h>
#include <visp3/core/vpImageConvert.h>
using namespace std;
using namespace cv;
using namespace dnn;

////// Realsense camera initialization
rs2::pipeline camera;
rs2::decimation_filter dec_filter;
rs2::spatial_filter spat_filter;
rs2::config cfg;
rs2::align align_to_depth(RS2_STREAM_DEPTH);
rs2::align align_to_color(RS2_STREAM_COLOR);
float depth_scale = 0.0010000000474974513f; 
unsigned int frame_width = 640, frame_height = 480;
int fps = 30;
float ppx = 320.504 ; 
float ppy = 235.603 ; 
float fx =  383.970 ;    
float fy =  383.970 ;   
float depth = 0;

/// Multithreading in order to make the voice and detection work in parallel
vpMutex s_mutex_capture, s_mutex_sound;
typedef enum { capture_waiting, capture_started, capture_stopped } t_CaptureState;
t_CaptureState s_capture_state = capture_started;
bool s_sound_state = false; 

/// The voice will be only active when the depth to the nearest object is within 1 meter
char text[] = {"obstactle in 1 meter!"};

cv::Mat s_frame;

/// The first thread is used for the detction
vpThread::Return displayFunction(vpThread::Args args)
{
    (void)args; // Avoid warning: unused parameter args
    t_CaptureState capture_state_;
    std::vector<std::string> class_names;
    ifstream ifs(string("../input/object_detection_classes_coco.txt").c_str());
    string line;
    while (getline(ifs, line))
    {
        class_names.push_back(line);
    }  
    auto model = readNet("../input/frozen_inference_graph.pb", 
                        "../input/ssd_mobilenet_v2_coco_2018_03_29.pbtxt.txt", 
                        "TensorFlow");

    // capture the video
    cfg.enable_stream(RS2_STREAM_DEPTH, frame_width, frame_height, RS2_FORMAT_ANY, fps);
    cfg.enable_stream(RS2_STREAM_COLOR, frame_width, frame_height, RS2_FORMAT_ANY, fps);
    camera.start(cfg);
    VideoWriter out("../outputs/video_result.avi", VideoWriter::fourcc('M', 'J', 'P', 'G'), 30, 
                    Size(frame_width, frame_height));

    do {
    s_mutex_capture.lock();
    capture_state_ = s_capture_state;
    s_mutex_capture.unlock();
    vpImage<uint16_t> I_depth_raw(frame_height, frame_width);
    vpImage<vpRGBa> I_colour(frame_height, frame_width);
    rs2::frameset frameset = camera.wait_for_frames();
    auto  aligned_frames = align_to_color.process(frameset);
    auto aligned_depth_frame = aligned_frames.get_depth_frame();
    auto color_frame = aligned_frames.get_color_frame();
    const uint16_t* depth_data = reinterpret_cast<const uint16_t*>(aligned_depth_frame.get_data());
    auto colour_data = color_frame.get_data();
    for (int y = 0; y < frame_height; y++)
    {
        auto depth_pixel_index = y * frame_width;
        for (int x = 0; x < frame_width; x++, ++depth_pixel_index){
            I_depth_raw[y][x] =  depth_data[depth_pixel_index];
        }
    }
    cv::Mat image = cv::Mat(cv::Size(frame_width, frame_height), CV_8UC3, (void*)color_frame.get_data());
    cv::cvtColor(image, image, COLOR_BGR2RGB);

    int image_height = image.cols;
    int image_width = image.rows;
    //create blob from image for the center of the object detected
    Mat blob = blobFromImage(image, 1.0, Size(300, 300), Scalar(127.5, 127.5, 127.5), 
                            true, false);
    model.setInput(blob);
    //forward pass through the model to carry out the detection
    Mat output = model.forward();
    Mat detectionMat(output.size[2], output.size[3], CV_32F, output.ptr<float>());
    for (int i = 0; i < detectionMat.rows; i++){
        int class_id = detectionMat.at<float>(i, 1);
        float confidence = detectionMat.at<float>(i, 2);
        // Check if the detection is of good quality
        if (confidence > 0.4){
            int box_x = static_cast<int>(detectionMat.at<float>(i, 3) * image.cols);
            int box_y = static_cast<int>(detectionMat.at<float>(i, 4) * image.rows);
            int box_width = static_cast<int>(detectionMat.at<float>(i, 5) * image.cols - box_x);
            int box_height = static_cast<int>(detectionMat.at<float>(i, 6) * image.rows - box_y);
            circle(image, Point(int(box_x+box_width/2.0), int(box_y+box_height/2.0)), 10, Scalar( 0, 255,0), -1);
            rectangle(image, Point(box_x, box_y), Point(box_x+box_width, box_y+box_height), Scalar(255,255,255), 2);
            putText(image, class_names[class_id-1].c_str(), Point(box_x+10, box_y+20), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0,0,255), 1);
            depth = I_depth_raw[int(box_y+box_height/2.0)][int(box_x+box_width/2.0)]*depth_scale;
            putText(image, "distance: "+std::to_string(depth)+ " (m)", Point(box_x+10, box_y+40), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0,255,255), 1);
            if (depth >0.01 && depth < 1.3 ){
                vpMutex::vpScopedLock lock(s_mutex_sound);
                s_sound_state = true;
                strcpy(text,class_names[class_id-1].c_str() );
            }
        }
    }

    out.write(image);

    // Check if a frame is available
    if (capture_state_ == capture_started) {
        imshow("image", image);
        int k = waitKey(10);
        if (k == 113) {
            vpMutex::vpScopedLock lock(s_mutex_capture);
            s_capture_state = capture_stopped;
        }
    } 
    }while (capture_state_ != capture_stopped);
    std::cout << "End of display thread" << std::endl;
    return 0;
}


/// The second thread is used to enable voice in order to warn the user that an obstacle is within 1 meter 
vpThread::Return soundFunction(vpThread::Args args)
{
    (void)args; // Avoid warning: unused parameter args
    espeak_AUDIO_OUTPUT output = AUDIO_OUTPUT_SYNCH_PLAYBACK;
    char *path = NULL;
    void* user_data;
    unsigned int *identifier;
    int buflength = 500, options = 0;
    unsigned int position = 0, end_position = 0, flags = espeakCHARS_AUTO;
    espeak_VOICE voice;
    const char *langNativeString = "en"; // Set voice by properties
    voice.languages = langNativeString;
    voice.name = "US";
    voice.variant = 2;
    voice.gender = 2;
    espeak_Initialize(output, buflength, path, options );
    espeak_POSITION_TYPE position_type = espeak_POSITION_TYPE::POS_WORD;
    memset(&voice, 0, sizeof(espeak_VOICE)); // Zero out the voice first
    espeak_SetVoiceByProperties(&voice);
    bool sound_state;
    char previous [] = "";
    do { 
    s_mutex_sound.lock();
    sound_state = s_sound_state;
    s_mutex_sound.unlock();
    // Check if we need to talk
    if (sound_state == true && strcmp(text, previous) != 0) {
        vpMutex::vpScopedLock lock(s_mutex_sound);
        espeak_Synth(text, buflength, position, position_type, end_position, flags, identifier, user_data);
        s_sound_state = false;
        strcpy(previous,text);
    }
    } while (s_capture_state != capture_stopped);
  return 0;
}

int main(int, char**) {
    vpThread thread_display((vpThread::Fn)displayFunction);
    vpThread thread_sound((vpThread::Fn)soundFunction);
    thread_display.join();
    thread_sound.join();
    destroyAllWindows();
}
