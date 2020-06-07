#define _CRT_SECURE_NO_WARNINGS

// #if defined(_WIN32) | defined(__WIN32__) | defined(__WIN32) | defined(_WIN64) | defined(__WIN64)
// #include <windows.h>
// #endif

#include <cmath>
#include <cstdlib>
#include <cstdint>
#include <iostream>
#include <iomanip>
#include <thread>
#include <sstream>
#include <fstream>
#include <chrono>
#include <algorithm>
#include <string>
#include <utility>
#include <stdexcept>

// Custom includes
#include "obj_det_run.h"
#include "get_platform.h"
#include "get_current_time.h"
#include "num2string.h"
#include "file_ops.h"
#include "sleep_ms.h"

// object detector library header
#include "obj_det_lib.h"
extern const uint32_t array_depth = 1;

// OpenCV Includes
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

// -------------------------------GLOBALS--------------------------------------
std::string platform;
std::string version;
std::string logfileName = "obj_det_net_log_";

// ----------------------------------------------------------------------------
void get_platform_control(void)
{
    get_platform(platform);

    if (platform == "")
    {
        std::cout << "No Platform could be identified... defaulting to Windows." << std::endl;
        platform = "Win";
    }

    version = version + platform;
    logfileName = version + "_log_";
}

// ----------------------------------------------------------------------------
void print_usage(void)
{
    std::cout << "Enter the following as arguments into the program:" << std::endl;
    std::cout << "<input file name> " << std::endl;
    std::cout << endl;
}

// ----------------------------------------------------------------------------
int main(int argc, char** argv)
{

    uint64_t idx = 0, jdx = 0;
    uint8_t HPC = 0;

    // timing variables
    typedef std::chrono::duration<double> d_sec;
    auto start_time = chrono::system_clock::now();
    auto stop_time = chrono::system_clock::now();
    auto elapsed_time = chrono::duration_cast<d_sec>(stop_time - start_time);
    std::string sdate, stime;

    // data IO variables
    const std::string os_file_sep = "/";
    std::string parse_filename;
    std::string program_root;
    std::string save_directory;

    std::string test_inputfile;
    std::string trained_net_file;

    std::pair<std::string, uint8_t> test_input;
    std::string test_data_directory;
    std::vector<std::vector<std::string>> test_file;

    std::ofstream data_log_stream;

    unsigned int num_classes, num_win;
    unsigned int num_dets = 0;
    unsigned int t_nr = 0, t_nc = 0;
    unsigned int nr = 0, nc = 0;

    window_struct* det_win;
    struct detection_center* detects;
    struct detection_struct* dets;
    
    unsigned char* tiled_img = NULL;
    unsigned char* det_img = NULL;

    cv::Mat img;
  
    // ----------------------------------------------------------------------------------------
    if (argc == 1)
    {
        print_usage();
        std::cin.ignore();
        return 0;
    }

    parse_filename = argv[1];

    // parse through the supplied csv file
    parse_input_file(parse_filename, version, trained_net_file, test_input, save_directory);

    // check the platform
    get_platform_control();

    // check for HPC <- set the environment variable PLATFORM to HPC
    if(platform.compare(0,3,"HPC") == 0)
    {
        std::cout << "HPC Platform Detected." << std::endl;
        HPC = 1;
    }

    save_directory = path_check(save_directory);

    // setup save variable locations
#if defined(_WIN32) | defined(__WIN32__) | defined(__WIN32) | defined(_WIN64) | defined(__WIN64)
    program_root = get_path(get_path(get_path(std::string(argv[0]), "\\"), "\\"), "\\") + os_file_sep;

#else
    if (HPC == 1)
    {
        //HPC version
        program_root = get_path(get_path(get_path(std::string(argv[0]), os_file_sep), os_file_sep), os_file_sep) + os_file_sep;
    }
    else
    {
        // Ubuntu
        program_root = get_ubuntu_path();
    }

#endif

    std::cout << "Reading Inputs... " << std::endl;
    std::cout << "Platform:              " << platform << std::endl;
    std::cout << "program_root:          " << program_root << std::endl;
    std::cout << "save_directory:        " << save_directory << std::endl;

    try {

        get_current_time(sdate, stime);
        logfileName = logfileName + sdate + "_" + stime + ".txt";

        std::cout << "Log File:              " << (save_directory + logfileName) << std::endl << std::endl;
        data_log_stream.open((save_directory + logfileName), ios::out | ios::app);

        // Add the date and time to the start of the log file
        data_log_stream << "------------------------------------------------------------------" << std::endl;
        data_log_stream << "Version: 2.0    Date: " << sdate << "    Time: " << stime << std::endl;
        data_log_stream << "------------------------------------------------------------------" << std::endl;
        data_log_stream << "Platform: " << platform << std::endl;

//-----------------------------------------------------------------------------
// Read in the testing images
//-----------------------------------------------------------------------------

        // parse through the supplied test input file
        switch (test_input.second)
        {
        case 0:
            std::cout << "parsing grouped csv file..." << std::endl;
            parse_group_csv_file(test_input.first, '{', '}', test_file);
            break;
        case 1:
            std::cout << "parsing standard csv file..." << std::endl;
            parse_csv_file(test_input.first, test_file);
            break;
        }

        //parse_group_csv_file(test_inputfile, '{', '}', test_file);
        if (test_file.size() == 0)
        {
            std::cout << test_input.first << ": ";
            throw std::runtime_error("Test file is empty");
        }
        
        // the data directory should be the first entry in the input file
#if defined(_WIN32) | defined(__WIN32__) | defined(__WIN32) | defined(_WIN64) | defined(__WIN64)
        test_data_directory = test_file[0][0];
#else
        if (HPC == 1)
        {
            test_data_directory = test_file[0][2];
        }
        else
        {
            test_data_directory = test_file[0][1];
        }
#endif

        test_file.erase(test_file.begin());

        std::cout << "------------------------------------------------------------------" << std::endl;
        std::cout << "input file:     " << test_input.first << std::endl;
        std::cout << "data_directory: " << test_data_directory << std::endl;
        std::cout << "test images:    " << test_file.size() << std::endl;

        data_log_stream << "------------------------------------------------------------------" << std::endl;
        data_log_stream << "Input file:     " << test_input.first << std::endl;
        data_log_stream << "data_directory: " << test_data_directory << std::endl;
        data_log_stream << "test images:    " << test_file.size() << std::endl;

//-----------------------------------------------------------------------------
//  EVALUATE THE NETWORK PERFORMANCE
//-----------------------------------------------------------------------------

        // initialize the network
        init_net(trained_net_file.c_str(), &num_classes, det_win, &num_win);
               
        for (idx = 0; idx < test_file.size(); ++idx)
        {
        
            img = cv::imread((test_data_directory + test_file[idx][0]), cv::IMREAD_GRAYSCALE);
            
            nr = img.rows;
            nc = img.cols;

            // get the rough classification time per image
            start_time = chrono::system_clock::now();
            //run_net(img.ptr<unsigned char>(0), nr, nc, det_img, &num_dets, dets);
            get_detections(img.ptr<unsigned char>(0), nr, nc, &num_dets, detects);
            stop_time = chrono::system_clock::now();
            
            elapsed_time = chrono::duration_cast<d_sec>(stop_time - start_time);
            
            sleep_ms(100);

            std::cout << "------------------------------------------------------------------" << std::endl;
            std::cout << "Image " << std::right << std::setw(5) << std::setfill('0') << idx << ": " << test_file[idx][0] << std::endl;
            std::cout << "Image Size (h x w): " << nr << "x" << nc << std::endl;
            std::cout << "Classification Time (s): " << elapsed_time.count() << std::endl;

            data_log_stream << "------------------------------------------------------------------" << std::endl;
            data_log_stream << "Image " << std::right << std::setw(5) << std::setfill('0') << idx << ": " << test_file[idx][0] << std::endl;
            data_log_stream << "Image Size (h x w): " << nr << "x" << nc << std::endl;
            data_log_stream << "Classification Time (s): " << elapsed_time.count() << std::endl;

            for (jdx = 0; jdx < num_dets; ++jdx)
            {
                std::cout << "Detection: " << std::string(detects[jdx].name) << ", Center (x, y): " << detects[jdx].x << "," << detects[jdx].y << std::endl;
                data_log_stream << "Detection: " << std::string(detects[jdx].name) << ", Center (x, y): " << detects[jdx].x << "," << detects[jdx].y << std::endl;
            }

        }

        close_lib();

        std::cout << "End of Program." << std::endl;
        data_log_stream.close();
        std::cin.ignore();
        
    }
    catch (std::exception& e)
    {
        std::cout << e.what() << std::endl;

        data_log_stream << std::endl;
        data_log_stream << "------------------------------------------------------------------" << std::endl;
        data_log_stream << e.what() << std::endl;
        data_log_stream << "------------------------------------------------------------------" << std::endl;
        data_log_stream.close();

        std::cout << "Press Enter to close..." << std::endl;
        std::cin.ignore();
    }

    return 0;

}   // end of main
