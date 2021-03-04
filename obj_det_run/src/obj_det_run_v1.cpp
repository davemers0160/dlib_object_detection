#define _CRT_SECURE_NO_WARNINGS

 #if defined(_WIN32) | defined(__WIN32__) | defined(__WIN32) | defined(_WIN64) | defined(__WIN64)
 #include <windows.h>
 #endif

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
//#include "get_platform.h"
#include "get_current_time.h"
#include "num2string.h"
#include "file_ops.h"
#include "sleep_ms.h"

// object detector library header
#include "obj_det_lib.h"

// OpenCV Includes
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

// -------------------------------GLOBALS--------------------------------------



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

    // timing variables
    typedef std::chrono::duration<double> d_sec;
    auto start_time = chrono::system_clock::now();
    auto stop_time = chrono::system_clock::now();
    auto elapsed_time = chrono::duration_cast<d_sec>(stop_time - start_time);
    std::string sdate, stime;

    // data IO variables
    const std::string os_file_sep = "/";
    std::string version;
    std::string parse_filename;
    std::string program_root;
    std::string save_directory;

    std::string test_inputfile;
    std::string trained_net_file;

    std::pair<std::string, uint8_t> test_input;
    std::string test_data_directory;
    std::vector<std::vector<std::string>> test_file;

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

    save_directory = path_check(save_directory);

    // setup save variable locations
#if defined(_WIN32) | defined(__WIN32__) | defined(__WIN32) | defined(_WIN64) | defined(__WIN64)
    program_root = get_path(get_path(get_path(std::string(argv[0]), "\\"), "\\"), "\\") + os_file_sep;

#else
    // Ubuntu
    program_root = get_ubuntu_path();

#endif

    std::cout << "Reading Inputs... " << std::endl;
    std::cout << "program_root:          " << program_root << std::endl;
    std::cout << "save_directory:        " << save_directory << std::endl;

    try {

        get_current_time(sdate, stime);

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
            get_detections(img.ptr<unsigned char>(0), nr, nc, &num_dets, dets);
            stop_time = chrono::system_clock::now();
            
            elapsed_time = chrono::duration_cast<d_sec>(stop_time - start_time);
            

            layer_struct ls_01;
            const float* ld_01 = NULL;

            //get_layer_01(&ls_01, ld_01);
            get_combined_output(&ls_01, ld_01);

            sleep_ms(100);

            std::cout << "------------------------------------------------------------------" << std::endl;
            std::cout << "Image " << std::right << std::setw(5) << std::setfill('0') << idx << ": " << test_file[idx][0] << std::endl;
            std::cout << "Image Size (h x w): " << nr << "x" << nc << std::endl;
            std::cout << "Classification Time (s): " << elapsed_time.count() << std::endl;

            for (jdx = 0; jdx < num_dets; ++jdx)
            {
                std::cout << "Detection: " << std::string(detects[jdx].name) << ", Center (x, y): " << detects[jdx].x << "," << detects[jdx].y << std::endl;
            }

        }

        close_lib();

        std::cout << "End of Program." << std::endl;
        std::cin.ignore();
        
    }
    catch (std::exception& e)
    {
        std::cout << e.what() << std::endl;

        std::cout << "Press Enter to close..." << std::endl;
        std::cin.ignore();
    }

    return 0;

}   // end of main
