#define _CRT_SECURE_NO_WARNINGS

#if defined(_WIN32) | defined(__WIN32__) | defined(__WIN32) | defined(_WIN64) | defined(__WIN64)
#include <windows.h>
#else
#include <dlfcn.h>
#endif

//#include <cmath>
#include <cstdlib>
#include <cstdint>
#include <iostream>
#include <iomanip>
//#include <thread>
//#include <sstream>
//#include <fstream>
#include <chrono>
#include <algorithm>
#include <string>
//#include <utility>
#include <stdexcept>

// Custom includes
#include "obj_det_run.h"
#include "num2string.h"
#include "file_ops.h"
#include "opencv_colormap_functions.h"


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

    // data IO variables
    const std::string os_file_sep = "/";
    std::string parse_filename;
    std::string program_root;
    std::string save_directory;

    std::string test_inputfile;
    std::string trained_net_file;
    std::string lib_filename;

    std::pair<std::string, uint8_t> test_input;
    std::string test_data_directory;
    std::vector<std::vector<std::string>> test_file;

    // parameters for OpenCV image write png compression
    std::vector<int> compression_params;
    compression_params.push_back(cv::IMWRITE_PNG_COMPRESSION);
    compression_params.push_back(2);

    unsigned int num_classes, num_win;
    unsigned int num_dets = 0;
    unsigned int t_nr = 0, t_nc = 0;
    unsigned int nr = 0, nc = 0;

    window_struct* det_win;
    detection_struct* detects;
    
    unsigned char* tiled_img = NULL;
    unsigned char* det_img = NULL;

    // image containers for the display of the detected image and the heatmaps
    cv::Mat img, jet_hm, gray_hm;

    // OpenCV image show window names
    std::string img_win_name = "Image";
    std::string jet_hm_name = "Jet Heatmap Output";
    std::string gray_hm_name = "Gray Heatmap Output";

    // ----------------------------------------------------------------------------------------
    if (argc == 1)
    {
        print_usage();
        std::cin.ignore();
        return 0;
    }

    parse_filename = argv[1];

    // parse through the supplied csv file
    parse_input_file(parse_filename, lib_filename, trained_net_file, test_input, save_directory);

    save_directory = path_check(save_directory);

    // setup save variable locations
#if defined(_WIN32) | defined(__WIN32__) | defined(__WIN32) | defined(_WIN64) | defined(__WIN64)
    program_root = get_path(get_path(get_path(std::string(argv[0]), "\\"), "\\"), "\\") + os_file_sep;

#else
    // Ubuntu
    program_root = get_ubuntu_path();

#endif

    std::cout << "Reading Inputs... " << std::endl << std::endl;
    std::cout << "program_root:        " << program_root << std::endl;
    std::cout << "library_filename:    " << lib_filename << std::endl;
    std::cout << "net_file:            " << trained_net_file << std::endl;
    std::cout << "save_directory:      " << save_directory << std::endl << std::endl;

    try {

//-----------------------------------------------------------------------------
// Read in the testing images
//-----------------------------------------------------------------------------

        std::cout << "------------------------------------------------------------------" << std::endl;

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

        // check for an empty data file
        if (test_file.size() == 0)
        {
            std::cout << test_input.first << ": ";
            throw std::runtime_error("Test file is empty");
        }
        
        // the data directory should be the first entry in the input file
#if defined(_WIN32) | defined(__WIN32__) | defined(__WIN32) | defined(_WIN64) | defined(__WIN64)
        test_data_directory = test_file[0][0];
#else
        test_data_directory = test_file[0][1];
#endif

        test_file.erase(test_file.begin());

        std::cout << "input file:     " << test_input.first << std::endl;
        std::cout << "data_directory: " << test_data_directory << std::endl;
        std::cout << "test images:    " << test_file.size() << std::endl;

        // load in the library
#if defined(_WIN32) | defined(__WIN32__) | defined(__WIN32) | defined(_WIN64) | defined(__WIN64)
        HINSTANCE obj_det_lib = LoadLibrary(lib_filename.c_str());

        if (obj_det_lib == NULL)
        {
            throw std::runtime_error("error loading library");
        }

        init_net lib_init_net = (init_net)GetProcAddress(obj_det_lib, "init_net");
        //run_net lib_run_net = (run_net)GetProcAddress(obj_det_lib, "run_net");
        get_detections lib_get_detections = (get_detections)GetProcAddress(obj_det_lib, "get_detections");
        get_cropped_detections lib_get_cropped_detections = (get_cropped_detections)GetProcAddress(obj_det_lib, "get_cropped_detections");
        get_combined_output lib_get_combined_output = (get_combined_output)GetProcAddress(obj_det_lib, "get_combined_output");
        close_lib lib_close_lib = (close_lib)GetProcAddress(obj_det_lib, "close_lib");

#else
        void* obj_det_lib = dlopen(lib_filename.c_str(), RTLD_NOW);

        if (obj_det_lib == NULL)
        {
            throw std::runtime_error("error loading library");
        }

        init_net lib_init_net = (init_net)dlsym(obj_det_lib, "init_net");
        //run_net lib_run_net = (run_net)dlsym(obj_det_lib, "run_net");
        get_detections lib_get_detections = (get_detections)dlsym(obj_det_lib, "get_detections");
        get_cropped_detections lib_get_detections = (get_cropped_detections)dlsym(obj_det_lib, "get_cropped_detections");
        get_combined_output lib_get_combined_output = (get_combined_output)dlsym(obj_det_lib, "get_combined_output");
        close_lib lib_close_lib = (close_lib)dlsym(obj_det_lib, "close_lib");

#endif

//-----------------------------------------------------------------------------
//  EVALUATE THE NETWORK PERFORMANCE
//-----------------------------------------------------------------------------

        // initialize the network
        lib_init_net(trained_net_file.c_str(), &num_classes, det_win, &num_win);

        // create the windows to display the results and the heatmap
        cv::namedWindow(img_win_name, cv::WINDOW_NORMAL | cv::WINDOW_KEEPRATIO);
        cv::namedWindow(jet_hm_name, cv::WINDOW_NORMAL | cv::WINDOW_KEEPRATIO);
        cv::namedWindow(gray_hm_name, cv::WINDOW_NORMAL | cv::WINDOW_KEEPRATIO);

        for (idx = 0; idx < test_file.size(); ++idx)
        {
        
            img = cv::imread((test_data_directory + test_file[idx][0]), cv::IMREAD_GRAYSCALE);
            
            nr = img.rows;
            nc = img.cols;

            // get the rough classification time per image
            start_time = chrono::system_clock::now();
            //run_net(img.ptr<unsigned char>(0), nr, nc, det_img, &num_dets, dets);
            lib_get_detections(img.ptr<unsigned char>(0), nr, nc, &num_dets, detects);

            stop_time = chrono::system_clock::now();
            
            elapsed_time = chrono::duration_cast<d_sec>(stop_time - start_time);
            
            layer_struct ls_hm;
            const float* ld_hm = NULL;

            lib_get_combined_output(&ls_hm, ld_hm);

            jet_hm = cv::Mat(ls_hm.nr, ls_hm.nc, CV_32FC1, (void*)ld_hm);
            gray_hm = cv_gray<float>(jet_hm, -1.5, 0.0);
            jet_hm = cv_jet<float>(jet_hm, -1.5, 0.0);

            cv::imshow(img_win_name, img);
            cv::imshow(jet_hm_name, jet_hm);
            cv::imshow(gray_hm_name, gray_hm);

            std::cout << "------------------------------------------------------------------" << std::endl;
            std::cout << "Image " << std::right << std::setw(5) << std::setfill('0') << idx << ": " << test_file[idx][0] << std::endl;
            std::cout << "Image Size (h x w): " << nr << "x" << nc << std::endl;
            std::cout << "Classification Time (s): " << elapsed_time.count() << std::endl;

            for (jdx = 0; jdx < num_dets; ++jdx)
            {
                std::cout << "Detection: " << std::string(detects[jdx].name) << ", (x, y, w, h): "
                    << detects[jdx].x << "," << detects[jdx].y << "," << detects[jdx].w << "," << detects[jdx].h << std::endl;
            }

            //save results to an image
            cv::imwrite(save_directory + "jet_hm_" + num2str(idx, "_%05d.png"), jet_hm, compression_params);
            cv::imwrite(save_directory + "gray_hm_" + num2str(idx, "_%05d.png"), gray_hm, compression_params);

            cv::waitKey(1);

            int bp = 1;
        }

        std::cout << std::endl;
        lib_close_lib();

    // close the library
#if defined(_WIN32) | defined(__WIN32__) | defined(__WIN32) | defined(_WIN64) | defined(__WIN64)
        FreeLibrary(obj_det_lib);
#else
        dlclose(obj_det_lib);
#endif

        
    }
    catch (std::exception& e)
    {
        std::cout << e.what() << std::endl;

        std::cout << std::endl << "Press Enter to close..." << std::endl;
        std::cin.ignore();
    }

    std::cout << std::endl << "End of Program.  Press Enter to close..." << std::endl;
    std::cin.ignore();

    cv::destroyAllWindows();

    return 0;

}   // end of main
