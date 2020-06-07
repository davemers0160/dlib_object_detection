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

// Net Version
#if defined(USE_OBJ_LIB)
#include "obj_det_lib.h"
extern const uint32_t array_depth = 1;

#else
extern const uint32_t array_depth;
#include "obj_det_net_v10.h"

#endif()

#include "load_data.h"
#include "load_oid_data.h"
//#include "eval_net_performance.h"

// dlib includes
#include <dlib/dnn.h>
#include <dlib/image_io.h>
#include <dlib/data_io.h>
#include <dlib/image_transforms.h>
#include <dlib/rand.h>

#if !defined(DLIB_NO_GUI_SUPPORT)
#include "overlay_bounding_box.h"
#include <dlib/gui_widgets.h>
#endif

// dlib-contrib includes
#include "array_image_operations.h"
#include "random_array_cropper.h"
#include "copy_dlib_net.h"
#include "dlib_set_learning_rates.h"

// -------------------------------GLOBALS--------------------------------------

//extern const uint32_t array_depth;
std::string platform;

//this will store the standard RGB images and groundtruth data for the bounding box labels
std::vector<std::array<dlib::matrix<uint8_t>, array_depth>> test_images;
std::vector<std::vector<dlib::mmod_rect>> test_labels;

// containers to store the random crops used during each training iteration and groundtruth data for the bounding box labels
//std::vector<std::array<dlib::matrix<uint8_t>, array_depth>> train_batch_samples, test_batch_samples;
//std::vector<std::vector<dlib::mmod_rect>> train_batch_labels, test_batch_labels;

std::string version;
//std::string net_name = "obj_det_net_";
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
    //net_sync_name = version + "_sync";
    logfileName = version + "_log_";
    //net_name = version +  "_final_net.dat";
}

// ----------------------------------------------------------------------------------------

void print_usage(void)
{
    std::cout << "Enter the following as arguments into the program:" << std::endl;
    std::cout << "<image file name> " << std::endl;
    std::cout << endl;
}

// ----------------------------------------------------------------------------------------

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
    //std::string sync_save_location;
    //std::string image_save_location;
    //std::string results_save_location;

    //std::string train_inputfile;
    std::string test_inputfile;
    //std::string train_class_name;
    std::string trained_net_file;

    std::pair<std::string, uint8_t> test_input;
    std::string test_data_directory;
    //std::vector<std::vector<std::string>> training_file;
    std::vector<std::vector<std::string>> test_file;
    std::vector<std::string> te_image_files;

    // these two matrices will contain the results of the training and testing
    //dlib::matrix<double, 1, 6> training_results = dlib::zeros_matrix<double>(1, 6);
    dlib::matrix<double, 1, 6> test_results = dlib::zeros_matrix<double>(1, 6);

    std::ofstream data_log_stream;

    std::vector<int32_t> gpu = { 0 };

    dlib::rgb_pixel color;
    dlib::matrix<dlib::rgb_pixel> rgb_img;

    uint32_t num_classes;
    std::set<std::string> tmp_names;
    std::vector<dlib::rgb_pixel> class_color;

#if !defined(DLIB_NO_GUI_SUPPORT)
    //create window to display images
    dlib::image_window win;
#endif

    dlib::rand rnd;
    rnd = dlib::rand(time(NULL));
    
    // ----------------------------------------------------------------------------------------
   
    if (argc == 1)
    {
        print_usage();
        std::cin.ignore();
        return 0;
    }

    parse_filename = argv[1];


    // parse through the supplied csv file
    parse_input_file(parse_filename, version, gpu, trained_net_file, test_input, save_directory);

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
    //results_save_location = save_directory + "results/";
    //image_save_location = save_directory + "result_images/";

#else
    if (HPC == 1)
    {
        //HPC version
        program_root = get_path(get_path(get_path(std::string(argv[0]), os_file_sep), os_file_sep), os_file_sep) + os_file_sep;
    }
    else
    {
        // Ubuntu
        program_root = "/home/owner/Projects/dlib_obj_detector/";
    }

    //results_save_location = save_directory + "results/";
    //image_save_location = save_directory + "result_images/";

#endif

    std::cout << "Reading Inputs... " << std::endl;
    std::cout << "Platform:              " << platform << std::endl;
    std::cout << "GPU:                   { ";
    for (idx = 0; idx < gpu.size(); ++idx)
        std::cout << gpu[idx] << " ";
    std::cout << "}" << std::endl;
    std::cout << "program_root:          " << program_root << std::endl;
    std::cout << "save_directory:        " << save_directory << std::endl;
    //std::cout << "results_save_location: " << results_save_location << std::endl;
    //std::cout << "image_save_location:   " << image_save_location << std::endl;


    try {

        get_current_time(sdate, stime);
        logfileName = logfileName + sdate + "_" + stime + ".txt";
        //cropper_stats_file = output_save_location + "cr_stats_" + version + "_" + sdate + "_" + stime + ".txt";

        //std::cout << "Log File:              " << (results_save_location + logfileName) << std::endl << std::endl;
        //data_log_stream.open((results_save_location + logfileName), ios::out | ios::app);
        std::cout << "Log File:              " << (save_directory + logfileName) << std::endl << std::endl;
        data_log_stream.open((save_directory + logfileName), ios::out | ios::app);

        // Add the date and time to the start of the log file
        data_log_stream << "------------------------------------------------------------------" << std::endl;
        data_log_stream << "Version: 2.0    Date: " << sdate << "    Time: " << stime << std::endl;
        data_log_stream << "Platform: " << platform << std::endl;
        data_log_stream << "GPU: { ";
        for (idx = 0; idx < gpu.size(); ++idx)
            data_log_stream << gpu[idx] << " ";
        data_log_stream << "}" << std::endl;
        data_log_stream << "------------------------------------------------------------------" << std::endl;

//-----------------------------------------------------------------------------
// Read in the testing images
//-----------------------------------------------------------------------------

        //-------------------------------------------------------------------------------------
        // parse through the supplied test input file
        switch (test_input.second)
        {
        case 0:
            parse_group_csv_file(test_input.first, '{', '}', test_file);
            break;
        case 1:
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
        std::cout << "input file: " << test_input.first << std::endl;
        std::cout << "data_directory:        " << test_data_directory << std::endl;
        std::cout << "Test image sets to parse: " << test_file.size() << std::endl;

        data_log_stream << "------------------------------------------------------------------" << std::endl;
        data_log_stream << "Input file: " << test_input.first << std::endl;
        data_log_stream << "data_directory:        " << test_data_directory << std::endl;
        data_log_stream << "Test image sets to parse: " << test_file.size() << std::endl;
       
        std::cout << "Loading test images... ";

        // load in the images and labels
        start_time = chrono::system_clock::now();
        switch (test_input.second)
        {
        case 0:
            load_data(test_file, test_data_directory, test_images, test_labels, te_image_files);
            break;
        case 1:
            load_oid_data(test_file, test_data_directory, test_images, test_labels, te_image_files);
            break;
        }
        //load_data(test_file, test_data_directory, test_images, test_labels, te_image_files);
        stop_time = chrono::system_clock::now();
        elapsed_time = chrono::duration_cast<d_sec>(stop_time - start_time);

        std::cout << "Loaded " << test_images.size() << " test image sets in " << elapsed_time.count() / 60 << " minutes." << std::endl;
        data_log_stream << "Loaded " << test_images.size() << " test image sets in " << elapsed_time.count() / 60 << " minutes." << std::endl;
        
//-----------------------------------------------------------------------------
// Setup the network
//-----------------------------------------------------------------------------

        // this sets th GPUs to use algorithms that are smaller in memory but may take a little longer to execute
        //dlib::set_dnn_prefer_smallest_algorithms();
        dlib::set_dnn_prefer_fastest_algorithms();

        // set the cuda device explicitly
        //if (gpu.size() == 1)
        //    dlib::cuda::set_device(gpu[0]);
     
//-----------------------------------------------------------------------------
//  EVALUATE THE NETWORK PERFORMANCE
//-----------------------------------------------------------------------------

#if defined(USE_OBJ_LIB)
        unsigned int num_classes, num_win;
        unsigned int num_dets = 0;
        unsigned int t_nr = 0, t_nc = 0;

        window_struct* det_win;
        struct detection_center* detects;
        struct detection_struct* dets;

        // init the network
        init_net(trained_net_file.c_str(), &num_classes, det_win, &num_win);

        unsigned char* tiled_img = NULL;
        unsigned char* det_img = NULL;

        long nr = 0, nc = 0;
        uint32_t index = 0;

        for (idx = 0; idx < test_images.size(); ++idx)
        {
            nr = test_images[idx][0].nr();
            nc = test_images[idx][0].nc();

            unsigned char *te = new unsigned char[nr * nc];
            index = 0;
            for (long r = 0; r < nr; ++r)
            {
                for (long c = 0; c < nc; ++c)
                {
                    te[index++] = test_images[idx][0](r, c);
                }
            }

            // get the rough classification time per image
            start_time = chrono::system_clock::now();
            get_detections(te, nr, nc, &num_dets, detects);
            run_net(te, nr, nc, det_img, &num_dets, dets);

            stop_time = chrono::system_clock::now();
            elapsed_time = chrono::duration_cast<d_sec>(stop_time - start_time);
            sleep_ms(100);

            std::cout << "------------------------------------------------------------------" << std::endl;
            std::cout << "Image " << std::right << std::setw(5) << std::setfill('0') << idx << ": " << te_image_files[idx] << std::endl;
            std::cout << "Image Size (h x w): " << test_images[idx][0].nr() << "x" << test_images[idx][0].nc() << std::endl;
            std::cout << "Classification Time (s): " << elapsed_time.count() << std::endl;

            for (jdx = 0; jdx < num_dets; ++jdx)
            {
                std::cout << "Detection: " << detects[jdx].name << ", Center (x, y): " << detects[jdx].x << "," << detects[jdx].y << std::endl;
            }

        }
#else
        // load the network from the saved file
        anet_type test_net;

        std::cout << std::endl << "------------------------------------------------------------------" << std::endl;
        std::cout << "Loading " << trained_net_file << std::endl;
        dlib::deserialize(trained_net_file) >> test_net;

        // get the details about the loss layer -> the number and names of the classes
        dlib::mmod_options options = dlib::layer<0>(test_net).loss_details().get_options();

        std::set<std::string> tmp_names;
        std::cout << std::endl << "------------------------------------------------------------------" << std::endl;
        for (idx = 0; idx < options.detector_windows.size(); ++idx)
        {
            std::cout << "detector window (w x h): " << options.detector_windows[idx].label << " - " << options.detector_windows[idx].width << " x " << options.detector_windows[idx].height << std::endl;

            //det_win[idx].w = options.detector_windows[idx].width;
            //det_win[idx].h = options.detector_windows[idx].height;
            //strcpy(det_win[idx].label, options.detector_windows[idx].label.c_str());

            tmp_names.insert(options.detector_windows[idx].label);
        }
        std::cout << std::endl;

        std::vector<std::string> class_names(tmp_names.begin(), tmp_names.end());
        num_classes = class_names.size();

        for (idx = 0; idx < num_classes; ++idx)
        {
            class_color.push_back(dlib::rgb_pixel(rnd.get_random_8bit_number(), rnd.get_random_8bit_number(), rnd.get_random_8bit_number()));
        }

        std::vector<label_stats> test_label_stats(num_classes, label_stats(0, 0));

        // show the network to verify that it looks correct
        std::cout << "------------------------------------------------------------------" << std::endl;
        //std::cout << "Net Name: " << net_name << std::endl;
        std::cout << test_net << std::endl;

        //data_log_stream << "Net Name: " << net_name << std::endl;
        data_log_stream << test_net << std::endl;
        data_log_stream << "------------------------------------------------------------------" << std::endl;

        // In this section we want to evaluate the performance of the network against the test data
        // this should be displayed and then saved to the log file
        // - This can also include displaying the input image along with the ground truth bounding box, name and dnn results
        std::cout << "Analyzing Test Results..." << std::endl;

        // testResults = eval_all_net_performance(test_net, test_images, test_labels, dnn_test_labels, min_target_size);
        test_results = dlib::zeros_matrix<double>(1, 6);
        //dnn_test_labels.clear();

        for (idx = 0; idx < test_images.size(); ++idx)
        {

            std::vector<dlib::mmod_rect> dnn_labels;
            std::vector<label_stats> ls(num_classes, label_stats(0, 0));

            // get the rough classification time per image
            start_time = chrono::system_clock::now();
            //dlib::matrix<double, 1, 6> tr = eval_net_performance(test_net, test_images[idx], test_labels[idx], dnn_labels, target_size.first, fda_test_box_overlap(0.4, 1.0), class_names, ls);           
            dnn_labels = test_net(test_images[idx]);
            stop_time = chrono::system_clock::now();
            elapsed_time = chrono::duration_cast<d_sec>(stop_time - start_time);

            std::cout << "------------------------------------------------------------------" << std::endl;
            std::cout << "Image " << std::right << std::setw(5) << std::setfill('0') << idx << ": " << te_image_files[idx] << std::endl;
            std::cout << "Image Size (h x w): " << test_images[idx][0].nr() << "x" << test_images[idx][0].nc() << std::endl;
            std::cout << "Classification Time (s): " << elapsed_time.count() << std::endl;
            ////std::cout << "Results: " << std::fixed << std::setprecision(4) << tr(0, 0) << ", " << tr(0, 3) << ", " << tr(0, 4) << ", " << tr(0, 5) << std::endl;

            //data_log_stream << "------------------------------------------------------------------" << std::endl;
            //data_log_stream << "Image " << std::right << std::setw(5) << std::setfill('0') << idx << ": " << te_image_files[idx] << std::endl;
            //data_log_stream << "Image Size (h x w): " << test_images[idx][0].nr() << "x" << test_images[idx][0].nc() << std::endl;
            //data_log_stream << "Classification Time (s): " << elapsed_time.count() << std::endl;
            ////data_log_stream << "Results: " << std::fixed << std::setprecision(4) << tr(0, 0) << ", " << tr(0, 3) << ", " << tr(0, 4) << ", " << tr(0, 5) << std::endl;

            //for (jdx = 0; jdx < num_classes; ++jdx)
            //{
            //    test_label_stats[jdx].count += ls[jdx].count;
            //    test_label_stats[jdx].match_count += ls[jdx].match_count; 
            //    
            //    double acc = (ls[jdx].count == 0) ? 0.0 : ls[jdx].match_count / (double)ls[jdx].count;
            //    std::cout << std::left << std::setw(15) << std::setfill(' ') << (class_names[jdx] + ":") << std::fixed << std::setprecision(4) << acc << ", " << ls[jdx].match_count << ", " << ls[jdx].count << std::endl;
            //    data_log_stream << std::left << std::setw(15) << std::setfill(' ') << (class_names[jdx] + ":") << std::fixed << std::setprecision(4) << acc << ", " << ls[jdx].match_count << ", " << ls[jdx].count  << std::endl;
            //}
            //std::cout << std::left << std::setw(15) << std::setfill(' ') << "Results: " << std::fixed << std::setprecision(4) << tr(0, 0) << ", " << tr(0, 3) << ", " << tr(0, 4) << ", " << tr(0, 5) << std::endl;
            //data_log_stream << std::left << std::setw(15) << std::setfill(' ') << "Results: " << std::fixed << std::setprecision(4) << tr(0, 0) << ", " << tr(0, 3) << ", " << tr(0, 4) << ", " << tr(0, 5) << std::endl;

            if (array_depth < 3)
                dlib::assign_image(rgb_img, test_images[idx][0]);
            else
                merge_channels(test_images[idx], rgb_img);

#if !defined(DLIB_NO_GUI_SUPPORT)
            win.clear_overlay();

            //overlay the dnn detections on the image
            for (jdx = 0; jdx < dnn_labels.size(); ++jdx)
            {
                auto class_index = std::find(class_names.begin(), class_names.end(), dnn_labels[jdx].label);
                
                overlay_bounding_box(rgb_img, dnn_labels[jdx], class_color[std::distance(class_names.begin(), class_index)], false);
                dlib::point center = dlib::center(dnn_labels[jdx].rect);

                std::cout << "Detection: " << dnn_labels[jdx].label << ", Center (x, y): " << center.x() << "," << center.y();
                std::cout << ", Confidence Level: " << dnn_labels[jdx].detection_confidence << std::endl;

                data_log_stream << "Detection: " << dnn_labels[jdx].label << ", Center (x, y): " << center.x() << "," << center.y();
                data_log_stream << ", Confidence Level: " << dnn_labels[jdx].detection_confidence << std::endl;
            }
            win.set_image(rgb_img);
#endif

            //save results to an image save_directory
            //std::string image_save_name = image_save_location + "test_img_" + version + num2str(idx, "_%05d.png");
            std::string image_save_name = save_directory + "test_img_" + version + num2str(idx, "_%05d.png");
            //save_png(rgb_img, image_save_name);

            //test_results += tr;
            //dlib::sleep(50);
            std::cin.ignore();

        }
        data_log_stream << "------------------------------------------------------------------" << std::endl << std::endl;
        std::cout << "------------------------------------------------------------------" << std::endl << std::endl;

        // output the training results
        //std::cout << "------------------------------------------------------------------" << std::endl;
        //for (jdx = 0; jdx < num_classes; ++jdx)
        //{
        //    double acc = (train_label_stats[jdx].count == 0) ? 0.0 : train_label_stats[jdx].match_count / (double)train_label_stats[jdx].count;
        //    std::cout << std::left << std::setw(15) << std::setfill(' ') << (class_names[jdx] + ":") << std::fixed << std::setprecision(4) << acc << ", " << train_label_stats[jdx].match_count << ", " << train_label_stats[jdx].count << std::endl;
        //}
        //std::cout << "Training Results (detction_accuracy, correct_detects, false_positives, missing_detections):  " << std::fixed << std::setprecision(4) << training_results(0, 0) / (double)training_file.size() << ", " << training_results(0, 3) << ", " << training_results(0, 4) << ", " << training_results(0, 5) << std::endl;
        //std::cout << "------------------------------------------------------------------" << std::endl << std::endl;


        // output the test results
        std::cout << "------------------------------------------------------------------" << std::endl;
        for (jdx = 0; jdx < num_classes; ++jdx)
        {
            double acc = (test_label_stats[jdx].count == 0) ? 0.0 : test_label_stats[jdx].match_count / (double)test_label_stats[jdx].count;
            std::cout << std::left << std::setw(15) << std::setfill(' ') << (class_names[jdx] + ":") << std::fixed << std::setprecision(4) << acc << ", " << test_label_stats[jdx].match_count << ", " << test_label_stats[jdx].count << std::endl;
        }
        std::cout << "Testing Results (detction_accuracy, correct_detects, false_positives, missing_detections):  " << std::fixed << std::setprecision(4) << test_results(0, 0) / (double)test_file.size() << ", " << test_results(0, 3) << ", " << test_results(0, 4) << ", " << test_results(0, 5) << std::endl;
        std::cout << "------------------------------------------------------------------" << std::endl;


        // save the results to the log file
        //data_log_stream << "------------------------------------------------------------------" << std::endl;
        //data_log_stream << "Training Results (detction_accuracy, correct_detects, false_positives, missing_detections): " << std::fixed << std::setprecision(4) << training_results(0, 0) / (double)training_file.size() << ", " << training_results(0, 3) << ", " << training_results(0, 4) << ", " << training_results(0, 5) << std::endl;
        //data_log_stream << "class_name, detction_accuracy, correct_detects, groundtruth" << std::endl;
        //for (jdx = 0; jdx < num_classes; ++jdx)
        //{
        //    double acc = (train_label_stats[jdx].count == 0) ? 0.0 : train_label_stats[jdx].match_count / (double)train_label_stats[jdx].count;
        //    data_log_stream << std::left << std::setw(15) << std::setfill(' ') << (class_names[jdx] + ":") << std::fixed << std::setprecision(4) << acc<< ", " << train_label_stats[jdx].match_count << ", " << train_label_stats[jdx].count << std::endl;
        //}
        //data_log_stream << "------------------------------------------------------------------" << std::endl << std::endl;

        data_log_stream << "------------------------------------------------------------------" << std::endl;
        data_log_stream << "Testing Results (detction_accuracy, correct_detects, false_positives, missing_detections):  " << std::fixed << std::setprecision(4) << test_results(0, 0) / (double)test_file.size() << ", " << test_results(0, 3) << ", " << test_results(0, 4) << ", " << test_results(0, 5) << std::endl;
        data_log_stream << "class_name, detction_accuracy, correct_detects, groundtruth" << std::endl;
        for (jdx = 0; jdx < num_classes; ++jdx)
        {
            double acc = (test_label_stats[jdx].count == 0) ? 0.0 : test_label_stats[jdx].match_count / (double)test_label_stats[jdx].count;
            data_log_stream << std::left << std::setw(15) << std::setfill(' ') << (class_names[jdx] + ":") << std::fixed << std::setprecision(4) << acc << ", " << test_label_stats[jdx].match_count << ", " << test_label_stats[jdx].count << std::endl;
        }
        data_log_stream << "------------------------------------------------------------------" << std::endl;

#endif

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
