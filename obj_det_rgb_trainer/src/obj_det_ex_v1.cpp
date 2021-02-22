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
#include "get_platform.h"
#include "get_current_time.h"
#include "num2string.h"
#include "file_ops.h"

// Net Version
//#include "yj_net_v10.h"
#include "obj_det_net_rgb_v13.h"
//#include "tfd_net_v04.h"

#include "obj_det_dnn.h"
#include "load_data.h"
#include "run_rgb_network_performance.h"
#include "eval_net_performance.h"
//#include "enhanced_array_cropper.h"
//#include "random_channel_swap.h"
//#include "enhanced_channel_swap.h"

// dlib includes
#include <dlib/dnn.h>
#include <dlib/image_io.h>
#include <dlib/data_io.h>
#include <dlib/image_transforms.h>
#include <dlib/rand.h>

#if !defined(DLIB_NO_GUI_SUPPORT)
#include "dlib_overlay_bbox.h"
#include <dlib/gui_widgets.h>
#endif

// dlib-contrib includes
//#include "array_image_operations.h"
//#include "random_array_cropper.h"
#include "image_noise_functions.h"
#include "random_channel_swap.h"


// new copy and set learning rate includes
//#include "copy_dlib_net.h"
//#include "dlib_set_learning_rates.h"

// -------------------------------GLOBALS--------------------------------------

//extern const uint32_t array_depth;
std::string platform;

//this will store the standard RGB images and groundtruth data for the bounding box labels
std::vector<dlib::matrix<dlib::rgb_pixel>> train_images, test_images;
std::vector<std::vector<dlib::mmod_rect>> train_labels, test_labels;

// containers to store the random crops used during each training iteration and groundtruth data for the bounding box labels
std::vector<dlib::matrix<dlib::rgb_pixel>> train_batch_samples, test_batch_samples;
std::vector<std::vector<dlib::mmod_rect>> train_batch_labels, test_batch_labels;

std::string version;
std::string net_name = "obj_det_net_";
std::string net_sync_name = "obj_det_sync_";
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
    net_sync_name = version + "_sync";
    logfileName = version + "_log_";
    net_name = version +  "_final_net.dat";
}

// ----------------------------------------------------------------------------------------

void print_usage(void)
{
    std::cout << "The wrong number of parameters was entered..." << std::endl;
    std::cout << "Enter the following as arguments into the program:" << std::endl;
    std::cout << "<training_input_filename> " << std::endl;
    std::cout << endl;
}

// ----------------------------------------------------------------------------------------

int main(int argc, char** argv)
{

    uint64_t idx = 0, jdx = 0;
    uint8_t HPC = 0;
    std::string sdate, stime;

    // data IO variables
    const std::string os_file_sep = "/";
    std::string program_root;
    std::string save_directory;
    std::string sync_save_location;
    std::string image_save_location;
    std::string results_save_location;
    std::string train_inputfile;
    std::string test_inputfile;
    std::pair<std::string, uint8_t> train_input, test_input;
    std::string train_data_directory, test_data_directory;
    std::vector<std::vector<std::string>> training_file;
    std::vector<std::vector<std::string>> test_file;
    std::vector<std::string> tr_image_files, te_image_files;
    std::ofstream data_log_stream;

    // timing variables
    typedef std::chrono::duration<double> d_sec;
    auto start_time = chrono::system_clock::now();
    auto stop_time = chrono::system_clock::now();
    auto elapsed_time = chrono::duration_cast<d_sec>(stop_time - start_time);

    // training variables
    int32_t stop = -1;
    std::vector<std::string> stop_codes = { "Minimum Learning Rate Reached.", "Max Training Time Reached", "Max Training Steps Reached" };
    std::vector<double> stop_criteria;
    training_params tp;
    std::vector<float> avg_color;
    std::vector<uint32_t> filter_num;

    crop_info ci;

    std::pair<uint32_t, uint32_t> target_size;  // min_target_size, max_target_size
    double min_window_overlap;

    std::vector<int32_t> gpu;
    uint64_t one_step_calls = 0;
    uint64_t epoch = 0;
    uint64_t index = 0;   
    
    double std = 3.0;

    dlib::rgb_pixel color;
    dlib::matrix<dlib::rgb_pixel> rgb_img;

#if !defined(DLIB_NO_GUI_SUPPORT)
    //create window to display images
    //dlib::image_window win;
#endif

    dlib::rand rnd;
    rnd = dlib::rand(time(NULL));

    // set the learning rate multipliers: 0 means freeze the layers; r1 = learning rate multiplier, r2 = learning rate bias multiplier
    //double r1 = 1.0, r2 = 1.0;
    
    // ----------------------------------------------------------------------------------------   
    if (argc == 1)
    {
        print_usage();
        std::cin.ignore();
        return 0;
    }

    std::string parse_filename = argv[1];

    // parse through the supplied csv file
    parse_input_file(parse_filename, version, gpu, stop_criteria, tp, train_input, test_input, ci, \
                     target_size, min_window_overlap, avg_color, filter_num, save_directory);

    // check the platform
    get_platform_control();

    // check for HPC <- set the environment variable PLATFORM to HPC
    if(platform.compare(0,3,"HPC") == 0)
    {
        std::cout << "HPC Platform Detected." << std::endl;
        HPC = 1;
    }

    // setup save variable locations
#if defined(_WIN32) | defined(__WIN32__) | defined(__WIN32) | defined(_WIN64) | defined(__WIN64)
    //program_root = get_path(get_path(get_path(std::string(argv[0]), "\\"), "\\"), "\\") + os_file_sep;
    sync_save_location = save_directory + "nets/";
    results_save_location = save_directory + "results/";
    image_save_location = save_directory + "result_images/";

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

    sync_save_location = save_directory + "nets/";
    results_save_location = save_directory + "results/";
    image_save_location = save_directory + "result_images/";

#endif

    std::cout << "Reading Inputs... " << std::endl;
    std::cout << "Platform:              " << platform << std::endl;
    std::cout << "GPU:                   { ";
    for (idx = 0; idx < gpu.size(); ++idx)
        std::cout << gpu[idx] << " ";
    std::cout << "}" << std::endl;
    //std::cout << "program_root:          " << program_root << std::endl;
    std::cout << "save_directory:        " << save_directory << std::endl;
    std::cout << "sync_save_location:    " << sync_save_location << std::endl;
    std::cout << "results_save_location: " << results_save_location << std::endl;
    std::cout << "image_save_location:   " << image_save_location << std::endl;


    try {

        get_current_time(sdate, stime);
        logfileName = logfileName + sdate + "_" + stime + ".txt";
        //cropper_stats_file = output_save_location + "cr_stats_" + version + "_" + sdate + "_" + stime + ".txt";

        std::cout << "Log File:              " << (results_save_location + logfileName) << std::endl << std::endl;
        data_log_stream.open((results_save_location + logfileName), ios::out | ios::app);

        // Add the date and time to the start of the log file
        data_log_stream << "------------------------------------------------------------------" << std::endl;
        data_log_stream << "Version: 2.0    Date: " << sdate << "    Time: " << stime << std::endl;
        data_log_stream << "Platform: " << platform << std::endl;
        data_log_stream << "GPU: { ";
        for (idx = 0; idx < gpu.size(); ++idx)
            data_log_stream << gpu[idx] << " ";
        data_log_stream << "}" << std::endl << std::endl;

//-----------------------------------------------------------------------------
// Read in the training and testing images
//-----------------------------------------------------------------------------

        // parse through the supplied training input file
        parse_group_csv_file(train_input.first, '{', '}', training_file);

        if (training_file.size() == 0)
        {
            std::cout << train_input.first << ": ";
            throw std::runtime_error("Training file is empty");
        }

        // the first line in this file is now the data directory
#if defined(_WIN32) | defined(__WIN32__) | defined(__WIN32) | defined(_WIN64) | defined(__WIN64)
        train_data_directory = training_file[0][0];
#else
        if (HPC == 1)
        {
            train_data_directory = training_file[0][2];
        }
        else
        {
            train_data_directory = training_file[0][1];
        }
#endif

        training_file.erase(training_file.begin());

        std::cout << "------------------------------------------------------------------" << std::endl;
        std::cout << "data_directory:        " << train_data_directory << std::endl;

        std::cout << train_input.first << std::endl;
        std::cout << "Training image sets to parse: " << training_file.size() << std::endl;

        data_log_stream << "------------------------------------------------------------------" << std::endl;
        data_log_stream << train_input.first << std::endl;
        data_log_stream << "Training image sets to parse: " << training_file.size() << std::endl;
        
        std::cout << "Loading training images... ";

        // load in the images and labels
        start_time = chrono::system_clock::now();

        load_rgb_data(training_file, train_data_directory, train_images, train_labels, tr_image_files, 1);


        // this is placeholder code for selecting a specific class, removing all other classes
        //std::string class_name = "test2";
        //uint32_t img_size = train_images.size() - 1;
        //for (int32_t idx = img_size; idx >=0 ; --idx)
        //{
        //    uint32_t label_size = train_labels[idx].size() - 1;
        //    for (int32_t jdx = label_size; jdx >=0 ; --jdx)
        //    {
        //        if (train_labels[idx][jdx].label != class_name)
        //        {
        //            train_labels[idx].erase(train_labels[idx].begin() + jdx);
        //        }
        //    }

        //    if (train_labels[idx].size() == 0)
        //    {
        //        train_labels.erase(train_labels.begin() + idx);
        //        train_images.erase(train_images.begin() + idx);
        //        tr_image_files.erase(tr_image_files.begin() + idx);
        //    }
        //}

        stop_time = chrono::system_clock::now();
        elapsed_time = chrono::duration_cast<d_sec>(stop_time - start_time);

        std::cout << "Loaded " << train_images.size() << " training image sets in " << elapsed_time.count() / 60 << " minutes." << std::endl;
        data_log_stream << "Loaded " << train_images.size() << " training image sets in " << elapsed_time.count() / 60 << " minutes." << std::endl << std::endl;
        
        //data_log_stream << "the following objects were ignored: " << std::endl << std::endl;

        //int num_ignored_train_images = 0;
        //int num_found_train_images = 0;

        //for (int idx = 0; idx < train_labels.size(); ++idx) 
        //{
        //    for (int jdx = 0; jdx < train_labels[idx].size(); ++jdx) 
        //    {
        //        if (train_labels[idx][jdx].ignore == true) 
        //        {
        //            ++num_ignored_train_images;
        //            data_log_stream << training_file[idx][1] << " " << training_file[idx][6] << std::endl;
        //        }
        //        else 
        //        {
        //            ++num_found_train_images;
        //        }
        //    }
        //}

        //std::cout << "Number of Found Train Objects: " << num_found_train_images << std::endl;
        //std::cout << "Number of Ignored Train Objects : " << num_ignored_train_images << std::endl;
        //std::cout << "------------------------------------------------------------------" << std::endl << std::endl;

        //data_log_stream << "Number of Found Train Objects: " << num_found_train_images << std::endl;
        //data_log_stream << "Number of Ignored Train Objects: " << num_ignored_train_images << std::endl<<std::endl;

        // for debugging to view the images
        //for (idx = 0; idx < training_file.size(); ++idx)
        //{   

        //    win.clear_overlay();
        //    win.set_image(train_images[idx]);

        //    for (jdx = 0; jdx < train_labels[idx].size(); ++jdx)
        //    {
        //        color = train_labels[idx][jdx].ignore ? dlib::rgb_pixel(0, 0, 255) : dlib::rgb_pixel(0, 255, 0);
        //        win.add_overlay(train_labels[idx][jdx].rect, color);
        //    }

        //    win.set_title(("Training Image: " + num2str(idx+1,"%05u")));

        //    //std::cin.ignore();
        //    dlib::sleep(500);
        //}


        //-------------------------------------------------------------------------------------
        // parse through the supplied test input file
        parse_group_csv_file(test_input.first, '{', '}', test_file);

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
        std::cout << std::endl << "------------------------------------------------------------------" << std::endl;
        std::cout << "data_directory:        " << test_data_directory << std::endl;
        std::cout << test_input.first << std::endl;
        std::cout << "Test image sets to parse: " << test_file.size() << std::endl;

        data_log_stream << "------------------------------------------------------------------" << std::endl;
        data_log_stream << test_input.first << std::endl;
        data_log_stream << "Test image sets to parse: " << test_file.size() << std::endl;
       
        std::cout << "Loading test images... ";

        // load in the images and labels
        start_time = chrono::system_clock::now();
        load_rgb_data(test_file, test_data_directory, test_images, test_labels, te_image_files, 1);

        //load_data(test_file, test_data_directory, test_images, test_labels, te_image_files);
        stop_time = chrono::system_clock::now();
        elapsed_time = chrono::duration_cast<d_sec>(stop_time - start_time);

        std::cout << "Loaded " << test_images.size() << " test image sets in " << elapsed_time.count() / 60 << " minutes." << std::endl;
        data_log_stream << "Loaded " << test_images.size() << " test image sets in " << elapsed_time.count() / 60 << " minutes." << std::endl << std::endl;
        
        //data_log_stream << "the following objects were ignored: " << std::endl << std::endl;

        //for (int idx = 0; idx < test_labels.size(); ++idx) 
        //{
        //    for (int jdx = 0; jdx < test_labels[idx].size(); ++jdx) 
        //    {
        //        if (test_labels[idx][jdx].ignore == true) 
        //        {
        //            ++num_ignored_test_images;
        //            data_log_stream << test_file[idx][1] << " " << test_file[idx][6] << std::endl;
        //        }
        //        else 
        //        {
        //            ++num_found_test_images;
        //        }
        //    }
        //}

        //std::cout << "Number of Found Test Objects : " << num_found_test_images << std::endl;
        //std::cout << "Number of Ignored Test Objects : " << num_ignored_test_images << std::endl;
        //std::cout << "------------------------------------------------------------------" << std::endl << std::endl;

        //data_log_stream << "Number of Found Test Objects : " << num_found_test_images << std::endl;
        //data_log_stream << "Number of Ignored Test Objects: " << num_ignored_test_images << std::endl << std::endl;
        //data_log_stream << "------------------------------------------------------------------" << std::endl<<std::endl;

        // ------------------------------------------------------------------------------------

        // for debugging to view the images
        //for (idx = 0; idx < test_images.size(); ++idx)
        //{

        //    win.clear_overlay();
        //    win.set_image(test_images[idx]);

        //    for (jdx = 0; jdx < test_labels[idx].size(); ++jdx)
        //    {
        //        color = test_labels[idx][jdx].ignore ? dlib::rgb_pixel(0, 0, 255) : dlib::rgb_pixel(0, 255, 0);
        //        win.add_overlay(test_labels[idx][jdx].rect, color);
        //    }

        //    win.set_title(("Training Image: " + num2str(idx+1,"%05u")));

        //    std::cin.ignore();
        //    //dlib::sleep(800);
        //}

//-----------------------------------------------------------------------------
// Setup the network
//-----------------------------------------------------------------------------

        // this sets th GPUs to use algorithms that are smaller in memory but may take a little longer to execute
        dlib::set_dnn_prefer_smallest_algorithms();

        // set the cuda device explicitly
        if (gpu.size() == 1)
            dlib::cuda::set_device(gpu[0]);

        // For further details see the mmod_options documentation.
        dlib::mmod_options options(train_labels, target_size.second, target_size.first, min_window_overlap);

        // example of how to push back a custion window
        // options.detector_windows.push_back(dlib::mmod_options::detector_window_details(114, 103));

        options.loss_per_false_alarm = 1.0;
        options.loss_per_missed_target = 2.0;
        options.truth_match_iou_threshold = 0.40;
        options.overlaps_nms = dlib::test_box_overlap(0.4, 1.0);

        std::cout << std::endl << "------------------------------------------------------------------" << std::endl;
        data_log_stream << std::endl << "------------------------------------------------------------------" << std::endl;

        std::cout << "num detector windows: " << options.detector_windows.size() << std::endl;
        data_log_stream << "num detector windows: " << options.detector_windows.size() << std::endl;

        std::cout << "bounding box configuration (min, max, overlap): " << target_size.first << ", " << target_size.second << ", " << min_window_overlap << std::endl;;
        data_log_stream << "bounding box configuration (min, max, overlap): " << target_size.first << ", " << target_size.second << ", " << min_window_overlap << std::endl;;

        std::set<std::string> tmp_names;

        for (auto& w : options.detector_windows)
        {
            std::cout << "detector window (w x h): " << w.label << " - " << w.width << " x " << w.height << std::endl;
            data_log_stream << "detector window (w x h): " << w.label << " - " << w.width << " x " << w.height << std::endl;
            tmp_names.insert(w.label);
        }
        
        std::vector<std::string> class_names(tmp_names.begin(), tmp_names.end());
        uint32_t num_classes = class_names.size();

        std::vector<dlib::rgb_pixel> class_color;
        for (idx = 0; idx < num_classes; ++idx)
        {
            class_color.push_back(dlib::rgb_pixel(rnd.get_random_8bit_number(), rnd.get_random_8bit_number(), rnd.get_random_8bit_number()));
        }
        
        std::cout << "overlap NMS IOU thresh:             " << options.overlaps_nms.get_iou_thresh() << std::endl;
        std::cout << "overlap NMS percent covered thresh: " << options.overlaps_nms.get_percent_covered_thresh() << std::endl;

        data_log_stream << "overlap NMS IOU thresh:             " << options.overlaps_nms.get_iou_thresh() << std::endl;
        data_log_stream << "overlap NMS percent covered thresh: " << options.overlaps_nms.get_percent_covered_thresh() << std::endl;
        data_log_stream << std::endl << "------------------------------------------------------------------"  << std::endl;


        // ------------------------------------------------------------------------------------
        /*
        for (idx = 0; idx < train_images.size(); ++idx)
        {
            merge_channels(train_images[idx], rgb_img);
            win.clear_overlay();

            for (jdx = 0; jdx < train_labels[idx].size(); ++jdx)
            {
                auto& class_index = std::find(class_names.begin(), class_names.end(), train_labels[idx][jdx].label);
                dlib_overlay_bbox(rgb_img, train_labels[idx][jdx], class_color[std::distance(class_names.begin(), class_index)]);
            }
            win.set_image(rgb_img);
            //dlib::sleep(800);
            std::cin.ignore();
        }
        */
        // ------------------------------------------------------------------------------------

        // Now we are ready to create our network and trainer.
        net_type net = config_net<net_type>(options, avg_color, filter_num);

        dlib::dnn_trainer<net_type, dlib::adam> trainer(net, dlib::adam(0.0001, 0.9, 0.99),  gpu);
        trainer.set_learning_rate(tp.intial_learning_rate);
        trainer.be_verbose();
        trainer.set_synchronization_file((sync_save_location + net_sync_name), std::chrono::minutes(5));
        trainer.set_iterations_without_progress_threshold(tp.steps_wo_progess);
        trainer.set_test_iterations_without_progress_threshold(5000);
        trainer.set_learning_rate_shrink_factor(tp.learning_rate_shrink_factor);

        // set the batch normalization stats window to something big
        dlib::set_all_bn_running_stats_window_sizes(net, 1000);

        dlib::random_cropper cropper;

        cropper.set_seed(time(NULL));

        // set the rows, cols for the cropped image size
        cropper.set_chip_dims(ci.crop_height, ci.crop_width);

        // Usually you want to give the cropper whatever min sizes you passed to the
        // mmod_options constructor, which is what we do here.
        cropper.set_min_object_size(target_size.second+2, target_size.first+2);

        cropper.set_max_object_size(1.0);   // 0.8

        // percetange of crops that don't contain an object of interest
        cropper.set_background_crops_fraction(0.4);

        // randomly flip left-right
        cropper.set_randomly_flip(true);

        // maximum allowed rotation +/-
        cropper.set_max_rotation_degrees(ci.angle);

        // set the cropper stats recorder
        //cropper.set_stats_filename(cropper_stats_file);

        // display a few hits and also save them to the log file for later analysis
        std::cout << std::endl << "------------------------------------------------------------------" << std::endl;
        std::cout << "Crop count: " << ci.crop_num << std::endl;
        data_log_stream << "Crop count: " << ci.crop_num << std::endl;

        // show all of the cropper settings
        std::cout << cropper << std::endl;
        data_log_stream << cropper << std::endl;
        data_log_stream << "------------------------------------------------------------------" << std::endl;

        // show all of the trainer settings
        std::cout << "------------------------------------------------------------------" << std::endl;
        std::cout << trainer << std::endl;
        data_log_stream << trainer << std::endl;
        data_log_stream << "------------------------------------------------------------------" << std::endl;

        // show the network to verify that it looks correct
        std::cout << "------------------------------------------------------------------" << std::endl;
        std::cout << "Net Name: " << net_name << std::endl;
        std::cout << net << std::endl;

        data_log_stream << "Net Name: " << net_name << std::endl;
        data_log_stream << net << std::endl;

//-----------------------------------------------------------------------------
// TRAINING START
//-----------------------------------------------------------------------------

        // these two matrices will contain the results of the training and testing
        dlib::matrix<double, 1, 6> training_results = dlib::zeros_matrix<double>(1, 6);
        dlib::matrix<double, 1, 6> test_results = dlib::zeros_matrix<double>(1, 6);
        std::vector<label_stats> train_label_stats(num_classes, label_stats(0,0));
        std::vector<label_stats> test_label_stats(num_classes, label_stats(0,0));
        
        uint64_t test_step_count = 2000;

        std::cout << "------------------------------------------------------------------" << std::endl;
        std::cout << "Starting Training..." << std::endl;
        start_time = chrono::system_clock::now();

        while(stop < 0)
        {
            // first check to make sure that the final_learning_rate hasn't been exceeded
            if (trainer.get_learning_rate() >= tp.final_learning_rate)
            {
                //cropper.file_append(num_crops, train_data_directory, training_file, mini_batch_samples, mini_batch_labels);
                cropper(ci.crop_num, train_images, train_labels, train_batch_samples, train_batch_labels);

                // apply some noise to the image
                for (auto&& tc : train_batch_samples)
                {
                    apply_poisson_noise(tc, std, rnd, (uint8_t)0, (uint8_t)255);
                    //swap_channels(tc, rnd);
                }

#if defined(_DEBUG)
/*                
                for (idx = 0; idx < train_batch_samples.size(); ++idx)
                {

                    merge_channels(train_batch_samples[idx], rgb_img);
                    win.clear_overlay();
                    win.set_image(rgb_img);

                    for (jdx = 0; jdx < train_batch_labels[idx].size(); ++jdx)
                    {
                        color = train_batch_labels[idx][jdx].ignore ? dlib::rgb_pixel(0, 0, 255) : dlib::rgb_pixel(0, 255, 0);
                        win.add_overlay(train_batch_labels[idx][jdx].rect, color);
                    }
                    std::cin.ignore();
                }
*/
#endif

                trainer.train_one_step(train_batch_samples, train_batch_labels);

            }
            else
            {
                stop = 0;
            }

            one_step_calls = trainer.get_train_one_step_calls();

            if((one_step_calls % test_step_count) == 0)
            {
                // this is where we will perform any needed evaluations of the network
                // detction_accuracy, correct_hits, false_positives, missing_detections

                cropper(ci.crop_num, test_images, test_labels, test_batch_samples, test_batch_labels);

                trainer.test_one_step(test_batch_samples, test_batch_labels);
/*
                test_results = dlib::zeros_matrix<double>(1, 6);

                for (idx = 0; idx < test_file.size(); ++idx)
                {    
                    test_label.clear();
                    load_single_set(test_data_directory, test_file[idx], test_image, test_label);

                    merge_channels(test_image, rgb_img);
                    //std::cout << te_image_files[idx].first;
                    //win.clear_overlay();
                    //win.set_image(rgb_img);
                    v_win[idx].clear_overlay();
                    v_win[idx].set_image(rgb_img);
                 
                    //v_win[idx].clear_overlay();
                    //v_win[idx].set_image(tmp_img);

                    std::vector<dlib::mmod_rect> dnn_labels;           

                    // get the rough classification time per image
                    start_time = chrono::system_clock::now();
                    dlib::matrix<double, 1, 6> tr = eval_net_performance(net, test_image, test_label, dnn_labels, min_target_size, fda_test_box_overlap(0.3, 1.0));
                    stop_time = chrono::system_clock::now();

                    elapsed_time = chrono::duration_cast<d_sec>(stop_time - start_time);
                    dnn_test_labels.push_back(dnn_labels);

                    //overlay the dnn detections on the image
                    for (jdx = 0; jdx < dnn_labels.size(); ++jdx)
                    {
                    v_win[idx].add_overlay(dnn_labels[jdx].rect, dlib::rgb_pixel(255, 0, 0));
                    //draw_rectangle(tmp_img, dnn_labels[jdx].rect, dlib::rgb_pixel(255, 0, 0), 2);
                    //data_log_stream << "Detect Confidence Level (" << dnn_test_labels[idx][jdx].label << "): " << dnn_test_labels[idx][jdx].detection_confidence << std::endl;
                    //std::cout << "Detect Confidence Level (" << dnn_test_labels[idx][jdx].label << "): " << dnn_test_labels[idx][jdx].detection_confidence << std::endl;
                    }
                    
                    std::cout << ".";
                    // overlay the ground truth boxes on the image
                    for (jdx = 0; jdx < test_label.size(); ++jdx)
                    {
                        v_win[idx].add_overlay(test_label[jdx].rect, dlib::rgb_pixel(0, 255, 0));
                        draw_rectangle(rgb_img, test_label[jdx].rect, dlib::rgb_pixel(0, 255, 0), 2);
                    }
                    
                    //save results in image form
                    //std::string image_save_name = output_save_location + "test_save_image_" + version + num2str(idx, "_%03d.png");
                    //save_png(rgb_img, image_save_name);
                    std::cout << std::endl;

                    test_results += tr;
                }

                test_results(0, 0) = test_results(0, 0) / (double)test_file.size();

                std::cout << "------------------------------------------------------------------" << std::endl;
                std::cout << "Results (DA, CH, FP, MD): " << std::fixed << std::setprecision(4) << test_results(0, 0) << ", " << test_results(0, 3) << ", " << test_results(0, 4) << ", " << test_results(0, 5) << std::endl;
                std::cout << "------------------------------------------------------------------" << std::endl;
*/                

                data_log_stream << std::setw(6) << std::setfill('0') << one_step_calls << ", " << std::fixed << std::setprecision(9) << trainer.get_learning_rate() << ", ";
                data_log_stream << std::setprecision(5) << trainer.get_average_loss() << ", " << trainer.get_average_test_loss() << std::endl;

            }

            // now check to see if we've trained long enough according to the input time limit
            stop_time = chrono::system_clock::now();
            elapsed_time = chrono::duration_cast<d_sec>(stop_time - start_time);

            if((double)elapsed_time.count()/(double)3600.0 > stop_criteria[0])
            {
                stop = 1;
            }

            // finally check to see if we've exceeded the max number of one step training calls
            // according to the input file
            if (one_step_calls >= stop_criteria[1])
            {
                stop = 2;
            }

        }   // end of while(stop<0)


//-----------------------------------------------------------------------------
// TRAINING STOP
//-----------------------------------------------------------------------------

        stop_time = chrono::system_clock::now();
        elapsed_time = chrono::duration_cast<d_sec>(stop_time - start_time);

        // wait for training threads to stop
        trainer.get_net();

        std::cout << std::endl << "------------------------------------------------------------------" << std::endl;
        std::cout << "Elapsed Training Time: " << elapsed_time.count() / 3600 << " hours" << std::endl;
        std::cout << "Stop Code: " << stop_codes[stop] << std::endl;
        std::cout << "Final Average Loss: " << trainer.get_average_loss() << std::endl;
        std::cout << "------------------------------------------------------------------" << std::endl << std::endl;

        data_log_stream << "------------------------------------------------------------------" << std::endl;
        data_log_stream << "Elapsed Training Time: " << elapsed_time.count() / 3600 << " hours" << std::endl;
        data_log_stream << "Stop Code: " << stop_codes[stop] << std::endl;
        data_log_stream << "Final Average Loss: " << trainer.get_average_loss() << std::endl;
        data_log_stream << "------------------------------------------------------------------" << std::endl << std::endl;

        // Save the network to disk
        net.clean();
        dlib::serialize(sync_save_location + net_name) << net;

//-----------------------------------------------------------------------------
//  EVALUATE THE FINAL NETWORK PERFORMANCE
//-----------------------------------------------------------------------------

        // load the network from the saved file
        anet_type test_net;

        std::cout << std::endl << "Loading " << (sync_save_location + net_name) << std::endl;
        dlib::deserialize(sync_save_location + net_name) >> test_net;

        //---------------------------------------------------------------------------------------
        // In this section we want to evaluate the performance of the network against the training data
        // this should be displayed and then saved to the log file
        // - This can also include displaying the input image along with the ground truth bounding box, name and dnn results       
        std::cout << "Analyzing Training Results..." << std::endl;

        training_results = run_net_performace(data_log_stream, test_net, train_images, train_labels, class_names, class_color, tr_image_files, image_save_location);

        //---------------------------------------------------------------------------------------
        // In this section we want to evaluate the performance of the network against the test data
        // this should be displayed and then saved to the log file
        // - This can also include displaying the input image along with the ground truth bounding box, name and dnn results
        std::cout << "Analyzing Test Results..." << std::endl;

        test_results = run_net_performace(data_log_stream, test_net, test_images, test_labels, class_names, class_color, te_image_files, image_save_location);

        //---------------------------------------------------------------------------------------
        // output the final results to the std out
        std::cout << "-------------------------------------------------------------------------------" << std::endl;
        std::cout << "Training Results (detction_accuracy, correct_hits, false_positives, missing_detections): " << std::fixed << std::setprecision(4) << training_results(0, 0);
        std::cout << ", " << training_results(0, 3) << ", " << training_results(0, 1) << ", " << training_results(0, 4) << ", " << training_results(0, 5) << std::endl;
        std::cout << "-------------------------------------------------------------------------------" << std::endl << std::endl;

        std::cout << "-------------------------------------------------------------------------------" << std::endl;
        std::cout << "Testing Results (detction_accuracy, correct_detects, false_positives, missing_detections):  " << std::fixed << std::setprecision(4) << test_results(0, 0);
        std::cout << ", " << test_results(0, 3) << ", " << test_results(0, 1) << ", " << test_results(0, 4) << ", " << test_results(0, 5) << std::endl;
        std::cout << "-------------------------------------------------------------------------------" << std::endl;

        // save the results to the log file
        data_log_stream << "-------------------------------------------------------------------------------" << std::endl;
        data_log_stream << "Training Results (detction_accuracy, correct_detects, false_positives, missing_detections): " << std::fixed << std::setprecision(4) << training_results(0, 0);
        data_log_stream << ", " << training_results(0, 3) << ", " << training_results(0, 1) << ", " << training_results(0, 4) << ", " << training_results(0, 5) << std::endl;
        data_log_stream << "-------------------------------------------------------------------------------" << std::endl << std::endl;

        data_log_stream << "-------------------------------------------------------------------------------" << std::endl;
        data_log_stream << "Testing Results (detction_accuracy, correct_detects, false_positives, missing_detections):  " << std::fixed << std::setprecision(4) << test_results(0, 0);
        data_log_stream << ", " << test_results(0, 3) << ", " << test_results(0, 1) << ", " << test_results(0, 4) << ", " << test_results(0, 5) << std::endl;
        data_log_stream << "-------------------------------------------------------------------------------" << std::endl;

        //------------------------------------------------------------------------------------------
        if (HPC == 1)
        {
            std::cout << "-------------------------------------------------------------------------------" << std::endl;
            std::cout << std::fixed << std::setprecision(4) << training_results(0, 0) << ", " << training_results(0, 3) << ", " << training_results(0, 1) << ", " << training_results(0, 4) << ", " << training_results(0, 5) << ", ";
            std::cout << std::fixed << std::setprecision(4) << test_results(0, 0) << ", " << test_results(0, 3) << ", " << test_results(0, 1) << ", " << test_results(0, 4) << ", " << test_results(0, 5) << std::endl;
            std::cout << "-------------------------------------------------------------------------------" << std::endl;
            
            data_log_stream << "-------------------------------------------------------------------------------" << std::endl;
            data_log_stream << std::fixed << std::setprecision(4) << training_results(0, 0) << ", " << training_results(0, 3) << ", " << training_results(0, 1) << ", " << training_results(0, 4) << ", " << training_results(0, 5) << ", ";
            data_log_stream << std::fixed << std::setprecision(4) << test_results(0, 0) << ", " << test_results(0, 3) << ", " << test_results(0, 1) << ", " << test_results(0, 4) << ", " << test_results(0, 5) << std::endl;
            data_log_stream << "-------------------------------------------------------------------------------" << std::endl;
        }

        //------------------------------------------------------------------------------------------
        std::cout << std::endl << "End of Program." << std::endl;
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
