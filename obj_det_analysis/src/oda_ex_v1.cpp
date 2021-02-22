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
#include "obj_detector.h"
#include "get_platform.h"
#include "file_ops.h"
#include "get_current_time.h"
#include "num2string.h"
#include "dlib_overlay_bbox.h"

// Net Version
#include "obj_det_net_v13.h"
//#include "tfd_net_v03.h"
#include "load_data.h"
#include "run_network_performance.h"
//#include "eval_net_performance.h"
//#include "enhanced_array_cropper.h"
//#include "random_array_cropper.h"
//#include "random_channel_swap.h"
//#include "enhanced_channel_swap.h"

#include "array_image_operations.h"


// dlib includes
#include <dlib/dnn.h>
#include <dlib/image_io.h>
#include <dlib/data_io.h>
#include <dlib/gui_widgets.h>
#include <dlib/image_transforms.h>
#include <dlib/rand.h>

// -------------------------------GLOBALS--------------------------------------
extern const uint32_t array_depth;

std::string platform;

//this will store the standard RGB images
std::vector<std::array<dlib::matrix<uint8_t>, array_depth>> test_images;

// this will store the ground truth data for the bounding box labels
std::vector<std::vector<dlib::mmod_rect>> test_labels;

std::string version;
std::string logfileName = "oda_log_";

// ----------------------------------------------------------------------------
void get_platform_control(void)
{
	get_platform(platform);

	if (platform == "")
	{
		std::cout << "No Platform could be identified... defaulting to Windows." << std::endl;
		platform = "Win";
	}

	logfileName = logfileName + version;
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
	std::string sdate, stime;

    // data IO variables
	const std::string os_file_sep = "/";
	std::string program_root;
    //std::string &data_file;
    std::string network_weights_file;
	std::string image_save_location;
	std::string results_save_location;
	std::string test_inputfile;
	std::string test_data_directory;
	std::vector<std::vector<std::string>> test_file;
	std::vector<std::string> te_image_files;
	std::ofstream data_log_stream;

    // timing variables
	typedef std::chrono::duration<double> d_sec;
	auto start_time = chrono::system_clock::now();
	auto stop_time = chrono::system_clock::now();
	auto elapsed_time = chrono::duration_cast<d_sec>(stop_time - start_time);

    std::pair<uint32_t, uint32_t> target_size(45, 100);
 
    //create window to display images
    //dlib::image_window win;
    //dlib::rgb_pixel color;
    //dlib::matrix<dlib::rgb_pixel> rgb_img;

    dlib::rand rnd;
    rnd = dlib::rand(time(NULL));

    //uint64_t missing_detections = 0;
    //uint64_t false_positives = 0;
    //uint64_t match_count = 0;
    //uint64_t num_gt = 0;
    //double acc = 0.0;

    // ---------------------------------------------------------------------------------------- 
	if (argc == 1)
	{
		print_usage();
		std::cin.ignore();
		return 0;
	}

	std::string parse_filename = argv[1];

	// parse through the supplied csv file
	parse_input_file(parse_filename, test_inputfile, network_weights_file, version, results_save_location);

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
	program_root = get_path(get_path(get_path(std::string(argv[0]), "\\"), "\\"), "\\") + os_file_sep;

#else
	if (HPC == 1)
	{
		//HPC version
		program_root = get_path(get_path(get_path(std::string(argv[0]), os_file_sep), os_file_sep), os_file_sep) + os_file_sep;
	}
	else
	{
        program_root = get_ubuntu_path();
	}

	//sync_save_location = program_root + "nets/";
	//results_save_location = program_root + "results/";

#endif
	image_save_location = results_save_location;

	std::cout << "Reading Inputs... " << std::endl;
	std::cout << "Platform:              " << platform << std::endl;
	std::cout << "program_root:          " << program_root << std::endl;
	std::cout << "results_save_location: " << results_save_location << std::endl;
	std::cout << "image_save_location:   " << image_save_location << std::endl;

	try {

		get_current_time(sdate, stime);
		logfileName = logfileName + sdate + "_" + stime + ".txt";

		std::cout << "Log File:              " << (results_save_location + logfileName) << std::endl << std::endl;
		data_log_stream.open((results_save_location + logfileName), ios::out | ios::app);

		// Add the date and time to the start of the log file
		data_log_stream << "-------------------------------------------------------------------------------" << std::endl;
		data_log_stream << "Version: 2.0    Date: " << sdate << "    Time: " << stime << std::endl;
		data_log_stream << "Platform: " << platform << std::endl;
		data_log_stream << std::endl;

		///////////////////////////////////////////////////////////////////////////////
		// Step 1: Read in the test images
		///////////////////////////////////////////////////////////////////////////////

        parse_group_csv_file(test_inputfile, '{', '}', test_file);
        if (test_file.size() == 0)
        {
            throw std::runtime_error("The data input file is empty or unreadable (" + test_inputfile + ")");
        }

        // the data directory should be the first entry in the input file
#if defined(_WIN32) | defined(__WIN32__) | defined(__WIN32) | defined(_WIN64) | defined(__WIN64)
        test_data_directory = test_file[0][0];
#else
        if (HPC == 1)
        {
            test_data_directory = test_file[0][2];
        }
        else if (platform.compare(0,7,"SL02319") == 0)
        {
            test_data_directory = test_file[0][2];
        }
        else
        {
            test_data_directory = test_file[0][1];
        }
#endif

		test_file.erase(test_file.begin());
        std::cout << std::endl << "-------------------------------------------------------------------------------" << std::endl;
		std::cout << "data_directory:        " << test_data_directory << std::endl;
        std::cout << "test input file:       " << test_inputfile << std::endl;
		std::cout << "Test image sets to parse: " << test_file.size() << std::endl;

        data_log_stream << "-------------------------------------------------------------------------------" << std::endl;
        data_log_stream << test_inputfile << std::endl;
        data_log_stream << "Test image sets to parse: " << test_file.size() << std::endl;


        std::cout << "Loading test images... ";

		// load in the images and labels
        start_time = chrono::system_clock::now();
        load_data(test_file, test_data_directory, test_images, test_labels, te_image_files);
        stop_time = chrono::system_clock::now();
        elapsed_time = chrono::duration_cast<d_sec>(stop_time - start_time);

        std::cout << "Loaded " << test_images.size() << " test image sets in " << elapsed_time.count() / 60 << " minutes." << std::endl;
        data_log_stream << "Loaded " << test_images.size() << " test image sets in " << elapsed_time.count() / 60 << " minutes." << std::endl;

        ///////////////////////////////////////////////////////////////////////////////
        // Step 2: Setup the network
        ///////////////////////////////////////////////////////////////////////////////

        // this sets th GPUs to use algorithms that are smaller in memory but may take a little longer to execute
        dlib::set_dnn_prefer_smallest_algorithms();

        // load the network from the saved file
        anet_type test_net;

        std::cout << std::endl << "-------------------------------------------------------------------------------" << std::endl; 
        std::cout << "Loading network: " << (network_weights_file) << std::endl;
        dlib::deserialize(network_weights_file) >> test_net;

        // show the network to verify that it looks correct
        std::cout << std::endl << "-------------------------------------------------------------------------------" << std::endl;
        std::cout << test_net << std::endl;

        data_log_stream << std::endl << "-------------------------------------------------------------------------------" << std::endl;
        data_log_stream << network_weights_file << std::endl;
        data_log_stream << test_net << std::endl;
        data_log_stream << "-------------------------------------------------------------------------------" << std::endl;

        // get the details about the loss layer -> the number and names of the classes
        dlib::mmod_options options = dlib::layer<0>(test_net).loss_details().get_options();

        std::cout << std::endl << "-------------------------------------------------------------------------------" << std::endl;

        std::cout << "num detector windows: " << options.detector_windows.size() << std::endl;
        data_log_stream << "num detector windows: " << options.detector_windows.size() << std::endl;

        //std::cout << "bounding box configuration (min, max): " << target_size.first << ", " << target_size.second << std::endl;;
        //data_log_stream << "bounding box configuration (min, max): " << target_size.first << ", " << target_size.second << std::endl;;

        std::set<std::string> tmp_names;
        for (auto& w : options.detector_windows)
        {
            std::cout << "detector window (w x h): " << w.label << " - " << w.width << " x " << w.height << std::endl;
            data_log_stream << "detector window (w x h): " << w.label << " - " << w.width << " x " << w.height << std::endl;
            tmp_names.insert(w.label);
        }

        std::vector<std::string> class_names(tmp_names.begin(), tmp_names.end());
        uint32_t num_classes = class_names.size();
        //std::vector<label_stats> test_label_stats(num_classes, label_stats(0, 0, 0, 0));

        std::vector<dlib::rgb_pixel> class_color;
        for (idx = 0; idx < num_classes; ++idx)
        {
            //test_label_stats[idx].label = class_names[idx];
            class_color.push_back(dlib::rgb_pixel(rnd.get_random_8bit_number(), rnd.get_random_8bit_number(), rnd.get_random_8bit_number()));
        }

        std::cout << "overlap NMS IOU thresh:             " << options.overlaps_nms.get_iou_thresh() << std::endl;
        std::cout << "overlap NMS percent covered thresh: " << options.overlaps_nms.get_percent_covered_thresh() << std::endl;

        data_log_stream << "overlap NMS IOU thresh:             " << options.overlaps_nms.get_iou_thresh() << std::endl;
        data_log_stream << "overlap NMS percent covered thresh: " << options.overlaps_nms.get_percent_covered_thresh() << std::endl;
        data_log_stream << std::endl;

//-----------------------------------------------------------------------------
//          EVALUATE THE FINAL NETWORK PERFORMANCE
//-----------------------------------------------------------------------------
        std::cout << std::endl << "Analyzing Test Results..." << std::endl;

        dlib::matrix<double, 1, 6> test_results = run_net_performace(data_log_stream, test_net, test_images, test_labels, class_names, class_color, te_image_files, image_save_location);

/*
        // this matrix will contain the results of the training and testing
		dlib::matrix<double, 1, 6> test_results = dlib::zeros_matrix<double>(1, 6);

		for (idx = 0; idx < test_images.size(); ++idx)
        {
            std::vector<dlib::mmod_rect> dnn_labels;
            std::vector<label_stats> ls(num_classes, label_stats(0, 0, 0, 0));

            // get the rough classification time per image
            start_time = chrono::system_clock::now();
            dlib::matrix<double, 1, 6> tr = eval_net_performance(test_net, test_images[idx], test_labels[idx], dnn_labels, target_size.first, fda_test_box_overlap(0.4, 1.0), class_names, ls);
            stop_time = chrono::system_clock::now();

            elapsed_time = chrono::duration_cast<d_sec>(stop_time - start_time);

            std::cout << "------------------------------------------------------------------" << std::endl;
            std::cout << "Image " << std::right << std::setw(5) << std::setfill('0') << idx << ": " << te_image_files[idx] << std::endl;
            std::cout << "Image Size (h x w): " << test_images[idx][0].nr() << "x" << test_images[idx][0].nc() << std::endl;
            std::cout << "Classification Time (s): " << elapsed_time.count() << std::endl;
            //std::cout << "Results: " << std::fixed << std::setprecision(4) << tr(0, 0) << ", " << tr(0, 3) << ", " << tr(0, 4) << ", " << tr(0, 5) << std::endl;

            data_log_stream << "------------------------------------------------------------------" << std::endl;
            data_log_stream << "Image " << std::right << std::setw(5) << std::setfill('0') << idx << ": " << te_image_files[idx] << std::endl;
            data_log_stream << "Image Size (h x w): " << test_images[idx][0].nr() << "x" << test_images[idx][0].nc() << std::endl;
            data_log_stream << "Classification Time (s): " << elapsed_time.count() << std::endl;
            //data_log_stream << "Results: " << std::fixed << std::setprecision(4) << tr(0, 0) << ", " << tr(0, 3) << ", " << tr(0, 4) << ", " << tr(0, 5) << std::endl;

            for (jdx = 0; jdx < num_classes; ++jdx)
            {
                test_label_stats[jdx].count += ls[jdx].count;
                test_label_stats[jdx].match_count += ls[jdx].match_count;
                test_label_stats[jdx].false_positives += ls[jdx].false_positives;
                test_label_stats[jdx].missed_detects += ls[jdx].missed_detects;

                acc = (ls[jdx].count == 0) ? 0.0 : ls[jdx].match_count / (double)ls[jdx].count;
                std::cout << std::left << std::setw(15) << std::setfill(' ') << (class_names[jdx] + ":") << std::fixed << std::setprecision(4) << acc;
                std::cout << ", " << ls[jdx].match_count << ", " << ls[jdx].count << ", " << ls[jdx].false_positives << ", " << ls[jdx].missed_detects << std::endl;
                data_log_stream << std::left << std::setw(15) << std::setfill(' ') << (class_names[jdx] + ":") << std::fixed << std::setprecision(4) << acc;
                data_log_stream << ", " << ls[jdx].match_count << ", " << ls[jdx].count << ls[jdx].false_positives << ", " << ls[jdx].missed_detects << std::endl;
            }
            std::cout << std::left << std::setw(15) << std::setfill(' ') << "Results: " << std::fixed << std::setprecision(4) << tr(0, 0) << ", " << tr(0, 3) << ", " << tr(0,1) << ", " << tr(0, 4) << ", " << tr(0, 5) << std::endl;
            data_log_stream << std::left << std::setw(15) << std::setfill(' ') << "Results: " << std::fixed << std::setprecision(4) << tr(0, 0) << ", " << tr(0, 3) << ", " << tr(0, 1) << ", " << tr(0, 4) << ", " << tr(0, 5) << std::endl;

            if (array_depth < 3)
                dlib::assign_image(rgb_img, test_images[idx][0]);
            else
                merge_channels(test_images[idx], rgb_img);

#if !defined(DLIB_NO_GUI_SUPPORT)
            win.clear_overlay();

            //overlay the dnn detections on the image
            for (jdx = 0; jdx < dnn_labels.size(); ++jdx)
            {
                vector<string>::iterator class_index = std::find(class_names.begin(), class_names.end(), dnn_labels[jdx].label);
                dlib_overlay_bbox(rgb_img, dnn_labels[jdx], class_color[std::distance(class_names.begin(), class_index)]);

                data_log_stream << "Detect Confidence Level (" << dnn_labels[jdx].label << "): " << dnn_labels[jdx].detection_confidence << std::endl;
                std::cout << "Detect Confidence Level (" << dnn_labels[jdx].label << "): " << dnn_labels[jdx].detection_confidence << std::endl;
            }
            win.set_image(rgb_img);
#endif

            //save results to an image
            std::string image_save_name = image_save_location + "test_img_" + version + num2str(idx, "_%05d.png");
            //save_png(rgb_img, image_save_name);

            test_results += tr;
            dlib::sleep(100);
            //std::cin.ignore();

		}

        data_log_stream << "------------------------------------------------------------------" << std::endl << std::endl;
        std::cout << "------------------------------------------------------------------" << std::endl << std::endl;

        // output the combined test results
        std::cout << "------------------------------------------------------------------" << std::endl;
        std::cout << "class_name, detction_accuracy, correct_detects, groundtruth, false_positives, missing_detections" << std::endl;
        
        data_log_stream << "------------------------------------------------------------------" << std::endl;
        data_log_stream << "class_name, detction_accuracy, correct_detects, groundtruth, false_positives, missing_detections" << std::endl;

        match_count = 0;
        num_gt = 0;
        for (jdx = 0; jdx < num_classes; ++jdx)
        {
            acc = (test_label_stats[jdx].count == 0) ? 0.0 : test_label_stats[jdx].match_count / (double)test_label_stats[jdx].count;
            std::cout << std::left << std::setw(15) << std::setfill(' ') << (class_names[jdx] + ":") << std::fixed << std::setprecision(4) << acc << ", ";
            std::cout << test_label_stats[jdx].match_count << ", " << test_label_stats[jdx].count << ", " << test_label_stats[jdx].false_positives << ", " << test_label_stats[jdx].missed_detects << std::endl;

            data_log_stream << std::left << std::setw(15) << std::setfill(' ') << (class_names[jdx] + ":") << std::fixed << std::setprecision(4) << acc << ", ";
            data_log_stream << test_label_stats[jdx].match_count << ", " << test_label_stats[jdx].count << ", " << test_label_stats[jdx].false_positives << ", " << test_label_stats[jdx].missed_detects << std::endl;

            match_count += test_label_stats[jdx].match_count;
            num_gt += test_label_stats[jdx].count;
        }

        acc = (num_gt == 0) ? 0.0 : match_count / (double)num_gt;
*/

        std::cout << "-------------------------------------------------------------------------------" << std::endl;
        std::cout << "Testing Results (detction_accuracy, correct_detects, groundtruth, false_positives, missing_detections):  " << std::fixed << std::setprecision(4) << test_results(0, 0);
        std::cout << ", " << test_results(0, 3) << ", " << test_results(0, 1) << ", " << test_results(0, 4) << ", " << test_results(0, 5) << std::endl;
        std::cout << "-------------------------------------------------------------------------------" << std::endl;

        data_log_stream << "-------------------------------------------------------------------------------" << std::endl;
        data_log_stream << "Testing Results (detction_accuracy, correct_detects, groundtruth, false_positives, missing_detections):  " << std::fixed << std::setprecision(4) << test_results(0, 0);
        data_log_stream << ", " << test_results(0, 3) << ", " << test_results(0, 1) << ", " << test_results(0, 4) << ", " << test_results(0, 5) << std::endl;
        data_log_stream << "-------------------------------------------------------------------------------" << std::endl;



        std::cout << std::endl << "End of Program." << std::endl;
        data_log_stream.close();
        std::cin.ignore();
        
	}
	catch (std::exception& e)
	{
		std::cout << e.what() << std::endl;

		data_log_stream << std::endl;
		data_log_stream << "-------------------------------------------------------------------------------" << std::endl;
		data_log_stream << e.what() << std::endl;
		data_log_stream << "-------------------------------------------------------------------------------" << std::endl;
		data_log_stream.close();

		std::cout << "Press Enter to close..." << std::endl;
		std::cin.ignore();
	}

	return 0;

}
