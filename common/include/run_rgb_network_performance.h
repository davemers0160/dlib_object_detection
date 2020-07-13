#ifndef _RUN_NET_PERFORMANCE_
#define _RUN_NET_PERFORMANCE_

// C/C++ Includes
//#include <cstdlib>
#include <cstdint>
#include <iostream>
#include <iomanip>
#include <sstream>
#include <fstream>
#include <string>

// Custom Includes
#include "eval_net_performance.h"

// dlib Includes
#include <dlib/dnn.h>

//-----------------------------------------------------------------------------
template <typename net_type>
dlib::matrix<double, 1, 6> run_net_performace(std::ofstream &data_log_stream,
	net_type &net,
    std::vector<dlib::matrix<dlib::rgb_pixel>> &test_images,
    std::vector<std::vector<dlib::mmod_rect>> &test_labels,
    std::vector<std::string> class_names,
    std::vector<dlib::rgb_pixel> class_color,
    std::vector<std::string> te_image_files,
    std::string image_save_location
)
{
	uint64_t idx, jdx;

    uint64_t match_count = 0;
    uint64_t num_gt = 0;
    double acc = 0.0;

    // timing variables
    typedef std::chrono::duration<double> d_sec;
    auto start_time = chrono::system_clock::now();
    auto stop_time = chrono::system_clock::now();
    auto elapsed_time = chrono::duration_cast<d_sec>(stop_time - start_time);

#if !defined(DLIB_NO_GUI_SUPPORT)
    dlib::image_window win;
#endif

    dlib::matrix<dlib::rgb_pixel> rgb_img;

    uint32_t num_classes = class_names.size();
    std::vector<label_stats> test_label_stats(num_classes, label_stats(0, 0, 0, 0));

    for (idx = 0; idx < num_classes; ++idx)
    {
        test_label_stats[idx].label = class_names[idx];
    }

    // this matrix will contain the results of the training and testing
    dlib::matrix<double, 1, 6> test_results = dlib::zeros_matrix<double>(1, 6);

    for (idx = 0; idx < test_images.size(); ++idx)
    {
        std::vector<dlib::mmod_rect> dnn_labels;
        std::vector<label_stats> ls(num_classes, label_stats(0, 0, 0, 0));

        // get the rough classification time per image
        start_time = chrono::system_clock::now();
        dlib::matrix<double, 1, 6> tr = eval_net_performance(net, test_images[idx], test_labels[idx], dnn_labels, fda_test_box_overlap(0.4, 1.0), class_names, ls);
        stop_time = chrono::system_clock::now();

        elapsed_time = chrono::duration_cast<d_sec>(stop_time - start_time);

        std::cout << "-------------------------------------------------------------------------------" << std::endl;
        std::cout << "Image " << std::right << std::setw(5) << std::setfill('0') << idx << ": " << te_image_files[idx] << std::endl;
        std::cout << "Image Size (h x w): " << test_images[idx].nr() << "x" << test_images[idx].nc() << std::endl;
        std::cout << "Classification Time (s): " << elapsed_time.count() << std::endl;
        //std::cout << "Results: " << std::fixed << std::setprecision(4) << tr(0, 0) << ", " << tr(0, 3) << ", " << tr(0, 4) << ", " << tr(0, 5) << std::endl;

        data_log_stream << "-------------------------------------------------------------------------------" << std::endl;
        data_log_stream << "Image " << std::right << std::setw(5) << std::setfill('0') << idx << ": " << te_image_files[idx] << std::endl;
        data_log_stream << "Image Size (h x w): " << test_images[idx].nr() << "x" << test_images[idx].nc() << std::endl;
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
            data_log_stream << ", " << ls[jdx].match_count << ", " << ls[jdx].count << ", " << ls[jdx].false_positives << ", " << ls[jdx].missed_detects << std::endl;
        }
        std::cout << std::left << std::setw(15) << std::setfill(' ') << "Results: " << std::fixed << std::setprecision(4) << tr(0, 0) << ", " << tr(0, 3) << ", " << tr(0, 1) << ", " << tr(0, 4) << ", " << tr(0, 5) << std::endl;
        data_log_stream << std::left << std::setw(15) << std::setfill(' ') << "Results: " << std::fixed << std::setprecision(4) << tr(0, 0) << ", " << tr(0, 3) << ", " << tr(0, 1) << ", " << tr(0, 4) << ", " << tr(0, 5) << std::endl;

        dlib::assign_image(rgb_img, test_images[idx]);

#if !defined(DLIB_NO_GUI_SUPPORT)
        win.clear_overlay();

        //overlay the dnn detections on the image
        for (jdx = 0; jdx < dnn_labels.size(); ++jdx)
        {
            vector<string>::iterator class_index = std::find(class_names.begin(), class_names.end(), dnn_labels[jdx].label);
            overlay_bounding_box(rgb_img, dnn_labels[jdx], class_color[std::distance(class_names.begin(), class_index)]);

            data_log_stream << "Detect Confidence Level (" << dnn_labels[jdx].label << "): " << dnn_labels[jdx].detection_confidence << std::endl;
            std::cout << "Detect Confidence Level (" << dnn_labels[jdx].label << "): " << dnn_labels[jdx].detection_confidence << std::endl;
        }
        win.set_image(rgb_img);
#endif

        //save results to an image
        //std::string image_save_name = image_save_location + "test_img_" + version + num2str(idx, "_%05d.png");
        //save_png(rgb_img, image_save_name);

        test_results += tr;
        dlib::sleep(100);
        //std::cin.ignore();

    }

    data_log_stream << "-------------------------------------------------------------------------------" << std::endl << std::endl;
    std::cout << "-------------------------------------------------------------------------------" << std::endl << std::endl;

    // output the combined test results
    std::cout << "-------------------------------------------------------------------------------" << std::endl;
    std::cout << "class_name, detction_accuracy, correct_detects, groundtruth, false_positives, missing_detections" << std::endl;

    data_log_stream << "-------------------------------------------------------------------------------" << std::endl;
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

    std::cout << "-------------------------------------------------------------------------------" << std::endl << std::endl;
    data_log_stream << "-------------------------------------------------------------------------------" << std::endl << std::endl;

    test_results(0, 0) = (num_gt == 0) ? 0.0 : match_count / (double)num_gt;

    return test_results;

}	// end of run_net_performace

//-----------------------------------------------------------------------------

#endif	// _RUN_NET_PERFORMANCE_