#ifndef OBJ_DET_RUN_H_
#define OBJ_DET_RUN_H_

#include <cstdint>

// custom includes
#include "file_parser.h"

// ----------------------------------------------------------------------------------------
// The common function definitions that are in the dynamic library 
// ----------------------------------------------------------------------------------------
const int label_size = 256;

struct layer_struct
{
    unsigned int k;
    unsigned int n;
    unsigned int nr;
    unsigned int nc;
    unsigned int size;
};

// ----------------------------------------------------------------------------------------
struct window_struct
{
    unsigned int w;
    unsigned int h;
    char label[label_size];
};

// ----------------------------------------------------------------------------------------
struct detection_struct
{
    unsigned int x;
    unsigned int y;
    unsigned int w;
    unsigned int h;
    char name[label_size];

    detection_struct()
    {
        x = 0;
        y = 0;
        w = 0;
        h = 0;
        name[0] = 0;
    }

    detection_struct(unsigned int x_, unsigned int y_, unsigned int w_, unsigned int h_, const char name_[])
    {
        x = x_;
        y = y_;
        w = w_;
        h = h_;
        strcpy(name, name_);
    }

};

// ----------------------------------------------------------------------------------------
typedef void (*init_net)(const char* net_name, unsigned int* num_classes, struct window_struct*& det_win, unsigned int* num_win);
//typedef void (*run_net)(unsigned char* image, unsigned int nr, unsigned int nc, unsigned char*& det_img, unsigned int* num_dets, struct detection_struct*& dets);
typedef void (*get_detections)(unsigned char* input_img, unsigned int nr, unsigned int nc, unsigned int* num_dets, struct detection_struct*& dets);
typedef void (*get_cropped_detections)(unsigned char* input_img, unsigned int nr, unsigned int nc, unsigned int x, unsigned int y, unsigned int w, unsigned int h, unsigned int* num_dets, struct detection_struct*& dets);
typedef void (*get_combined_output)(struct layer_struct* data, const float*& data_params);
typedef void (*close_lib)();

// ----------------------------------------------------------------------------------------
void parse_input_file(std::string parse_filename, 
    std::string & lib_filename,
    std::string &trained_net_file,
    std::pair<std::string, uint8_t> &test_input,
    std::string &save_directory
)
{
    // parse the input file
    std::vector<std::vector<std::string>> params;
    parse_csv_file(parse_filename, params);

    for (uint64_t idx = 0; idx<params.size(); ++idx)
    {
        switch (idx)
        {

            // get the dynamic library filename
            case 0:
                lib_filename = params[idx][0];
                break;

            // get the trained network data file
            case 1:
                trained_net_file = params[idx][0];
                break;

            // get the file that contains the test data
            case 2:
                try {
                    test_input = std::make_pair(params[idx][0], std::stoi(params[idx][1]));
                }
                catch (std::exception & e) {
                    std::cout << e.what() << std::endl;
                }
                break;

            // get the location to save to data
            case 3:
                save_directory = params[idx][0];
                break;

            default:
                break;

        }   // end of switch

    }   // end of for loop

}   // end of parse_dnn_data_file

#endif  // OBJ_DET_RUN_H_
