#ifndef OBJ_DET_RUN_H_
#define OBJ_DET_RUN_H_


#include <cstdint>
#include <tuple>

// custom includes
#include "file_parser.h"

// dlib includes

extern const uint32_t array_depth;

// ----------------------------------------------------------------------------------------
//
//typedef struct training_params {
//
//    training_params() = default;
//    training_params(double ilr, double flr, double lrsf, uint32_t step) : intial_learning_rate(ilr), final_learning_rate(flr), learning_rate_shrink_factor(lrsf), steps_wo_progess(step){}
//
//    double intial_learning_rate;
//    double final_learning_rate;
//    double learning_rate_shrink_factor;
//    uint32_t steps_wo_progess;
//
//} training_params;

//// ----------------------------------------------------------------------------------------
//
//typedef struct crop_info {
//
//    crop_info() = default;
//    crop_info(uint64_t n, uint64_t h, uint64_t w, double a) : crop_num(n), crop_height(h), crop_width(w), angle(a) {}
//
//    uint64_t crop_num;
//    uint64_t crop_height;
//    uint64_t crop_width;
//    double angle;
//
//} crop_info;

typedef struct label_stats {

    label_stats() = default;

    label_stats(uint32_t c, uint32_t mc) : count(c), match_count(mc) {}

    uint32_t count;
    uint32_t match_count;

} label_stats;

// ----------------------------------------------------------------------------------------
void parse_input_file(std::string parse_filename, 
    std::string &version, 
    std::vector<int32_t> &gpu,
    std::string &trained_net_file,
    std::pair<std::string, uint8_t> &test_input,
    std::string &save_directory
)
{

    std::vector<std::vector<std::string>> params;
    parse_csv_file(parse_filename, params);

    for (uint64_t idx = 0; idx<params.size(); ++idx)
    {
        switch (idx)
        {

            // get the version name of the network - used for naming various files
            case 0:
                version = params[idx][0];
                break;

            // select which gpu to use
		    case 1:
                try {
                    gpu.clear();
                    for (int jdx = 0; jdx < params[idx].size(); ++jdx)
                    {
                        gpu.push_back(stol(params[idx][jdx]));
                    }
                }
                catch (std::exception &e) {
                    std::cout << e.what() << std::endl;
                    gpu.clear();
                    gpu.push_back(0);
                }
			    break;

            // get the trained network data file
            case 2:
                trained_net_file = params[idx][0];
                break;

            // get the file that contains the test data
            case 3:
                try {
                    test_input = std::make_pair(params[idx][0], std::stoi(params[idx][1]));
                }
                catch (std::exception & e) {
                    std::cout << e.what() << std::endl;
                }
                break;

            case 4:
                save_directory = params[idx][0];
                break;

            default:
                break;
        }   // end of switch

    }   // end of for

}   // end of parse_dnn_data_file

#endif  // OBJ_DET_RUN_H_
