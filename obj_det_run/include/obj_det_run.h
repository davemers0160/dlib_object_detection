#ifndef OBJ_DET_RUN_H_
#define OBJ_DET_RUN_H_


#include <cstdint>
#include <tuple>

// custom includes
#include "file_parser.h"

extern const uint32_t array_depth;

typedef struct label_stats {

    label_stats() = default;

    label_stats(uint32_t c, uint32_t mc) : count(c), match_count(mc) {}

    uint32_t count;
    uint32_t match_count;

} label_stats;

// ----------------------------------------------------------------------------------------
void parse_input_file(std::string parse_filename, 
    std::string &version, 
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

            case 3:
                save_directory = params[idx][0];
                break;

            default:
                break;
        }   // end of switch

    }   // end of for

}   // end of parse_dnn_data_file

#endif  // OBJ_DET_RUN_H_
