#ifndef _OBJECT_DETECTOR_H_
#define _OBJECT_DETECTOR_H_

#include <cstdint>
#include <string>

// custom includes
#include "file_parser.h"

// ----------------------------------------------------------------------------------------
void parse_input_file(std::string parse_filename, 
                      std::string &data_file,
                      std::string &network_weights_file,
                      std::string &results_name, 
                      std::string &save_location
                      )
{

    std::vector<std::vector<std::string>> params;
    parse_csv_file(parse_filename, params);

    network_weights_file = "";
    results_name = "";
    save_location = "";

    for (uint64_t idx = 0; idx<params.size(); ++idx)
    {
        switch (idx)
        {

            case 0:
                data_file = params[idx][0];
                break;

            case 1:
                network_weights_file = params[idx][0];
                break;

            case 2:
                results_name = params[idx][0];
                break;

            case 3:
                save_location = params[idx][0];
                break;

            default:
                break;
        }   // end of switch

    }   // end of for

}   // end of parse_dnn_data_file

#endif  // _OBJECT_DETECTOR_H_
