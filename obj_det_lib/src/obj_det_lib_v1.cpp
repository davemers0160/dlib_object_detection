#define _CRT_SECURE_NO_WARNINGS

#include <cstdint>
#include <string>
#include <vector>

//#if !defined(BUILD_LIB)
// Custom includes
#if defined(USE_OPENCV)
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include "dlib_overlay_bbox.h"
#endif
//#endif

// Custom includes
#include "file_ops.h"
#include "obj_det_lib.h"
//#include "tfd_net_v03.h"
#include "obj_det_net_v13.h"
#include "prune_detects.h"

// dlib includes
#include <dlib/dnn.h>
#include <dlib/data_io.h>
#include <dlib/image_transforms.h>

//----------------------------------------------------------------------------------
// library internal state variables:
anet_type net;
double pyr_scale;
unsigned long outer_padding;
unsigned long padding;
std::vector<std::string> class_names;
std::vector<dlib::rgb_pixel> class_color;

//----------------------------------------------------------------------------------
void init_net(const char *net_name, unsigned int *num_classes, struct window_struct* &det_win, unsigned int *num_win)
{
    uint32_t idx;
    int gpu = 0;

    dlib::cuda::set_device(gpu);

    dlib::deserialize(net_name) >> net;

    // Get the type of pyramid the CNN used
    using pyramid_type = std::remove_reference<decltype(dlib::input_layer(net))>::type::pyramid_type;        
    pyramid_type tmp_pyr;
    pyr_scale = dlib::pyramid_rate(tmp_pyr);    
    
    outer_padding = dlib::input_layer(net).get_pyramid_outer_padding();
    padding = dlib::input_layer(net).get_pyramid_padding();

    // get the details about the loss layer -> the number and names of the classes
    dlib::mmod_options options = dlib::layer<0>(net).loss_details().get_options();
    
    *num_win = options.detector_windows.size();
    det_win = new window_struct[options.detector_windows.size()];
    
    std::set<std::string> tmp_names;
    std::cout << std::endl << "------------------------------------------------------------------" << std::endl;
    for (idx = 0; idx < options.detector_windows.size(); ++idx)
    {
        std::cout << "detector window (w x h): " << options.detector_windows[idx].label << " - " << options.detector_windows[idx].width << " x " << options.detector_windows[idx].height << std::endl;

        det_win[idx].w = options.detector_windows[idx].width;
        det_win[idx].h = options.detector_windows[idx].height;
        std::string label = options.detector_windows[idx].label.substr(0, std::min((size_t)255, options.detector_windows[idx].label.length()));
        strcpy(det_win[idx].label, label.c_str());

        tmp_names.insert(options.detector_windows[idx].label);
    }
    std::cout << std::endl;

    // pull out the class names
    for (const auto &it : tmp_names) {
        class_names.push_back(it);
    }
    *num_classes = class_names.size();

    dlib::rand rnd(time(NULL));

    for (idx = 0; idx < *num_classes; ++idx)
    {
        class_color.push_back(dlib::rgb_pixel(rnd.get_random_8bit_number(), rnd.get_random_8bit_number(), rnd.get_random_8bit_number()));
    }

}

//----------------------------------------------------------------------------------
void base_run_net(std::array<dlib::matrix<uint8_t>, array_depth> a_img, std::vector<dlib::mmod_rect> &detects)
{

}

//----------------------------------------------------------------------------------
void get_pyramid_tiled_input(unsigned char* input_img, unsigned int nr, unsigned int nc, unsigned char*& tiled_img, unsigned int* t_nr, unsigned int* t_nc)
{
    uint64_t index = 0;
    dlib::matrix<dlib::rgb_pixel> ti;
    std::vector<dlib::rectangle> rects;
    using pyramid_type = std::remove_reference<decltype(dlib::input_layer(net))>::type::pyramid_type;

    // create the pyramid using pyramid type
    dlib::create_tiled_pyramid<pyramid_type>(dlib::mat(input_img, nr, nc), ti, rects,
        dlib::input_layer(net).get_pyramid_padding(),
        dlib::input_layer(net).get_pyramid_outer_padding());


    // bring out the tiled image version of the input image
    tiled_img = new unsigned char[(uint64_t)ti.nr() * (uint64_t)ti.nc() * 3];
    *t_nr = ti.nr();
    *t_nc = ti.nc();

    index = 0;
    for (unsigned int r = 0; r < *t_nr; ++r)
    {
        for (unsigned int c = 0; c < *t_nc; ++c)
        {
            tiled_img[index++] = ti(r, c).red;
            tiled_img[index++] = ti(r, c).green;
            tiled_img[index++] = ti(r, c).blue;
        }
    }
}

//----------------------------------------------------------------------------------
void run_net(unsigned char* input_img, unsigned int nr, unsigned int nc, unsigned char* &det_img, unsigned int *num_dets, struct detection_struct* &dets)
{

    uint64_t r, c;
    uint64_t idx = 0;
    uint64_t index = 0;
    std::string label;

    dlib::matrix<dlib::rgb_pixel> img(nr, nc);
    std::array<dlib::matrix<uint8_t>, array_depth> a_img;

    uint64_t size = (uint64_t)nr * (uint64_t)nc;

    try 
    {
        // copy the pointer into the input image.  The format is assumed to be row-major order
        // and if the array_depth (number of channels) is greater than 1 then channels are not interleaved
        for (idx = 0; idx < array_depth; ++idx)
        {
            a_img[idx] = dlib::mat<unsigned char>((input_img + (idx * (size))), (long)nr, (long)nc);
        }

        if (array_depth < 3)
        {
            for (r = 0; r < nr; ++r)
            {
                for (c = 0; c < nc; ++c)
                {
                    img(r, c).red = a_img[0](r, c);
                    img(r, c).green = a_img[0](r, c);
                    img(r, c).blue = a_img[0](r, c);
                }
            }
        }
        else
        {
            for (r = 0; r < nr; ++r)
            {
                for (c = 0; c < nc; ++c)
                {
                    img(r, c).red = a_img[0](r, c);
                    img(r, c).green = a_img[1](r, c);
                    img(r, c).blue = a_img[2](r, c);
                }
            }
        }


        std::vector<dlib::mmod_rect> d = net(a_img);
        prune_detects(d, 0.3);

        *num_dets = d.size();
        dets = new detection_struct[d.size()];

        // copy the image into tmp_img so that the original data is not modified
        dlib::matrix<dlib::rgb_pixel> tmp_img;
        dlib::assign_image(tmp_img, img);

        //overlay the dnn detections on the image
#if defined(USE_OPENCV)
        for (idx = 0; idx < d.size(); ++idx)
        {
            auto class_index = std::find(class_names.begin(), class_names.end(), d[idx].label);
            dlib_overlay_bbox(tmp_img, d[idx], class_color[std::distance(class_names.begin(), class_index)]);
            label = d[idx].label.substr(0, std::min((size_t)255, d[idx].label.length()));
            dets[idx] = detection_struct(d[idx].rect.left(), d[idx].rect.top(), d[idx].rect.width(), d[idx].rect.height(), label.c_str());
        }
#else
        for (idx = 0; idx < d.size(); ++idx)
        {
            label = d[idx].label.substr(0, std::min((size_t)255, d[idx].label.length()));
            dets[idx] = detection_struct(d[idx].rect.left(), d[idx].rect.top(), d[idx].rect.width(), d[idx].rect.height(), label.c_str());
        }
#endif

        det_img = new unsigned char[tmp_img.nr() * tmp_img.nc() * 3L];

        idx = 0;
        for (r = 0; r < nr; ++r)
        {
            for (c = 0; c < nc; ++c)
            {
                det_img[idx++] = tmp_img(r, c).red;
                det_img[idx++] = tmp_img(r, c).green;
                det_img[idx++] = tmp_img(r, c).blue;
            }
        }
    }
    catch (std::exception e)
    {
        std::cout << "Error in run_net function:" << std::endl;
        std::cout << e.what() << std::endl;
    }
}   // end of run_net

//----------------------------------------------------------------------------------
void get_detections(unsigned char* input_img, 
    unsigned int nr, 
    unsigned int nc, 
    unsigned int* num_dets, 
    struct detection_struct*& dets
)
{
    uint64_t idx = 0;
    std::string label;

    dlib::matrix<dlib::rgb_pixel> img(nr, nc);
    std::array<dlib::matrix<uint8_t>, array_depth> a_img;

    uint64_t size = (uint64_t)nr * (uint64_t)nc;

    try 
    {
        // copy the pointer into the input image.  The format is assumed to be row-major order
        // and if the array_depth (number of channels) is greater than 1 then channels are not interleaved
        for (idx = 0; idx < array_depth; ++idx)
        {
            a_img[idx] = dlib::mat<unsigned char>((input_img + (idx * (size))), (long)nr, (long)nc);
        }

        std::vector<dlib::mmod_rect> d = net(a_img);

        prune_detects(d, 0.3);

        *num_dets = d.size();
        dets = new detection_struct[d.size()];

        for (idx = 0; idx < d.size(); ++idx)
        {
            //dlib::point c = dlib::center(d[idx].rect);
            label = d[idx].label.substr(0, std::min((size_t)255, d[idx].label.length()));
            dets[idx] = detection_struct(d[idx].rect.left(), d[idx].rect.top(), d[idx].rect.width(), d[idx].rect.height(), label.c_str());
        }
    }
    catch (std::exception e)
    {
        std::cout << "Error in get_detections function:" << std::endl;
        std::cout << e.what() << std::endl;
    }

}   // end of get_detections

//----------------------------------------------------------------------------------
void get_cropped_detections(unsigned char* input_img, 
    unsigned int nr, 
    unsigned int nc, 
    unsigned int x,
    unsigned int y,
    unsigned int w,
    unsigned int h,
    unsigned int* num_dets, 
    struct detection_struct*& dets
)
{
    uint64_t idx = 0;
    std::string label;

    dlib::matrix<dlib::rgb_pixel> img(nr, nc);
    std::array<dlib::matrix<uint8_t>, array_depth> a_img;

    uint64_t size = (uint64_t)nr * (uint64_t)nc;

    dlib::rectangle rect(x, y, w + x - 1, h + y - 1);

    try
    {
        // copy the pointer into the input image.  The format is assumed to be row-major order
        // and if the array_depth (number of channels) is greater than 1 then channels are not interleaved
        for (idx = 0; idx < array_depth; ++idx)
        {
            dlib::matrix<unsigned char> tmp = dlib::mat<unsigned char>((input_img + (idx * (size))), (long)nr, (long)nc);
            a_img[idx] = dlib::subm(tmp, rect);
        }

        std::vector<dlib::mmod_rect> d = net(a_img);

        prune_detects(d, 0.3);

        *num_dets = d.size();
        dets = new detection_struct[d.size()];

        for (idx = 0; idx < d.size(); ++idx)
        {
            // move the rect back to the original image reference frame
            d[idx].rect = dlib::translate_rect(d[idx].rect, x, y);
            // get the center of the rect
            dlib::point c = dlib::center(d[idx].rect);
            label = d[idx].label.substr(0, std::min((size_t)255, label.length()));

            dets[idx] = detection_struct(d[idx].rect.left(), d[idx].rect.top(), d[idx].rect.width(), d[idx].rect.height(), label.c_str());
        }
    }
    catch (std::exception e)
    {
        std::cout << "Error in get_detections function:" << std::endl;
        std::cout << e.what() << std::endl;
    }

}   // end of get_detections



//----------------------------------------------------------------------------------
void close_lib()
{
    std::cout << "Closing..." << std::endl;
    net.clean();

    class_names.clear();
    class_color.clear();

    
}   // end of close_lib

//----------------------------------------------------------------------------------
//void get_combined_output(unsigned char* input_img, unsigned int nr, unsigned int nc, float*& data_params)
void get_combined_output(struct layer_struct* data, const float*& data_params)
{
    //// copy the first output window into a float matrix
    dlib::matrix<float> network_output = dlib::image_plane(net.subnet().get_output(), 0, 0);

    //// get the remaining window outputs and max pool across windows
    for (long k = 1; k < net.subnet().get_output().k(); ++k)
        network_output = dlib::max_pointwise(network_output, dlib::image_plane(net.subnet().get_output(), 0, k));

    data_params = new float[network_output.size()];
    memcpy((void*)data_params, &network_output(0, 0), network_output.size() * sizeof(float));

    data->k = 1;
    data->n = 1;
    data->nr = network_output.nr();
    data->nc = network_output.nc();
    data->size = network_output.size();

/*
    const double network_output_scale = nc / (double)network_output.nc();
    dlib::resize_image(network_output_scale, network_output);

    data_params = new float[nr*nc];
   
    std::array<dlib::matrix<unsigned char>, 1> img;
    img[0] = dlib::mat(input_img, nr, nc);
    //dlib::matrix<float> collapsed(nr, nc);
    dlib::resizable_tensor input_tensor;
    input_layer(net).to_tensor(&img, &img +1, input_tensor);

    long r, c;
    for (r = 0; r < nr; ++r)
    {
        for (c = 0; c < nc; ++c)
        {
            // Loop over a bunch of scale values and look up what part of network_output
            // corresponds to the point(c,r) in the original image, then take the max
            // detection score over all the scales and save it at pixel point(c,r).
            float max_score = -1e30;
            for (double scale = 1; scale > 0.2; scale *= pyr_scale)
            {
                // Map from input image coordinates to tiled pyramid coordinates.
                dlib::dpoint tmp = center(input_layer(net).image_space_to_tensor_space(input_tensor, scale, dlib::drectangle(dlib::dpoint(c, r))));
                
                // Now map from pyramid coordinates to network_output coordinates.
                tmp = dlib::point(network_output_scale * input_tensor_to_output_tensor(net, tmp));

                if (dlib::get_rect(network_output).contains(tmp))
                {
                    float val = network_output(tmp.y(), tmp.x());
                    if (val > max_score)
                        max_score = val;
                }
            }
            //collapsed(r, c) = max_score;
            data_params[r*nc + c] = max_score;
        }
    }

    //data_params = new float[collapsed.nr() * collapsed.nc()];
    //memcpy((void*)data_params, &collapsed(0, 0), collapsed.size() * sizeof(float));
*/
}   // end of get_combined_output


//----------------------------------------------------------------------------------
void get_layer_01(struct layer_struct *data, const float* &data_params)
{
    auto& lo = dlib::layer<1>(net).get_output();
    data->k = lo.k();
    data->n = lo.num_samples();
    data->nr = lo.nr();
    data->nc = lo.nc();
    data->size = lo.size();
    data_params = lo.host();
}

//----------------------------------------------------------------------------------
void get_input_layer(layer_struct *data, const float* &data_params)
{
    auto& lo = dlib::layer<net_type::num_layers - 2>(net).get_output();
    data->k = lo.k();
    data->n = lo.num_samples();
    data->nr = lo.nr();
    data->nc = lo.nc();
    data->size = lo.size();
    data_params = lo.host();
}

//----------------------------------------------------------------------------------
//void get_layer_05(layer_struct *data, const float **data_params)
//{
//    auto& lo = dlib::layer<5>(net).get_output();
//    data->k = lo.k();
//    data->n = lo.num_samples();
//    data->nr = lo.nr();
//    data->nc = lo.nc();
//    data->size = lo.size();
//    *data_params = lo.host();
//}
//
////----------------------------------------------------------------------------------
//void get_layer_08(layer_struct *data, const float **data_params)
//{
//    auto& lo = dlib::layer<8>(net).get_output();
//    data->k = lo.k();
//    data->n = lo.num_samples();
//    data->nr = lo.nr();
//    data->nc = lo.nc();
//    data->size = lo.size();
//    *data_params = lo.host();
//}
//
////----------------------------------------------------------------------------------
//void get_layer_09(layer_struct *data, const float **data_params)
//{
//    auto& lo = dlib::layer<9>(net).get_output();
//    data->k = lo.k();
//    data->n = lo.num_samples();
//    data->nr = lo.nr();
//    data->nc = lo.nc();
//    data->size = lo.size();
//    *data_params = lo.host();
//}
//
////----------------------------------------------------------------------------------
//void get_layer_12(layer_struct *data, const float **data_params)
//{
//    auto& lo = dlib::layer<12>(net).get_output();
//    data->k = lo.k();
//    data->n = lo.num_samples();
//    data->nr = lo.nr();
//    data->nc = lo.nc();
//    data->size = lo.size();
//    *data_params = lo.host();
//}

//----------------------------------------------------------------------------------
// check to see if we are building the library or a standalone executable
#if !defined(BUILD_LIB)

int main(int argc, char** argv)
{
    uint32_t idx;
    std::string program_root;
    std::string net_directory;
    std::string image_directory;
    std::string test_net_name;

    // timing variables
    typedef std::chrono::duration<double> d_sec;
    auto start_time = std::chrono::system_clock::now();
    auto stop_time = std::chrono::system_clock::now();
    auto elapsed_time = std::chrono::duration_cast<d_sec>(stop_time - start_time);

    //std::vector<std::string> test_images = { "test1.png", "test2.png", "test3.png", "test4.png", "test5.png", "test6.png", "test7.png", "test8.png", "test9.png", "test10.png" };
    std::vector<std::string> test_images = { "mframe_00156.png", "mframe_00163.png", "mframe_00279.png", "mframe_00353.png", "mframe_05042.png" };

    unsigned int num_classes, num_win;

    unsigned char* tiled_img = NULL;
    unsigned char* det_img = NULL;

    unsigned int t_nr = 0, t_nc = 0;
    window_struct* det;
    unsigned int num_dets = 0;
    struct detection_struct* dets;
    detection_center* detects;
    cv::Mat img;
    long nr, nc;

    // setup save variable locations
#if defined(_WIN32) | defined(__WIN32__) | defined(__WIN32) | defined(_WIN64) | defined(__WIN64)
    program_root = path_check(get_path(get_path(get_path(std::string(argv[0]), "\\"), "\\"), "\\"));
#else 
    program_root = get_ubuntu_path();
#endif

    net_directory = program_root + "../common/nets/";
    image_directory = "../images/";

    try
    {
        test_net_name = (net_directory + "tfd_v03_20_20_100_HPC_final_net.dat");

        // initialize the network
        init_net(test_net_name.c_str(), &num_classes, det, &num_win);

        dlib::matrix<uint8_t> ti;

        // run through some images to test the code
        for (idx = 0; idx < test_images.size(); ++idx)
        {
            img = cv::imread(image_directory + test_images[idx], cv::IMREAD_GRAYSCALE);
            nr = img.rows;
            nc = img.cols;

            unsigned char* image = new unsigned char[nr * nc]{ 0 };

            start_time = std::chrono::system_clock::now();

            run_net(img.ptr<unsigned char>(0), nr, nc, det_img, &num_dets, dets);

            get_detections(img.ptr<unsigned char>(0), nr, nc, &num_dets, detects);

            get_cropped_detections(img.ptr<unsigned char>(0), nr, nc, 128, 0, 256, 256, &num_dets, detects);

            layer_struct ls_all, ls_in;
            const float* ld_all, *ld_in;

            // get the input after the pyramid
            get_input_layer(&ls_in, ld_in);

            // get the combined detection map
            get_combined_output(&ls_all, ld_all);

            stop_time = std::chrono::system_clock::now();
            elapsed_time = std::chrono::duration_cast<d_sec>(stop_time - start_time);

            std::cout << "Runtime (s): " << elapsed_time.count() << std::endl;
        }

        close_lib();

    }
    catch (std::exception & e)
    {
        std::cout << std::endl;
        std::cout << e.what() << std::endl;
    }

    std::cout << std::endl << "Program complete.  Press Enter to close." << std::endl;
    std::cin.ignore();
    return 0;

}   // end of main

#endif  // BUILD_LIB
