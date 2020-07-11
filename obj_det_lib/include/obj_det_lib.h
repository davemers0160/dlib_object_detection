#ifndef OBJ_DET_DLL_H
#define OBJ_DET_DLL_H

//#define EXTERN_C
//#include <cstdint>
//#include <string>
//#include <vector>

#if defined(_WIN32) | defined(__WIN32__) | defined(__WIN32) | defined(_WIN64) | defined(__WIN64)

#if defined(BUILD_LIB)

#ifdef OBJ_DLL_EXPORTS
#define OBJ_DLL_API __declspec(dllexport)
#else
#define OBJ_DLL_API __declspec(dllimport)
#endif

#else

#define OBJ_DLL_API

#endif

#else
#define OBJ_DLL_API

#endif

// ----------------------------------------------------------------------------------------
struct layer_struct
{
    unsigned int k;
    unsigned int n;
    unsigned int nr;
    unsigned int nc;
    unsigned int size;
};

// ----------------------------------------------------------------------------------------
struct detection_struct
{
    unsigned int x;
    unsigned int y;
    unsigned int w;
    unsigned int h;
    char name[256];

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
struct detection_center
{
    unsigned int x;
    unsigned int y;
    char name[256];

    detection_center()
    {
        x = 0;
        y = 0;
        name[0] = 0;
    }

    detection_center(unsigned int x_, unsigned int y_, const char name_[])
    {
        x = x_;
        y = y_;
        strcpy(name, name_);
    }

};

// ----------------------------------------------------------------------------------------
struct window_struct
{
    unsigned int w;
    unsigned int h;
    char label[256];    
};

// ----------------------------------------------------------------------------------------
#ifdef __cplusplus
extern "C" {
#endif
    // This function will initialize the network and load the required weights
    OBJ_DLL_API void init_net(const char *net_name, unsigned int *num_classes, struct window_struct* &det_win, unsigned int *num_win);
#ifdef __cplusplus
}
#endif

// ----------------------------------------------------------------------------------------
#ifdef __cplusplus
extern "C" {
#endif
    // This function creates a tiled image pyramid like the one that the network creates to run the detections on
    OBJ_DLL_API void get_pyramid_tiled_input(unsigned char* input_img, unsigned int nr, unsigned int nc, unsigned char*& tiled_img, unsigned int* t_nr, unsigned int* t_nc);
#ifdef __cplusplus
}
#endif

// ----------------------------------------------------------------------------------------
#ifdef __cplusplus
extern "C" {
#endif
    // This function will take an grayscale image in unsigned char row major order [r0,c0, r0,c1, r0,c2,...]
    // as an input and will return the bounding boxes of the detections in the image
    OBJ_DLL_API void run_net(unsigned char* image, unsigned int nr, unsigned int nc, unsigned char* &det_img, unsigned int *num_dets, struct detection_struct* &dets);
#ifdef __cplusplus
}
#endif

// ----------------------------------------------------------------------------------------
#ifdef __cplusplus
extern "C" {
#endif
    // This function will take an grayscale image in unsigned char row major order [r0,c0, r0,c1, r0,c2,...]
    // as an input and will return the centers of the detections in the image
    OBJ_DLL_API void get_detections(unsigned char* input_img, unsigned int nr, unsigned int nc, unsigned int* num_dets, struct detection_center*& dets);
#ifdef __cplusplus
}
#endif

// ----------------------------------------------------------------------------------------
#ifdef __cplusplus
extern "C" {
#endif
    // This function will take an grayscale image in unsigned char row major order [r0,c0, r0,c1, r0,c2,...]
    // as an input and will return the centers of the detections in the image
    OBJ_DLL_API void get_cropped_detections(unsigned char* input_img, unsigned int nr, unsigned int nc, unsigned int x, unsigned int y, unsigned int w, unsigned int h, unsigned int* num_dets, struct detection_center*& dets);
#ifdef __cplusplus
}
#endif

// ----------------------------------------------------------------------------------------
#ifdef __cplusplus
extern "C" {
#endif
    // This function will output a vector of the output layer for the final classification layer
    OBJ_DLL_API void close_lib();
#ifdef __cplusplus
}
#endif


// ----------------------------------------------------------------------------------------
#ifdef __cplusplus
extern "C" {
#endif
    // This function will output a vector of the output layer for the final classification layer
    //OBJ_DLL_API void get_combined_output(unsigned char* input_img, unsigned int nr, unsigned int nc, float* &data_params);
    OBJ_DLL_API void get_combined_output(struct layer_struct* data, const float*& data_params);
#ifdef __cplusplus
}
#endif
// ----------------------------------------------------------------------------------------
#ifdef __cplusplus
extern "C" {
#endif
    // This function will output a vector of the output layer for the final classification layer
    OBJ_DLL_API void get_layer_01(struct layer_struct *data, const float* &data_params);
#ifdef __cplusplus
}
#endif

//// ----------------------------------------------------------------------------------------
#ifdef __cplusplus
extern "C" {
#endif
    OBJ_DLL_API void get_input_layer(struct layer_struct *data, const float* &data_params);
#ifdef __cplusplus
}
#endif

// ----------------------------------------------------------------------------------------
//#ifdef __cplusplus
//extern "C" {
//#endif
//    //MNIST_DLL_API void get_layer_05(struct layer_struct &data, const float* &data_params);
//    OBJ_DLL_API void get_layer_05(layer_struct *data, const float **data_params);
//#ifdef __cplusplus
//}
//#endif
//
//// ----------------------------------------------------------------------------------------
//#ifdef __cplusplus
//extern "C" {
//#endif
//    //MNIST_DLL_API void get_layer_08(struct layer_struct &data, const float* &data_params);
//    OBJ_DLL_API void get_layer_08(layer_struct *data, const float **data_params);
//#ifdef __cplusplus
//}
//#endif
//
//// ----------------------------------------------------------------------------------------
//#ifdef __cplusplus
//extern "C" {
//#endif
//    //MNIST_DLL_API void get_layer_09(struct layer_struct &data, const float* &data_params);
//    OBJ_DLL_API void get_layer_09(layer_struct *data, const float **data_params);
//#ifdef __cplusplus
//}
//#endif
//
//// ----------------------------------------------------------------------------------------
//#ifdef __cplusplus
//extern "C" {
//#endif
//    //MNIST_DLL_API void get_layer_12(struct layer_struct &data, const float* &data_params);
//    OBJ_DLL_API void get_layer_12(layer_struct *data, const float **data_params);
//#ifdef __cplusplus
//}
//#endif

#endif  // OBJ_DET_DLL_H
