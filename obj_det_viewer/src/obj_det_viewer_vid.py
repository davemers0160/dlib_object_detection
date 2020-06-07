import platform
import os
# this handles the *.dll/*.so file reading
import time

from cffi import FFI
# numpy/opencv
import numpy as np
import cv2 as cv
# File dialog stuff
from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import QFileDialog, QWidget, QApplication

import pandas as pd
import bokeh
from bokeh.io import curdoc
from bokeh.models import ColumnDataSource, HoverTool, Button, Div
from bokeh.plotting import figure, show
from bokeh.layouts import column, row, Spacer

# set up some global variables that will be used throughout the code
script_path = os.path.realpath(__file__)
image_path = os.path.dirname(os.path.dirname(script_path))
update_time = 100
num_det_wins = 0
detection_windows = []
app = QApplication([""])

# variable to store the final feature space detections
fs = []
ffi = FFI()

# This section allows cffi to get the info about the dll and the specialized data types
ffi.cdef('''
struct layer_struct{
    unsigned int k;
    unsigned int n;
    unsigned int nr;
    unsigned int nc;
    unsigned int size;
};

struct detection_struct{
    unsigned int x;
    unsigned int y;
    unsigned int w;
    unsigned int h;
    char label[256];
};
    
struct window_struct{
    unsigned int w;
    unsigned int h;
    char label[256];    
};

void init_net(const char *net_name, unsigned int *num_classes, struct window_struct** det_win, unsigned int *num_win);
void run_net(unsigned char* image, unsigned int nr, unsigned int nc, unsigned char** tiled_img, unsigned int *t_nr, unsigned int *t_nc, unsigned char** det_img, unsigned int *num_dets, struct detection_struct** dets);
void get_layer_01(struct layer_struct *data, const float** data_params);
''')

# modify these to point to the right locations
if platform.system() == "Windows":
    libname = "obj_det_lib.dll"
    lib_location = "D:/Projects/obj_det_lib/build/Release/" + libname
    weights_file = "D:/Projects/obj_det_lib/nets/yj_v4_s4_v4_BEAST_final_net.dat"
elif platform.system() == "Linux":
    libname = "obj_det_lib.so"
    home = os.path.expanduser('~')
    lib_location = home + "/Projects/obj_det_lib/build/" + libname
    weights_file = home + "/Projects/obj_det_lib/nets/yj_v4_s4_v4_BEAST_final_net.dat"
else:
    quit()


# read and write global
obj_det_lib = []
x_r = y_r = min_img = max_img = 0

sd = dict(input_img=[], pyr_img=[], det_view=[])

# ----------------------------------------------------------------------


def jet_clamp(v):
    v[v < 0] = 0
    v[v > 1] = 1
    return v

def jet_color(t, t_min, t_max):

    t_range = t_max - t_min
    t_avg = (t_max + t_min) / 2.0
    t_m = (t_max - t_avg) / 2.0

    rgba = np.empty((t.shape[0], t.shape[1], 4), dtype=np.uint8)
    rgba[:, :, 0] = (255*jet_clamp(1.5 - abs((4 / t_range)*(t - t_avg - t_m)))).astype(np.uint8)
    rgba[:, :, 1] = (255*jet_clamp(1.5 - abs((4 / t_range)*(t - t_avg)))).astype(np.uint8)
    rgba[:, :, 2] = (255*jet_clamp(1.5 - abs((4 / t_range)*(t - t_avg + t_m)))).astype(np.uint8)
    rgba[:, :, 3] = np.full((t.shape[0], t.shape[1]), 255, dtype=np.uint8)
    return rgba


def build_layer_image(ls, ld, cell_dim, padding, map_length):

    min_v = np.amin(ld)
    max_v = np.amax(ld)
    img_array = np.floor((map_length)*(ld - min_v)/(max_v - min_v)) + 10

    t_min = np.amin(img_array)
    t_max = np.amax(img_array)

    img_length = ls.nr * ls.nc

    img_h = (ls.nr + padding)*(cell_dim[0]-1) + ls.nr + 2*padding
    img_w = (ls.nc + padding)*(cell_dim[1]-1) + ls.nc + 2*padding
    layer_img = np.zeros((img_h, img_w), dtype=np.float)

    r = padding
    c = padding

    for idx in range(ls.k):
        p1 = (idx * img_length)
        p2 = ((idx+1) * img_length)

        layer_img[r:r+ls.nr, c:c+ls.nc] = np.reshape(img_array[p1:p2], [ls.nr, ls.nc])

        c = c + (ls.nc + padding)
        if(c >= img_w):
            c = padding
            r = r + (ls.nr + padding)

    return layer_img


def init_lib():
    global obj_det_lib, ls_01, ld_01, num_classes, detection_windows, \
        t_nr, t_nc, tiled_img, det_img, num_dets, dets

    obj_det_lib = ffi.dlopen(lib_location)

    # initialize the network with the weights file
    # void init_net(const char *net_name, unsigned int *num_classes, window** det_win, unsigned int *num_win)
    det_win = ffi.new('struct window_struct**')
    num_classes = ffi.new('unsigned int *')
    num_win = ffi.new('unsigned int *')
    obj_det_lib.init_net(weights_file.encode("utf-8"), num_classes, det_win, num_win)

    detection_windows = pd.DataFrame(columns=["h", "w", "label"])
    for idx in range(num_win[0]):
        detection_windows = detection_windows.append({"h": det_win[0][idx].h, "w": det_win[0][idx].w, "label": ffi.string(det_win[0][idx].label).decode("utf-8")}, ignore_index=True)

    # instantiate the run_net function
    # void run_net(unsigned char* image, unsigned int nr, unsigned int nc, unsigned char** tiled_img, unsigned int *t_nr, unsigned int *t_nc, unsigned char** det_img, unsigned int *num_dets, struct detection_struct** dets);
    tiled_img = ffi.new('unsigned char**')
    t_nr = ffi.new('unsigned int *')
    t_nc = ffi.new('unsigned int *')
    det_img = ffi.new('unsigned char**')
    num_dets = ffi.new('unsigned int *')
    dets = ffi.new('struct detection_struct**')

    # instantiate the get_layer_01 function
    # void get_layer_01(struct layer_struct *data, const float** data_params);
    ls_01 = ffi.new('struct layer_struct*')
    ld_01 = ffi.new('float**')

    bp = 1

def get_input():
    image_name = QFileDialog.getOpenFileName(None, "Select a file",  image_path, "Image files (*.mp4 *.mpeg *.avi);;All files (*.*)")

    if(image_name[0] == ""):
        return

    video_file = cv.VideoCapture(image_name[0])
    scale = 0.25
    file_id = open("test_detection_results.txt", "w")

    frame_count = 0
    while (video_file.isOpened()):
        ret, color_img = video_file.read()

        if ret:
            # resize the image
            color_img = cv.resize(color_img, (int(scale * color_img.shape[0]), int(scale * color_img.shape[1])), interpolation=cv.INTER_CUBIC)

            # run the image through the detector and get the results
            dets_text, det_img_view = run_images(cv.cvtColor(color_img, cv.COLOR_BGR2RGB))

            file_id.write("{:04d}".format(frame_count) + "," + dets_text + "\n")
            frame_count += 1

            cv.imshow("Results", det_img_view)
            if cv.waitKey(1) & 0xFF == ord('q'):
                break

    file_id.close()
    video_file.release()


def run_images(color_img):
    global tiled_img, t_nr, t_nc, det_img, num_dets, dets, ls_01, ld_01, results_div, p1, p2, ti, fs

    # print("Processing File: ", image_name[0])
    # load in an image
    # image_path = os.path.dirname(image_name[0])
    # color_img = cv.imread(image_name[0])
    # color_img = cv.cvtColor(color_img, cv.COLOR_BGR2RGB)
    img_nr = color_img.shape[0]
    img_nc = color_img.shape[1]

    # convert the image to RGBA for display
    # rgba_img = cv.cvtColor(color_img, cv.COLOR_RGB2RGBA)
    # p1.image_rgba(image=[np.flipud(rgba_img)], x=0, y=0, dw=400, dh=400)

    # run the image on network and get the results
    obj_det_lib.run_net(color_img.tobytes(), img_nr, img_nc, tiled_img, t_nr, t_nc, det_img, num_dets, dets)

    detections = pd.DataFrame(columns=["x", "y", "h", "w", "label"])
    # det_results = "<font size='3'>"
    dets_text = ""
    for idx in range(num_dets[0]):
        detections = detections.append({"x": dets[0][idx].x, "y": dets[0][idx].y, "h": dets[0][idx].h, "w": dets[0][idx].w, "label": ffi.string(dets[0][idx].label).decode("utf-8")}, ignore_index=True)
        # det_results += str(detections["x"][idx]) + ", " + str(detections["y"][idx]) + ", " + str(detections["h"][idx]) + ", " + str(detections["w"][idx]) + ", " + detections["label"][idx] + "<br>"
        dets_text += "{" + str(detections["x"][idx]) + ", " + str(detections["y"][idx]) + ", " + str(detections["h"][idx]) + ", " + str(detections["w"][idx]) + ", " + detections["label"][idx] + "},"

    # det_results += "</font>"
    # results_div.text = det_results  # "<font size='4'>" + str(index) + ": " + file_path[0] + "</font>"

    # det_img_alpha = np.full((img_nr, img_nc), 255, dtype=np.uint8)
    # det_img_view = np.dstack([np.reshape(np.frombuffer(ffi.buffer(det_img[0], img_nr * img_nc * 3), dtype=np.uint8), [img_nr, img_nc, 3]), det_img_alpha])

    det_img_view = np.reshape(np.frombuffer(ffi.buffer(det_img[0], img_nr * img_nc * 3), dtype=np.uint8), [img_nr, img_nc, 3])

    # tiled_img_alpha = np.full((t_nr[0], t_nc[0]), 255, dtype=np.uint8)
    # tiled_img_view = np.dstack([np.reshape(np.frombuffer(ffi.buffer(tiled_img[0], t_nr[0] * t_nc[0] * 3), dtype=np.uint8), [t_nr[0], t_nc[0], 3]),
    #                             tiled_img_alpha])

    # p2.image_rgba(image=[np.flipud(det_img_view)], x=0, y=0, dw=400, dh=400)
    # ti.image_rgba(image=[np.flipud(tiled_img_view)], x=0, y=0, dw=400, dh=400)

    # start to create the source data based on the static/know inputs
    # source.data = {'input_img': [np.flipud(rgba_img)], 'det_view': [np.flipud(det_img_view)],
    #                'tiled_img': [np.flipud(tiled_img_view)]}

    # get the Layer 01 data and shape it correctly
    # obj_det_lib.get_layer_01(ls_01, ld_01)
    # l01_data = np.frombuffer(ffi.buffer(ld_01[0], ls_01.size * 4), dtype=np.float32)
    # l01_min = -4.0  # hardcoded but can use this instead -> np.amin(l01_data)
    # l01_max = 0.0   # hardcoded but can use this instead -> np.amax(l01_data)
    # img_length = ls_01.nr * ls_01.nc

    # l01_list = []
    # for idx in range(ls_01.k):
    #     sd_key = "l01_img_" + "{:04d}".format(idx)
    #     fl_data = "l01_imgf_" + "{:04d}".format(idx)
    #     s1 = idx*img_length
    #     s2 = (idx+1)*img_length
    #     l01_list.append(np.reshape(l01_data[s1:s2], [ls_01.nr, ls_01.nc]))
    #     l01_jet = jet_color(l01_list[idx], l01_min, l01_max)
    #     fs[idx].image_rgba(image=[np.flipud(l01_jet)], x=0, y=0, dw=400, dh=400)
    #     # source.data[sd_key] = [np.flipud(l01_jet)]
    #     # source.data[fl_data] = [l01_list[idx]]
    #     #fs[idx].tools[0].tooltips = {"Value": "$l01_list[idx]"}

    # bp = 1
    return dets_text, det_img_view

# the main entry point into the code
# if __name__ == '__main__':


# file_select_btn = Button(label='Select File', width=100)
# file_select_btn.on_click(get_input)
# header_div = Div(width=800, text="Results: (x, y, h, w, label)", style={'font-size': '125%', 'font-weight': 'bold'})
# results_div = Div(width=800)

init_lib()
num_det_wins = detection_windows.shape[0]

# dynamically build the ColumnDataSource by first building the dict
# for idx in range(num_det_wins):
#     sd_key = "l01_img_" + "{:04d}".format(idx)
#     sd[sd_key] = []
#     fl_data = "l01_imgf_" + "{:04d}".format(idx)
#     sd[fl_data] = []
#
# source = ColumnDataSource(data=sd)

# setup the bokeh figures dynamically
# for idx in range(num_det_wins):
#     # ht = HoverTool(tooltips=[("Value", "0")])
#     sd_key = "l01_img_" + "{:04d}".format(idx)
#     fl_data = "@l01_imgf_" + "{:04d}".format(idx)
#     fs_title = detection_windows["label"][idx] + " (" + str(detection_windows["h"][idx]) + "x" + str(detection_windows["w"][idx]) + ")"
#     fs_tmp = figure(plot_height=600, plot_width=300, title=fs_title, tools=['wheel_zoom', 'reset', 'save', 'pan'])
#     # fs_tmp.image_rgba(image=sd_key, x=0, y=0, dw=400, dh=400, source=source)
#     fs_tmp.image_rgba(image=[], x=0, y=0, dw=400, dh=400)
#     fs_tmp.axis.visible = False
#     fs_tmp.grid.visible = False
#     fs_tmp.x_range.range_padding = 0
#     fs_tmp.y_range.range_padding = 0
#     # fs_tmp.add_tools(HoverTool(tooltips=[("value ", fl_data)]))
#     fs.append(fs_tmp)

# p1 = figure(plot_height=350, plot_width=500, title="Input image")
# p1.image_rgba(image=[], x=0, y=0, dw=400, dh=400)
# p1.axis.visible = False
# p1.grid.visible = False
# p1.x_range.range_padding = 0
# p1.y_range.range_padding = 0

# p2 = figure(plot_height=350, plot_width=500, title="Detection Results")
# p2.image_rgba(image=[], x=0, y=0, dw=400, dh=400)
# p2.axis.visible = False
# p2.grid.visible = False
# p2.x_range.range_padding = 0
# p2.y_range.range_padding = 0

# ti = figure(plot_height=600, plot_width=300, title="Tiled Image Input")
# ti.image_rgba(image=[], x=0, y=0, dw=400, dh=400)
# ti.axis.visible = False
# ti.grid.visible = False
# ti.x_range.range_padding = 0
# ti.y_range.range_padding = 0

# ol = [ti]
# ol.extend(fs)

# layout = column([row([p1, p2, column([file_select_btn, header_div, results_div])]), row(ol)])
# show(layout)

get_input()

# doc = curdoc()
# doc.title = "Object Detection Viewer"
# doc.add_root(layout)





