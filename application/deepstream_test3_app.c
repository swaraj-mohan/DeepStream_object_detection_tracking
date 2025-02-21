/*
 * SPDX-FileCopyrightText: Copyright (c) 2018-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: MIT
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */

#include <gst/gst.h>
#include <glib.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <math.h>
#include <time.h>
#include <string.h>
#include <sys/time.h>
#include <ctype.h>
#include <cuda_runtime_api.h>

#include "gstnvdsmeta.h"
#include "nvds_yml_parser.h"
#include "gst-nvmessage.h"

#include "nvbufsurface.h"
#include "nvds_obj_encode.h"

#define MAX_DISPLAY_LEN 64

#define PGIE_CLASS_ID_LIFT 2
#define PGIE_CLASS_ID_TRUCK 0

#define TRACKER_CONFIG_FILE "tracker_config.txt"
#define MAX_TRACKING_ID_LEN 16

/* By default, OSD process-mode is set to CPU_MODE. To change mode, set as:
 * 1: GPU mode (for Tesla only)
 * 2: HW mode (For Jetson only)
 */
#define OSD_PROCESS_MODE 0

/* By default, OSD will not display text. To display text, change this to 1 */
#define OSD_DISPLAY_TEXT 0

/* The muxer output resolution must be set if the input streams will be of
 * different resolution. The muxer will scale all the input frames to this
 * resolution. */
#define MUXER_OUTPUT_WIDTH 1920
#define MUXER_OUTPUT_HEIGHT 1080

/* Muxer batch formation timeout, for e.g. 40 millisec. Should ideally be set
 * based on the fastest source's framerate. */
#define MUXER_BATCH_TIMEOUT_USEC 40000

#define TILED_OUTPUT_WIDTH 1280
#define TILED_OUTPUT_HEIGHT 720

/* NVIDIA Decoder source pad memory feature. This feature signifies that source
 * pads having this capability will push GstBuffers containing cuda buffers. */
#define GST_CAPS_FEATURES_NVMM "memory:NVMM"

/* Check for parsing error. */
#define RETURN_ON_PARSER_ERROR(parse_expr) \
  if (NVDS_YAML_PARSER_SUCCESS != parse_expr) { \
    g_printerr("Error in parsing configuration file.\n"); \
    return -1; \
  }

gchar pgie_classes_str[8][32] = {"truck", "humans", "forklift", "loads", "cover", "number_plate", "straps", "vest"};

static gboolean PERF_MODE = FALSE;
bool save_image = FALSE;


// Function to calculate area of a rectangle
int area(int x1, int y1, int x2, int y2) {
    return (x2 - x1) * (y2 - y1);
}

// Function to calculate the intersection area
int intersection_area(int x1_1, int y1_1, int x2_1, int y2_1, int x1_2, int y1_2, int x2_2, int y2_2) {
    // Calculate the coordinates of the intersection rectangle
    int inter_left = (x1_1 > x1_2) ? x1_1 : x1_2;
    int inter_right = (x2_1 < x2_2) ? x2_1 : x2_2;
    int inter_top = (y1_1 > y1_2) ? y1_1 : y1_2;
    int inter_bottom = (y2_1 < y2_2) ? y2_1 : y2_2;

    // If there is no intersection
    if (inter_left < inter_right && inter_top < inter_bottom) {
        return area(inter_left, inter_top, inter_right, inter_bottom);
    }
    return 0; // No intersection
}


/* tracker_sink_pad_buffer_probe  will extract metadata received on OSD sink pad
 * and update params for drawing rectangle, object information etc. */

static GstPadProbeReturn
tracker_src_pad_buffer_probe (GstPad * pad, GstPadProbeInfo * info, gpointer ctx)
{
    FILE *file;
    GstBuffer *buf = (GstBuffer *) info->data;
    
    GstMapInfo inmap = GST_MAP_INFO_INIT;
    if (!gst_buffer_map (buf, &inmap, GST_MAP_READ)) {
      GST_ERROR ("input buffer mapinfo failed");
      return GST_PAD_PROBE_DROP;
    }
    NvBufSurface *ip_surf = (NvBufSurface *) inmap.data;
    gst_buffer_unmap (buf, &inmap);
    
    guint num_rects = 0; 
    NvDsObjectMeta *obj_meta = NULL;
    int lift_x1 = 0; int lift_y1 = 0; int lift_x2 = 0; int lift_y2 = 0; bool found_lift = FALSE;
    int truck_x1 = 0; int truck_y1 = 0; int truck_x2 = 0; int truck_y2 = 0; bool found_truck = FALSE;
    //bool save_image = FALSE;
    guint vehicle_count = 0;
    guint person_count = 0;
    NvDsMetaList * l_frame = NULL;
    NvDsMetaList * l_obj = NULL;
    //NvDsDisplayMeta *display_meta = NULL;

    NvDsBatchMeta *batch_meta = gst_buffer_get_nvds_batch_meta (buf);

    for (l_frame = batch_meta->frame_meta_list; l_frame != NULL;
      l_frame = l_frame->next) {
        save_image = FALSE;
        NvDsFrameMeta *frame_meta = (NvDsFrameMeta *) (l_frame->data);
        //int offset = 0;
        for (l_obj = frame_meta->obj_meta_list; l_obj != NULL;
                l_obj = l_obj->next) {
            obj_meta = (NvDsObjectMeta *) (l_obj->data);
            
            char string1[100];  // Make sure the buffer is large enough to hold the formatted string
            sprintf(string1, "%lu, %lu, %s, %d, %d, %d, %d", frame_meta->ntp_timestamp, obj_meta->object_id, pgie_classes_str[obj_meta->class_id], (int) obj_meta->rect_params.left, (int) obj_meta->rect_params.top, (int) obj_meta->rect_params.width, (int) obj_meta->rect_params.height);
            FILE *file = fopen("detection_data.txt", "a");
            if (file == NULL) {
                perror("Error opening file");
                return 1;
            }
            fprintf(file, "%s\n", string1);
            fclose(file);
            
            if (obj_meta->class_id == PGIE_CLASS_ID_LIFT) {
                lift_x1 = (int) obj_meta->rect_params.left;
                lift_y1 = (int) obj_meta->rect_params.top;
                lift_x2 = (int) obj_meta->rect_params.left + (int) obj_meta->rect_params.width;
                lift_y2 = (int) obj_meta->rect_params.top + (int) obj_meta->rect_params.height;
                found_lift = TRUE;
            }
            if (obj_meta->class_id == PGIE_CLASS_ID_TRUCK) {
                truck_x1 = (int) obj_meta->rect_params.left;
                truck_y1 = (int) obj_meta->rect_params.top;
                truck_x2 = (int) obj_meta->rect_params.left + (int) obj_meta->rect_params.width;
                truck_y2 = (int) obj_meta->rect_params.top + (int) obj_meta->rect_params.height;
                found_truck = TRUE;
            }
            if (found_lift && found_truck) {
                //g_print("found truck and lift\n");
                int int_area = intersection_area(lift_x1, lift_y1, lift_x2, lift_y2, truck_x1, truck_y1, truck_x2, truck_y2);
                //printf("IoU is: %d\n", int_area);
                if (int_area > 40000) {
                    save_image = TRUE;
                }
            }
        }
    }
    return GST_PAD_PROBE_OK;
}


/* tracker_sink_pad_buffer_probe  will extract metadata received on OSD sink pad
 * and update params for drawing rectangle, object information etc. */

static GstPadProbeReturn
osd_src_pad_buffer_probe (GstPad * pad, GstPadProbeInfo * info, gpointer ctx)
{
    GstBuffer *buf = (GstBuffer *) info->data;
    
    GstMapInfo inmap = GST_MAP_INFO_INIT;
    if (!gst_buffer_map (buf, &inmap, GST_MAP_READ)) {
      GST_ERROR ("input buffer mapinfo failed");
      return GST_PAD_PROBE_DROP;
    }
    NvBufSurface *ip_surf = (NvBufSurface *) inmap.data;
    gst_buffer_unmap (buf, &inmap);

    NvDsMetaList * l_frame = NULL;
    NvDsBatchMeta *batch_meta = gst_buffer_get_nvds_batch_meta (buf);

    for (l_frame = batch_meta->frame_meta_list; l_frame != NULL;
      l_frame = l_frame->next) {
        NvDsFrameMeta *frame_meta = (NvDsFrameMeta *) (l_frame->data);
        //int offset = 0;
        if (save_image) {
            //g_print("going to save\n");
            NvDsObjEncUsrArgs frameData = { 0 };
            /* Preset */
            frameData.isFrame = 1;
            /* To be set by user */
            frameData.saveImg = TRUE;
            time_t current_time;
            struct tm *local_time;
            char buffer[80];
            time(&current_time);
            local_time = localtime(&current_time);
            strftime(buffer, sizeof(buffer), "%Y-%m-%d %H:%M:%S", local_time);
            sprintf(frameData.fileNameImg, "images/forklift_near_truck_%s.jpg", buffer);
            frameData.attachUsrMeta = TRUE;
            /* Set if Image scaling Required */
            frameData.scaleImg = FALSE;
            frameData.scaledWidth = 0;
            frameData.scaledHeight = 0;
            /* Quality */
            frameData.quality = 80;
            /* Main Function Call */
            nvds_obj_enc_process (ctx, &frameData, ip_surf, NULL, frame_meta);
        }
    }
    nvds_obj_enc_finish (ctx);
    return GST_PAD_PROBE_OK;
}



static gboolean
bus_call (GstBus * bus, GstMessage * msg, gpointer data)
{
  GMainLoop *loop = (GMainLoop *) data;
  switch (GST_MESSAGE_TYPE (msg)) {
    case GST_MESSAGE_EOS:
      g_print ("End of stream\n");
      g_main_loop_quit (loop);
      break;
    case GST_MESSAGE_WARNING:
    {
      gchar *debug;
      GError *error;
      gst_message_parse_warning (msg, &error, &debug);
      g_printerr ("WARNING from element %s: %s\n",
          GST_OBJECT_NAME (msg->src), error->message);
      g_free (debug);
      g_printerr ("Warning: %s\n", error->message);
      g_error_free (error);
      break;
    }
    case GST_MESSAGE_ERROR:
    {
      gchar *debug;
      GError *error;
      gst_message_parse_error (msg, &error, &debug);
      g_printerr ("ERROR from element %s: %s\n",
          GST_OBJECT_NAME (msg->src), error->message);
      if (debug)
        g_printerr ("Error details: %s\n", debug);
      g_free (debug);
      g_error_free (error);
      g_main_loop_quit (loop);
      break;
    }
    case GST_MESSAGE_ELEMENT:
    {
      if (gst_nvmessage_is_stream_eos (msg)) {
        guint stream_id;
        if (gst_nvmessage_parse_stream_eos (msg, &stream_id)) {
          g_print ("Got EOS from stream %d\n", stream_id);
        }
      }
      break;
    }
    default:
      break;
  }
  return TRUE;
}

/* Tracker config parsing */

#define CHECK_ERROR(error) \
    if (error) { \
        g_printerr ("Error while parsing config file: %s\n", error->message); \
        goto done; \
    }

#define CONFIG_GROUP_TRACKER "tracker"
#define CONFIG_GROUP_TRACKER_WIDTH "tracker-width"
#define CONFIG_GROUP_TRACKER_HEIGHT "tracker-height"
#define CONFIG_GROUP_TRACKER_LL_CONFIG_FILE "ll-config-file"
#define CONFIG_GROUP_TRACKER_LL_LIB_FILE "ll-lib-file"
#define CONFIG_GROUP_TRACKER_ENABLE_BATCH_PROCESS "enable-batch-process"
#define CONFIG_GPU_ID "gpu-id"

static gchar *
get_absolute_file_path (gchar *cfg_file_path, gchar *file_path)
{
  gchar abs_cfg_path[PATH_MAX + 1];
  gchar *abs_file_path;
  gchar *delim;

  if (file_path && file_path[0] == '/') {
    return file_path;
  }

  if (!realpath (cfg_file_path, abs_cfg_path)) {
    g_free (file_path);
    return NULL;
  }

  // Return absolute path of config file if file_path is NULL.
  if (!file_path) {
    abs_file_path = g_strdup (abs_cfg_path);
    return abs_file_path;
  }

  delim = g_strrstr (abs_cfg_path, "/");
  *(delim + 1) = '\0';

  abs_file_path = g_strconcat (abs_cfg_path, file_path, NULL);
  g_free (file_path);

  return abs_file_path;
}

static gboolean
set_tracker_properties (GstElement *nvtracker)
{
  gboolean ret = FALSE;
  GError *error = NULL;
  gchar **keys = NULL;
  gchar **key = NULL;
  GKeyFile *key_file = g_key_file_new ();

  if (!g_key_file_load_from_file (key_file, TRACKER_CONFIG_FILE, G_KEY_FILE_NONE,
          &error)) {
    g_printerr ("Failed to load config file: %s\n", error->message);
    return FALSE;
  }

  keys = g_key_file_get_keys (key_file, CONFIG_GROUP_TRACKER, NULL, &error);
  CHECK_ERROR (error);

  for (key = keys; *key; key++) {
    if (!g_strcmp0 (*key, CONFIG_GROUP_TRACKER_WIDTH)) {
      gint width =
          g_key_file_get_integer (key_file, CONFIG_GROUP_TRACKER,
          CONFIG_GROUP_TRACKER_WIDTH, &error);
      CHECK_ERROR (error);
      g_object_set (G_OBJECT (nvtracker), "tracker-width", width, NULL);
    } else if (!g_strcmp0 (*key, CONFIG_GROUP_TRACKER_HEIGHT)) {
      gint height =
          g_key_file_get_integer (key_file, CONFIG_GROUP_TRACKER,
          CONFIG_GROUP_TRACKER_HEIGHT, &error);
      CHECK_ERROR (error);
      g_object_set (G_OBJECT (nvtracker), "tracker-height", height, NULL);
    } else if (!g_strcmp0 (*key, CONFIG_GPU_ID)) {
      guint gpu_id =
          g_key_file_get_integer (key_file, CONFIG_GROUP_TRACKER,
          CONFIG_GPU_ID, &error);
      CHECK_ERROR (error);
      g_object_set (G_OBJECT (nvtracker), "gpu_id", gpu_id, NULL);
    } else if (!g_strcmp0 (*key, CONFIG_GROUP_TRACKER_LL_CONFIG_FILE)) {
      char* ll_config_file = get_absolute_file_path (TRACKER_CONFIG_FILE,
                g_key_file_get_string (key_file,
                    CONFIG_GROUP_TRACKER,
                    CONFIG_GROUP_TRACKER_LL_CONFIG_FILE, &error));
      CHECK_ERROR (error);
      g_object_set (G_OBJECT (nvtracker), "ll-config-file", ll_config_file, NULL);
    } else if (!g_strcmp0 (*key, CONFIG_GROUP_TRACKER_LL_LIB_FILE)) {
      char* ll_lib_file = get_absolute_file_path (TRACKER_CONFIG_FILE,
                g_key_file_get_string (key_file,
                    CONFIG_GROUP_TRACKER,
                    CONFIG_GROUP_TRACKER_LL_LIB_FILE, &error));
      CHECK_ERROR (error);
      g_object_set (G_OBJECT (nvtracker), "ll-lib-file", ll_lib_file, NULL);
    } else if (!g_strcmp0 (*key, CONFIG_GROUP_TRACKER_ENABLE_BATCH_PROCESS)) {
      gboolean enable_batch_process =
          g_key_file_get_integer (key_file, CONFIG_GROUP_TRACKER,
          CONFIG_GROUP_TRACKER_ENABLE_BATCH_PROCESS, &error);
      CHECK_ERROR (error);
      g_object_set (G_OBJECT (nvtracker), "enable_batch_process",
                    enable_batch_process, NULL);
    } else {
      g_printerr ("Unknown key '%s' for group [%s]", *key,
          CONFIG_GROUP_TRACKER);
    }
  }

  ret = TRUE;
done:
  if (error) {
    g_error_free (error);
  }
  if (keys) {
    g_strfreev (keys);
  }
  if (!ret) {
    g_printerr ("%s failed", __func__);
  }
  return ret;
}


static void
cb_newpad (GstElement * decodebin, GstPad * decoder_src_pad, gpointer data)
{
  GstCaps *caps = gst_pad_get_current_caps (decoder_src_pad);
  if (!caps) {
    caps = gst_pad_query_caps (decoder_src_pad, NULL);
  }
  const GstStructure *str = gst_caps_get_structure (caps, 0);
  const gchar *name = gst_structure_get_name (str);
  GstElement *source_bin = (GstElement *) data;
  GstCapsFeatures *features = gst_caps_get_features (caps, 0);

  /* Need to check if the pad created by the decodebin is for video and not
   * audio. */
  if (!strncmp (name, "video", 5)) {
    /* Link the decodebin pad only if decodebin has picked nvidia
     * decoder plugin nvdec_*. We do this by checking if the pad caps contain
     * NVMM memory features. */
    if (gst_caps_features_contains (features, GST_CAPS_FEATURES_NVMM)) {
      /* Get the source bin ghost pad */
      GstPad *bin_ghost_pad = gst_element_get_static_pad (source_bin, "src");
      if (!gst_ghost_pad_set_target (GST_GHOST_PAD (bin_ghost_pad),
              decoder_src_pad)) {
        g_printerr ("Failed to link decoder src pad to source bin ghost pad\n");
      }
      gst_object_unref (bin_ghost_pad);
    } else {
      g_printerr ("Error: Decodebin did not pick nvidia decoder plugin.\n");
    }
  }
}

static void
decodebin_child_added (GstChildProxy * child_proxy, GObject * object,
    gchar * name, gpointer user_data)
{
  g_print ("Decodebin child added: %s\n", name);
  if (g_strrstr (name, "decodebin") == name) {
    g_signal_connect (G_OBJECT (object), "child-added",
        G_CALLBACK (decodebin_child_added), user_data);
  }
  if (g_strrstr (name, "source") == name) {
        g_object_set(G_OBJECT(object),"drop-on-latency",true,NULL);
  }

}

static GstElement *
create_source_bin (guint index, gchar * uri)
{
  GstElement *bin = NULL, *uri_decode_bin = NULL;
  gchar bin_name[16] = { };

  g_snprintf (bin_name, 15, "source-bin-%02d", index);
  /* Create a source GstBin to abstract this bin's content from the rest of the
   * pipeline */
  bin = gst_bin_new (bin_name);

  /* Source element for reading from the uri.
   * We will use decodebin and let it figure out the container format of the
   * stream and the codec and plug the appropriate demux and decode plugins. */
  if (PERF_MODE) {
    uri_decode_bin = gst_element_factory_make ("nvurisrcbin", "uri-decode-bin");
    g_object_set (G_OBJECT (uri_decode_bin), "file-loop", TRUE, NULL);
    g_object_set (G_OBJECT (uri_decode_bin), "cudadec-memtype", 0, NULL);
  } else {
    uri_decode_bin = gst_element_factory_make ("uridecodebin", "uri-decode-bin");
  }

  if (!bin || !uri_decode_bin) {
    g_printerr ("One element in source bin could not be created.\n");
    return NULL;
  }

  /* We set the input uri to the source element */
  g_object_set (G_OBJECT (uri_decode_bin), "uri", uri, NULL);

  /* Connect to the "pad-added" signal of the decodebin which generates a
   * callback once a new pad for raw data has beed created by the decodebin */
  g_signal_connect (G_OBJECT (uri_decode_bin), "pad-added",
      G_CALLBACK (cb_newpad), bin);
  g_signal_connect (G_OBJECT (uri_decode_bin), "child-added",
      G_CALLBACK (decodebin_child_added), bin);

  gst_bin_add (GST_BIN (bin), uri_decode_bin);

  /* We need to create a ghost pad for the source bin which will act as a proxy
   * for the video decoder src pad. The ghost pad will not have a target right
   * now. Once the decode bin creates the video decoder and generates the
   * cb_newpad callback, we will set the ghost pad target to the video decoder
   * src pad. */
  if (!gst_element_add_pad (bin, gst_ghost_pad_new_no_target ("src",
              GST_PAD_SRC))) {
    g_printerr ("Failed to add ghost pad in source bin\n");
    return NULL;
  }

  return bin;
}

int
main (int argc, char *argv[])
{
  GMainLoop *loop = NULL;
  GstElement *pipeline = NULL, *streammux = NULL, *sink = NULL, *pgie = NULL, *nvtracker = NULL,
      *queue1, *queue2, *queue3, *queue4, *queue5, *nvvidconv = NULL, *dsexample = NULL,
      *nvosd = NULL, *tiler = NULL, *nvdslogger = NULL;
  GstBus *bus = NULL;
  guint bus_watch_id;
  GstPad *tracker_src_pad = NULL;
  GstPad *osd_sink_pad = NULL;
  guint i =0, num_sources = 0;
  guint tiler_rows, tiler_columns;
  guint pgie_batch_size;
  gboolean yaml_config = FALSE;
  NvDsGieType pgie_type = NVDS_GIE_PLUGIN_INFER;
  PERF_MODE = g_getenv("NVDS_TEST3_PERF_MODE") &&
      !g_strcmp0(g_getenv("NVDS_TEST3_PERF_MODE"), "1");

  int current_device = -1;
  cudaGetDevice(&current_device);
  struct cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, current_device);

  /* Check input arguments */
  if (argc < 2) {
    g_printerr ("Usage: %s <yml file>\n", argv[0]);
    g_printerr ("OR: %s <uri1> [uri2] ... [uriN] \n", argv[0]);
    return -1;
  }

  /* Standard GStreamer initialization */
  gst_init (&argc, &argv);
  loop = g_main_loop_new (NULL, FALSE);

  /* Parse inference plugin type */
  yaml_config = (g_str_has_suffix (argv[1], ".yml") ||
          g_str_has_suffix (argv[1], ".yaml"));

  if (yaml_config) {
    RETURN_ON_PARSER_ERROR(nvds_parse_gie_type(&pgie_type, argv[1],
                "primary-gie"));
  }

  /* Create gstreamer elements */
  /* Create Pipeline element that will form a connection of other elements */
  pipeline = gst_pipeline_new ("dstest3-pipeline");

  /* Create nvstreammux instance to form batches from one or more sources. */
  streammux = gst_element_factory_make ("nvstreammux", "stream-muxer");

  if (!pipeline || !streammux) {
    g_printerr ("One element could not be created. Exiting.\n");
    return -1;
  }
  gst_bin_add (GST_BIN (pipeline), streammux);

  GList *src_list = NULL ;

  if (yaml_config) {

    RETURN_ON_PARSER_ERROR(nvds_parse_source_list(&src_list, argv[1], "source-list"));

    GList * temp = src_list;
    while(temp) {
      num_sources++;
      temp=temp->next;
    }
    g_list_free(temp);
  } else {
      num_sources = argc - 1;
  }

  for (i = 0; i < num_sources; i++) {
    GstPad *sinkpad, *srcpad;
    gchar pad_name[16] = { };

    GstElement *source_bin= NULL;
    if (g_str_has_suffix (argv[1], ".yml") || g_str_has_suffix (argv[1], ".yaml")) {
      g_print("Now playing : %s\n",(char*)(src_list)->data);
      source_bin = create_source_bin (i, (char*)(src_list)->data);
    } else {
      source_bin = create_source_bin (i, argv[i + 1]);
    }
    if (!source_bin) {
      g_printerr ("Failed to create source bin. Exiting.\n");
      return -1;
    }

    gst_bin_add (GST_BIN (pipeline), source_bin);

    g_snprintf (pad_name, 15, "sink_%u", i);
    sinkpad = gst_element_get_request_pad (streammux, pad_name);
    if (!sinkpad) {
      g_printerr ("Streammux request sink pad failed. Exiting.\n");
      return -1;
    }

    srcpad = gst_element_get_static_pad (source_bin, "src");
    if (!srcpad) {
      g_printerr ("Failed to get src pad of source bin. Exiting.\n");
      return -1;
    }

    if (gst_pad_link (srcpad, sinkpad) != GST_PAD_LINK_OK) {
      g_printerr ("Failed to link source bin to stream muxer. Exiting.\n");
      return -1;
    }

    gst_object_unref (srcpad);
    gst_object_unref (sinkpad);

    if (yaml_config) {
      src_list = src_list->next;
    }
  }

  if (yaml_config) {
    g_list_free(src_list);
  }

  /* Use nvinfer or nvinferserver to infer on batched frame. */
  if (pgie_type == NVDS_GIE_PLUGIN_INFER_SERVER) {
    pgie = gst_element_factory_make ("nvinferserver", "primary-nvinference-engine");
  } else {
    pgie = gst_element_factory_make ("nvinfer", "primary-nvinference-engine");
  }
  
  /* We need to have a tracker to track the identified objects */
  nvtracker = gst_element_factory_make ("nvtracker", "tracker");

  /* Add queue elements between every two elements */
  queue1 = gst_element_factory_make ("queue", "queue1");
  queue2 = gst_element_factory_make ("queue", "queue2");
  queue3 = gst_element_factory_make ("queue", "queue3");
  queue4 = gst_element_factory_make ("queue", "queue4");
  queue5 = gst_element_factory_make ("queue", "queue5");

  /* Use nvdslogger for perf measurement. */
  nvdslogger = gst_element_factory_make ("nvdslogger", "nvdslogger");

  /* Use nvtiler to composite the batched frames into a 2D tiled array based
   * on the source of the frames. */
  tiler = gst_element_factory_make ("nvmultistreamtiler", "nvtiler");

  /* Use convertor to convert from NV12 to RGBA as required by nvosd */
  nvvidconv = gst_element_factory_make ("nvvideoconvert", "nvvideo-converter");
  
  dsexample = gst_element_factory_make ("dsexample", "example-plugin");

  /* Create OSD to draw on the converted RGBA buffer */
  nvosd = gst_element_factory_make ("nvdsosd", "nv-onscreendisplay");

  if (PERF_MODE) {
    sink = gst_element_factory_make ("fakesink", "nvvideo-renderer");
  } else {
    /* Finally render the osd output */
    if(prop.integrated) {
      sink = gst_element_factory_make ("nv3dsink", "nv3d-sink");
    } else {
      sink = gst_element_factory_make ("nveglglessink", "nvvideo-renderer");
    }
  }

  if (!pgie || !nvtracker || !nvdslogger || !tiler || !nvvidconv || !dsexample || !nvosd || !sink) {
    g_printerr ("One element could not be created. Exiting.\n");
    return -1;
  }

  if (yaml_config) {

    RETURN_ON_PARSER_ERROR(nvds_parse_streammux(streammux, argv[1],"streammux"));

    RETURN_ON_PARSER_ERROR(nvds_parse_gie(pgie, argv[1], "primary-gie"));

    g_object_get (G_OBJECT (pgie), "batch-size", &pgie_batch_size, NULL);
    if (pgie_batch_size != num_sources) {
      g_printerr
          ("WARNING: Overriding infer-config batch-size (%d) with number of sources (%d)\n",
          pgie_batch_size, num_sources);
      g_object_set (G_OBJECT (pgie), "batch-size", num_sources, NULL);
    }
    
    RETURN_ON_PARSER_ERROR(nvds_parse_tracker(nvtracker, argv[1], "tracker"));

    RETURN_ON_PARSER_ERROR(nvds_parse_osd(nvosd, argv[1],"osd"));

    tiler_rows = (guint) sqrt (num_sources);
    tiler_columns = (guint) ceil (1.0 * num_sources / tiler_rows);
    g_object_set (G_OBJECT (tiler), "rows", tiler_rows, "columns", tiler_columns, NULL);

    RETURN_ON_PARSER_ERROR(nvds_parse_tiler(tiler, argv[1], "tiler"));
    if(prop.integrated) {
      RETURN_ON_PARSER_ERROR(nvds_parse_3d_sink(sink, argv[1], "sink"));
    } else {
      RETURN_ON_PARSER_ERROR(nvds_parse_egl_sink(sink, argv[1], "sink"));
    }

  }
  else {

    g_object_set (G_OBJECT (streammux), "batch-size", num_sources, NULL);

    g_object_set (G_OBJECT (streammux), "width", MUXER_OUTPUT_WIDTH, "height",
        MUXER_OUTPUT_HEIGHT,
        "batched-push-timeout", MUXER_BATCH_TIMEOUT_USEC, NULL);

    /* Configure the nvinfer element using the nvinfer config file. */
    g_object_set (G_OBJECT (pgie),
        "config-file-path", "dstest3_pgie_config.txt", NULL);
        
    if(!prop.integrated) {
    /* Set properties of the nvvideoconvert element
    * requires unified cuda memory for opencv blurring on CPU
    */
      g_object_set (G_OBJECT (nvvidconv), "nvbuf-memory-type", 1, NULL);
    }
        
    /* Set properties of the dsexample element */
    g_object_set (G_OBJECT (dsexample), "full-frame", FALSE, NULL);
    g_object_set (G_OBJECT (dsexample), "blur-objects", TRUE, NULL);
        
    /* Set necessary properties of the tracker element. */
    if (!set_tracker_properties(nvtracker)) {
      g_printerr ("Failed to set tracker properties. Exiting.\n");
      return -1;
    }

    /* Override the batch-size set in the config file with the number of sources. */
    g_object_get (G_OBJECT (pgie), "batch-size", &pgie_batch_size, NULL);
    if (pgie_batch_size != num_sources) {
      g_printerr
          ("WARNING: Overriding infer-config batch-size (%d) with number of sources (%d)\n",
          pgie_batch_size, num_sources);
      g_object_set (G_OBJECT (pgie), "batch-size", num_sources, NULL);
    }

    tiler_rows = (guint) sqrt (num_sources);
    tiler_columns = (guint) ceil (1.0 * num_sources / tiler_rows);
    /* we set the tiler properties here */
    g_object_set (G_OBJECT (tiler), "rows", tiler_rows, "columns", tiler_columns,
        "width", TILED_OUTPUT_WIDTH, "height", TILED_OUTPUT_HEIGHT, NULL);

    g_object_set (G_OBJECT (nvosd), "process-mode", OSD_PROCESS_MODE,
        "display-text", OSD_DISPLAY_TEXT, NULL);

    g_object_set (G_OBJECT (sink), "qos", 0, NULL);

  }

  if (PERF_MODE) {
      if(prop.integrated) {
          g_object_set (G_OBJECT (streammux), "nvbuf-memory-type", 4, NULL);
      } else {
          g_object_set (G_OBJECT (streammux), "nvbuf-memory-type", 2, NULL);
      }
  }

  /* we add a message handler */
  bus = gst_pipeline_get_bus (GST_PIPELINE (pipeline));
  bus_watch_id = gst_bus_add_watch (bus, bus_call, loop);
  gst_object_unref (bus);

  /* Set up the pipeline */
  /* we add all elements into the pipeline */
  gst_bin_add_many (GST_BIN (pipeline), queue1, pgie, nvtracker, queue2, nvdslogger, tiler,
      queue3, nvvidconv, dsexample, queue4, nvosd, queue5, sink, NULL);
  /* we link the elements together
  * nvstreammux -> nvinfer -> nvdslogger -> nvtiler -> nvvidconv -> nvosd
  * -> video-renderer */
  if (!gst_element_link_many (streammux, queue1, pgie, nvtracker, queue2, nvdslogger, tiler,
        queue3, nvvidconv, dsexample, queue4, nvosd, queue5, sink, NULL)) {
    g_printerr ("Elements could not be linked. Exiting.\n");
    return -1;
  }

  /* Lets add probe to get informed of the meta data generated, we add probe to
   * the sink pad of the osd element, since by that time, the buffer would have
   * had got all the metadata. */
  tracker_src_pad = gst_element_get_static_pad (nvtracker, "src");
  
  /* Create Context for Object Encoding.
   * Takes GPU ID as a parameter. Passed by user through commandline.
   * Initialized as 0. */
  NvDsObjEncCtxHandle obj_ctx_handle = nvds_obj_enc_create_context (0); //gpu_id
  if (!obj_ctx_handle) {
    g_print ("Unable to create context\n");
    return -1;
  }
  
  if (!tracker_src_pad)
    g_print ("Unable to get src pad\n");
  else
    gst_pad_add_probe (tracker_src_pad, GST_PAD_PROBE_TYPE_BUFFER,
        tracker_src_pad_buffer_probe, (gpointer) obj_ctx_handle, NULL);
  gst_object_unref (tracker_src_pad);
  
  
  /* Lets add probe to get informed of the meta data generated, we add probe to
   * the sink pad of the osd element, since by that time, the buffer would have
   * had got all the metadata. */
  osd_sink_pad = gst_element_get_static_pad (nvosd, "src");
  if (!osd_sink_pad)
    g_print ("Unable to get sink pad\n");
  else
    gst_pad_add_probe (osd_sink_pad, GST_PAD_PROBE_TYPE_BUFFER,
        osd_src_pad_buffer_probe, (gpointer) obj_ctx_handle, NULL);
  gst_object_unref (osd_sink_pad);
  

  /* Set the pipeline to "playing" state */
  if (yaml_config) {
    g_print ("Using file: %s\n", argv[1]);
  }
  else {
    g_print ("Now playing:");
    for (i = 0; i < num_sources; i++) {
      g_print (" %s,", argv[i + 1]);
    }
    g_print ("\n");
  }
  gst_element_set_state (pipeline, GST_STATE_PLAYING);

  /* Wait till pipeline encounters an error or EOS */
  g_print ("Running...\n");
  g_main_loop_run (loop);
  
  /* Destroy context for Object Encoding */
  nvds_obj_enc_destroy_context (obj_ctx_handle);

  /* Out of the main loop, clean up nicely */
  g_print ("Returned, stopping playback\n");
  gst_element_set_state (pipeline, GST_STATE_NULL);
  g_print ("Deleting pipeline\n");
  gst_object_unref (GST_OBJECT (pipeline));
  g_source_remove (bus_watch_id);
  g_main_loop_unref (loop);
  return 0;
}
