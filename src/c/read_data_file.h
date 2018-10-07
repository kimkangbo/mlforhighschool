#ifndef __READ_DATA_FILE_H
#define __READ_DATA_FILE_H

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <stdbool.h>

#include "data_types.h"

struct mnist_data
{
	FILE *fp_data;
	FILE *fp_label;
	
	ui_32 data_num;
	
	ui_32 pix_row;
	ui_32 pix_col;
};

FILE * open_mnist_file(const c_8 *file_name);
i_32 get_mnist_file_points(struct mnist_data *mnist_obj, const c_8 *files[]);
ui_32 close_file(FILE *fp);
ui_32 close_mnist_file_points(struct mnist_data *mnist_obj);
void print_array_by_hex(uc_8 *buff, ui_32 n_byte);
ui_32 change_memory_to_ui_32(uc_8 *buff, ui_32 byte);
ui_32 read_data(uc_8 *buff, ui_32 n_byte, FILE *fp, bool convert_data);
ui_32 read_files_info(struct mnist_data *mnist_obj);
ui_32 print_pix(uc_8 pix);
ui_32 read_image(struct mnist_data mnist_obj, uc_8 *pixs, ui_32 n_byte);
ui_32 read_label(struct mnist_data mnist_obj);
ui_32 read_images(struct mnist_data mnist_obj);
ui_32 get_image_data(struct mnist_data mnist_obj, uc_8 *pixs);
ui_32 get_label(struct mnist_data mnist_obj, uc_8 *label);

#endif