#ifndef __NEURAL_NETWORKS_STRUCTURE_H
#define __NEURAL_NETWORKS_STRUCTURE_H

#include "data_types.h"

#define INPUTS				784		//24*24 = 784
#define HIDDEN_LAYERS		1
#define NEURONS_1			64
//#define NEURONS_2			64
//...
#define OUTPUTS				10		// 0~9

struct nn_info
{
	ui_32 input_num;
	ui_32 w_num;
	ui_32 w_array_num_for_input;
	ui_32 w_array_num_for_output;
	ui_32 hl_num;
	ui_32 neuron_1_num;
	ui_32 output_num;

	ui_32 input_num_size;
	ui_32 w_num_size;
	ui_32 w_array_num_for_input_size;
	ui_32 w_array_num_for_output_size;
	ui_32 hl_num_size;
	ui_32 neuron_1_num_size;
	ui_32 output_num_size;
};

void learning_image(uc_8 *pixs,	uc_8 label);

#endif