#include "read_data_file.h"
#include "neural_networks_structure.h"

ui_32 init_Variables(uc_8 *input, f_32 **w, f_32 **hl, f_32 *ouput)
{
	ui_32 result = 0;
	ui_32 input_num = INPUTS+1;						// becase of bias, add 1 to input_num, Z = x*w + b = x'*w' 
	ui_32 w_num = HIDDEN_LAYERS+1;					// becase of bias, add 1 to input_num, Z = h*w + b = h'*w' 
	ui_32 w_array_num_for_input = (INPUTS*NEURONS_1) + 1;	// becase of bias, add 1 to input_num, Z = x*w + b = x'*w'
	ui_32 neuron_1_num = NEURONS_1 + 1;				// becase of bias, add 1 to input_num, Z = h*w + b = h'*w' 
	ui_32 w_array_num_for_output = (OUTPUTS*NEURONS_1) + 1;	// becase of bias, add 1 to input_num, Z = h*w + b = h'*w' 
	
	input = malloc(input_num*sizeof(uc_8));
	
	w = malloc(w_num*sizeof(f_32 *));
	w[1] = malloc(w_array_num_for_input*sizeof(f_32));
	w[2] = malloc(w_array_num_for_output*sizeof(f_32));

	hl = malloc(HIDDEN_LAYERS*sizeof(f_32 *));	
	hl[1] = malloc(neuron_1_num*sizeof(f_32));
	
	output = malloc(OUTPUTS*sizeof(f_32));
	
	return result;
}

void set_Variables(void)
{
	uc_8 *input = NULL;
	f_32 **w = NULL;
	f_32 **hl = NULL;
	f_32 *ouput = NULL;
	
	init_Variables(input, w, hl, ouput);
	
}