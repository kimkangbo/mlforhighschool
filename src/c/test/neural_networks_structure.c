#include "read_data_file.h"
#include "neural_networks_structure.h"

struct nn_info malloc_variables(uc_8 **input, f_32 ***w, f_32 ***hl, f_32 **output, uc_8 **y_data)
{
	struct nn_info nn_stc;
	
	nn_stc.input_num = INPUTS+1;						// becase of bias, add 1 to input_num, Z = x*w + b = x'*w' 
	nn_stc.w_num = HIDDEN_LAYERS+1;					// becase of bias, add 1 to input_num, Z = h*w + b = h'*w' 
	nn_stc.w_array_num_for_input = (INPUTS*NEURONS_1) + 1;	// becase of bias, add 1 to input_num, Z = x*w + b = x'*w'
	nn_stc.w_array_num_for_output = (OUTPUTS*NEURONS_1) + 1;	// becase of bias, add 1 to input_num, Z = h*w + b = h'*w' 	
	nn_stc.hl_num = HIDDEN_LAYERS;
	nn_stc.neuron_1_num = NEURONS_1 + 1;				// becase of bias, add 1 to input_num, Z = h*w + b = h'*w' 
	nn_stc.output_num = OUTPUTS;

	nn_stc.input_num_size = nn_stc.input_num*sizeof(uc_8);						
	nn_stc.w_num_size = nn_stc.w_num*sizeof(f_32 *);					
	nn_stc.w_array_num_for_input_size = nn_stc.w_array_num_for_input*sizeof(f_32);	
	nn_stc.w_array_num_for_output_size = nn_stc.w_array_num_for_output*sizeof(f_32);		
	nn_stc.hl_num_size = nn_stc.hl_num*sizeof(f_32); 
	nn_stc.neuron_1_num_size = nn_stc.neuron_1_num*sizeof(f_32);				
	nn_stc.output_num_size = nn_stc.output_num*sizeof(f_32);
	
	*input = (uc_8 *)malloc(nn_stc.input_num_size);	
	*w = (f_32 **)malloc(nn_stc.w_num_size);
	(*w)[0] = (f_32 *)malloc(nn_stc.w_array_num_for_input_size);
	(*w)[1] = (f_32 *)malloc(nn_stc.w_array_num_for_output_size);
	*hl = (f_32 **)malloc(nn_stc.hl_num_size);	
	(*hl)[0] = (f_32 *)malloc(nn_stc.neuron_1_num_size);	
	*output = (f_32 *)malloc(nn_stc.output_num_size);
	*y_data = (uc_8 *)malloc(nn_stc.output_num_size);	

	// for f_32 ****tr
/*	
	*tr = (f_32 ***)malloc(2*sizeof(f_32 **));
	(*tr)[0] = (f_32 **)malloc(2*sizeof(f_32 *));
	(*tr)[1] = (f_32 **)malloc(2*sizeof(f_32 *));
	(*tr)[0][0] = (f_32 *)malloc(3*sizeof(f_32));
	(*tr)[0][1] = (f_32 *)malloc(3*sizeof(f_32));
	(*tr)[1][0] = (f_32 *)malloc(3*sizeof(f_32));
	(*tr)[1][1] = (f_32 *)malloc(3*sizeof(f_32));
*/	
	return nn_stc;
}

void init_variables(uc_8 *input, f_32 **w, f_32 **hl, f_32 *output, uc_8 *y_data, struct nn_info nn_stc, uc_8 *pixs, uc_8 label)
{
	memset(input, 0, nn_stc.input_num_size);
	memset(w[0], 0, nn_stc.w_array_num_for_input_size);			
	memset(w[1], 0, nn_stc.w_array_num_for_output_size);	
	memset(hl[0], 0, nn_stc.neuron_1_num_size);	
	memset(output, 0, nn_stc.output_num_size);
	memset(y_data, 0, nn_stc.output_num_size);
	
	memcpy(input, pixs, nn_stc.input_num_size);
	input[nn_stc.input_num_size-1] = 1; 		// for bias of input and w[0]
	hl[0][nn_stc.neuron_1_num_size-1] = 1;		// for bias of hl and w[1]
	
	y_data[label] = 1;
}

void print_nn_info(struct nn_info nn_stc)
{
	printf("# input_num: %d\n", nn_stc.input_num);
	printf("# w_num: %d\n", nn_stc.w_num);
	printf("# w_array_num_for_input: %d\n", nn_stc.w_array_num_for_input);
	printf("# w_array_num_for_output: %d\n", nn_stc.w_array_num_for_output);
	printf("# hl_num: %d\n", nn_stc.hl_num);
	printf("# neuron_1_num: %d\n", nn_stc.neuron_1_num);
	printf("# output_num: %d\n", nn_stc.output_num);	
	puts("");
	printf("# input_num_size: %d\n", nn_stc.input_num_size);
	printf("# w_num_size: %d\n", nn_stc.w_num_size);
	printf("# w_array_num_for_input_size: %d\n", nn_stc.w_array_num_for_input_size);
	printf("# w_array_num_for_output_size: %d\n", nn_stc.w_array_num_for_output_size);
	printf("# hl_size: %d\n", nn_stc.hl_num_size);
	printf("# neuron_1_num_size: %d\n", nn_stc.neuron_1_num_size);
	printf("# output_num_size: %d\n", nn_stc.output_num_size);	
}

void free_variables(uc_8 *input, f_32 **w, f_32 **hl, f_32 *output, uc_8 *y_data)
{
	free(input);
	free(w[0]);
	free(w[1]);
	free(w);
	free(hl[0]);
	free(hl);
	free(output);
	free(y_data);
}

void print_data_uc_8(const char* title, uc_8 *data, ui_32 size)
{
	ui_32 i=0;
	printf("# %s\n", title);
	for(i=0; i<size; i++){
		printf("%d,", data[i]);
	}
	puts("");
}

void forward_propagation(uc_8 *input, f_32 **w, f_32 **hl, f_32 *output, uc_8 *y_data, struct nn_info nn_stc)
{
}

void backward_propagation(uc_8 *input, f_32 **w, f_32 **hl, f_32 *output, uc_8 *y_data, struct nn_info nn_stc)
{
}

void learning_image(uc_8 *pixs,	uc_8 label)
{
	uc_8 *input = NULL;
	f_32 **w = NULL;
	f_32 **hl = NULL;
	f_32 *output = NULL;
	uc_8 *y_data = NULL;
	
	struct nn_info nn_stc;
		
	nn_stc = malloc_variables(&input, &w, &hl, &output, &y_data);
//	print_nn_info(nn_stc);
//	Setting Data
	init_variables(input, w, hl, output, y_data, nn_stc, pixs, label);
	
// Read images
//	print_data_uc_8("input", input, nn_stc.input_num_size);
//	print_data_uc_8("y_data", y_data, nn_stc.output_num);
//	printf("#label: %d\n", label);
	
	free_variables(input, w, hl, output, y_data);
}