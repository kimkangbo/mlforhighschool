#include <stdio.h>
#include "data_types.h"
#include "read_data_file.h"
#include "neural_networks_structure.h"
#include "forward_propagation.h"

const char *train_files[] = {	
								"../../data/train-images.idx3-ubyte", 
								"../../data/train-labels.idx1-ubyte"
							};
							
const char *test_files[] = {	
								"../../../data/t10k-images.idx3-ubyte",
								"../../../data/t10k-labels.idx1-ubyte"
							};							

int main()
{
	int i=0;
	struct mnist_data train_data = {NULL, NULL, 0, 0, 0};
//	struct mnist_data test_data = {NULL, NULL, 0, 0, 0};
	ui_32 n_byte = 0;		
	uc_8 *pixs = NULL;
	uc_8 label = 0;
	
	get_mnist_file_points(&train_data, train_files);
	read_files_info(&train_data);
	
	n_byte = (train_data.pix_row * train_data.pix_col) * sizeof(uc_8);		
	pixs = malloc(n_byte);
	
	for(i=0; i<train_data.data_num; i++){
        printf("# %d th\n", i);
		get_image_data(train_data, pixs);
		get_label(train_data, &label);
		learning_image(pixs, label);
	}
	
	close_mnist_file_points(&train_data);
	
	return 0;
}