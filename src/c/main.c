#include <stdio.h>
#include "data_types.h"
#include "read_data_file.h"

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
	struct mnist_data train_data = {NULL, NULL, 0, 0, 0};
//	struct mnist_data test_data = {NULL, NULL, 0, 0, 0};	
	
	get_mnist_file_points(&train_data, train_files);
	read_files_info(&train_data);
	read_images(train_data);
	close_mnist_file_points(&train_data);
	
	return 0;
}