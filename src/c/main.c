#include <stdio.h>
#include "data_types.h"
#include "read_data_file.h"

i_32 main(){
	const c_8 *train_img = "";
	const c_8 *train_label = "";
	const c_8 *w_set = "";
	const c_8 *test_img = "";
	const c_8 *test_label = "";
	int data_count = 0;
	
	data_count = read_data_file(train_img, train_label);
	printf("#train_data_set:%d",data_count);
	
	data_count = read_data_file(test_img, test_label);
	printf("#test_data_set:%d",data_count);	
	
	return 0;
}