/**

This code is under Apache License 2.0.
This code is refered to https://github.com/kernel-bz/ml written by JungJaeJoon(rgbi3307@nate.com) on the www.kernel.bz

Copyright by kimkangbo@gmail.com @2018
https://github.com/kimkangbo/mlforhighschool
 
**/

#include "read_data_file.h"

#ifdef DEBUG
    #define dprintf(fmt,args...)     printf(fmt,##args)
#else
    #define dprintf(fmt,args...)
#endif

FILE * open_mnist_file(const c_8 *file_name)
{
	FILE *fp = fopen(file_name, "rb");
	if(fp == NULL)
	{
		printf("# Error open %s\n", file_name);
	} 
	
	return fp;
}

i_32 get_mnist_file_points(struct mnist_data *mnist_obj, const c_8 *files[])
{
	i_32 result = 0;
	
	mnist_obj->fp_data = open_mnist_file(files[0]);
	mnist_obj->fp_label = open_mnist_file(files[1]);
	
	if(mnist_obj->fp_data != NULL || mnist_obj->fp_label != NULL)
	{
		result = -1;
	}
	
	return result;
}

ui_32 close_file(FILE *fp)
{
	ui_32 file_count = 0;
	i_32 result =  0;
		
	if( result == 0 )
	{
		file_count = 1;
	} 
	else if( result == EOF ) 
	{
		printf("# EOF Error to close file: Err %d\n", result);
	}
	
	return file_count;
}
	
ui_32 close_mnist_file_points(struct mnist_data *mnist_obj)
{	
	ui_32 closed_file_count = 0;
	
	closed_file_count += close_file(mnist_obj->fp_data);
	closed_file_count += close_file(mnist_obj->fp_label);		
	
	dprintf("# closed_file_number: %d\n", closed_file_count);
	
	return closed_file_count;
}

void print_array_by_hex(uc_8 *buff, ui_32 n_byte)
{
	ui_32 i = 0;

	printf("# hex data for %d bytes\n", n_byte);	
	for(i=0; i<n_byte; i++)
	{
		printf("%x,", buff[i]);
	}
	
	printf("\n");
}

ui_32 change_memory_to_ui_32(uc_8 *buff, ui_32 byte)
{
	ui_32 result = 0;
	ui_32 i = 0;
	
	for(i=0; i<byte; i++)
	{
		result = result | ((ui_32) buff[i] << 8*(byte-1-i));
		dprintf("#shifted result : %08x\n", result);
	}
	
	return result;
}

ui_32 read_data(uc_8 *buff, ui_32 n_byte, FILE *fp, bool convert_data)
{
 	ui_32 n_size = 0;
	ui_32 result = 0;
	
	memset(buff, 0, n_byte);
	n_size = fread(buff, n_byte, 1, fp);
	
	if(n_size < 1)
	{
		printf("# Error read data(size: %d) from fp_train_data\n", n_size);
		result = -1;
	}else 
	{
//		print_array_by_hex(buff, n_byte);
		if(convert_data == true)
		{
			result = change_memory_to_ui_32(buff, n_byte);
		}
		dprintf("%x, %d\n", result, result);		
	}

	return result;
}

ui_32 read_files_info(struct mnist_data *mnist_obj)
{
	uc_8 buff[5] = {0,};
	ui_32 result = 0;
	
	read_data(buff, 4, mnist_obj->fp_data, false);
	mnist_obj->data_num = read_data(buff, 4, mnist_obj->fp_data, true);	
	mnist_obj->pix_row = read_data(buff, 4, mnist_obj->fp_data, true);
	mnist_obj->pix_col = read_data(buff, 4, mnist_obj->fp_data, true);	

	read_data(buff, 4, mnist_obj->fp_label, false);
	read_data(buff, 4, mnist_obj->fp_label, false);	
	
	printf("# data set: %dea, pixel %d*%d\n", mnist_obj->data_num, mnist_obj->pix_row, mnist_obj->pix_col); 
	
	return result;
}

ui_32 print_pix(uc_8 pix)
{
	uc_8 pattern = 0;
	
	if(pix > 224)
	{
		pattern = '@';
	}
	else if(pix > 192)
	{
		pattern = '#';
	}
	else if(pix > 160)
	{
		pattern = '$';
	}
	else if(pix > 128)
	{
		pattern = '%';
	}
	else if(pix > 96)
	{
		pattern = '&';
	}
	else if(pix > 64)
	{
		pattern = '*';
	}
	else if(pix > 32)
	{
		pattern = '+';
	}
	else if( pix > 16 )
	{
		pattern = '^';
	}
	else if( pix > 8 )
	{
		pattern = '-';
	}		
	else if( pix == 0 )
	{
		pattern= 32;
	}
	
	printf("%c", pattern);	
	
	return pattern;
}

ui_32 read_image(struct mnist_data mnist_obj, uc_8 *pixs, ui_32 n_byte)
{
	ui_32 result = 0;
	ui_32 i = 0;
	
	printf("#n_byte: %d\n",n_byte);
	result = read_data(pixs, n_byte, mnist_obj.fp_data, false);
	
	for(i=0; i<n_byte; i++)
	{
		print_pix(pixs[i]);
		if((i+1)%(mnist_obj.pix_col) == 0)
		{
			puts("");
		}
	}
	
	return result;
}

ui_32 read_label(struct mnist_data mnist_obj)
{
	uc_8 label = 0;
	ui_32 n_byte = 1;
	
	read_data(&label, n_byte, mnist_obj.fp_label, false);
	printf("# picture label: %d\n", label);
	
	return label;
}

ui_32 read_images(struct mnist_data mnist_obj)
{
	uc_8 *pixs = NULL;
	ui_32 data_num = 0;
	ui_32 n_byte = 0, i = 0, read_count = 0;
	
	n_byte = (mnist_obj.pix_row * mnist_obj.pix_col) * sizeof(uc_8);		
	pixs = malloc(n_byte);
	data_num = mnist_obj.data_num;
	
	for(i=0; i<data_num; i++)
	{
		read_image(mnist_obj, pixs, n_byte);
		read_label(mnist_obj);
		read_count++;
		
		printf("# If you want to quit hit 'q'. Or not, hit other keys: ");		
		if(getchar() == 'q')
		{
			break;
		}
	}
	
	free(pixs);
	pixs = NULL;
	
	return read_count;
}

ui_32 get_image_data(struct mnist_data mnist_obj, uc_8 *pixs)
{
	ui_32 n_byte = 0, result = 0;
	
	n_byte = (mnist_obj.pix_row * mnist_obj.pix_col) * sizeof(uc_8);		
	result = read_data(pixs, n_byte, mnist_obj.fp_data, false);
		
	return result;
}

ui_32 get_label(struct mnist_data mnist_obj, uc_8 *label)
{
	ui_32 n_byte = 1, result = 0;
	
	read_data(label, n_byte, mnist_obj.fp_label, false);
	
	return result;
}