/**
 *  file name:  nn.c
 *  function:   Neural Network for Machine Learning
 *  1st author:     JungJaeJoon(rgbi3307@nate.com) on the www.kernel.bz
 *  2nd author: 	KimKangBo(kimkangbo@naver.com) 
 *
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <fcntl.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>

#define NUM_INPUTS     784          ///28 * 28
#define NUM_HIDDEN_1   64           ///Accuracy: 0.9259, 0.9358, 0.936, 0.9371, 0.9425, 0.9429, 0.9402 RunTime: ?m
#define NUM_HIDDEN_2   64           ///Accuracy: ?, RunTime: ?m
#define NUM_OUTPUTS    10
#define ACT_TANH        1
#define ACT_RELU        2
#define ACT_TYPE		ACT_RELU

///Weights
float W0[NUM_INPUTS+1][NUM_HIDDEN_1];
float W1[NUM_HIDDEN_1+1][NUM_HIDDEN_2];
float W2[NUM_HIDDEN_2+1][NUM_OUTPUTS];


int debug = 1; //0 for now debugging, 1 for the loss each iteration, 2 for all vectors/matrices each iteration

static void nn_debug (const char *label, float *m, int rows, int cols)
{
    int i;

	printf ("   %s:\n", label);
	for (i=0; i<rows*cols; i++)
        printf ("%10.5f%c", m[i], (cols>1 && i%cols==cols-1) ? '\n' : ' ');
	if (cols==1) printf ("\n");
}

static float activation(float x, int activation_type){
	float result = 0.0;
	
	switch(activation_type){
		case ACT_TANH: // tahn
			result = tanh(x); //apply activation function on other hidden nodes
			break;
		case ACT_RELU: // ReLU
			if(x>0.0){
				result = x;
			}else{
				result = 0;
			}
			break;
		default:
			result = tanh(result); //apply activation function on other hidden nodes
			break;
	}
	
	return result;
}

static float diff_actv(float x, int activation_type){
	float result = 0.0;
	
	switch(activation_type){
		case ACT_TANH: // tahn
			result = 1-pow(tanh(x),2); //apply activation function gradient temporary; //apply activation function on other hidden nodes
			break;
		case ACT_RELU: // ReLU
			if(x>0.0){
				result = 1;
			}else{
				result = 0;
			}
			break;
		default:
			result = 1-pow(tanh(x),2); //apply activation function on other hidden nodes
			break;
	}
	
	return result;
}

static float nn_learning(float *x, float *y, float learningrate)
{
	float Z0[NUM_HIDDEN_1]; //weighted sums for the 1st hidden nodes
	float Z1[NUM_HIDDEN_2]; //weighted sums for the 2nd hidden nodes
	float Z2[NUM_OUTPUTS];  //weighted sums for the output nodes
	float p[NUM_OUTPUTS]; //activation values of the output nodes

	float h0[NUM_HIDDEN_1+1]; //activation values of the 1st hidden nodes, including one extra for the bias
	float h1[NUM_HIDDEN_2+1]; //activation values of the 2nd hidden nodes, including one extra for the bias	
	float err[NUM_OUTPUTS]; //error in the output

	float dW0[NUM_INPUTS+1][NUM_HIDDEN_1]; //adjustments to weights between inputs x and 1st hidden nodes
	float dW1[NUM_HIDDEN_1+1][NUM_HIDDEN_2]; //adjustments to weights between inputs x and 2nd hidden nodes
	float dW2[NUM_HIDDEN_2+1][NUM_OUTPUTS]; //adjustments to weights between hidden nodes and output y

	float temp_sum_1[NUM_HIDDEN_1+1]; //temparary sums for hidden 1 layer
	float temp_sum_2[NUM_HIDDEN_2+1]; //temparary sums for hidden 2 layer

	float loss, sum; //for storing the loss
	int in, ih0, ih1, io; //looping variables for iterations, input nodes, hidden nodes, output nodes

    ///Forward propagation ------------------------------------------------
    //Start the forward pass by calculating the weighted sums and activation values for the hidden layer	
	// 1st hidden layer
    memset (Z0, 0, sizeof (Z0)); //set all the weighted sums to zero
    for (ih0=0; ih0<NUM_HIDDEN_1; ih0++){
        for (in=0; in<NUM_INPUTS+1; in++){
            Z0[ih0] += x[in] * W0[in][ih0]; //multiply and sum inputs * weights
		}
	}
    if (debug>=2) nn_debug ("input/hidden weights", (float*)W0, NUM_INPUTS+1, NUM_HIDDEN_1);
    if (debug>=2) nn_debug ("hidden weighted sums", Z0, NUM_HIDDEN_1, 1);

    h0[NUM_HIDDEN_1]=1.0; //set the bias for the first hidden node to 1
    for (ih0=0; ih0<NUM_HIDDEN_1; ih0++){
        h0[ih0] = activation(Z0[ih0], ACT_TYPE); //apply activation function on other hidden nodes
	}
    if (debug>=2) nn_debug ("hidden node activation values", h0, NUM_HIDDEN_1+1, 1);
	
	// 2nd hidden layer
    memset (Z1, 0, sizeof (Z1)); //set all the weighted sums to zero
    for (ih1=0; ih1<NUM_HIDDEN_2; ih1++){
        for (ih0=0; ih0<NUM_HIDDEN_1+1; ih0++){
            Z1[ih1] += h0[ih0] * W1[ih0][ih1]; //multiply and sum inputs * weights
		}
//		if (debug>=1) printf ("Z1[%d]: %10.5f\n", ih1, Z1[ih1]); 	
	}
    if (debug>=2) nn_debug ("input/hidden weights", (float*)W1, NUM_HIDDEN_1+1, NUM_HIDDEN_2);
    if (debug>=2) nn_debug ("hidden weighted sums", Z1, NUM_HIDDEN_2, 1);

    h1[NUM_HIDDEN_2]=1.0; //set the bias for the second hidden node to 1
    for (ih1=0; ih1<NUM_HIDDEN_2; ih1++){
		h1[ih1] = activation(Z1[ih1], ACT_TYPE); //apply activation function on other hidden nodes
	}
    if (debug>=2) nn_debug ("hidden node activation values", h1, NUM_HIDDEN_2+1, 1);
	
	// Output layer
    memset (Z2, 0, sizeof (Z2)); //set all the weighted sums to zero
    for (io=0; io<NUM_OUTPUTS; io++){
        for (ih1=0; ih1<NUM_HIDDEN_2+1; ih1++){
            Z2[io] += h1[ih1] * W2[ih1][io]; //multiply and sum inputs * weights
		}
	}
    if (debug>=2) nn_debug ("hidden/output weights", (float*)W2, NUM_HIDDEN_2+1, NUM_OUTPUTS);
    if (debug>=2) nn_debug ("output weighted sums", Z2, NUM_OUTPUTS, 1);

    for (sum=0, io=0; io<NUM_OUTPUTS; io++) {
        p[io] = exp (Z2[io]);
//		if (debug>=1) printf ("p[io]: %10.5f, Z2[io]: %10.5f\n", p[io], Z2[io]); //output the loss		
        sum += p[io];		
    } //compute exp(z) for softmax
    for (io=0; io<NUM_OUTPUTS; io++){		
		p[io] /= sum; //apply softmax by dividing by the the sum all the exps
	}
    if (debug>=2) nn_debug ("softmax probabilities", p, NUM_OUTPUTS, 1);

    for (io=0; io<NUM_OUTPUTS; io++){
        err[io] = p[io] - y[io]; //the error for each output
	}
    if (debug>=2) nn_debug ("output error", err, NUM_OUTPUTS, 1);

    for (loss=0, io=0; io<NUM_OUTPUTS; io++){
        loss -= y[io] * log (p[io]); //the loss
//		if (debug>=1) printf ("y[io]: %10.5f, p[io]: %10.5f\n", y[io], p[io]); //output the loss
	}
	
//    if (debug>=1) printf ("loss(cost): %10.5f\n", loss); //output the loss

    /// Back propagation --------------------------------------------------
    //Multiply h1*e to get the adjustments to dW2
    for (ih1=0; ih1<NUM_HIDDEN_2+1; ih1++){
        for (io=0; io<NUM_OUTPUTS; io++){
            dW2[ih1][io] = h1[ih1] * err[io];
		}
	}
    if (debug>=2) nn_debug ("hidden/output weights gradient", (float*)dW2, NUM_HIDDEN_2+1, NUM_OUTPUTS);

	// for dW1
    //Backward propagate the errors and store in the temp_sum_2 temporally
    memset (temp_sum_2, 0, sizeof (temp_sum_2)); //set all the weighted sums to zero
    for (ih1=0; ih1<NUM_HIDDEN_2+1; ih1++){
        for (io=0; io<NUM_OUTPUTS; io++){
            temp_sum_2[ih1] += W2[ih1][io] * err[io]; //multiply and sum inputs * weights
		}
	}
    if (debug>=2) nn_debug ("back propagated error values", temp_sum_2, NUM_HIDDEN_2+1, 1);

    for (ih1=0; ih1<NUM_HIDDEN_2; ih1++){
        Z1[ih1] = temp_sum_2[ih1] * diff_actv(Z1[ih1], ACT_TYPE); //apply activation function gradient by using Z1 temporally
	}
    if (debug>=2) nn_debug ("hidden weighted sums after gradient", Z1, NUM_HIDDEN_2, 1);

    //Multiply x*eh*zh to get the adjustments to dW1, this does not include the bias node
    for (ih0=0; ih0<NUM_HIDDEN_1+1; ih0++){
        for (ih1=0; ih1<NUM_HIDDEN_2; ih1++){
            dW1[ih0][ih1] = h0[ih0] * Z1[ih1];
		}
	}
    if (debug>=2) nn_debug ("hidden_1/hidden_2 weights gradient", (float*)dW1, NUM_HIDDEN_1+1, NUM_HIDDEN_2);

	// for dW0	
    //Backward propagate the errors and store in the temp_sum_1 vector temporally	
    memset (temp_sum_1, 0, sizeof (temp_sum_1)); //set all the weighted sums to zero	
    for (ih0=0; ih0<NUM_HIDDEN_1+1; ih0++){
        for (ih1=0; ih1<NUM_HIDDEN_2; ih1++){
            temp_sum_1[ih0] += W1[ih0][ih1]*Z1[ih1]; //apply activation function gradient temporary
		}
	}	
	
    for (ih0=0; ih0<NUM_HIDDEN_1; ih0++){
		Z0[ih0] = diff_actv(Z0[ih0], ACT_TYPE)*temp_sum_1[ih0]; //apply activation function gradient by using Z0 temporally
	}
    if (debug>=2) nn_debug ("hidden weighted sums after gradient", Z0, NUM_HIDDEN_1, 1);

    //Multiply x*h*zh*sum(zh) to get the adjustments to dW0, this does not include the bias node	
    for (in=0; in<NUM_INPUTS+1; in++){
        for (ih0=0; ih0<NUM_HIDDEN_1; ih0++){
            dW0[in][ih0] = x[in] * Z0[ih0];
		}
	}
    if (debug>=2) nn_debug ("input/hidden weights gradient", (float*)W0, NUM_INPUTS+1, NUM_HIDDEN_1);	
	
    /// Now add in the adjustments ----------------------------------------
    for (ih1=0; ih1<NUM_HIDDEN_2+1; ih1++){
        for (io=0; io<NUM_OUTPUTS; io++){
            W2[ih1][io] -= learningrate * dW2[ih1][io];
		}
	}

    for (ih0=0; ih0<NUM_HIDDEN_1+1; ih0++){
        for (ih1=0; ih1<NUM_HIDDEN_2; ih1++){
            W1[ih0][ih1] -= learningrate * dW1[ih0][ih1];
		}
	}

    for (in=0; in<NUM_INPUTS+1; in++){
        for (ih0=0; ih0<NUM_HIDDEN_1; ih0++){
            W0[in][ih0] -= learningrate * dW0[in][ih0];
		}
	}

    return loss;    ///cost
}

int nn_answer(float *x, float *y)
{
	float Z0[NUM_HIDDEN_1]; //weighted sums for the 1st hidden nodes
	float Z1[NUM_HIDDEN_2]; //weighted sums for the 2nd hidden nodes
	float Z2[NUM_OUTPUTS];  //weighted sums for the output nodes		
	float p[NUM_OUTPUTS]; //activation values of the output nodes

	float h0[NUM_HIDDEN_1+1]; //activation values of the 1st hidden nodes, including one extra for the bias
	float h1[NUM_HIDDEN_2+1]; //activation values of the 2nd hidden nodes, including one extra for the bias	
	
	float sum; //for storing the loss
	int in, ih0, ih1, io, ir; //looping variables for iterations, input nodes, hidden nodes, output nodes, result number

    if (debug>=2) {
        nn_debug ("input/hidden1 weights(W0)", (float*)W0, NUM_INPUTS+1, NUM_HIDDEN_1);
        nn_debug ("hidden1/hidden2 weights(W1)", (float*)W1, NUM_HIDDEN_1+1, NUM_HIDDEN_2);		
        nn_debug ("hidden2/output weights(W2)", (float*)W2, NUM_HIDDEN_2+1, NUM_OUTPUTS);
    }

    ///Forward propagation ------------------------------------------------
    //Start the forward pass by calculating the weighted sums and activation values for the 1st hidden layer
    memset (Z0, 0, sizeof (Z0)); //set all the weighted sums to zero
    for (ih0=0; ih0<NUM_HIDDEN_1; ih0++){
        for (in=0; in<NUM_INPUTS+1; in++){
            Z0[ih0] += x[in] * W0[in][ih0]; //multiply and sum inputs * weights
		}
	}

    h0[NUM_HIDDEN_1]=1; //set the bias for the 1st hidden node to 1
    for (ih0=0; ih0<NUM_HIDDEN_1; ih0++){
        h0[ih0] = activation(Z0[ih0], ACT_TYPE); //apply activation function on other hidden nodes
	}

    //Continue the forward pass by calculating the weighted sums and activation values for the 2nd hidden layer	
    memset (Z1, 0, sizeof (Z1)); //set all the weighted sums to zero
    for (ih1=0; ih1<NUM_HIDDEN_2; ih1++){
        for (ih0=0; ih0<NUM_HIDDEN_1+1; ih0++){
            Z1[ih1] += h0[ih0] * W1[ih0][ih1]; //multiply and sum h0 * weights
		}
	}

    h1[NUM_HIDDEN_2]=1; //set the bias for the 2nd hidden node to 1
    for (ih1=0; ih1<NUM_HIDDEN_2; ih1++){
        h1[ih1] = activation(Z1[ih1], ACT_TYPE); //apply activation function on other hidden nodes
	}
	
    //Continue the forward pass by calculating the weighted sums for the output layer			
    memset (Z2, 0, sizeof (Z2)); //set all the weighted sums to zero
    for (io=0; io<NUM_OUTPUTS; io++)
        for (ih1=0; ih1<NUM_HIDDEN_2+1; ih1++)
            Z2[io] += h1[ih1] * W2[ih1][io]; //multiply and sum inputs * weights

    for (sum=0, io=0; io<NUM_OUTPUTS; io++) {
        p[io] = exp (Z2[io]);
        sum += p[io];
    } //compute exp(z) for softmax
    for (io=0; io<NUM_OUTPUTS; io++){		
		p[io] /= sum; //apply softmax by dividing by the the sum all the exps
	}
    if (debug>=2) nn_debug ("softmax probabilities", p, NUM_OUTPUTS, 1);

    ir = 0;
    sum = 0.0;
    for (io=0; io<NUM_OUTPUTS; io++) {
        y[io] = p[io];
        if (y[io] > sum) {
            sum = y[io];
            ir = io;
        }
    }
    return ir;   ///answer
}

void nn_fwrite(char *fname)
{
	int in, ih0, ih1, io;
	FILE *fp;

	fp = fopen(fname, "w+");
	if (!fp) {
        printf("file open error in the nn_write()\n");
        return;
	}

	for (in=0; in<NUM_INPUTS+1; in++){
        for (ih0=0; ih0<NUM_HIDDEN_1; ih0++){
            fprintf(fp, "%f", W0[in][ih0]);
		}
	}

	for (ih0=0; ih0<NUM_HIDDEN_1+1; ih0++){
        for (ih1=0; ih1<NUM_HIDDEN_2; ih1++){
            fprintf(fp, "%f", W1[ih0][ih1]);
		}
	}

	for (ih1=0; ih1<NUM_HIDDEN_2+1; ih1++){
        for (io=0; io<NUM_OUTPUTS; io++){
            fprintf(fp, "%f", W2[ih1][io]);
		}
	}

    fclose(fp);

    for (in=0; in<10; in++){
        printf("%f, ", W0[in][0]);
	}
    printf("\n");

    printf("weight have write to file(%s)\n", fname);
}

static int nn_fread(char *fname)
{
	int in, ih0, ih1, io;
	FILE *fp;

	fp = fopen(fname, "r");
	if (!fp) {
        printf("file open error in the nn_read()\n");
        return 0;
	}

	for (in=0; in<NUM_INPUTS+1; in++){
        for (ih0=0; ih0<NUM_HIDDEN_1; ih0++){
            fscanf(fp, "%f", &W0[in][ih0]);
		}
	}

	for (ih0=0; ih0<NUM_HIDDEN_1+1; ih0++){
        for (ih1=0; ih1<NUM_HIDDEN_2; ih1++){
            fscanf(fp, "%f", &W1[ih0][ih1]);
		}
	}

	for (ih1=0; ih1<NUM_HIDDEN_2+1; ih1++){
        for (io=0; io<NUM_OUTPUTS; io++){
            fscanf(fp, "%f", &W2[ih1][io]);
		}
	}

    fclose(fp);

    for (in=0; in<10; in++){
        printf("%f, ", W0[in][0]);
	}
    printf("\n");

    printf("weight have read from file(%s)\n", fname);

    return 1;
}

int nn_init(int flag)
{
	int in, ih0, ih1, io;
	int done = 0;
    ///int irange;
	///float frange;
	///frange = sqrt(6.0 / (NUM_INPUTS + NUM_HIDDEN)); ///0.084
	///irange = frange * 1000;

    if (flag){
        done = nn_fread("nn.wb");
	}

    if (!done)
    {
		for (in=0; in<NUM_INPUTS+1; in++){
			for (ih0=0; ih0<NUM_HIDDEN_1; ih0++){
					W0[in][ih0] = ((float)rand() / (double)RAND_MAX) * 0.2 - 0.1; ///+-0.0x
					///W0[in][ih0] = (float)(rand() % irange) / 500 - 0.06;
			}
		}

        for (ih0=0; ih0<NUM_HIDDEN_1+1; ih0++){
            for (ih1=0; ih1<NUM_HIDDEN_2; ih1++) {
                W1[ih0][ih1] = ((float)rand() / (double)RAND_MAX) * 0.2 - 0.1;
                ///W1[ih0][ih1] = (float)(rand() % irange) / 500 - 0.06;
                ///printf("%f  ", W1[ih0][ih1]);
            }
		}

        for (ih1=0; ih1<NUM_HIDDEN_2+1; ih1++){
            for (io=0; io<NUM_OUTPUTS; io++) {
                W2[ih1][io] = ((float)rand() / (double)RAND_MAX) * 0.2 - 0.1;
                ///W2[ih1][io] = (float)(rand() % irange) / 500 - 0.06;
                ///printf("%f  ", W2[ih1][io]);
            }
		}

        return 0;
    }

    return done;
}

/**
    @isize: size of xdata
    @rate: learning rate
*/
float nn_running (unsigned char *xdata, int ydata, int isize, float rate)
{
    int i;
	float x[NUM_INPUTS+1];
	float y[NUM_OUTPUTS] = {0.0,};

	x[isize] = 1.0; ///bias
	///memcpy ((float *)&x[1], (float *)xdata, isize);
	for (i=0; i<isize; i++) {
        ///x[i] = (float)xdata[i] / 128.0;
        x[i] = (float)xdata[i] / 255.0;
    }

	y[ydata] = 1.0;

	return nn_learning(x, y, rate);
}

int nn_question(unsigned char *xdata, int ydata, int isize)
{
    int i, ans;
	float x[NUM_INPUTS+1];
	float y[NUM_OUTPUTS] = {0.0,};

	x[isize] = 1.0; ///bias
	///memcpy ((float *)&x[1], (float *)xdata, isize);
	for (i=0; i<isize; i++) {
        ///x[i] = (float)xdata[i] / 128.0;
        x[i] = (float)xdata[i] / 255.0;
    }

	y[ydata] = 1.0;

	///printf ("------------------------------- Answer --------------------------------\n");
    ///nn_debug ("x input", &x[1], NUM_INPUTS, 1);
    ans = nn_answer(x, y);
    ///nn_debug ("y answer", y, NUM_OUTPUTS, 1);
    printf("What is this(%d)?, It is %d.\n", ydata, ans);

    return (ydata == ans) ? 1 : 0;
}
