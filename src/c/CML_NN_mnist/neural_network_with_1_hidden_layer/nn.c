/**
 *  file name:  nn.c
 *  function:   Neural Network for Machine Learning
 *  author:     JungJaeJoon(rgbi3307@nate.com) on the www.kernel.bz
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
//#define NUM_HIDDEN_1     256       ///Accuracy: 0.9596, RunTime: 71m, 0.9474, 0.9618, 0.9691, 0.9699, 0.971
///#define NUM_HIDDEN_1     128       ///Accuracy: 0.9536, RunTime: 28m
#define NUM_HIDDEN_1      64          ///Accuracy: 0.9508, RunTime: 12m, 0.9697
#define NUM_OUTPUTS     10
#define ACT_TANH        1
#define ACT_RELU        2
#define ACT_TYPE		ACT_RELU

///Weights
float W0[NUM_INPUTS+1][NUM_HIDDEN_1];
float W1[NUM_HIDDEN_1+1][NUM_OUTPUTS];


int debug = 0; //0 for now debugging, 1 for the loss each iteration, 2 for all vectors/matrices each iteration

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
	float Z0[NUM_HIDDEN_1]; //weighted sums for the hidden nodes
	float Z1[NUM_OUTPUTS]; //weighted sums for the output nodes
	float p[NUM_OUTPUTS]; //probabilities: activation values of the output nodes

	float h0[NUM_HIDDEN_1+1]; //activation values of the hidden nodes, including one extra for the bias
	float err[NUM_OUTPUTS]; //outputErrors: error in the output

	float dW0[NUM_INPUTS+1][NUM_HIDDEN_1]; //adjustments to weights between inputs x and hidden nodes
	float dW1[NUM_HIDDEN_1+1][NUM_OUTPUTS]; //adjustments to weights between hidden nodes and output y

	float loss, sum; //for storing the loss
	int in, ih0, io; //looping variables for iterations, input nodes, hidden nodes, output nodes

    ///Forward propagation ------------------------------------------------
    //Start the forward pass by calculating the weighted sums and activation values for the hidden layer
    memset (Z0, 0, sizeof (Z0)); //set all the weighted sums to zero
    for (ih0=0; ih0<NUM_HIDDEN_1; ih0++){
        for (in=0; in<NUM_INPUTS+1; in++){
            Z0[ih0] += x[in] * W0[in][ih0]; //multiply and sum inputs * weights
		}
	}
    if (debug>=2) nn_debug ("input/hidden weights", (float*)W0, NUM_INPUTS+1, NUM_HIDDEN_1);
    if (debug>=2) nn_debug ("hidden weighted sums", Z0, NUM_HIDDEN_1, 1);

    h0[NUM_HIDDEN_1]=1; //set the bias for the last hidden node to 1
    for (ih0=0; ih0<NUM_HIDDEN_1; ih0++){		
		h0[ih0] = activation(Z0[ih0], ACT_TYPE); //apply activation function on other hidden nodes
	}
    if (debug>=2) nn_debug ("hidden node activation values", h0, NUM_HIDDEN_1+1, 1);

    memset (Z1, 0, sizeof (Z1)); //set all the weighted sums to zero
    for (io=0; io<NUM_OUTPUTS; io++){
        for (ih0=0; ih0<NUM_HIDDEN_1+1; ih0++){
            Z1[io] += h0[ih0] * W1[ih0][io]; //multiply and sum inputs * weights
		}
	}
    if (debug>=2) nn_debug ("hidden/output weights", (float*)W1, NUM_HIDDEN_1+1, NUM_OUTPUTS);
    if (debug>=2) nn_debug ("output weighted sums", Z1, NUM_OUTPUTS, 1);

    for (sum=0, io=0; io<NUM_OUTPUTS; io++) {
        p[io] = exp (Z1[io]);
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
	}

    if (debug>=1) printf ("loss(cost): %10.5f\n", loss); //output the loss

    /// Back propagation --------------------------------------------------
    //Multiply h*e to get the adjustments to deltaW1
    for (ih0=0; ih0<NUM_HIDDEN_1+1; ih0++){
        for (io=0; io<NUM_OUTPUTS; io++){
            dW1[ih0][io] = h0[ih0] * err[io];
		}
	}
    if (debug>=2) nn_debug ("hidden/output weights gradient", (float*)dW1, NUM_HIDDEN_1+1, NUM_OUTPUTS);


    //Backward propogate the errors and store in the h0 vector temporary
    memset (h0, 0, sizeof (h0)); //set all the weighted sums to zero
    for (ih0=1; ih0<NUM_HIDDEN_1+1; ih0++){
        for (io=0; io<NUM_OUTPUTS; io++){
            h0[ih0] += W1[ih0][io] * err[io]; //multiply and sum inputs * weights
		}
	}
    if (debug>=2) nn_debug ("back propagated error values", h0, NUM_HIDDEN_1+1, 1);

    for (ih0=0; ih0<NUM_HIDDEN_1; ih0++){
        Z0[ih0] = h0[ih0] * diff_actv(Z0[ih0], ACT_TYPE); //apply activation function gradient temporary
	}
    if (debug>=2) nn_debug ("hidden weighted sums after gradient", Z0, NUM_HIDDEN_1, 1);

    //Multiply x*eh*zh to get the adjustments to dW0, this does not include the bias node
    for (in=0; in<NUM_INPUTS+1; in++){
        for (ih0=0; ih0<NUM_HIDDEN_1; ih0++){
            dW0[in][ih0] = x[in] * Z0[ih0];
		}
	}
    if (debug>=2) nn_debug ("input/hidden weights gradient", (float*)W0, NUM_INPUTS+1, NUM_HIDDEN_1);


    /// Now add in the adjustments ----------------------------------------
    for (ih0=0; ih0<NUM_HIDDEN_1+1; ih0++){
        for (io=0; io<NUM_OUTPUTS; io++){
            W1[ih0][io] -= learningrate * dW1[ih0][io];
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
	float Z0[NUM_HIDDEN_1]; //weighted sums for the hidden nodes
	float Z1[NUM_OUTPUTS]; //weighted sums for the output nodes
	float probabilities[NUM_OUTPUTS]; //activation values of the output nodes

	float h0[NUM_HIDDEN_1+1]; //activation values of the hidden nodes, including one extra for the bias

	float sum; //for storing the loss
	int i, h, o; //looping variables for iterations, input nodes, hidden nodes, output nodes

    if (debug>=2) {
        nn_debug ("input/hidden weights(W0)", (float*)W0, NUM_INPUTS+1, NUM_HIDDEN_1);
        nn_debug ("hidden/output weights(W1)", (float*)W1, NUM_HIDDEN_1+1, NUM_OUTPUTS);
    }

    ///Forward propagation ------------------------------------------------
    //Start the forward pass by calculating the weighted sums and activation values for the hidden layer
    memset (Z0, 0, sizeof (Z0)); //set all the weighted sums to zero
    for (h=0; h<NUM_HIDDEN_1; h++)
        for (i=0; i<NUM_INPUTS+1; i++)
            Z0[h] += x[i] * W0[i][h]; //multiply and sum inputs * weights

    h0[NUM_HIDDEN_1]=1; //set the bias for the last hidden node to 1
    for (h=0; h<NUM_HIDDEN_1; h++)
        h0[h] = activation(Z0[h], ACT_TYPE); //apply activation function on other hidden nodes

    memset (Z1, 0, sizeof (Z1)); //set all the weighted sums to zero
    for (o=0; o<NUM_OUTPUTS; o++)
        for (h=0; h<NUM_HIDDEN_1+1; h++)
            Z1[o] += h0[h] * W1[h][o]; //multiply and sum inputs * weights

    for (sum=0, o=0; o<NUM_OUTPUTS; o++) {
        probabilities[o] = exp (Z1[o]);
        sum += probabilities[o];
    } //compute exp(z) for softmax
    for (o=0; o<NUM_OUTPUTS; o++) probabilities[o] /= sum; //apply softmax by dividing by the the sum all the exps
    if (debug>=2) nn_debug ("softmax probabilities", probabilities, NUM_OUTPUTS, 1);

    i = 0;
    sum = 0.0;
    for (o=0; o<NUM_OUTPUTS; o++) {
        y[o] = probabilities[o];
        if (y[o] > sum) {
            sum = y[o];
            i = o;
        }
    }
    return i;   ///answer
}

void nn_write(char *fname)
{
	int i, h, o;
	int fd;

	fd = open(fname, O_RDWR | O_CREAT | O_TRUNC, S_IRUSR | S_IWUSR);
	if (fd < 0) {
        printf("file open error in the nn_write()\n");
        return;
	}

	for (i=0; i<NUM_INPUTS+1; i++)
        for (h=0; h<NUM_HIDDEN_1; h++)
            write(fd, &W0[i][h], sizeof(float));

	for (h=0; h<NUM_HIDDEN_1+1; h++)
        for (o=0; o<NUM_OUTPUTS; o++)
            write(fd, &W1[h][o], sizeof(float));

    close(fd);

    for (i=0; i<10; i++)
        printf("%f, ", W0[i][0]);
    printf("\n");

    printf("weight have write to file(%s)\n", fname);
}

void nn_fwrite(char *fname)
{
	int i, h, o;
	FILE *fp;

	fp = fopen(fname, "w+");
	if (!fp) {
        printf("file open error in the nn_write()\n");
        return;
	}

	for (i=0; i<NUM_INPUTS+1; i++)
        for (h=0; h<NUM_HIDDEN_1; h++)
            fprintf(fp, "%f", W0[i][h]);

	for (h=0; h<NUM_HIDDEN_1+1; h++)
        for (o=0; o<NUM_OUTPUTS; o++)
            fprintf(fp, "%f", W1[h][o]);

    fclose(fp);

    for (i=0; i<10; i++)
        printf("%f, ", W0[i][0]);
    printf("\n");

    printf("weight have write to file(%s)\n", fname);
}

static int nn_read(char *fname)
{
	int i, h, o;
	int fd;

	fd = open(fname, O_RDONLY);
	if (fd < 0) {
        printf("file open error in the nn_read()\n");
        return 0;
	}

	for (i=0; i<NUM_INPUTS+1; i++)
        for (h=0; h<NUM_HIDDEN_1; h++)
            read(fd, &W0[i][h], sizeof(float));

	for (h=0; h<NUM_HIDDEN_1+1; h++)
        for (o=0; o<NUM_OUTPUTS; o++)
            read(fd, &W1[h][o], sizeof(float));

    close(fd);

    for (i=0; i<10; i++)
        printf("%f, ", W0[i][0]);
    printf("\n");

    printf("weight have read from file(%s)\n", fname);

    return 1;
}

static int nn_fread(char *fname)
{
	int i, h, o;
	FILE *fp;

	fp = fopen(fname, "r");
	if (!fp) {
        printf("file open error in the nn_read()\n");
        return 0;
	}

	for (i=0; i<NUM_INPUTS+1; i++)
        for (h=0; h<NUM_HIDDEN_1; h++)
            fscanf(fp, "%f", &W0[i][h]);

	for (h=0; h<NUM_HIDDEN_1+1; h++)
        for (o=0; o<NUM_OUTPUTS; o++)
            fscanf(fp, "%f", &W1[h][o]);

    fclose(fp);

    for (i=0; i<10; i++)
        printf("%f, ", W0[i][0]);
    printf("\n");

    printf("weight have read from file(%s)\n", fname);

    return 1;
}

int nn_init(int flag)
{
	int i, h, o;
	int done = 0;
    ///int irange;
	///float frange;
	///frange = sqrt(6.0 / (NUM_INPUTS + NUM_HIDDEN_1)); ///0.084
	///irange = frange * 1000;

    if (flag)
        ///done = nn_read("nn.wb");
        done = nn_fread("nn.wb");

    if (!done)
    {
        for (i=0; i<NUM_INPUTS+1; i++)
            for (h=0; h<NUM_HIDDEN_1; h++)
                W0[i][h] = ((float)rand() / (double)RAND_MAX) * 0.2 - 0.1; ///+-0.0x
                ///W0[i][h] = (float)(rand() % irange) / 500 - 0.06;

        for (h=0; h<NUM_HIDDEN_1+1; h++)
            for (o=0; o<NUM_OUTPUTS; o++) {
                W1[h][o] = ((float)rand() / (double)RAND_MAX) * 0.2 - 0.1;
                ///W1[h][o] = (float)(rand() % irange) / 500 - 0.06;
                ///printf("%f  ", W1[h][o]);
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
