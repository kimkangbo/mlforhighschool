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
#define NUM_HIDDEN_1   64           ///Accuracy: ?, RunTime: ?m
#define NUM_HIDDEN_2   99           ///Accuracy: ?, RunTime: ?m
#define NUM_OUTPUTS    10

///Weights
float W0[NUM_INPUTS+1][NUM_HIDDEN_1];
float W1[NUM_HIDDEN_1+1][NUM_HIDDEN_2];
float W2[NUM_HIDDEN_2+1][NUM_OUTPUTS];


int debug = 0; //0 for now debugging, 1 for the loss each iteration, 2 for all vectors/matrices each iteration

static void nn_debug (const char *label, float *m, int rows, int cols)
{
    int i;

	printf ("   %s:\n", label);
	for (i=0; i<rows*cols; i++)
        printf ("%10.5f%c", m[i], (cols>1 && i%cols==cols-1) ? '\n' : ' ');
	if (cols==1) printf ("\n");
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

	float loss, sum; //for storing the loss
	int in, ih0, ih1, io; //looping variables for iterations, input nodes, hidden nodes, output nodes

    ///Forward propagation ------------------------------------------------
    //Start the forward pass by calculating the weighted sums and activation values for the hidden layer	
	// 1st hidden layer
    memset (W0, 0, sizeof (W0)); //set all the weighted sums to zero
    for (ih0=0; ih0<NUM_HIDDEN_1; ih0++){
        for (in=0; in<NUM_INPUTS+1; in++){
            Z0[ih0] += x[in] * W0[in][ih0]; //multiply and sum inputs * weights
		}
	}
    if (debug>=2) nn_debug ("input/hidden weights", (float*)W0, NUM_INPUTS+1, NUM_HIDDEN_1);
    if (debug>=2) nn_debug ("hidden weighted sums", Z0, NUM_HIDDEN_1, 1);

    h0[NUM_HIDDEN_1]=1; //set the bias for the first hidden node to 1
    for (ih0=0; ih0<NUM_HIDDEN_1; ih0++){
        h0[ih0] = tanh (Z0[ih0]); //apply activation function on other 1st hidden nodes
	}
    if (debug>=2) nn_debug ("hidden node activation values", h0, NUM_HIDDEN_1+1, 1);
	
	// 2nd hidden layer
    memset (W1, 0, sizeof (W1)); //set all the weighted sums to zero
    for (ih1=0; ih1<NUM_HIDDEN_2; ih1++){
        for (ih0=0; ih0<NUM_HIDDEN_1+1; ih0++){
            Z1[ih1] += h0[ih0] * W1[ih0][ih1]; //multiply and sum inputs * weights
		}
	}
    if (debug>=2) nn_debug ("input/hidden weights", (float*)W1, NUM_HIDDEN_1+1, NUM_HIDDEN_2);
    if (debug>=2) nn_debug ("hidden weighted sums", Z1, NUM_HIDDEN_2, 1);

    h1[NUM_HIDDEN_2]=1; //set the bias for the second hidden node to 1
    for (ih1=0; ih1<NUM_HIDDEN_2; ih1++){
        h1[h] = tanh (Z1[ih1]); //apply activation function on other 1st hidden nodes
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
    if (debug>=2) nn_debug ("output weighted sums", Zl, NUM_OUTPUTS, 1);

    for (sum=0, io=0; io<NUM_OUTPUTS; io++) {
        p[io] = exp (Z2[io]);
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
    //Multiply h1*e to get the adjustments to dW2
    for (ih1=0; ih1<NUM_HIDDEN_2+1; ih1++){
        for (io=0; io<NUM_OUTPUTS; io++){
            dW2[ih1][io] = h1[ih1] * err[io];
		}
	}
    if (debug>=2) nn_debug ("hidden/output weights gradient", (float*)dW2, NUM_HIDDEN_2+1, NUM_OUTPUTS);

    //Backward propagate the errors and store in the h1 vector by using h1 temporally
    memset (h1, 0, sizeof (h1)); //set all the weighted sums to zero
    for (ih1=0; ih1<NUM_HIDDEN_2+1; ih1++){
        for (io=0; io<NUM_OUTPUTS; io++){
            h1[ih1] += W2[ih1][io] * err[io]; //multiply and sum inputs * weights
		}
	}
    if (debug>=2) nn_debug ("back propagated error values", h1, NUM_HIDDEN_2+1, 1);

	// for dW1
    for (ih1=0; ih1<NUM_HIDDEN_2; ih1++){
        Z1[ih1] = h1[ih1] * (1 - pow (tanh (Z1[h]), 2)); //apply activation function gradient by using Z1 temporally
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
    for (ih0=0; ih0<NUM_HIDDEN_1; ih0++){
		for (ih1=0; ih1<NUM_HIDDEN_2; ih1++){
			Z0[ih0] = h0[ih0] * (1 - pow (tanh (Z0[ih0]), 2))*Z1[ih1]; //apply activation function gradient temporary
		}
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
    for (h=0; h<NUM_HIDDEN_2+1; h++){
        for (o=0; o<NUM_OUTPUTS; o++){
            W2[h][o] -= learningrate * dW2[h][o];
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

    h0[NUM_HIDDEN_1]=1; //set the bias for the first hidden node to 1
    for (ih0=0; ih0<NUM_HIDDEN_1; ih0++){
        h0[ih0] = tanh(Z0[ih0]); //apply activation function on other hidden nodes
	}

    //Continue the forward pass by calculating the weighted sums and activation values for the 2nd hidden layer	
    memset (Z1, 0, sizeof (Z1)); //set all the weighted sums to zero
    for (ih1=0; ih1<NUM_HIDDEN_2; ih1++){
        for (ih0=0; ih0<NUM_HIDDEN_1+1; ih0++){
            Z1[ih1] += h0[ih0] * W1[ih0][ih1]; //multiply and sum h0 * weights
		}
	}

    h1[NUM_HIDDEN_2]=1; //set the bias for the first hidden node to 1
    for (ih1=0; ih1<NUM_HIDDEN_2; ih1++){
        h1[ih1] = tanh(Z1[ih1]); //apply activation function on other hidden nodes
	}
	
    //Continue the forward pass by calculating the weighted sums for the output layer			
    memset (Z2, 0, sizeof (Z2)); //set all the weighted sums to zero
    for (io=0; io<NUM_OUTPUTS; io++)
        for (ih1=0; ih1<NUM_HIDDEN_2+1; ih1++)
            Z2[io] += h1[ih1] * W2[ih1][io]; //multiply and sum inputs * weights

    for (sum=0, io=0; o<NUM_OUTPUTS; io++) {
        p[io] = exp (Z2[io]);
        sum += p[io];
    } //compute exp(z) for softmax
    for (io=0; io<NUM_OUTPUTS; io++){		
		probabilities[io] /= sum; //apply softmax by dividing by the the sum all the exps
	}
    if (debug>=2) nn_debug ("softmax probabilities", probabilities, NUM_OUTPUTS, 1);

    ir = 0;
    sum = 0.0;
    for (io=0; io<NUM_OUTPUTS; io++) {
        y[io] = probabilities[io];
        if (y[io] > sum) {
            sum = y[io];
            ir = io;
        }
    }
    return ir;   ///answer
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
        for (h=0; h<NUM_HIDDEN; h++)
            write(fd, &Wij[i][h], sizeof(float));

	for (h=0; h<NUM_HIDDEN+1; h++)
        for (o=0; o<NUM_OUTPUTS; o++)
            write(fd, &Wjk[h][o], sizeof(float));

    close(fd);

    for (i=0; i<10; i++)
        printf("%f, ", Wij[i][0]);
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
        for (h=0; h<NUM_HIDDEN; h++)
            fprintf(fp, "%f", Wij[i][h]);

	for (h=0; h<NUM_HIDDEN+1; h++)
        for (o=0; o<NUM_OUTPUTS; o++)
            fprintf(fp, "%f", Wjk[h][o]);

    fclose(fp);

    for (i=0; i<10; i++)
        printf("%f, ", Wij[i][0]);
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
        for (h=0; h<NUM_HIDDEN; h++)
            read(fd, &Wij[i][h], sizeof(float));

	for (h=0; h<NUM_HIDDEN+1; h++)
        for (o=0; o<NUM_OUTPUTS; o++)
            read(fd, &Wjk[h][o], sizeof(float));

    close(fd);

    for (i=0; i<10; i++)
        printf("%f, ", Wij[i][0]);
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
        for (h=0; h<NUM_HIDDEN; h++)
            fscanf(fp, "%f", &Wij[i][h]);

	for (h=0; h<NUM_HIDDEN+1; h++)
        for (o=0; o<NUM_OUTPUTS; o++)
            fscanf(fp, "%f", &Wjk[h][o]);

    fclose(fp);

    for (i=0; i<10; i++)
        printf("%f, ", Wij[i][0]);
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
	///frange = sqrt(6.0 / (NUM_INPUTS + NUM_HIDDEN)); ///0.084
	///irange = frange * 1000;

    if (flag)
        ///done = nn_read("nn.wb");
        done = nn_fread("nn.wb");

    if (!done)
    {
        for (i=0; i<NUM_INPUTS+1; i++)
            for (h=0; h<NUM_HIDDEN; h++)
                Wij[i][h] = ((float)rand() / (double)RAND_MAX) * 0.2 - 0.1; ///+-0.0x
                ///Wij[i][h] = (float)(rand() % irange) / 500 - 0.06;

        for (h=0; h<NUM_HIDDEN+1; h++)
            for (o=0; o<NUM_OUTPUTS; o++) {
                Wjk[h][o] = ((float)rand() / (double)RAND_MAX) * 0.2 - 0.1;
                ///Wjk[h][o] = (float)(rand() % irange) / 500 - 0.06;
                ///printf("%f  ", Wjk[h][o]);
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
