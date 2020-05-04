/**
 *  file name:  nn.h
 *  function:   Neural Network Header
 *  author:     JungJaeJoon(rgbi3307@nate.com) on the www.kernel.bz
 */

#ifndef __NN_H
#define __NN_H

#define LEARNING_RATE   0.01
#define REG_OVERFITTING_RATE 0.1

int nn_init(int flag);
float nn_running (unsigned char *xdata, int ydata, int isize, float rate, float lamda);
int nn_question(unsigned char *xdata, int ydata, int isize);

void nn_write(char *fname);
void nn_fwrite(char *fname);

#endif
