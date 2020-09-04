#include "lenet.h"
#include "model.h"
#include <time.h>
#include <stdlib.h>
#include <math.h>
#include "include/DSPLib.h"

short int i,j,o0,o1,i0,i1,l0,l1,x,w0,w1,y,k,tl1,tl2,length1,length2;
int temp1,temp2,temp3;

#define GETLENGTH(array) (sizeof(array)/sizeof(*(array)))

#define GETCOUNT(array)  (sizeof(array)/sizeof(float))

#define FOREACH(i,count) for (i = 0; i < count; ++i)

#define CONVOLUTE_VALID(input,output,weight)                                            \
{                                                                                       \
    FOREACH(o0,GETLENGTH(output))                                                       \
        FOREACH(o1,GETLENGTH(*(output)))                                                \
            FOREACH(w0,GETLENGTH(weight))                                               \
                FOREACH(w1,GETLENGTH(*(weight)))                                        \
                    (output)[o0][o1] += (input)[o0 + w0][o1 + w1] * (weight)[w0][w1];   \
}

#define CONVOLUTE_FULL(input,output,weight)												\
{																						\
	FOREACH(i0,GETLENGTH(input))														\
		FOREACH(i1,GETLENGTH(*(input)))													\
			FOREACH(w0,GETLENGTH(weight))												\
				FOREACH(w1,GETLENGTH(*(weight)))										\
				(output)[i0 + w0][i1 + w1] += (input)[i0][i1] * (weight)[w0][w1];   \
}

#define CONVOLUTION_FORWARD(input,output,weight,bias,action)                    \
{                                                                               \
    for (x = 0; x < GETLENGTH(weight); ++x)                                 \
        for (y = 0; y < GETLENGTH(*weight); ++y)                            \
            CONVOLUTE_VALID(input[x], output[y], weight[x][y]);                 \
    FOREACH(j, GETLENGTH(output))                                               \
        FOREACH(i, GETCOUNT(output[j]))                                         \
        ((float *)output[j])[i] = action(((float *)output[j])[i] + bias[j]);  \
}

#define CONVOLUTION_FIRST_PART(input,output,weight,bias,action,floop,sloop,finit,sinit)                    \
{ \
    for (x = finit; x <floop ; x++)                                 \
        for (y = sinit; y <sloop ; y++)                            \
            CONVOLUTE_VALID(input[x], output[y], weight[x%2][y]);                 \
}

#define CONVOLUTION_FORWARD_Last(input,output,weight,bias,action)                    \
{                                                                               \
    length1 = GETLENGTH(weight); \
    length2 = GETLENGTH(*weight);\
    CONVOLUTION_FIRST_PART(input,output,weight,bias,action,2,120,0,0) \
    CONVOLUTION_FIRST_PART(input,output,weight4_5_1,bias,action,4,120,2,0) \
    CONVOLUTION_FIRST_PART(input,output,weight4_5_2,bias,action,6,120,4,0) \
    CONVOLUTION_FIRST_PART(input,output,weight4_5_3,bias,action,8,120,6,0) \
    CONVOLUTION_FIRST_PART(input,output,weight4_5_4,bias,action,10,120,8,0) \
    CONVOLUTION_FIRST_PART(input,output,weight4_5_5,bias,action,12,120,10,0) \
    CONVOLUTION_FIRST_PART(input,output,weight4_5_6,bias,action,14,120,12,0) \
    CONVOLUTION_FIRST_PART(input,output,weight4_5_7,bias,action,16,120,14,0) \
    FOREACH(j, GETLENGTH(output))                                               \
        FOREACH(i, GETCOUNT(output[j]))                                         \
        ((float *)output[j])[i] = action(((float *)output[j])[i] + bias[j]);  \
}



#define SUBSAMP_MAX_FORWARD(input,output)														\
{																								\
	const int len0 = GETLENGTH(*(input)) / GETLENGTH(*(output));								\
	const int len1 = GETLENGTH(**(input)) / GETLENGTH(**(output));								\
	FOREACH(i, GETLENGTH(output))																\
	FOREACH(o0, GETLENGTH(*(output)))															\
	FOREACH(o1, GETLENGTH(**(output)))															\
	{																							\
		int x0 = 0, x1 = 0, ismax;																\
		FOREACH(l0, len0)																		\
			FOREACH(l1, len1)																	\
		{																						\
			ismax = input[i][o0*len0 + l0][o1*len1 + l1] > input[i][o0*len0 + x0][o1*len1 + x1];\
			x0 += ismax * (l0 - x0);															\
			x1 += ismax * (l1 - x1);															\
		}																						\
		output[i][o0][o1] = input[i][o0*len0 + x0][o1*len1 + x1];								\
	}																							\
}

#define DOT_PRODUCT_FORWARD(input,output,weight,bias,action)				\
{																			\
	for (x = 0; x < GETLENGTH(weight); ++x)								\
		for (y = 0; y < GETLENGTH(*weight); ++y)						\
			((float *)output)[y] += ((float *)input)[x] * weight[x][y];	\
	FOREACH(j, GETLENGTH(bias))												\
		((float *)output)[j] = action(((float *)output)[j] + bias[j]);	\
}

void DOT_PRODUCT_FORWARD_UPDATED(float(*action)(float))
{
    for (x = 0; x < GETLENGTH(n_weight5_6); ++x)
        for (y = 0; y < GETLENGTH(*n_weight5_6); ++y)
            f_output[y] += f_layer5[x][0][0] * n_weight5_6[x][y];
    for(j=0;j< GETLENGTH(n_bias5_6); j++)
        f_output[j] = action(f_output[j] + n_bias5_6[j]);
}

float relu(float x)
{
	return x*(x > 0);
}

float relugrad(float y)
{
	return y > 0;
}
static void convolutionFunction2()
{
    for (; x < tl1; ++x)
        for (y = 0; y < tl2; ++y)
            for (o0 = 0; o0 < LENGTH_FEATURE5; ++o0)
                for (o1 = 0; o1 < LENGTH_FEATURE5; ++o1)
                    for (w0 = 0; w0 < LENGTH_KERNEL; ++w0)
                        for (w1 = 0; w1 < LENGTH_KERNEL; ++w1){
                            temp1  = weight4_5_2[x][y][w0][w1];
                            temp2 = f_layer4[x][o0 + w0][o1 + w1];
                            temp3 = f_layer5[y][o0][o1];
                            f_layer5[y][o0][o1] = temp3 + (temp2 *temp1);
                        }
}

static void CONVOLUTION_Last(float(*action)(float))
{
    CONVOLUTION_FIRST_PART(f_layer4,f_layer5,weight4_5,n_bias4_5,action,2,120,0,0)
    CONVOLUTION_FIRST_PART(f_layer4,f_layer5,weight4_5_1,n_bias4_5,action,4,120,2,0)
    CONVOLUTION_FIRST_PART(f_layer4,f_layer5,weight4_5_2,n_bias4_5,action,6,120,4,0)
    CONVOLUTION_FIRST_PART(f_layer4,f_layer5,weight4_5_3,n_bias4_5,action,8,120,6,0)
    CONVOLUTION_FIRST_PART(f_layer4,f_layer5,weight4_5_4,n_bias4_5,action,10,120,8,0)
    CONVOLUTION_FIRST_PART(f_layer4,f_layer5,weight4_5_5,n_bias4_5,action,12,120,10,0)
    CONVOLUTION_FIRST_PART(f_layer4,f_layer5,weight4_5_6,n_bias4_5,action,14,120,12,0)
    CONVOLUTION_FIRST_PART(f_layer4,f_layer5,weight4_5_7,n_bias4_5,action,16,120,14,0)

    for(j=0;j <120; j++)
            for(i=0;i <1; i++)
                f_layer5[j][i][0] = action((f_layer5[j][i][0] + n_bias4_5[j]));
}



static void convolutionFunction1(float(*action)(float))
{
    for (x = 0; x < INPUT; ++x)
        for (y = 0; y < LAYER1; ++y)
            for (o0 = 0; o0 < LENGTH_FEATURE1; ++o0)
                for (o1 = 0; o1 < LENGTH_FEATURE2; ++o1)
                    for (w0 = 0; w0 < LENGTH_KERNEL; ++w0)
                        for (w1 = 0; w1 < LENGTH_KERNEL; ++w1){
                            temp1  = n_weight0_1[x][y][w0][w1];
                            temp2 = f_input[x][o0 + w0][o1 + w1];
                            temp3 = f_layer1[y][o0][o1];
                            f_layer1[y][o0][o1] = temp3 + (temp2 *temp1);
                        }

    for (j = 0; j < LAYER1; ++j)
        for (i = 0; i < LENGTH_FEATURE1; ++i)
        ((float *)f_layer1[j])[i] = action(((float *)f_layer1[j])[i] + n_bias0_1[j]);
}

static void forward(float(*action)(float))
{
    CONVOLUTION_FORWARD(f_input,f_layer1, n_weight0_1, n_bias0_1, action);
	SUBSAMP_MAX_FORWARD(f_layer1, f_layer2);



    CONVOLUTION_FORWARD(f_layer2, f_layer3, n_weight2_3, n_bias2_3, action);
	SUBSAMP_MAX_FORWARD(f_layer3, f_layer4);



    //freeMemory();
    //weight2(weight4_5);

	CONVOLUTION_Last(action);//CONVOLUTION_FORWARD_Last(f_layer4, f_layer5, weight4_5, n_bias4_5, action);
	//DOT_PRODUCT_FORWARD(f_layer5, f_output, n_weight5_6, n_bias5_6, action);
	DOT_PRODUCT_FORWARD_UPDATED(action);
}

static inline void load_input(image input)
{
	float (*layer0)[LENGTH_FEATURE0][LENGTH_FEATURE0] = f_input;
	const long sz = sizeof(image) / sizeof(**input);
	float mean = 0, std = 0;
	for(j=0;j<28;j++)
	    for(k=0;k<28;k++)
	{
		mean =mean + (float)input[j][k];
		std =std + (float)input[j][k] * (float)input[j][k];
	}
	mean /= sz;
	std = sqrt(std / sz - mean*mean);
	FOREACH(j, sizeof(image) / sizeof(*input))
		FOREACH(k, sizeof(*input) / sizeof(**input))
	{
	    layer0[0][j + PADDING][k + PADDING] = (input[j][k] - mean) / std;
	}
}



static uint8 get_result(uint8 count)
{
	float *output = (float *)f_output;
	//const int outlen = GETCOUNT(features->output);
	uint8 result = 0;
	float maxvalue = *output;
	for (i = 1; i < count; ++i)
	{
		if (output[i] > maxvalue)
		{
			maxvalue = output[i];
			result = i;
		}
	}
	return result;
}


uint8 Predict(image input,uint8 count)
{
	load_input(input);
	forward(relu);
	return get_result(count);
}

void freeMemory()
{
    for(i=0;i<LAYER1;i++)
        for(j=0;j<LENGTH_FEATURE1;j++)
                free(f_layer1[i][j]);
    free(f_layer1);

    for(i=0;i<INPUT;i++)
        for(j=0;j<LENGTH_FEATURE0;j++)
                free(f_input[i][j]);
    free(f_input);

    for(i=0;i<LAYER2;i++)
        for(j=0;j<LENGTH_FEATURE2;j++)
                free(f_layer2[i][j]);
    free(f_layer2);

    for(i=0;i<LAYER3;i++)
        for(j=0;j<LENGTH_FEATURE3;j++)
                free(f_layer3[i][j]);
    free(f_layer3);

    for(i=0;i<INPUT;i++)
            for(j=0;j<LAYER1;j++)
                for(x=0;x<LENGTH_KERNEL;x++)
                        free(n_weight0_1[i][j][x]);
    free(n_weight0_1);

    for(i=0;i<LAYER2;i++)
                for(j=0;j<LAYER3;j++)
                    for(x=0;x<LENGTH_KERNEL;x++)
                            free(n_weight2_3[i][j][x]);
    free(n_weight2_3);
}

