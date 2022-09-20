#ifndef __NETWORK__
#define __NETWORK__
#include <gsl/gsl_matrix_float.h>
#include <gsl/gsl_vector_float.h>
#include <math.h>


#define gsv gsl_vector_float
#define gsm gsl_matrix_float
#define gsv_alloc gsl_vector_float_alloc
#define gsm_alloc gsl_matrix_float_alloc
#define gsv_free gsl_vector_float_free
#define gsm_free gsl_matrix_float_free
#define gsv_mul gsl_vector_float_mul
#define gsv_add gsl_vector_float_add
#define gsm_add gsl_matrix_float_add
#define gsv_add_constant gsl_vector_float_add_constant
#define gsv_sub gsl_vector_float_sub
#define gsm_sub gsl_matrix_float_sub
#define gsv_scale gsl_vector_float_scale
#define gsm_scale gsl_matrix_float_scale
#define gsv_set gsl_vector_float_set
#define gsm_set gsl_matrix_float_set
#define gsv_get gsl_vector_float_get
#define gsm_get gsl_matrix_float_get
#define sgemm gsl_blas_sgemm

struct Layer {
    gsl_matrix_float* weights;
    gsl_vector_float* biases;
    gsl_matrix_float* a;
    int num_neurons;
};
struct Network {
    struct Layer* layers;
    float rate;
    int num_layers;
};

void seed_network(struct Network* net);
struct Network* init_network(int layers, float rate, int* layerSz, int batchSz);
gsm* applySigmoid(gsm* matrix);
void feedforward(struct Network* net, gsm* input);
gsm* getOutputError(gsm* a, gsm* y);
gsm* getPreviousError(gsm* a, gsm* w, gsm* error);
gsv* getBiasError(gsm* error);
gsm* getWeightsError(gsm* a, gsm* error);
void backprop(struct Network* net, gsm* input, gsm* y);
void freeOutputs(struct Network* net);
void train(struct Network* net, gsm** input, gsm** y, int epochs, int batchSz);
void deleteNetwork(struct Network* net);
int getDigit(struct Network* net, gsv* input);

#endif