#include "Network.h"

#include <gsl/gsl_blas.h>
#include <gsl/gsl_matrix_float.h>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_vector_float.h>
#include <math.h>
#include <stdlib.h>

#include "Debug.h"

static inline float sigmoid(float x) { return 1 / (1 + exp(-x)); }
/**
 * @brief Initializes a network with the given parameters
 *
 * @param layers - number of layers
 * @param rate - learning rate
 * @param layerSz - int[] size of each layer
 * @param batchSz - batch size
 * @return struct Network*
 */
struct Network* init_network(int layers, float rate, int* layerSz,
    int batchSz)
{
#ifdef DEBUG
    printf("Network Starting Initialization. \n");
#endif  // DEBUG
    // initialize network
    struct Network* net;
    // malloc network and layers
    if (!(net = malloc(sizeof(struct Network)))) return NULL;
    if (!(net->layers = malloc(sizeof(struct Layer) * layers))) return NULL;

    // set network parameters
    net->num_layers = layers;
    net->rate = rate;

    // set layer parameters
    for (int i = 1; i < layers; i++) {
        net->layers[i].num_neurons = layerSz[i];
        net->layers[i].weights = gsm_alloc(layerSz[i], layerSz[i - 1]);
        net->layers[i].biases = gsv_alloc(layerSz[i]);
        if (net->layers[i].weights == NULL || net->layers[i].biases == NULL)
            return NULL;
    }
    seed_network(net);
#ifdef DEBUG
    printf("Network Finished Initialization.\n");
#endif  // DEBUG
    return net;
}
void seed_network(struct Network* net)
{
    gsl_rng* rng = gsl_rng_alloc(gsl_rng_mt19937);
    gsl_rng_set(rng, 89712349);

    for (int i = 1; i < net->num_layers; i++) {
        for (int j = 0; j < net->layers[i].weights->size1; j++) {
            for (int k = 0; k < net->layers[i].weights->size2; k++) {
                gsl_matrix_float_set(net->layers[i].weights, j, k,
                    (gsl_rng_uniform(rng) - .5) / 2);
            }
            gsl_vector_float_set(net->layers[i].biases, j,
                (gsl_rng_uniform(rng) - .5) / 1.2);
        }
    }
    gsl_rng_free(rng);
}
//temporary solution, non multithreaded
gsm* applySigmoid(gsm* matrix)
{
    gsm* output = gsm_alloc(matrix->size1, matrix->size2);
    for (int i = 0; i < matrix->size1; i++) {
        for (int j = 0; j < matrix->size2; j++) {
            gsm_set(output, i, j, sigmoid(gsm_get(matrix, i, j)));
        }
    }
    return output;
}

void printMatrix(gsm* matrix)
{
    for (int i = 0; i < matrix->size1; i++) {
        for (int j = 0; j < matrix->size2; j++) {
            printf("%.1f ", gsm_get(matrix, i, j));
        }
        printf("\n");
    }
}
void printVector(gsv* vector)
{
    for (int i = 0; i < vector->size; i++) {
        printf("%.1f ", gsv_get(vector, i));
    }
    printf("\n");
}
void feedforward(struct Network* net, gsm* input)
{
    gsm* prev = input;
    net->layers[0].a = gsm_alloc(input->size1, input->size2);
    gsl_matrix_float_memcpy(net->layers[0].a, input);
    for (int i = 0; i < net->num_layers; i++) {
        struct Layer* curr = &net->layers[i];
        if (i != 0) {
            // z = w * a + b
            gsm* z = gsm_alloc(curr->weights->size1, prev->size2);
            sgemm(CblasNoTrans, CblasNoTrans, 1, curr->weights, prev, 0, z);
            for (int j = 0; j < z->size2; j++) {
                gsl_vector_float_view col = gsl_matrix_float_column(z, j);
                gsv_add(&col.vector, curr->biases);
            }
            if (i != 1)
                gsm_free(prev);
            prev = z;
            curr->a = applySigmoid(prev);
        }
    }
    gsm_free(prev);
}

gsm* getOutputError(gsm* a, gsm* y)
{
    // printf("y1: %ld, y2: %ld, a1: %ld, a2: %ld", y->size1, y->size2, a->size1,
    //     a->size2);
    //(a - y)
    gsm* error = gsm_alloc(a->size1, a->size2);
    for (int i = 0; i < a->size1; i++) {
        for (int j = 0; j < a->size2; j++) {
            float aVal = gsm_get(a, i, j);
            gsm_set(error, i, j, (aVal - gsm_get(y, i, j)) * (aVal * (1 - aVal)));
        }
    }
    return error;
}
gsm* getPreviousError(gsm* a, gsm* w, gsm* error)
{
    //(w^T * error) * (a * (1 - a))
    gsm* output = gsm_alloc(a->size1, a->size2);
    gsm* temp = gsm_alloc(a->size1, a->size2);
    sgemm(CblasTrans, CblasNoTrans, 1, w, error, 0, temp);
    for (int i = 0; i < a->size1; i++) {
        for (int j = 0; j < a->size2; j++) {
            float aVal = gsm_get(a, i, j);
            gsm_set(output, i, j, gsm_get(temp, i, j) * (aVal * (1 - aVal)));
        }
    }
    gsm_free(temp);
    return output;
}
gsv* getBiasError(gsm* error)
{
    gsv* output = gsv_alloc(error->size1);
    for (int i = 0; i < error->size1; i++) {
        float sum = 0;
        for (int j = 0; j < error->size2; j++) {
            sum += gsm_get(error, i, j);
        }
        gsv_set(output, i, sum);
    }
    return output;
}
gsm* getWeightsError(gsm* a, gsm* error)
{
    gsm* out = gsm_alloc(error->size1, a->size1);
    sgemm(CblasNoTrans, CblasTrans, 1, error, a, 0, out);
    return out;
}
void backprop(struct Network* net, gsm* input, gsm* y)
{
#ifdef DEBUG
    printf("Backprop Starting.\n");
#endif  // DEBUG
    feedforward(net, input);
#ifdef DEBUG
    printf("Feedforward Finished.\n");
#endif  // DEBUG
    gsm* error = getOutputError(net->layers[net->num_layers - 1].a, y);
#ifdef DEBUG
    printf("Output Error Finished.\n");
    double sum = 0;
    for (int i = 0; i < error->size1; i++) {
        for (int j = 0; j < error->size2; j++) {
            sum += gsm_get(error, i, j);
        }
    }
    printf("Error Sum: %f\n", sum);
#endif // DEBUG
    for (int i = net->num_layers - 1; i > 0; i--) {
        struct Layer* curr = &net->layers[i];
        gsv* biasError = getBiasError(error);
#ifdef DEBUG
        printf("Bias Error Finished.\n");
#endif // DEBUG
        gsm* weightError = getWeightsError(net->layers[i - 1].a, error);
#ifdef DEBUG
        printf("Weight Error Finished.\n");
#endif // DEBUG
        gsv_scale(biasError, net->rate / input->size2);
        gsm_scale(weightError, net->rate / input->size2);

        gsv_sub(curr->biases, biasError);
        gsm_sub(curr->weights, weightError);
        gsv_free(biasError);
        gsm_free(weightError);
        gsm* temp = getPreviousError(net->layers[i - 1].a, curr->weights, error);
        gsm_free(error);
#ifdef DEBUG
        printf("Previous Error Finished.\n");
#endif // DEBUG
        error = temp;
    }
    gsm_free(error);
    freeOutputs(net);
}
void freeOutputs(struct Network* net)
{
    for (int i = 0; i < net->num_layers; i++) {
        gsm_free(net->layers[i].a);
    }
}
void train(struct Network* net, gsm** input, gsm** y, int epochs, int batches)
{
    for (int i = 0; i < epochs; i++) {
        printf("Epoch %d\n", i);
        for (int j = 0; j < batches; j++) {
            backprop(net, input[j], y[j]);
        }
    }
}
void deleteNetwork(struct Network* net)
{
    for (int i = 0; i < net->num_layers; i++) {
        gsm_free(net->layers[i].weights);
        gsv_free(net->layers[i].biases);
    }
    free(net->layers);
    free(net);
}
int getDigit(struct Network* net, gsv* input)
{
    gsm* in = gsm_alloc(input->size, 1);
    for (int i = 0; i < input->size; i++) {
        gsm_set(in, i, 0, gsv_get(input, i));
    }
    feedforward(net, in);
    gsm* output = net->layers[net->num_layers - 1].a;
    int max = 0;
    for (int i = 0; i < output->size1; i++) {
        if (gsm_get(output, i, 0) > gsm_get(output, max, 0)) {
            max = i;
        }
    }
    gsm_free(in);
    return max;
}