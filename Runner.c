#include <gsl/gsl_blas.h>
#include <gsl/gsl_matrix_float.h>
#include <gsl/gsl_vector_float.h>
#include <math.h>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>

#include "Debug.h"
#include "Network.h"
#define gsm_calloc gsl_matrix_float_calloc
int main()
{
    int layers = 4;
    float rate = .63;
    int dataWidth = 784;
    int outputSz = 10;
    int layerSz[] = { dataWidth, 100, 100, outputSz };
    int batchSz = 100;
    int epochs = 10;
    int dataSz = 60000;
    struct Network* net = init_network(layers, rate, layerSz, batchSz);
    if (net == NULL) {
        printf("Error initializing network\n");
        return -1;
    }
#ifdef DEBUG
    printf("Starting Input Processing.\n");
#endif  // DEBUG
    int batches = dataSz / batchSz;
    gsm** inputs;
    inputs = malloc(sizeof(gsm*) * batches);
    gsm** targets;
    targets = malloc(sizeof(gsm*) * batches);
    if (inputs == NULL || targets == NULL) {
        printf("Error allocating inputs or targets\n");
        return -1;
    }
    FILE* stream = fopen("mnist_train.csv", "r");
    char* buf = NULL;
    size_t len = 0;
    getline(&buf, &len, stream);
    if (buf == NULL) return -1;
    for (int i = 0; i < batches; i++) {
        inputs[i] = gsm_alloc(dataWidth, batchSz);
        targets[i] = gsm_alloc(outputSz, batchSz);
        for (int j = 0; j < batchSz; j++) {
            getline(&buf, &len, stream);
            if (buf == NULL) return -1;
            char* tok = strtok(buf, ",");
            gsl_matrix_float_set(targets[i], atoi(tok), j, 1.0);
            for (int k = 0; k < dataWidth; k++) {
                tok = strtok(NULL, ",");
                gsl_matrix_float_set(inputs[i], k, j, atof(tok) / 255.0);
            }
        }
    }
    free(buf);
    fclose(stream);
#ifdef DEBUG
    printf("Finished Input Processing\n");
#endif  // DEBUG
    train(net, inputs, targets, epochs, batchSz);
    free(inputs);
    free(targets);
    stream = fopen("mnist_test.csv", "r");
    buf = NULL;
    len = 0;
    getline(&buf, &len, stream);
    if (buf == NULL) return -1;
    int correct = 0;
    int total = 0;
    while (getline(&buf, &len, stream) != -1) {
        gsv* input = gsv_alloc(dataWidth);
        char* tok = strtok(buf, ",");
        int target = atoi(tok);
        for (int k = 0; k < dataWidth; k++) {
            tok = strtok(NULL, ",");
            gsv_set(input, k, atof(tok) / 255.0);
        }
        int output = getDigit(net, input);
        if (target == output) {
            correct++;
        }
        total++;
        gsv_free(input);
    }
    printf("Accuracy: %f, Correct: %d, Total: %d\n", (float)correct / total, correct, total);
    deleteNetwork(net);
    return 0;
}