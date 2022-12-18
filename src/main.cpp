#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include "active_func.h"

#define LR 0.5
#define EPOCHS 5000
#define ZERO 0

void init();
double rand_num();
void shuffle(int *array, size_t n);
void forward_prop(double *rand_training_pt);
void test_nn(void);

// define the structure of neural network
static const int input_num = 2;
static const int hidden_nodes = 2;
static const int output_num = 1;

double *hidden_layer;
double *output_layer;
double *hidden_bias;
double *output_bias;

double **hidden_weights;
double **output_weights;

// define the training set
static const int training_sets_num = 4;
double training_inputs[training_sets_num][input_num] = {{0, 0}, {1, 0}, {0, 1}, {1, 1}};
double training_outputs[training_sets_num][output_num] = {{0}, {1}, {1}, {0}};

int main()
{
    srand(time(NULL)); // time random

    hidden_layer = (double *)malloc(hidden_nodes * sizeof(double));
    output_layer = (double *)malloc(output_num * sizeof(double));
    hidden_bias = (double *)malloc(hidden_nodes * sizeof(double));
    output_bias = (double *)malloc(output_num * sizeof(double));

    hidden_weights = (double **)malloc(input_num * sizeof(double *));
    for (int i = 0; i < input_num; i += 1)
    {
        hidden_weights[i] = (double *)malloc(hidden_nodes * sizeof(double));
    }

    output_weights = (double **)malloc(hidden_nodes * sizeof(double *));
    for (int i = 0; i < input_num; i += 1)
    {
        output_weights[i] = (double *)malloc(output_num * sizeof(double));
    }

    init(); // Initialize weights and biases

    // Training Process
    int training_sets_order[] = {0, 1, 2, 3};
    double rand_training_pt[input_num] = {};
    int rand_training_out = 0;
    for (int iteration = 0; iteration < EPOCHS; iteration++)
    {
        shuffle(training_sets_order, training_sets_num); // shuffle the order of training points
        for (int order = 0; order < training_sets_num; order++)
        {
            int i = training_sets_order[order];
            // rand_training_out = training_outputs[i][];
            for (int j = 0; j < input_num; j++)
            {
                rand_training_pt[j] = training_inputs[i][j];
            }

            // Foward propagation
            forward_prop(rand_training_pt);

            // printf("Input: %f %f   Output: %f   Expected Output: %f \n", training_inputs[i][0], training_inputs[i][1], output_layer[0], training_outputs[i][0]);

            // Backward propagation
            // calculate cost
            double output_delta[output_num] = {};
            double output_error = 0;
            for (int j = 0; j < output_num; j++)
            {
                output_error = (training_outputs[i][j] - output_layer[j]); // Error

                output_delta[j] = output_error * dsigmoid(output_layer[j]);
            }
            double hidden_delta[hidden_nodes] = {};
            for (int j = 0; j < hidden_nodes; j++)
            {
                double hidden_error = 0;
                for (int k = 0; k < output_num; k++)
                {
                    hidden_error += output_delta[k] * output_weights[j][k];
                }

                hidden_delta[j] = hidden_error * dsigmoid(hidden_layer[j]);
            }
            // update weights
            for (int j = 0; j < output_num; j++)
            {
                output_bias[j] += output_delta[j] * LR;
                for (int k = 0; k < hidden_nodes; k++)
                {
                    output_weights[k][j] += hidden_layer[k] * output_delta[j] * LR;
                }
            }
            for (int j = 0; j < hidden_nodes; j++)
            {
                hidden_bias[j] += hidden_delta[j] * LR;
                for (int k = 0; k < input_num; k++)
                {
                    hidden_weights[k][j] += rand_training_pt[k] * hidden_delta[j] * LR;
                }
            }
        }
    }

    test_nn();
    free(hidden_layer);
    free(output_layer);
    free(hidden_bias);
    free(output_bias);
    free(hidden_weights);
    free(output_weights);

    return 0;
}

double rand_num()
{
    // srand(time(NULL));
    double num = 0;
    num = ((double)rand()) / ((double)RAND_MAX);
    return num;
}

void init(void)
{
    for (int i = 0; i < input_num; i++)
    {
        for (int j = 0; j < hidden_nodes; j++)
        {
            hidden_weights[i][j] = rand_num(); // input layer to hideen layer -> 2x2 matrix
        }
    }
    for (int i = 0; i < hidden_nodes; i++)
    {
        hidden_bias[i] = rand_num();
        for (int j = 0; j < output_num; j++)
        {
            output_weights[i][j] = rand_num(); // hidden layer to output layer -> 2x1 matrix
        }
    }
    for (int i = 0; i < output_num; i++)
    {
        output_bias[i] = rand_num();
    }
}
void shuffle(int *array, size_t n)
{
    if (n > 1)
    {
        size_t i;
        for (i = 0; i < n - 1; i++)
        {
            srand(time(NULL));
            size_t j = i + rand() / (RAND_MAX / (n - i) + 1);
            int tmp = array[j];
            array[j] = array[i];
            array[i] = tmp;
        }
    }
}

void forward_prop(double *rand_training_pt)
{
    for (int j = 0; j < hidden_nodes; j++)
    {
        double activation = 0;
        for (int k = 0; k < input_num; k++)
        {
            activation += rand_training_pt[k] * hidden_weights[k][j]; // sum of weight*input
        }
        activation += hidden_bias[j];
        hidden_layer[j] = sigmoid(activation);
    }
    for (int j = 0; j < output_num; j++)
    {
        double activation = 0;
        for (int k = 0; k < hidden_nodes; k++)
        {
            activation += hidden_layer[k] * output_weights[k][j]; // sum of hidden*output
        }
        activation += output_bias[j];
        output_layer[j] = sigmoid(activation);
    }
}

void test_nn(void)
{
    static const int testing_num = 4;
    double testing_input[testing_num][input_num] = {{0, 0}, {1, 0}, {0, 1}, {1, 1}};
    double expect_output[testing_num] = {{0}, {1}, {1}, {0}};

    for (int i = 0; i < testing_num; i++)
    {
        double testing_pt[input_num] = {};
        for (int j = 0; j < input_num; j++)
        {
            testing_pt[j] = testing_input[i][j];
        }

        forward_prop(testing_pt);

        printf("Input: %f %f    Output: %f    Expect: %f\n", testing_input[i][0], testing_input[i][1], output_layer[0], expect_output[i]);
    }
}