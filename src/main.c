#include <stdio.h>
#include <stdlib.h>

#include <math.h>
#include <time.h>
#include "sine.h"
#include "cosine.h"
#include "inv_matrix.h"

// reference: [1]Liu JinKun. Robot Control System Design and MATLAB Simulation[M]. Tsinghua University Press, 2008.
// [2]王耀南. 机器人智能控制工程[M]. 科学出版社, 2004.

// global variables declaration
#define PI 3.14159
#define H 5              // hidden layer neurons number
#define IN 4             // input layer neurons number
#define n_joint 2        // output layer neurons number
#define ARRAY_SIZE 50000 // sampling times

static double Ts = 0.001; // sampling period
static double t0 = 0.0;   // start time
static double t1 = 50.0;  // end time
static double center[IN][H] = {{-1.1, -0.4, 0.5, 1.4, 2.1},
                               {-1.1, -0.4, 0.5, 1.4, 2.1},
                               {-0.05, 0.4, 1, 1.6, 2.05},
                               {-0.05, 0.4, 1, 1.6, 2.05}}; // center of RBF neural network
static double width = 0.9;                                   // width of RBF neural network

double sign(double x){ // symbolic function
    if (x > 0){
        return 1.0;
    } else if (x < 0){
        return -1.0;
    } else{
        return 0.0;
    }
}

// neural network structure
struct _neural_network{
    double input[IN];                     // RBF neural network's input
    double neural_output[n_joint];        // RBF neural network's output
    double hidden_output[H];              // RBF neural network's hidden layer output
    double weight[n_joint][H];            // RBF network's weight estimate
    double weight_derivative[n_joint][H]; // RBF network's weight estimate's derivative
} neural_network;

struct _archive{
    double q1_archive[ARRAY_SIZE];
    double dq1_archive[ARRAY_SIZE];
    double q2_archive[ARRAY_SIZE];
    double dq2_archive[ARRAY_SIZE];
    double error1_archive[ARRAY_SIZE];
    double error2_archive[ARRAY_SIZE];
    double error1_velocity_archive[ARRAY_SIZE];
    double error2_velocity_archive[ARRAY_SIZE];
    double neural_output1_archive[ARRAY_SIZE];
    double neural_output2_archive[ARRAY_SIZE];
    double model_uncertainty1_archive[ARRAY_SIZE];
    double model_uncertainty2_archive[ARRAY_SIZE];
    double torque1_archive[ARRAY_SIZE];
    double torque2_archive[ARRAY_SIZE];
} archive;

Data q1_desired, dq1_desired, ddq1_desired;
Data q2_desired, dq2_desired, ddq2_desired;

struct Amp{
    double q1_desired;
    double dq1_desired;
    double ddq1_desired;
    double q2_desired;
    double dq2_desired;
    double ddq2_desired;
};

struct M0{
    double q1_desired;
    double dq1_desired;
    double ddq1_desired;
    double q2_desired;
    double dq2_desired;
    double ddq2_desired;
};

struct B0{
    double q1_desired;
    double dq1_desired;
    double ddq1_desired;
    double q2_desired;
    double dq2_desired;
    double ddq2_desired;
};

void SystemInput(Data *q1_desired, Data *dq1_desired, Data *ddq1_desired, Data *q2_desired, Data *dq2_desired, Data *ddq2_desired, double Ts, double t0, double t1){

    struct Amp amp; // amplitude
    amp.q1_desired = 1.5;
    amp.dq1_desired = -1.5;
    amp.ddq1_desired = -1.5;
    amp.q2_desired = 1;
    amp.dq2_desired = -1;
    amp.ddq2_desired = -1;

    struct M0 m0; // angular frequency
    m0.q1_desired = 1;
    m0.dq1_desired = 1;
    m0.ddq1_desired = 1;
    m0.q2_desired = 1;
    m0.dq2_desired = 1;
    m0.ddq2_desired = 1;

    struct B0 b0; // vertical shift
    b0.q1_desired = 0.5;
    b0.dq1_desired = 0;
    b0.ddq1_desired = 0;
    b0.q2_desired = 1;
    b0.dq2_desired = 0;
    b0.ddq2_desired = 0;

    cosine(q1_desired, Ts, t0, t1, amp.q1_desired, m0.q1_desired, b0.q1_desired);           // desired angular displacement of link 1
    sine(dq1_desired, Ts, t0, t1, amp.dq1_desired, m0.dq1_desired, b0.dq1_desired);     // desired angular velocity of link 1
    cosine(ddq1_desired, Ts, t0, t1, amp.ddq1_desired, m0.ddq1_desired, b0.ddq1_desired);   // desired angular acceleration of link 1
    cosine(q2_desired, Ts, t0, t1, amp.q2_desired, m0.q2_desired, b0.q2_desired);         // desired angular displacement of link 2
    sine(dq2_desired, Ts, t0, t1, amp.dq2_desired, m0.dq2_desired, b0.dq2_desired);       // desired angular velocity of link 2
    cosine(ddq2_desired, Ts, t0, t1, amp.ddq2_desired, m0.ddq2_desired, b0.ddq2_desired); // desired angular acceleration of link 2
}

struct _system_state{
    double q[n_joint];   // actual angular displacement
    double dq[n_joint];  // actual angular velocity
    double ddq[n_joint]; // actual angular acceleration
} system_state;

double torque[n_joint]; // control input torque

struct _dynamics{
    double M[n_joint][n_joint];  // manipulator's inertia matrix
    double G[n_joint];           // manipulator's gravity matrix
    double Vm[n_joint][n_joint]; // manipulator's Coriolis matrix
    double D[n_joint];           // disturbance term
    double F[n_joint];           // model uncertainty term
} dynamics;

double l[n_joint]; // length of link
double m[n_joint]; // mass of link

struct _controller{
    double controller_u[10];
    double controller_out[4];
    double error[n_joint];          // angular displacement error
    double error_velocity[n_joint]; // angular velocity error
    double alpha;                   // parameter of second state variable, Eq. 3.160 define
    double x2[n_joint];             // second state variable, Eq. 3.160 define
    double eta;                     // learning rate
    double gamma;                   // desired disturbance suppression level
    double omega[n_joint];          // defined in Eq. 3.162
    int CONTROLLER;                 // control method
    double controller[n_joint];     // control volume
} controller;

void CONTROLLER_init(){
    system_state.q[0] = -2.0;
    system_state.dq[0] = 0.0;
    system_state.q[1] = -2.0;
    system_state.dq[1] = 0.0;
    controller.controller_u[0] = q1_desired.y[0];
    controller.controller_u[1] = dq1_desired.y[0];
    controller.controller_u[2] = ddq1_desired.y[0];
    controller.controller_u[3] = q2_desired.y[0];
    controller.controller_u[4] = dq2_desired.y[0];
    controller.controller_u[5] = ddq2_desired.y[0];
    controller.controller_u[6] = system_state.q[0];
    controller.controller_u[7] = system_state.dq[0];
    controller.controller_u[8] = system_state.q[1];
    controller.controller_u[9] = system_state.dq[1];
    controller.alpha = 30;
    controller.eta = 1000;
    controller.gamma = 0.05;
    controller.CONTROLLER = 2;
    for (int j = 0; j < n_joint; j++){
        for (int k = 0; k < H; k++){
            neural_network.weight[j][k] = 0.5;
        }
    }

    l[0] = 1; l[1] = 1;
    m[0] = 1; m[1] = 10;
}

double CONTROLLER_realize(int i){
    controller.controller_u[0] = q1_desired.y[i];
    controller.controller_u[1] = dq1_desired.y[i];
    controller.controller_u[2] = ddq1_desired.y[i];
    controller.controller_u[3] = q2_desired.y[i];
    controller.controller_u[4] = dq2_desired.y[i];
    controller.controller_u[5] = ddq2_desired.y[i];
    controller.controller_u[6] = system_state.q[0];
    controller.controller_u[7] = system_state.dq[0];
    controller.controller_u[8] = system_state.q[1];
    controller.controller_u[9] = system_state.dq[1];
    archive.q1_archive[i] = controller.controller_u[6];
    archive.dq1_archive[i] = controller.controller_u[7];
    archive.q2_archive[i] = controller.controller_u[8];
    archive.dq2_archive[i] = controller.controller_u[9];

    neural_network.input[0] = controller.controller_u[6]; // RBF neural network's input
    neural_network.input[1] = controller.controller_u[7];
    neural_network.input[2] = controller.controller_u[8];
    neural_network.input[3] = controller.controller_u[9];

    // hidden layer output of RBF neural network
    for (int j = 0; j < H; j++){
        double sum = 0.0;
        for (int k = 0; k < IN; k++) {
            sum += pow(neural_network.input[k] - center[k][j], 2);
        }
        neural_network.hidden_output[j] = exp(-sum / (2 * width * width));
    }
    // for (int j = 0; j < H; j++){
    //     printf("neural_network.hidden_output[%d]: %f\n", j, neural_network.hidden_output[j]);
    // }

    // output of output layer of RBF neural network
    for (int j = 0; j < n_joint; j++){
        double sum = 0.0;
        for (int k = 0; k < H; k++){
            sum += neural_network.weight[j][k] * neural_network.hidden_output[k];
        }
        neural_network.neural_output[j] = sum;
    }

    archive.neural_output1_archive[i] = neural_network.neural_output[0];
    archive.neural_output2_archive[i] = neural_network.neural_output[1];

    controller.error[0] = system_state.q[0] - q1_desired.y[i];            // angular position tracking error of link 1
    controller.error_velocity[0] = system_state.dq[0] - dq1_desired.y[i]; // angular velocity tracking error of link 1
    controller.error[1] = system_state.q[1] - q2_desired.y[i];            // angular position tracking error of link 2
    controller.error_velocity[1] = system_state.dq[1] - dq2_desired.y[i]; // angular velocity tracking error of link 2

    archive.error1_archive[i] = controller.error[0];
    archive.error1_velocity_archive[i] = controller.error_velocity[0];
    archive.error2_archive[i] = controller.error[1];
    archive.error2_velocity_archive[i] = controller.error_velocity[1];

    for (int j = 0; j < n_joint; j++){
        controller.x2[j] = controller.error_velocity[j] + controller.alpha * controller.error[j]; // second state variable, Eq. 3.160 define
    }
    // for (int j = 0; j < n_joint; j++){
    //     printf("controller.x2[%d]: %f\n", j, controller.x2[j]);
    // }

    // inertia matrix of robot manipulator dynamics model
    dynamics.M[0][0] = (m[0] + m[1]) * pow(l[0], 2) + m[1] * pow(l[1], 2) + 2 * m[1] * l[0] * l[1] * cos(controller.controller_u[8]);
    dynamics.M[0][1] = m[1] * pow(l[1], 2) + m[1] * l[0] * l[1] * cos(controller.controller_u[8]);
    dynamics.M[0][1] = dynamics.M[1][0];
    dynamics.M[1][1] = m[1] * pow(l[1], 2);

    // coriolis/centrifugal force matrix of robot manipulator dynamics model
    double Vm12 = m[1] * l[0] * sin(controller.controller_u[8]);
    dynamics.Vm[0][0] = -Vm12 * controller.controller_u[9];
    dynamics.Vm[0][1] = -Vm12 * (controller.controller_u[7] + controller.controller_u[9]);
    dynamics.Vm[1][0] = Vm12 * controller.controller_u[6];
    dynamics.Vm[2][2] = 0;

    // gravitational matrix of robot manipulator dynamics model
    dynamics.G[0] = (m[0] + m[1]) * l[0] * cos(controller.controller_u[8]) + m[1] * l[1] * cos(controller.controller_u[6] + controller.controller_u[8]);
    dynamics.G[1] = m[1] * l[1] * cos(controller.controller_u[6] + controller.controller_u[8]);

    for (int j = 0; j < n_joint; j++){
        controller.omega[j] = controller.alpha * (dynamics.M[j][0] * controller.error_velocity[0] + dynamics.M[j][1] * controller.error_velocity[1] +
                                                  dynamics.Vm[j][0] * controller.error[0] + dynamics.Vm[j][1] * controller.error[1]);
    }

    // without neural network compensation
    if (controller.CONTROLLER == 1){
        for (int j = 0; j < n_joint; j++){
            controller.controller[j] = -controller.error[j] - controller.omega[j] - 0.5 / pow(controller.gamma, 2) * (controller.error_velocity[j] + controller.alpha * controller.error[j]);
        }
    }
    // within neural network compensation
    else if (controller.CONTROLLER == 2){
        for (int j = 0; j < n_joint; j++){
            controller.controller[j] = -controller.error[j] - controller.omega[j] + neural_network.neural_output[j] - 0.5 / pow(controller.gamma, 2) * (controller.error_velocity[j] + controller.alpha * controller.error[j]);
        }
    }

    for (int j = 0; j < n_joint; j++){
        torque[j] = controller.controller[j] + dynamics.M[j][0] * controller.controller_u[2] + dynamics.M[j][1] * controller.controller_u[5] + dynamics.Vm[j][0] * controller.controller_u[1] + dynamics.Vm[j][1] * controller.controller_u[4] + dynamics.G[j]; // control law
    }

    archive.torque1_archive[i] = torque[0];
    archive.torque2_archive[i] = torque[1];

    // weight adaptive updating law, RBF neural network weight's derivative
    for (int j = 0; j < n_joint; j++){
        for (int k = 0; k < H; k++){
            neural_network.weight_derivative[j][k] = -controller.eta * controller.x2[j] * neural_network.hidden_output[k];
        }
    }

    for (int j = 0; j < n_joint; j++){
        for (int k = 0; k < H; k++){
            neural_network.weight[j][k] += neural_network.weight_derivative[j][k] * Ts;
        }
    }

    controller.controller_out[0] = torque[0];
    controller.controller_out[1] = torque[1];
    controller.controller_out[2] = neural_network.neural_output[0];
    controller.controller_out[3] = neural_network.neural_output[1];
}

struct _plant{
    double plant_u[4];
    double plant_out[4];
} plant;

void PLANT_init(){
    system_state.q[0] = -2.0;
    system_state.dq[0] = 0.0;
    system_state.q[1] = -2.0;
    system_state.dq[1] = 0.0;
    plant.plant_out[0] = system_state.q[0];
    plant.plant_out[1] = system_state.dq[0];
    plant.plant_out[2] = system_state.q[1];
    plant.plant_out[3] = system_state.dq[1];
}

double PLANT_realize(int i){
    plant.plant_u[0] = torque[0];
    plant.plant_u[1] = torque[1];
    srand((unsigned int)time(NULL));
    dynamics.D[0] = (double)(rand()) / RAND_MAX * 20 - 10; // disturbance term
    dynamics.D[1] = (double)(rand()) / RAND_MAX * 20 - 10;

    for (int j = 0; j < n_joint; j++){
        dynamics.F[j] = 10 * sign(system_state.dq[j]) * (0.1 + exp(-fabs(system_state.dq[j]))); // model uncertainty term
    }

    archive.model_uncertainty1_archive[i] = dynamics.F[0];
    archive.model_uncertainty2_archive[i] = dynamics.F[1];

    double model_uncertainty1_archive[ARRAY_SIZE];
    double model_uncertainty2_archive[ARRAY_SIZE];
    double inv_M[n_joint][n_joint], torque_Vmdq_D_G_F[n_joint];
    inv_matrix(inv_M, dynamics.M, 2);
    for (int j = 0; j < n_joint; j++){
        torque_Vmdq_D_G_F[j] = plant.plant_u[j] - (dynamics.Vm[j][0] * system_state.dq[0] + dynamics.Vm[j][1] * system_state.dq[1]) + dynamics.D[j] - dynamics.G[j] - dynamics.F[j];
    }

    for (int j = 0; j < n_joint; j++){
        system_state.ddq[j] = inv_M[j][0] * torque_Vmdq_D_G_F[0] + inv_M[j][1] * torque_Vmdq_D_G_F[1];
    }

    system_state.dq[0] = system_state.dq[0] + system_state.ddq[0] * Ts;
    system_state.dq[1] = system_state.dq[1] + system_state.ddq[1] * Ts;
    system_state.q[0] = system_state.q[0] + system_state.dq[0] * Ts;
    system_state.q[1] = system_state.q[1] + system_state.dq[1] * Ts;

    plant.plant_out[0] = system_state.q[0];
    plant.plant_out[1] = system_state.dq[0];
    plant.plant_out[2] = system_state.q[1];
    plant.plant_out[3] = system_state.dq[1];
}

void saveArchiveToTxt(double *archive, int size, const char *filename){

    FILE *file = fopen(filename, "w+");

    if (file == NULL){
        perror("Failed to open file");
        exit(1);
    }
    else{
        for (int i = 0; i < size; i++){
            fprintf(file, "%lf\n", archive[i]);
        }
        fclose(file);
        printf("Saved to file %s\n", filename);
    }
}

void saveArchive(){

    saveArchiveToTxt(q1_desired.y, ARRAY_SIZE, "../report/qd1.txt");
    saveArchiveToTxt(archive.q1_archive, ARRAY_SIZE, "../report/q1.txt");
    saveArchiveToTxt(archive.dq1_archive, ARRAY_SIZE, "../report/dq1.txt");
    saveArchiveToTxt(q2_desired.y, ARRAY_SIZE, "../report/qd2.txt");
    saveArchiveToTxt(archive.q2_archive, ARRAY_SIZE, "../report/q2.txt");
    saveArchiveToTxt(archive.dq2_archive, ARRAY_SIZE, "../report/dq2.txt");
    saveArchiveToTxt(archive.error1_archive, ARRAY_SIZE, "../report/error1.txt");
    saveArchiveToTxt(archive.error1_velocity_archive, ARRAY_SIZE, "../report/error1_velocity.txt");
    saveArchiveToTxt(archive.error2_archive, ARRAY_SIZE, "../report/error2.txt");
    saveArchiveToTxt(archive.error2_velocity_archive, ARRAY_SIZE, "../report/error2_velocity.txt");
    saveArchiveToTxt(archive.neural_output1_archive, ARRAY_SIZE, "../report/neural_output1.txt");
    saveArchiveToTxt(archive.neural_output2_archive, ARRAY_SIZE, "../report/neural_output2.txt");
    saveArchiveToTxt(archive.model_uncertainty1_archive, ARRAY_SIZE, "../report/model_uncertainty1.txt");
    saveArchiveToTxt(archive.model_uncertainty2_archive, ARRAY_SIZE, "../report/model_uncertainty2.txt");
    saveArchiveToTxt(archive.torque1_archive, ARRAY_SIZE, "../report/torque1.txt");
    saveArchiveToTxt(archive.torque2_archive, ARRAY_SIZE, "../report/torque2.txt");
}

int main(){

    SystemInput(&q1_desired, &dq1_desired, &ddq1_desired, &q2_desired, &dq2_desired, &ddq2_desired, Ts, t0, t1);
    CONTROLLER_init(); // initialize controller parameter
    PLANT_init();      // initialize plant parameter

    for (int i = 0; i < ARRAY_SIZE; i++){
    // for (int i = 0; i < 5; i++){
        double time = i * Ts + t0;
        printf("time at step %d: %f\n", i, time);
        CONTROLLER_realize(i);
        PLANT_realize(i);
    }

    saveArchive();

    return 0;
}
