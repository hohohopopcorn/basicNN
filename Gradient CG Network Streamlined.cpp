#define _CRT_SECURE_NO_WARNINGS
#define _CRT_NONSTDC_NO_WARNINGS
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <float.h>
#include <limits.h>
#include "mkl.h"

//CHANGEABLE--------------------------------------------------------------------------------
#define layers 2  //amount of hidden layers
#define input 2  //amount of inputs
#define output 1  //amount of outputs
int const amount[layers+2]={input, 20, 20, output};  // amount of nodes in layer n
#define MAX 21  //LIMIT OF amount[]                                       IMPORTANT
int ITERATIONS=0;   //amount of iterations for training (greater than size of weight array to make sure)
int SAVE=0;  //frequency of saving weights
int CUTOFF=0; //maximum number of conjugate gradients
#define pi 3.1415926535897932384626
#define PHI (1+sqrt(5))/2
#define naturalE 2.718281828459
char trainfile[200];
#define LOC sprintf(a, "")
const double goodrange[2]={3, 0.5};  //feature scaling targets (absolute value of range of input and output has to be equal)

double h(double x)   //activation function
{ // WATCH OUT FOR INFINITY!!!!!
	double temp;
	temp=(double)1/(x*x+(double)1);
	return temp;
}
double hprime(double x)
{ // WATCH OUT FOR INFINITY!!!!!
	double temp;
	temp=-(double)2*x/(x*x+(double)1)/(x*x+(double)1);
	return temp;
}
double hprimeprime(double x)
{ // WATCH OUT FOR INFINITY!!!!!
	double temp;
	temp=((double)6*x*x-(double)2)/(x*x+(double)1)/(x*x+(double)1)/(x*x+(double)1);
	return temp;
}

double errorfunction(double predict, double actual)
{
	return (predict-actual)*(predict-actual)/(double)2;
}
double errorfunctionprime(double predict, double actual)
{
	return (predict-actual);
}
//CHANGEABLE--------------------------------------------------------------------------------

double *w=(double *)mkl_malloc((layers+1)*MAX*MAX*sizeof(double), 128);  //w[layers+1][MAX][MAX];  //weights in the synapses
double *best=(double *)mkl_malloc((layers+1)*MAX*MAX*sizeof(double), 128);  //best[layers+1][MAX][MAX];  //best weights in the synapses
double *err=(double *)mkl_malloc((layers+1)*MAX*MAX*sizeof(double), 128);   //err[layers+1][MAX][MAX];  //error gradient
double *nodes=(double *)mkl_malloc((layers+2)*MAX*sizeof(double), 128);  //nodes[layers+2][MAX];  //x values in each node after networking once (a terms), h(a)=z terms
double *delta=(double *)mkl_malloc((layers+2)*MAX*MAX*sizeof(double), 128);  //delta[layers+2][MAX][MAX];  //dE/dx in node (delta[0][][] should all be empty)
double *Gnodes=(double *)mkl_malloc((layers+2)*MAX*MAX*sizeof(double), 128);  //Gnodes[layers+2][MAX][MAX];  //y values in each node after networking once (a terms), h(a)=z terms
double *Gdelta=(double *)mkl_malloc((layers+2)*MAX*MAX*sizeof(double), 128);  //Gdelta[layers+2][MAX][MAX];  //dE/dy in node (Gdelta[0][][] should all be empty)
double *Hv=(double *)(double *)mkl_malloc((layers+1)*MAX*MAX*sizeof(double), 128);  //Hv[layers+1][MAX][MAX];  //hessian gradient in error gradient (Herr[0][] should all be empty)
double *ZERO=(double *)mkl_malloc((layers+1)*MAX*MAX*sizeof(double), 128);   //ZERO[layers+1][MAX][MAX];  //reset matrices
double *trainingset, *checkset;   //data points
double training[input+output*input];  //temperary training set
double testing[input];  //given inputs
double const P_infinity=DBL_MAX*DBL_MAX;
double const N_infinity=-DBL_MAX*DBL_MAX;
double const SMALL_NUM=DBL_MIN;
int timesbefore=0;
double totalpoints=0; //total amount of points
double translation[input+output];  //translation for feature scaling
double stretch[input+output];  //stretch for feature scaling

void display_w(FILE *store, int times)
{
	fprintf(store, "%d\n", layers);
	for(int i=0; i<layers+2; i++)fprintf(store, "%d ", amount[i]);
	fprintf(store, "\n");
	for(int i=0; i<input+output; i++)fprintf(store, "%20e %20e ", translation[i], stretch[i]);
	fprintf(store, "\n%d\n", timesbefore+times+1);
	for(int i=0; i<layers+1; i++)
	{
		for(int j=0; j<=amount[i]; j++)
		{
			for(int k=1; k<=amount[i+1]; k++)
			{
				fprintf(store, "%15e ", w[i*MAX*MAX+j*MAX+k]);
			}
			fprintf(store, "\n");
		}
		fprintf(store, "\n\n");
	}
	fclose(store);
}

void hidden(int lay=1) // which layer
{
	double *temp=(double *)mkl_malloc(MAX*sizeof(double), 128);
	cblas_dcopy(MAX, ZERO, 1, temp, 1);
	temp[0]=1;
	for(int i=1; i<=amount[lay-1]&&lay==1; i++)temp[i]=nodes[(lay-1)*MAX+i];
	for(int i=1; i<=amount[lay-1]&&lay!=1; i++)temp[i]=h(nodes[(lay-1)*MAX+i]);
	cblas_dgemv(CblasRowMajor, CblasTrans, amount[lay-1]+1, amount[lay]+1, 1,
		&w[(lay-1)*MAX*MAX], MAX, temp, 1, 0, &nodes[lay*MAX], 1);
	nodes[lay*MAX]=1;

	//Gradient version
	for(int i=0; i<input; i++)
	{
		cblas_dcopy(MAX, &Gnodes[(lay-1)*MAX*MAX+i], MAX, temp, 1);
		for(int j=1; j<=amount[lay-1]&&lay!=1; j++)temp[j]=temp[j]*hprime(nodes[(lay-1)*MAX+j]);
		cblas_dgemv(CblasRowMajor, CblasTrans, amount[lay-1]+1, amount[lay]+1, 1,
			&w[(lay-1)*MAX*MAX], MAX, temp, 1, 0, &Gnodes[lay*MAX*MAX+i], MAX);
		Gnodes[lay*MAX*MAX+i]=0;
	}
	mkl_free(temp);
	if(lay<layers+1)hidden(lay+1);
}

void network()  //getting value from testing using neural networking
{
	nodes[0]=1;
	for(int i=1; i<=input; i++)nodes[i]=(testing[i-1]-translation[i-1])*stretch[i-1];
	for(int i=0; i<input; i++)
	{
		for(int j=0; j<=input; j++)Gnodes[j*MAX+i]=0;
		Gnodes[(i+1)*MAX+i]=1;
	}
	hidden();
	for(int i=1; i<=output; i++)nodes[(layers+1)*MAX+i]=(nodes[(layers+1)*MAX+i])/stretch[i-1+input]+translation[i-1+input];
	for(int i=1; i<=output; i++)
	{
		for(int j=0; j<input; j++)Gnodes[(layers+1)*MAX*MAX+i*MAX+j]=Gnodes[(layers+1)*MAX*MAX+i*MAX+j]/stretch[i-1+input]*stretch[j];
	}
}

void backprop(int lay=1)  //back propagation (which layer) first hidden layer is 1
{
	if(lay+1<=layers+1)backprop(lay+1); //not last layer yet (layers+1)
										//calculate deltas (delta[x][y] is respective to node[x][y])
										//bias node does not have error
	if(lay==layers+1)
	{
		for(int i=0; i<input; i++)
		{
			for(int j=1; j<=output; j++)Gdelta[lay*MAX*MAX+j*MAX+i]=
				errorfunctionprime(Gnodes[lay*MAX*MAX+j*MAX+i]*stretch[j-1+input]/stretch[i],
					training[input+(j-1)*input+i]*stretch[j-1+input]/stretch[i]);
			Gdelta[lay*MAX*MAX+i]=0;  //bias does not pass error
		}
		for(int i=0; i<input; i++)
		{
			for(int j=0; j<=output; j++)delta[lay*MAX*MAX+j*MAX+i]=0;  //error function not dependent on function value
		}
	}
	else
	{
		for(int i=0; i<input; i++)
		{
			cblas_dgemv(CblasRowMajor, CblasNoTrans, MAX, MAX, 1,
				&w[lay*MAX*MAX], MAX, &Gdelta[(lay+1)*MAX*MAX+i], MAX, 0, &Gdelta[lay*MAX*MAX+i], MAX);
			for(int j=1; j<=amount[lay]; j++)Gdelta[lay*MAX*MAX+j*MAX+i]*=hprime(nodes[lay*MAX+j]);
			Gdelta[lay*MAX*MAX+i]=0;  //bias does not pass error
		}
		double *temp=(double *)mkl_malloc(MAX*sizeof(double), 128);
		cblas_dcopy(MAX, ZERO, 1, temp, 1);
		for(int i=0; i<input; i++)
		{
			cblas_dgemv(CblasRowMajor, CblasNoTrans, MAX, MAX, 1,
				&w[lay*MAX*MAX], MAX, &delta[(lay+1)*MAX*MAX+i], MAX, 0, &delta[lay*MAX*MAX+i], MAX);
			for(int j=1; j<=amount[lay]; j++)delta[lay*MAX*MAX+j*MAX+i]*=hprime(nodes[lay*MAX+j]);
			delta[lay*MAX*MAX+i]=0;  //bias does not pass error

			cblas_dgemv(CblasRowMajor, CblasNoTrans, MAX, MAX, 1,
				&w[lay*MAX*MAX], MAX, &Gdelta[(lay+1)*MAX*MAX+i], MAX, 0, temp, 1);
			for(int j=1; j<=amount[lay]; j++)temp[j]*=hprimeprime(nodes[lay*MAX+j])*Gnodes[(lay)*MAX*MAX+j*MAX+i];
			temp[0]=0;  //bias does not pass error

			cblas_daxpby(MAX, 1, temp, 1, 1, &delta[lay*MAX*MAX+i], MAX);
		}
		mkl_free(temp);
	}
	//calculate error gradient
	double *temp=(double *)mkl_malloc(MAX*sizeof(double), 128);
	for(int i=0; i<=amount[lay-1]; i++)
	{
		cblas_dcopy(MAX, ZERO, 1, temp, 1);
		for(int j=0; j<input; j++)
		{
			double m;
			if(i==0)m=1;
			else if(lay-1==0)m=nodes[(lay-1)*MAX+i];
			else m=h(nodes[(lay-1)*MAX+i]);
			cblas_daxpby(MAX, m, &delta[(lay)*MAX*MAX+j], MAX, 1, temp, 1);

			if(i==0)m=0;
			else if(lay-1==0)m=Gnodes[(lay-1)*MAX*MAX+i*MAX+j];
			else m=hprime(nodes[(lay-1)*MAX+i])*Gnodes[(lay-1)*MAX*MAX+i*MAX+j];
			cblas_daxpby(MAX, m, &Gdelta[(lay)*MAX*MAX+j], MAX, 1, temp, 1);
		}
		cblas_daxpby(MAX, 1, temp, 1, 1, &err[(lay-1)*MAX*MAX+i*MAX], 1);
	}
	mkl_free(temp);
}

void trainresults(int iter)
{
	FILE *store=fopen("weight.txt", "w");
	display_w(store, iter);
}

void outputcheck()
{
	FILE *hypo=fopen("output.txt", "w");  //predicted values of the training set
	for(int n=0; n<totalpoints; n++)
	{
		memcpy(training, &trainingset[n*(input+output*input)], sizeof(training));

		for(int i=0; i<input; i++)
		{
			testing[i]=training[i];
			fprintf(hypo, "%15e ", training[i]);
		}
		network();
		for(int i=0; i<input; i++)
		{
			for(int j=1; j<=output; j++)fprintf(hypo, "%15e ", Gnodes[(layers+1)*MAX*MAX+j*MAX+i]);
		}
		for(int i=1; i<=output*input; i++)fprintf(hypo, "%15e ", training[input+i-1]);
		double cost=0;
		for(int i=0; i<input; i++)
		{
			for(int j=0; j<output; j++)cost+=errorfunction(Gnodes[(layers+1)*MAX*MAX+(j+1)*MAX+i], training[input+j*input+i]);
		}
		fprintf(hypo, "%15e\n", cost);
	}
	fclose(hypo);
}

void outputcost(double history, int iter)
{
	int length;
	char a[1000]={0}, ending[100]="_cost.txt";
	LOC;
	length=strlen(a);
	for(int i=0; (size_t)i<strlen(trainfile)-4; i++)a[i+length]=trainfile[i];
	length=strlen(a);
	for(int i=0; i<layers; i++)
	{
		a[length++]='_';
		sprintf(a+length, "%d", amount[1+i]);
		length=strlen(a);
	}
	strncpy(a+length,ending,strlen(ending));

	FILE *cost=fopen(a, "a");  //cost history
	if(cost==NULL||iter==5)FILE *cost=fopen(a, "w");  //cost history
	fprintf(cost, "%10d %20e\n", iter, history);
	fclose(cost);
}

double func(double weights[]=best)
{
	double cost=0;
	memcpy(w, weights, (layers+1)*MAX*MAX*sizeof(double));
	for(int n=0; n<totalpoints; n++)
	{
		memcpy(training, &trainingset[n*(input+output*input)], sizeof(training));
		for(int i=0; i<input; i++)testing[i]=training[i];
		network();
		double costing=0;
		for(int i=0; i<input; i++)
		{
			for(int j=0; j<output; j++)costing+=errorfunction(Gnodes[(layers+1)*MAX*MAX+(j+1)*MAX+i], training[input+j*input+i]);
		}
		cost+=costing/totalpoints;
	}
	if(cost!=cost||cost==P_infinity||cost==N_infinity)cost=DBL_MAX;
	return cost;
}

double truefunc(double weights[]=best)
{
	double cost=0;
	memcpy(w, weights, (layers+1)*MAX*MAX*sizeof(double));
	for(int n=0; n<totalpoints; n++)
	{
		memcpy(training, &trainingset[n*(input+output*input)], sizeof(training));
		for(int i=0; i<input; i++)testing[i]=training[i];
		network();
		double costing=0;
		for(int i=0; i<input; i++)
		{
			for(int j=0; j<output; j++)costing+=
				errorfunction(Gnodes[(layers+1)*MAX*MAX+(j+1)*MAX+i]*stretch[j+input]/stretch[i], training[input+j*input+i]*stretch[j+input]/stretch[i]);
		}
		cost+=costing/totalpoints;
	}
	if(cost!=cost||cost==P_infinity||cost==N_infinity)cost=DBL_MAX;
	return cost;
}

void grad(double weights[]=best)
{
	double rest=0;
	cblas_dcopy((layers+1)*MAX*MAX, weights, 1, w, 1);
	cblas_dcopy((layers+1)*MAX*MAX, ZERO, 0, err, 1);
	for(int n=0; n<totalpoints; n++)
	{
		memcpy(training, &trainingset[n*(input+output*input)], sizeof(training));
		for(int i=0; i<input; i++)testing[i]=training[i];
		network();
		backprop();  //error gradient
	}
	cblas_dscal((layers+1)*MAX*MAX, 1/totalpoints, err, 1);  //average gradient
}

//Hessian product with optimal direction
void hessproduct(double direction[])
{
	double small=1e-6;
	double *move=(double *)mkl_malloc((layers+1)*MAX*MAX*sizeof(double), 128);

	cblas_dcopy((layers+1)*MAX*MAX, best, 1, move, 1);
	cblas_daxpby((layers+1)*MAX*MAX, small, direction, 1, 1, move, 1);
	grad(move);
	cblas_dcopy((layers+1)*MAX*MAX, err, 1, Hv, 1);
	grad();
	cblas_daxpby((layers+1)*MAX*MAX, -1/small, err, 1, 1/small, Hv, 1);
	mkl_free(move);
}

void linemin(double *X, double direction[])
{
	double alpha=1e0;
	double gamma=0.5;
	double tau=0.1;//1-1/PHI;
	double *unit=(double *)mkl_malloc((layers+1)*MAX*MAX*sizeof(double), 128);
	double *here=(double *)mkl_malloc((layers+1)*MAX*MAX*sizeof(double), 128);
	double *start=(double *)mkl_malloc((layers+1)*MAX*MAX*sizeof(double), 128);
	double accounted=0;
	long double magnitude=0;

	magnitude=cblas_ddot((layers+1)*MAX*MAX, direction, 1, direction, 1);
	magnitude=sqrt(magnitude);
	if(magnitude==0)
	{
		mkl_free(unit);
		mkl_free(start);
		mkl_free(here);
		return;
	}
	cblas_dcopy((layers+1)*MAX*MAX, direction, 1, unit, 1);
	cblas_dscal((layers+1)*MAX*MAX, 1/magnitude, unit, 1);
	cblas_dcopy((layers+1)*MAX*MAX, X, 1, start, 1);
	cblas_daxpby((layers+1)*MAX*MAX, 1, best, 1, 1, start, 1);
	cblas_dcopy((layers+1)*MAX*MAX, start, 1, here, 1);
	cblas_daxpby((layers+1)*MAX*MAX, alpha, unit, 1, 1, here, 1);
	grad(start);
	accounted=cblas_ddot((layers+1)*MAX*MAX, unit, 1, err, 1);

	double a=truefunc(start);
	double b=truefunc(here);
	while(a-b<-alpha*gamma*accounted)
	{
		alpha=alpha*tau;
		if(alpha<1e-20)
		{
			mkl_free(unit);
			mkl_free(start);
			mkl_free(here);
			return;
		}
		cblas_dcopy((layers+1)*MAX*MAX, start, 1, here, 1);
		cblas_daxpby((layers+1)*MAX*MAX, alpha, unit, 1, 1, here, 1);
		b=truefunc(here);
	}
	cblas_daxpby((layers+1)*MAX*MAX, alpha, unit, 1, 1, X, 1);
	mkl_free(unit);
	mkl_free(start);
	mkl_free(here);
}

double fmincg(double *X, int iter)
{
	double thresh=1e-30;
	double cost=0;
	double *derivative=(double *)mkl_malloc((layers+1)*MAX*MAX*sizeof(double), 128);
	long double alpha=0;
	long double beta=0;
	long double denom=0;
	double pre=1e10, previous=1e10;
	double *direction=(double *)mkl_malloc((layers+1)*MAX*MAX*sizeof(double), 128);
	cblas_dcopy((layers+1)*MAX*MAX, ZERO, 0, direction, 1);
	cblas_dcopy((layers+1)*MAX*MAX, ZERO, 0, X, 1);

	for(int times=0; times<(layers+1)*MAX*MAX&&times<CUTOFF; times++)
	{
		//get gradient
		hessproduct(X); //<-calculates grad() as last line. err is the derivative at X0
		grad();
		cblas_dcopy((layers+1)*MAX*MAX, err, 1, derivative, 1);
		cblas_daxpby((layers+1)*MAX*MAX, 1, Hv, 1, 1, derivative, 1);

		//update direction cancelation factor (initial is 0)
		if(times!=0)
		{
			hessproduct(direction);
			denom=cblas_ddot((layers+1)*MAX*MAX, direction, 1, Hv, 1);
			if(denom>=-thresh&&denom<=thresh)
			{
				mkl_free(derivative);
				mkl_free(direction);
				return cost;
			}
			beta=cblas_ddot((layers+1)*MAX*MAX, derivative, 1, Hv, 1);
			beta=beta/denom;
		}
		//update direction
		cblas_daxpby((layers+1)*MAX*MAX, -1, derivative, 1, beta, direction, 1);

		//line minimize in direction
		linemin(X, direction);

		double *temp=(double *)mkl_malloc((layers+1)*MAX*MAX*sizeof(double), 128);
		cblas_dcopy((layers+1)*MAX*MAX, X, 1, temp, 1);
		cblas_daxpby((layers+1)*MAX*MAX, 1, best, 1, 1, temp, 1);
		cost=func(temp);
		printf("%10d | %10d / %10d ||  %15e %15e\n", iter+1, times+1, (layers+1)*MAX*MAX, cost, pre-cost);
		if((pre-cost)/cost<1e-6&&previous/pre<1e-6)
		{
			mkl_free(derivative);
			mkl_free(direction);
			return cost;
		}
		previous=pre-cost;
		pre=cost;
	}
	mkl_free(derivative);
	mkl_free(direction);
	return cost;
}

void train()
{
	double *change=(double *)mkl_malloc((layers+1)*MAX*MAX*sizeof(double), 128); //change in w
	double pre=1e10;
	double cost=0;
	clock_t t1=clock();
	cblas_dcopy((layers+1)*MAX*MAX, w, 1, best, 1);
	int times;
	for(times=0; times<ITERATIONS; times++)
	{
		cost=fmincg(change, times);
		cblas_daxpby((layers+1)*MAX*MAX, 1, change, 1, 1, best, 1);

		printf("%10d %15e | ETA: %5d seconds||  %15e %15e\n", times+1+timesbefore, 
			((double)((double)(clock()-t1)/CLOCKS_PER_SEC)/((double)times+1)),
			(int)((double)(ITERATIONS-times-1)*((double)((double)(clock()-t1)/CLOCKS_PER_SEC)/((double)times+1))),
			cost, pre-cost);
		pre=cost;
		if((times+1+timesbefore)%5==0)outputcost(cost, times+1+timesbefore);
		if((times+1+timesbefore)%SAVE==0)trainresults(times+timesbefore);
	}
	mkl_free(change);
}

int pull(FILE *previous)
{
	//check validity
	int number;
	if(previous!=NULL)
	{
		fscanf(previous, "%d", &number);
		if(number==layers)
		{
			for(int i=0; i<layers+2; i++)
			{
				fscanf(previous, "%d", &number);
				if(number!=amount[i])return -1;
			}
		}
		else return -1;
	}

	for(int i=0; i<input+output; i++)fscanf(previous, "%lf %lf", &translation[i], &stretch[i]);
	fscanf(previous, "%d", &timesbefore);
	for(int i=0; i<layers+1; i++)
	{
		for(int j=0; j<=amount[i]; j++)
		{
			for(int k=1; k<=amount[i+1]; k++)
			{
				w[i*MAX*MAX+j*MAX+k]=0;
				fscanf(previous, "%lf", &w[i*MAX*MAX+j*MAX+k]);
			}
		}
	}
	return 0;
}

int init(FILE *previous, int const OPTION)
{
	if(OPTION>0)
	{
		fclose(previous);
		//srand((unsigned int)time(NULL));
		for(int i=0; i<layers+1; i++)
		{
			double scale=(double)sqrt((double)6/(double)(amount[i]+amount[i+1]));
			for(int j=0; j<=amount[i]; j++)
			{
				for(int k=1; k<=amount[i+1]; k++)
				{
					w[i*MAX*MAX+j*MAX+k]=0;
					while(w[i*MAX*MAX+j*MAX+k]==0&&j!=0)w[i*MAX*MAX+j*MAX+k]=scale*((double)(rand()%201-100)/100);
				}
			}
		}
	}
	else
	{
		if(pull(previous)==-1)
		{
			fclose(previous);
			return -1;
		}
		fclose(previous);

		if(OPTION==-2)
		{
			printf("ADD NOISE TO WEIGHTS\n");
		}
	}
	cblas_dcopy((layers+1)*MAX*MAX, w, 1, best, 1);

	FILE *testfile=fopen(trainfile, "r");
	double range[input+output*input][2];
	while(true)
	{
		bool done=false;
		for(int i=0; i<input+output*input; i++)
		{
			if(fscanf(testfile, "%lf", &training[i])==-1)done=true;
		}
		if(done==true)break;

		for(int i=0; i<input+output*input&&OPTION>0; i++)
		{
			if(i<input)
			{
				if(totalpoints==0)translation[i]=0;
				translation[i]+=training[i];
			}
			if(totalpoints==0||range[i][0]>training[i])range[i][0]=training[i];
			if(totalpoints==0||range[i][1]<training[i])range[i][1]=training[i];
		}
		totalpoints++;
	}
	for(int i=0; i<input&&OPTION>0; i++)translation[i]=translation[i]/totalpoints;
	for(int i=0; i<input&&OPTION>0; i++)stretch[i]=goodrange[0]/((range[i][1]-range[i][0])/2);
	double yrange=0;
	for(int i=0; i<output&&OPTION>0; i++)
	{
		yrange=0;
		for(int j=0; j<input; j++)
		{
			if(yrange<range[input+i*j][0]||yrange<-range[input+i*j][0])yrange=range[input+i*j][0];
			if(yrange<0)yrange*=-1;
			if(yrange<range[input+i*j][1]||yrange<-range[input+i*j][1])yrange=range[input+i*j][1];
			if(yrange<0)yrange*=-1;
		}
		stretch[i+input]=goodrange[1]/yrange;
	}
	fclose(testfile);

	//get test points
	trainingset=(double *)malloc(((size_t)totalpoints+1)*(input+output*input)*sizeof(double));
	testfile=fopen(trainfile, "r");
	for(int n=0; n<(int)totalpoints; n++)
	{
		bool done=false;
		for(int i=0; i<input+output*input; i++)
		{
			if(fscanf(testfile, "%lf", &training[i])==-1)done=true;
		}
		if(done==true)break;
		memcpy(&trainingset[n*(input+output*input)], training, sizeof(training));
	}
	fclose(testfile);
}

int main()
{
	FILE *setting=fopen("INPUT.txt", "r");
	int OPTION=0;							//-2 : training network with old weights + noise
											//-1 : training network with old weights
											// 0 : output cost function of data with old weights
											// 1 : training network with new weights
	for(int i=0; i<(layers+1)*MAX*MAX; i++)w[i]=0;
	for(int i=0; i<(layers+1)*MAX*MAX; i++)err[i]=0;
	for(int i=0; i<(layers+2)*MAX; i++)nodes[i]=0;
	for(int i=0; i<(layers+2)*MAX; i++)delta[i]=0;
	for(int i=0; i<MAX*(layers+2)*MAX; i++)Gnodes[i]=0;
	for(int i=0; i<MAX*(layers+2)*MAX; i++)Gdelta[i]=0;
	for(int i=0; i<(layers+1)*MAX*MAX; i++)ZERO[i]=0;
	for(int i=0; i<layers+2; i++)
	{
		if(amount[i]+1>MAX)
		{		
			printf("CHECK MAX VALUE!\n");
			return 1;
		}
	}
	if(setting!=NULL)
	{
		fscanf(setting, "%s", trainfile);
		fscanf(setting, "%d", &OPTION);
		fscanf(setting, "%d", &ITERATIONS);
		fscanf(setting, "%d", &SAVE);
		fscanf(setting, "%d", &CUTOFF);
		switch (OPTION)
		{
		case  1: printf("Running the training job with new weights!\n");
			break;
		case -1: printf("Running the training job with old weights!\n");
			break;
		case -2: printf("Running the training job with old weights + noise!\n");
			break;
		default: printf("Running the checking job!\n");
			break;
		}
	}
	else
	{
		printf("######## ERROR ########\n");
		printf("# NO INPUT.txt found! #\n");
		printf("######## ERROR ########\n");
		return 1;
	}
	if(init(setting, OPTION)==-1)
	{
		printf("###### ERROR #####\n");
		printf("# WRONG WEIGHTS! #\n");
		printf("###### ERROR #####\n");
		return 1;
	}

	if(OPTION!=0)train();  //train the network
	else outputcheck();
	return 0;
}