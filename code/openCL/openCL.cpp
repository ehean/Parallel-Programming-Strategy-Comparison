#include <stdio.h>
#include <math.h>
#include <string.h>
#include <stdlib.h>
#ifdef WIN32
#include <windows.h>
#else
#include <unistd.h>
#endif
#include <omp.h>
#include <fstream>
#include <iostream>

#include "cl.h"
#include "cl_platform.h"

#define SIZE			32768
#define NUMTRIES       	10
#define LOCAL_SIZE		256
#define NUM_WORK_GROUPS SIZE / LOCAL_SIZE

double sumMegaCalcs, maxMegaCalcs;

const char *			CL_FILE_NAME = { "autoCorrelate.cl" };
const float			TOL = 0.0001f;

void				Wait(cl_command_queue);
int				LookAtTheBits(float);

int main()
{

	std::ofstream sums;
	sums.open("openCL_autocorrelation_sums.csv");
	size_t dataSize = SIZE * sizeof(float);

	for (int test = 0; test < NUMTRIES; ++test) {

		FILE *fp = fopen("signals.txt", "r");
		if (fp == NULL)
		{
			fprintf(stderr, "Cannot open file 'signals.txt'\n");
			exit(1);
		}
		int Size;
		fscanf(fp, "%d", &Size);
		Size = SIZE;


		FILE *clFile;
#ifdef WIN32
		errno_t err = fopen_s(&clFile, CL_FILE_NAME, "r");
		if (err != 0)
#else
		clFile = fopen(CL_FILE_NAME, "r");
		if (fp == NULL)
#endif
		{
			fprintf(stderr, "Cannot open OpenCL source file '%s'\n", CL_FILE_NAME);
			return 1;
		}

		cl_int status;		// returned status from opencl calls
							// test against CL_SUCCESS

							// get the platform id:

		cl_platform_id platform;
		status = clGetPlatformIDs(1, &platform, NULL);
		if (status != CL_SUCCESS)
			fprintf(stderr, "clGetPlatformIDs failed (2)\n");

		// get the device id:

		cl_device_id device;
		status = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);
		if (status != CL_SUCCESS)
			fprintf(stderr, "clGetDeviceIDs failed (2)\n");

		std::cout << " 2. allocate the host memory buffers\n";

		float *hArray = new float[2 * Size];
		float *hSums = new float[1 * Size];

		// fill the host memory buffers:


		for (int i = 0; i < Size; i++)
		{
			fscanf(fp, "%f", &hArray[i]);
			hArray[i + Size] = hArray[i];		// duplicate the array
		}
		std::cout << "allocated hArray\n";

		std::cout << "3. create an opencl context\n";

		cl_context context = clCreateContext(NULL, 1, &device, NULL, NULL, &status);
		if (status != CL_SUCCESS)
			fprintf(stderr, "clCreateContext failed\n");

		std::cout <<  "4. create an opencl command queue\n";

		cl_command_queue cmdQueue = clCreateCommandQueue(context, device, 0, &status);
		if (status != CL_SUCCESS)
			fprintf(stderr, "clCreateCommandQueue failed\n");

		std::cout << "5. allocate the device memory buffers\n";

		cl_mem dArray = clCreateBuffer(context, CL_MEM_READ_ONLY, 2 * Size * sizeof(cl_float), NULL, &status);
		if (status != CL_SUCCESS)
			fprintf(stderr, "clCreateBuffer failed (0)\n");

		cl_mem dSums = clCreateBuffer(context, CL_MEM_WRITE_ONLY, 1 * Size * sizeof(cl_float), NULL, &status);
		if (status != CL_SUCCESS)
			fprintf(stderr, "clCreateBuffer failed (1)\n");

		std::cout << "6. enqueue the 2 commands to write the data from the host buffers to the device buffers\n";

		status = clEnqueueWriteBuffer(cmdQueue, dArray, CL_FALSE, 0, dataSize * 2, hArray, 0, NULL, NULL);
		if (status != CL_SUCCESS)
			fprintf(stderr, "clEnqueueWriteBuffer failed (1)\n");

		status = clEnqueueWriteBuffer(cmdQueue, dSums, CL_FALSE, 0, dataSize, hSums, 0, NULL, NULL);
		if (status != CL_SUCCESS)
			fprintf(stderr, "clEnqueueWriteBuffer failed (2)\n");

		Wait(cmdQueue);

		std::cout << "7. read the kernel code from a file\n";

		fseek(clFile, 0, SEEK_END);
		size_t fileSize = ftell(clFile);
		fseek(clFile, 0, SEEK_SET);
		char *clProgramText = new char[fileSize + 1];		// leave room for '\0'
		size_t n = fread(clProgramText, 1, fileSize, clFile);
		clProgramText[fileSize] = '\0';
		fclose(clFile);
		if (n != fileSize)
			fprintf(stderr, "Expected to read %d bytes read from '%s' -- actually read %d.\n", fileSize, CL_FILE_NAME, n);

		// create the text for the kernel program:

		char *strings[1];
		strings[0] = clProgramText;
		cl_program program = clCreateProgramWithSource(context, 1, (const char **)strings, NULL, &status);
		if (status != CL_SUCCESS)
			fprintf(stderr, "clCreateProgramWithSource failed\n");
		delete[] clProgramText;

		std::cout << "8. compile and link the kernel code\n";

		char *options = { "" };
		status = clBuildProgram(program, 1, &device, options, NULL, NULL);
		if (status != CL_SUCCESS)
		{
			size_t size;
			clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, NULL, &size);
			cl_char *log = new cl_char[size];
			clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, size, log, NULL);
			fprintf(stderr, "clBuildProgram failed:\n%s\n", log);
			delete[] log;
		}

		std::cout<< "9. create the kernel object\n";

		cl_kernel kernel = clCreateKernel(program, "AutoCorrelate", &status);
		if (status != CL_SUCCESS)
			fprintf(stderr, "clCreateKernel failed\n");

		std::cout << "10. setup the arguments to the kernel object\n";

		status = clSetKernelArg(kernel, 0, sizeof(cl_mem), &dArray);
		if (status != CL_SUCCESS)
			fprintf(stderr, "clSetKernelArg failed (1)\n");

		status = clSetKernelArg(kernel, 1, sizeof(cl_mem), &dSums);
		if (status != CL_SUCCESS)
			fprintf(stderr, "clSetKernelArg failed (2)\n");


		std::cout << "11. enqueue the kernel object for execution\n";

		size_t globalWorkSize[3] = { Size, 1, 1 };
		size_t localWorkSize[3] = { LOCAL_SIZE,   1, 1 };

		Wait(cmdQueue);

		double time0 = omp_get_wtime();

		time0 = omp_get_wtime();

		status = clEnqueueNDRangeKernel(cmdQueue, kernel, 1, NULL, globalWorkSize, localWorkSize, 0, NULL, NULL);
		if (status != CL_SUCCESS)
			fprintf(stderr, "clEnqueueNDRangeKernel failed: %d\n", status);

		Wait(cmdQueue);

		double time1 = omp_get_wtime();

		std::cout << "12. read the results buffer back from the device to the host\n";

		status = clEnqueueReadBuffer(cmdQueue, dSums, CL_TRUE, 0, dataSize, hSums, 0, NULL, NULL);
		if (status != CL_SUCCESS)
			fprintf(stderr, "clEnqueueReadBuffer failed\n");

		Wait(cmdQueue);

		// did it work?

		fprintf(stderr, "%8d\t%4d\t%10d\t%10.3lf GigaMultsPerSecond\n",
			SIZE, LOCAL_SIZE, NUM_WORK_GROUPS, SIZE / (time1 - time0) / 1000000000.);


#ifdef WIN32
		Sleep(2000);
#endif


		std::cout << "13. clean everything up\n";

	
		for (int i = 0; i < SIZE; ++i) {
			sums << i << ", " << hSums[i] << '\n';
		}

		std::cout << "release kernel\n";
		clReleaseKernel(kernel);

		std::cout << "release program\n";
		clReleaseProgram(program);

		std::cout << "release command queue\n";
		clReleaseCommandQueue(cmdQueue);

		std::cout << "release dArray\n";
		clReleaseMemObject(dArray);

		std::cout << "release dSums\n";
		clReleaseMemObject(dSums);

		std::cout << "delete hArray\n";
		delete[] hArray;

		std::cout << "delete hSums\n";
		delete[] hSums;

		//std::cout << "close cl file\n";
		//fclose(clFile);

		std::cout << "calculate average calculations\n";
		double avgMegaCalcs = sumMegaCalcs / (double)NUMTRIES;
		printf("   Peak Performance = %8.2lf MegaCalcs/Sec\n", maxMegaCalcs);
		printf("Average Performance = %8.2lf MegaCalcs/Sec\n", avgMegaCalcs);

		return 0;
	}
}

void
Wait(cl_command_queue queue)
{
	cl_event wait;

	cl_int status = clEnqueueMarker(queue, &wait);
	if (status != CL_SUCCESS)
		fprintf(stderr, "Wait: clEnqueueMarker failed\n");

	status = clEnqueueWaitForEvents(queue, 1, &wait);
	if (status != CL_SUCCESS)
		fprintf(stderr, "Wait: clEnqueueWaitForEvents failed\n");
}


int
LookAtTheBits(float clFile)
{
	int *ip = (int *)&clFile;
	return *ip;
}
