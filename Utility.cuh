/*   Bluebird Library - High performance CPUs and GPUs computing library.
*    
*    Copyright (C) 2012-2013 Orange Owl Solutions.  
*
*    This file is part of Bluebird Library.
*    Bluebird Library is free software: you can redistribute it and/or modify
*    it under the terms of the Lesser GNU General Public License as published by
*    the Free Software Foundation, either version 3 of the License, or
*    (at your option) any later version.
*
*    Bluebird Library is distributed in the hope that it will be useful,
*    but WITHOUT ANY WARRANTY; without even the implied warranty of
*    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
*    Lesser GNU General Public License for more details.
*
*    You should have received a copy of the GNU General Public License
*    along with Bluebird Library.  If not, see <http://www.gnu.org/licenses/>.
*
*
*    For any request, question or bug reporting please visit http://www.orangeowlsolutions.com/
*    or send an e-mail to: info@orangeowlsolutions.com
*
*
*/

#ifndef __UTILITY_CUH__
#define __UTILITY_CUH__

namespace BB
{
	/**************/
	/* ONES CLASS */
	/**************/
	template <class T>	Expr<Scalar<T>,T> SetOnes(int Rows, int Columns)	{ return Expr<Scalar<T>,T>(Scalar<T>(T(1)),Rows,Columns,DEVICEAGNOSTIC); }

	Expr<Scalar<DefaultType>,DefaultType> SetOnes(int Rows, int Columns)	{ return Expr<Scalar<DefaultType>,DefaultType>(Scalar<DefaultType>(DefaultType(1)),Rows,Columns,DEVICEAGNOSTIC); }

	/***************/
	/* ZEROS CLASS */
	/***************/
	template <class T>	Expr<Scalar<T>,T> SetZeros(int Rows, int Columns)	{ return Expr<Scalar<T>,T>(Scalar<T>(T(0)),Rows,Columns,DEVICEAGNOSTIC); }

	Expr<Scalar<DefaultType>,DefaultType> SetZeros(int Rows, int Columns)	{ return Expr<Scalar<DefaultType>,DefaultType>(Scalar<DefaultType>(DefaultType(0)),Rows,Columns,DEVICEAGNOSTIC); }

	/*******************/
	/* GENERATOR CLASS */
	/*******************/
	template <class T>
	class Generator
	{

		private:
			double	start_;
			double	dx_;
			int		Rows_;
			int		Columns_;

		public:
			Generator(const double &start, const double &end, const int N): start_(start), dx_((end-start)/(N-1)), Rows_(1), Columns_(N) {}

			int GetRows()			const { return Rows_; }
			int GetColumns()		const { return Columns_; }
			//int GetNumElements()	const { return a_.GetNumElements(); }

			__host__ __device__ inline T operator[](int i) const { T a; a = start_+i*dx_; return a; }
    
	};

	/************/
	/* EQSPACE */
	/************/
	template <class T>
	Expr<Generator<T>,T> EqSpace(const double start, const double end, const int N)
	{
		return Expr<Generator<T>,T>(Generator<T>(start,end,N),N,1,DEVICEAGNOSTIC);
	};

	Expr<Generator<DefaultType>,DefaultType> EqSpace(const double start, const double end, const int N)
	{
		return Expr<Generator<DefaultType>,DefaultType>(Generator<DefaultType>(start,end,N),N,1,DEVICEAGNOSTIC);
	};

	/*************************/
	/* COLON - THREE INDICES */
	/*************************/
	template <class T>
	Expr<Generator<T>,T> Colon(const double start, const double step, const double end)
	{
		int N = floor((end-start)/step);
		return EqSpace<T>(start,end,N+1);
	};

	Expr<Generator<DefaultType>,DefaultType> Colon(const double start, const double step, const double end)
	{
		int N = floor((end-start)/step);
		return EqSpace<DefaultType>(start,end,N+1);
	};

	/***********************/
	/* COLON - TWO INDICES */
	/***********************/
	template <class T>
	Expr<Generator<T>,T> Colon(const double start, const double end)
	{
		const double step = 1.;
		int N = floor((end-start)/step);
		return EqSpace<T>(start,end,N+1);
	};

	Expr<Generator<DefaultType>,DefaultType> Colon(const double start, const double end)
	{
		const double step = 1.;
		int N = floor((end-start)/step);
		return EqSpace<DefaultType>(start,end,N+1);
	};

	
	//float MinVector(Matrixfloat &vett)
//{
//     int length = vett.GetNumElements();   establish size of array
//     float min = vett[0];        start with max = first element
//
//     for(int i = 1; ilength; i++)
//     {
//          if(vett[i]  min)
//                min = vett[i];
//     }
//     return min;                 return highest value in array
//}

}
	
//// Beginning of GPU Architecture definitions
//inline int _ConvertSMVer2Cores(int major, int minor)
//{
//    // Defines for GPU Architecture types (using the SM version to determine the # of cores per SM
//    typedef struct {
//       int SM; // 0xMm (hexidecimal notation), M = SM Major version, and m = SM minor version
//       int Cores;
//    } sSMtoCores;
//
//    sSMtoCores nGpuArchCoresPerSM[] = 
//    { { 0x10,  8 }, // Tesla Generation (SM 1.0) G80 class
//      { 0x11,  8 }, // Tesla Generation (SM 1.1) G8x class
//      { 0x12,  8 }, // Tesla Generation (SM 1.2) G9x class
//      { 0x13,  8 }, // Tesla Generation (SM 1.3) GT200 class
//      { 0x20, 32 }, // Fermi Generation (SM 2.0) GF100 class
//      { 0x21, 48 }, // Fermi Generation (SM 2.1) GF10x class
//      {   -1, -1 }
//    };
//
//    int index = 0;
//    while (nGpuArchCoresPerSM[index].SM != -1) {
//       if (nGpuArchCoresPerSM[index].SM == ((major << 4) + minor) ) {
//          return nGpuArchCoresPerSM[index].Cores;
//       }	
//       index++;
//    }
//    printf("MapSMtoCores undefined SM %d.%d is undefined (please update to the latest SDK)!\n", major, minor);
//    return -1;
//}

/***********************************************************/
/* THIS FUNCTION RETURNS THE BEST GPU (WITH MAXIMUM GLOPS) */
/***********************************************************/
//int gpuGetMaxGflopsDeviceId()
//{
//    int current_device     = 0, sm_per_multiproc  = 0;
//    int max_compute_perf   = 0, max_perf_device   = 0;
//    int device_count       = 0, best_SM_arch      = 0;
//    cudaDeviceProp deviceProp;
//    cudaGetDeviceCount( &device_count );
//    
//    // Find the best major SM Architecture GPU device
//    while (current_device < device_count)
//    {
//        cudaGetDeviceProperties( &deviceProp, current_device );
//        if (deviceProp.major > 0 && deviceProp.major < 9999)
//        {
//            best_SM_arch = MAX(best_SM_arch, deviceProp.major);
//        }
//        current_device++;
//    }
//
//    // Find the best CUDA capable GPU device
//    current_device = 0;
//    while( current_device < device_count )
//    {
//        cudaGetDeviceProperties( &deviceProp, current_device );
//        if (deviceProp.major == 9999 && deviceProp.minor == 9999)
//        {
//            sm_per_multiproc = 1;
//        }
//        else
//        {
//            sm_per_multiproc = _ConvertSMVer2Cores(deviceProp.major, deviceProp.minor);
//        }
//        
//        int compute_perf  = deviceProp.multiProcessorCount * sm_per_multiproc * deviceProp.clockRate;
//        
//    if( compute_perf  > max_compute_perf )
//    {
//            // If we find GPU with SM major > 2, search only these
//            if ( best_SM_arch > 2 )
//            {
//                // If our device==dest_SM_arch, choose this, or else pass
//                if (deviceProp.major == best_SM_arch)
//                {
//                    max_compute_perf  = compute_perf;
//                    max_perf_device   = current_device;
//                 }
//            }
//            else
//            {
//                max_compute_perf  = compute_perf;
//                max_perf_device   = current_device;
//             }
//        }
//        ++current_device;
//    }
//    return max_perf_device;
//}

/******************************************/
/* GENERAL GPU DEVICE CUDA INITIALIZATION */
/******************************************/
//int gpuDeviceInit(int devID)
//{
//    int deviceCount;
//    CudaSafeCall(cudaGetDeviceCount(&deviceCount));
//
//    if (deviceCount == 0)
//    {
//        fprintf(stderr, "gpuDeviceInit() CUDA error: no devices supporting CUDA.\n");
//        exit(-1);
//    }
//
//    if (devID < 0)
//       devID = 0;
//        
//    if (devID > deviceCount-1)
//    {
//        fprintf(stderr, "\n");
//        fprintf(stderr, ">> %d CUDA capable GPU device(s) detected. <<\n", deviceCount);
//        fprintf(stderr, ">> gpuDeviceInit (-device=%d) is not a valid GPU device. <<\n", devID);
//        fprintf(stderr, "\n");
//        return -devID;
//    }
//
//    cudaDeviceProp deviceProp;
//    CudaSafeCall(cudaGetDeviceProperties(&deviceProp, devID));
//
//    if (deviceProp.major < 1)
//    {
//        fprintf(stderr, "gpuDeviceInit(): GPU device does not support CUDA.\n");
//        exit(-1);                                                  
//    }
//    
//    CudaSafeCall( cudaSetDevice(devID) );
//    printf("gpuDeviceInit() CUDA Device [%d]: \"%s\n", devID, deviceProp.name);
//
//    return devID;
//}

/****************************************************/
/* INITIALIZATION CODE TO FIND THE BEST CUDA DEVICE */
/****************************************************/
//int findCudaDevice(void)
//{
//    cudaDeviceProp deviceProp;
//    int devID = 0;
//	devID = gpuDeviceInit(devID);
//    if (devID < 0) {
//		printf("Error. No CUDA enabled GPU.\n ");
//		getch();
//		exit(-1); }
//    else
//    {
//        // Otherwise pick the device with highest Gflops/s
//        devID = gpuGetMaxGflopsDeviceId();
//        CudaSafeCall(cudaSetDevice(devID));
//        CudaSafeCall(cudaGetDeviceProperties(&deviceProp,devID));
//        printf("GPU Device %d: \"%s\" with compute capability %d.%d\n\n", devID, deviceProp.name, deviceProp.major, deviceProp.minor);
//    }
//    return devID;
//}

#endif
