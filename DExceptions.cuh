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

#ifndef __DEXCEPTIONS_CUH__
#define __DEXCEPTIONS_CUH__

#include "cufft.h"
#include <cublas_v2.h>
#include <conio.h>		// Needed for getch()
#include <assert.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Define this to turn on error checking
#define CUDA_ERROR_CHECK

/*! \brief Catch CUDA runtime API errors.
 *
 *  Example:
 *
 *      CudaSafeCall(cudaMalloc((void **)&pointer,size*sizeof(Type)));
 *
*/
#ifndef CudaSafeCall
#define CudaSafeCall(err)		__cudaSafeCall(err, __FILE__, __LINE__)
#endif

/*! \brief Catch CUDA kernel runtime errors.
 *
 *  Example:
 *
 *	global_function<<<dimGrid,dimBlock>>>(...input_parameters...;
 *  CudaCheckError();
 *
*/
#ifndef CudaCheckError
#define CudaCheckError()		__cudaCheckError( __FILE__, __LINE__)
#endif

/*! \brief Catch CUDA CUFFT runtime errors.
 *
 *  Example:
 *
 *	NO - FIX!!!! global_function<<<dimGrid,dimBlock>>>(...input_parameters...;
 *  CudaCheckError();
 *
*/
#ifndef cufftSafeCall
#define cufftSafeCall(err)      __cufftSafeCall(err, __FILE__, __LINE__)
#endif

#ifndef cublasSafeCall
#define cublasSafeCall(err)     __cublasSafeCall(err, __FILE__, __LINE__)
#endif

// --- CUDASAFECALL
inline void __cudaSafeCall( cudaError err, const char *file, const int line )
{
#ifdef CUDA_ERROR_CHECK
	if ( cudaSuccess != err )
    {
        fprintf( stderr, "cudaSafeCall() failed at %s:%i : %s\n", file, line, cudaGetErrorString( err ) );
        getch(); exit( -1 );
    }
#endif
 
    return;
}

// --- CUDACHECKERROR
inline void __cudaCheckError( const char *file, const int line )
{
#ifdef CUDA_ERROR_CHECK
    cudaError err = cudaGetLastError();
    if ( cudaSuccess != err )
    {
        fprintf( stderr, "cudaCheckError() failed at %s:%i : %s\n",
                 file, line, cudaGetErrorString( err ) );
        getch(); exit(-1);
    }
 
    // More careful checking. However, this will affect performance.
    // Comment away if needed.
    err = cudaDeviceSynchronize();
    if( cudaSuccess != err )
    {
        fprintf( stderr, "cudaCheckError() with sync failed at %s:%i : %s\n",
                 file, line, cudaGetErrorString( err ) );
        getch(); exit( -1 );
    }
#endif
 
    return;
}

static const char *_cudaGetErrorEnum(cufftResult error)
{
    switch (error)
    {
        case CUFFT_SUCCESS:
            return "CUFFT_SUCCESS";

        case CUFFT_INVALID_PLAN:
            return "CUFFT_INVALID_PLAN";

        case CUFFT_ALLOC_FAILED:
            return "CUFFT_ALLOC_FAILED";

        case CUFFT_INVALID_TYPE:
            return "CUFFT_INVALID_TYPE";

        case CUFFT_INVALID_VALUE:
            return "CUFFT_INVALID_VALUE";

        case CUFFT_INTERNAL_ERROR:
            return "CUFFT_INTERNAL_ERROR";

        case CUFFT_EXEC_FAILED:
            return "CUFFT_EXEC_FAILED";

        case CUFFT_SETUP_FAILED:
            return "CUFFT_SETUP_FAILED";

        case CUFFT_INVALID_SIZE:
            return "CUFFT_INVALID_SIZE";

        case CUFFT_UNALIGNED_DATA:
            return "CUFFT_UNALIGNED_DATA";
    }

    return "<unknown>";
}

// --- CUFFTSAFECALL
inline void __cufftSafeCall(cufftResult err, const char *file, const int line)
{
    if( CUFFT_SUCCESS != err) {
		fprintf(stderr, "CUFFT error in file '%s', line %d\n \nerror %d: %s\nterminating!\n",__FILE__, __LINE__,err, _cudaGetErrorEnum(err)); 
		getch(); cudaDeviceReset(); assert(0); 
        //fprintf(stderr, "cufftSafeCall() CUFFT error in file <%s>, line %i.\n",
        //        file, line);
        //getch(); exit(-1);
    }
}

// --- CUBLASSAFECALL
inline void __cublasSafeCall(cublasStatus_t err, const char *file, const int line)
{
    if( CUBLAS_STATUS_SUCCESS != err) {
		fprintf(stderr, "CUBLAS error in file '%s', line %d\n \nerror %d \nterminating!\n",__FILE__, __LINE__,err); 
		getch(); cudaDeviceReset(); assert(0); 
        //fprintf(stderr, "cufftSafeCall() CUFFT error in file <%s>, line %i.\n",
        //        file, line);
        //getch(); exit(-1);
    }
}

// Round a / b to nearest higher integer value
//int iDivUp(const int a, const int b) { return (a % b != 0) ? (a / b + 1) : (a / b); }



#endif