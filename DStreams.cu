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


#include "BB.h"
#include "DStreams.cuh"

#include "HExceptions.h"
#include "DExceptions.cuh"					// Needed for CudaSafeCall

#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>

using namespace std;
using namespace BB;

struct BB::PrivateStreams {
	int				whichDevice_;
	int				nStreams_;
	cudaDeviceProp	prop_;
	cudaStream_t*	streams_;
	cudaStream_t	active_stream_;
};

/***********************/
/* DEFAULT CONSTRUCTOR */
/***********************/

DStreams::DStreams() { privateStreams = new BB::PrivateStreams; };

/****************/
/* INIT STREAMS */
/****************/

void DStreams::InitStreams(const int nStreams) {
	CudaSafeCall(cudaGetDevice(&((*privateStreams).whichDevice_)));
	CudaSafeCall(cudaGetDeviceProperties(&((*privateStreams).prop_),(*privateStreams).whichDevice_));
	if (!(*privateStreams).prop_.deviceOverlap) std::cout << "Device will not handle overlaps, so no speed up from streams\n"; 
			
	(*privateStreams).nStreams_ = nStreams;
	// allocate and initialize an array of stream handles
	(*privateStreams).streams_ = (cudaStream_t*) malloc((*privateStreams).nStreams_*sizeof(cudaStream_t));
	for(int i = 0; i < (*privateStreams).nStreams_; i++) CudaSafeCall(cudaStreamCreate(&((*privateStreams).streams_[i]))); 
			
	(*privateStreams).active_stream_ = (*privateStreams).streams_[0]; }

/*******************/
/* DESTROY STREAMS */
/*******************/

void DStreams::DestroyStreams() {
	// release resources
	for(int i = 0; i<(*privateStreams).nStreams_; i++) cudaStreamDestroy((*privateStreams).streams_[i]); 
	free((*privateStreams).streams_); 
			
	(*privateStreams).nStreams_ = 1;
	// allocate and initialize an array of stream handles
	(*privateStreams).streams_ = (cudaStream_t*) malloc((*privateStreams).nStreams_*sizeof(cudaStream_t));
	for(int i = 0; i < (*privateStreams).nStreams_; i++) CudaSafeCall(cudaStreamCreate(&((*privateStreams).streams_[i]))); 
			
	(*privateStreams).active_stream_ = (*privateStreams).streams_[0]; }

/***************/
/* SET STREAMS */
/***************/

void DStreams::SetStream(const int i) {
	if (i>=(*privateStreams).nStreams_) 
	{ char* str0 = "********************************\n"; 
	  char* str1 = "* Invalid stream access number *\n"; 
	  char* str2 = "Total number of available streams: "; 
	  char* str3 = "Stream access number: "; 
	  char* catString = (char*) malloc(2*strlen(str0)+strlen(str1)+strlen(str2)+strlen(str3)+15*sizeof(char)); 
	  sprintf(catString, "%s%s%s\n%s%i\n%s%i\n",str0,str1,str0,str2,(*privateStreams).nStreams_,str3,i); 
	  throw  GenericError(catString,__FILE__,__LINE__); 
	}
	else (*privateStreams).active_stream_ = (*privateStreams).streams_[i]; }

/***************************/
/* SYNCHRONIZE I-TH STREAM */
/***************************/

void DStreams::Synchronize(const int i) { 
	if (i>=(*privateStreams).nStreams_) 
	{ char* str0 = "********************************\n"; 
	  char* str1 = "* Invalid stream access number *\n"; 
	  char* str2 = "Total number of available streams: "; 
	  char* str3 = "Stream access number: "; 
	  char* catString = (char*) malloc(2*strlen(str0)+strlen(str1)+strlen(str2)+strlen(str3)+15*sizeof(char)); 
	  sprintf(catString, "%s%s%s\n%s%i\n%s%i\n",str0,str1,str0,str2,(*privateStreams).nStreams_,str3,i); 
	  throw  GenericError(catString,__FILE__,__LINE__); 
	}
	else cudaStreamSynchronize((*privateStreams).streams_[i]); }

/***************************/
/* SYNCHRONIZE ALL STREAMS */
/***************************/

void DStreams::SynchronizeAll() { for(int i = 0; i<(*privateStreams).nStreams_; i++) cudaStreamSynchronize((*privateStreams).streams_[i]); }

/**********************/
/* DEFAULT DESTRUCTOR */
/**********************/

DStreams::~DStreams() {}

/*************************************************/
/* RETURN THE TOTAL NUMBER OF STREAMS DESTRUCTOR */
/*************************************************/

int DStreams::GetNumStreams() const { return (*privateStreams).nStreams_; }

/*************************************************/
/* RETURN THE TOTAL NUMBER OF STREAMS DESTRUCTOR */
/*************************************************/

cudaStream_t DStreams::GetActiveStream() const { return (*privateStreams).active_stream_; };

