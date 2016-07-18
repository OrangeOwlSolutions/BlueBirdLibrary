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
#include "TimingGPU.cuh"

#include <cuda.h>
#include <cuda_runtime.h>

struct BB::PrivateTimingGPU {
		cudaEvent_t		start;
		cudaEvent_t		stop;
        };

// default constructor
BB::TimingGPU::TimingGPU() { privateTimingGPU = new BB::PrivateTimingGPU; }

// default destructor
BB::TimingGPU::~TimingGPU() { }

void BB::TimingGPU::StartCounter()
{
	cudaEventCreate(&((*privateTimingGPU).start));
	cudaEventCreate(&((*privateTimingGPU).stop));
	cudaEventRecord((*privateTimingGPU).start,0);
}

void BB::TimingGPU::StartCounterFlags()
{
	int eventflags = cudaEventBlockingSync;

	cudaEventCreateWithFlags(&((*privateTimingGPU).start),eventflags);
	cudaEventCreateWithFlags(&((*privateTimingGPU).stop),eventflags);
	cudaEventRecord((*privateTimingGPU).start,0);
}

// Gets the counter in ms
float BB::TimingGPU::GetCounter()
{
	float	time;
	cudaEventRecord((*privateTimingGPU).stop, 0);
	cudaEventSynchronize((*privateTimingGPU).stop);
	cudaEventElapsedTime(&time,(*privateTimingGPU).start,(*privateTimingGPU).stop);
	return time;
}

