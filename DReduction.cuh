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


#ifndef __DREDUCTION_CUH__
#define __DREDUCTION_CUH__

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/reduce.h>
#include <thrust/functional.h>

namespace BB
{
	/***************************/
	/* REDUCTION STRUCTS FOR + */
	/***************************/

	// int2 + struct
	struct add_int2 {
		__host__ __device__ int2 operator()(const int2& a, const int2& b) const {
			int2 r;
			r.x = a.x + b.x;
			r.y = a.y + b.y;
			return r;
		}
	};

	// float2 + struct
	struct add_float2 {
		__host__ __device__ float2 operator()(const float2& a, const float2& b) const {
			float2 r;
			r.x = a.x + b.x;
			r.y = a.y + b.y;
			return r;
		}
	};

	// customized double2
	struct my_double2 {
		double x, y;
	};

	// double2 + struct
	struct add_my_double2 {
		__host__ __device__ my_double2 operator()(const my_double2& a, const my_double2& b) const {
			my_double2 r;
			r.x = a.x + b.x;
			r.y = a.y + b.y;
			return r;
		}
	};

	/*****************************/
	/* REDUCTION FUNCTIONS FOR + */
	/*****************************/

	// int - Dmatrix
	int SumAll(const Dmatrix<int> &in)
	{
		thrust::device_ptr<int> dev_ptr_1((int*)in.GetDataPointer());
		thrust::device_ptr<int> dev_ptr_2((int*)in.GetDataPointer()+in.GetNumElements());
 
		int out = thrust::reduce(dev_ptr_1,dev_ptr_2,0,thrust::plus<int>());

		return out;
	}

	// float - Dmatrix
	float SumAll(const Dmatrix<float> &in)
	{
		thrust::device_ptr<float> dev_ptr_1((float*)in.GetDataPointer());
		thrust::device_ptr<float> dev_ptr_2((float*)in.GetDataPointer()+in.GetNumElements());
 
		float out = thrust::reduce(dev_ptr_1,dev_ptr_2,0,thrust::plus<float>());

		return out;
	}

	// double - Dmatrix
	double SumAll(const Dmatrix<double> &in)
	{
		thrust::device_ptr<double> dev_ptr_1((double*)in.GetDataPointer());
		thrust::device_ptr<double> dev_ptr_2((double*)in.GetDataPointer()+in.GetNumElements());
 
		double out = thrust::reduce(dev_ptr_1,dev_ptr_2,0.,thrust::plus<double>());

		return out;
	}

	// int2_ - Dmatrix
	int2_ SumAll(const Dmatrix<int2_> &in)
	{
		int2_ out;

		thrust::device_ptr<int2> dev_ptr_1((int2*)in.GetDataPointer());
		thrust::device_ptr<int2> dev_ptr_2((int2*)in.GetDataPointer()+in.GetNumElements());
 
		int2 init; init.x = init.y = 0;

		int2 sum = thrust::reduce(dev_ptr_1,dev_ptr_2,init,add_int2());

		out.c.x = sum.x; out.c.y = sum.y;

		return out;
	}
	
	// float2_ - Dmatrix
	float2_ SumAll(const Dmatrix<float2_> &in)
	{
		float2_ out;

		thrust::device_ptr<float2> dev_ptr_1((float2*)in.GetDataPointer());
		thrust::device_ptr<float2> dev_ptr_2((float2*)in.GetDataPointer()+in.GetNumElements());
 
		float2 init; init.x = init.y = 0.f;

		float2 sum = thrust::reduce(dev_ptr_1,dev_ptr_2,init,add_float2());

		out.c.x = sum.x; out.c.y = sum.y;

		return out;
	}

	// double2_ - Dmatrix
	double2_ SumAll(const Dmatrix<double2_> &in)
	{
		double2_ out;

		thrust::device_ptr<my_double2> dev_ptr_1((my_double2*)in.GetDataPointer());
		thrust::device_ptr<my_double2> dev_ptr_2((my_double2*)in.GetDataPointer()+in.GetNumElements());
 
		my_double2 init; init.x = init.y = 0.0;

		my_double2 sum = thrust::reduce(dev_ptr_1,dev_ptr_2,init,add_my_double2());

		out.c.x = sum.x; out.c.y = sum.y;

		return out;
	}

}

#endif
