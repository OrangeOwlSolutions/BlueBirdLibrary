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


#ifndef __HREDUCTION_H__
#define __HREDUCTION_H__

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/reduce.h>
#include <thrust/functional.h>

namespace BB
{
	/*****************************/
	/* REDUCTION FUNCTIONS FOR + */
	/*****************************/

	// int - Hmatrix
	int SumAll(const Hmatrix<int> &in)
	{
		int out = thrust::reduce((int*)in.GetDataPointer(),(int*)in.GetDataPointer()+in.GetNumElements(),0,thrust::plus<int>());

		return out;
	}

	// float - Hmatrix
	float SumAll(const Hmatrix<float> &in)
	{
		float out = thrust::reduce((float*)in.GetDataPointer(),(float*)in.GetDataPointer()+in.GetNumElements(),0.f,thrust::plus<float>());

		return out;
	}

	// double - Hmatrix
	double SumAll(const Hmatrix<double> &in)
	{
		double out = thrust::reduce((double*)in.GetDataPointer(),(double*)in.GetDataPointer()+in.GetNumElements(),0.,thrust::plus<double>());

		return out;
	}

	// int2_ - Hmatrix
	int2_ SumAll(const Hmatrix<int2_> &in)
	{
		int2_ out;

		int2 init; init.x = init.y = 0;

		int2 sum = thrust::reduce((int2*)in.GetDataPointer(),(int2*)in.GetDataPointer()+in.GetNumElements(),init,add_int2());

		out.c.x = sum.x; out.c.y = sum.y;

		return out;
	}
	
	// float2_ - Hmatrix
	float2_ SumAll(const Hmatrix<float2_> &in)
	{
		float2_ out;

		float2 init; init.x = init.y = 0.f;

		float2 sum = thrust::reduce((float2*)in.GetDataPointer(),(float2*)in.GetDataPointer()+in.GetNumElements(),init,add_float2());

		out.c.x = sum.x; out.c.y = sum.y;

		return out;
	}

	// double2_ - Hmatrix
	double2_ SumAll(const Hmatrix<double2_> &in)
	{
		double2_ out;

		my_double2 init; init.x = init.y = 0.0;

		my_double2 sum = thrust::reduce((my_double2*)in.GetDataPointer(),(my_double2*)in.GetDataPointer()+in.GetNumElements(),init,add_my_double2());

		out.c.x = sum.x; out.c.y = sum.y;

		return out;
	}

	// Expression
	template <class Q, class T>
	double SumAll(const Expr<Q,T> &e)
	{
		T out;

		if (e.IsDevice()) {
			Dmatrix<T> in(e);
			out = SumAll(in); }
		else {
			Hmatrix<T> in(e);
			out = SumAll(in); }
	
		return out;
	}

}

#endif
