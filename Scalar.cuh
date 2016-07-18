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


#ifndef __SCALAR_CUH__
#define __SCALAR_CUH__

namespace BB
{
	template <class Type>
	class Scalar
	{
		private:	
			Type	c_;
	
		public:
			//__host__ __device__ Scalar(const Type &);
			//__host__ __device__ __forceinline__ Scalar(const Type &c) { printf("Scalar constructor\n"); c_ = c; };
			__host__ __device__ __forceinline __forceinline__ Scalar(const Type &c) { c_ = c; };
			//__forceinline Scalar(const Type &c) { c_ = c; };

			//__host__ __device__ Type operator[](const int) const;
			//__host__ __device__ __forceinline__ Type operator[](const int i) const { printf("Scalar []\n"); return c_; }
			__host__ __device__ __forceinline __forceinline__ Type operator[](const int i) const { return c_; }
			//__forceinline Type operator[](const int i) const { return c_; }
	};
	
	//__host__ __device__ __forceinline__ Scalar<BB::double2_>::Scalar(const BB::double2_ &c) { printf("Scalar constructor\n"); c_.x = c.x; c_.y = c.y; };
	__host__ __device__ __forceinline __forceinline__ Scalar<BB::double2_>::Scalar(const BB::double2_ &c) { c_.c.x = c.c.x; c_.c.y = c.c.y; };

}

#endif 