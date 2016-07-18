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


#ifndef __CUDAFINITEDIFFERENCE_CUH__
#define __CUDAFINITEDIFFERENCE_CUH__

namespace BB
{

	/*******************/
	/* CUDA DIFF CLASS */
	/*******************/
	// A		- Type of input
	// Type		- Type of output
	template <class A, class Type>
	class  CudaDiff
	{
		private:
			A M_;

		public:
			CudaDiff(A M) : M_(M) { }

			__host__ __device__ inline const	Type& operator[](const int i) const { return M_[i+1]-M_[i]; }
	};

	// Diff on Hmatrix
	template <class T> 
	Expr<CudaDiff<T*,T>,T> Diff(const Hmatrix<T> &v) 
	{ 
		if(v.IsAvector() && ((v.GetRows()>1) || (v.GetColumns()>1)))
		{ 
			int RowDiff, ColumnDiff;
			if (v.GetRows()>1) { RowDiff=v.GetRows()-1; ColumnDiff=1; } else { RowDiff=1; ColumnDiff=v.GetColumns()-1; }
			typedef CudaDiff<T*,T> DiffExpr; 
			return Expr<CudaDiff<T*,T>,T>(DiffExpr(v.data_),RowDiff,ColumnDiff,ISHOST); 
		} else { char* str0 = "************************************************\n"; 
				 char* str1 = "* 1D Finite Difference operator not applicable *\n"; 
			char* str2 = "Operand size: "; 
			char* catString = (char*) malloc(2*strlen(str0)+strlen(str1)+strlen(str2)+50*sizeof(char)); 
			sprintf(catString, "%s%s%s\n%s%i x %i\n",str0,str1,str0,str2,v.GetRows(),v.GetColumns()); 
			throw  GenericError(catString,__FILE__,__LINE__); 
		} 
	}

	// Diff on Dmatrix
	template <class T> 
	Expr<CudaDiff<T*,T>,T> Diff(const Dmatrix<T> &v) 
	{ 
		if(v.IsAvector() && ((v.GetRows()>1) || (v.GetColumns()>1)))
		{ 
			int RowDiff, ColumnDiff;
			if (v.GetRows()>1) { RowDiff=v.GetRows()-1; ColumnDiff=1; } else { RowDiff=1; ColumnDiff=v.GetColumns()-1; }
			typedef CudaDiff<T*,T> DiffExpr; 
			return Expr<CudaDiff<T*,T>,T>(DiffExpr(v.data_),RowDiff,ColumnDiff,ISHOST); 
		} else { char* str0 = "************************************************\n"; 
				 char* str1 = "* 1D Finite Difference operator not applicable *\n"; 
			char* str2 = "Operand size: "; 
			char* catString = (char*) malloc(2*strlen(str0)+strlen(str1)+strlen(str2)+50*sizeof(char)); 
			sprintf(catString, "%s%s%s\n%s%i x %i\n",str0,str1,str0,str2,v.GetRows(),v.GetColumns()); 
			throw  GenericError(catString,__FILE__,__LINE__); 
		} 
	}

} // namespace

#endif
