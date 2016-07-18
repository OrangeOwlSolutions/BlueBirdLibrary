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


#ifndef __CUDAMATRIXEXPRESSION_CUH__
#define __CUDAMATRIXEXPRESSION_CUH__

namespace BB
{

			//template <class A, class T, class OutType>
			//const Dmatrix<OutType>::Dmatrix<OutType>& Dmatrix<OutType>::operator=(Expr<A,T>::Expr<A,T> e)
			//{   
			//	if((e.GetRows() == Rows_) && (e.GetColumns() == Columns_)) evaluation_matrix_function_expr(data_,e,GetNumElements());
			//	else 
			//	{  
			//		char* str0 = "**********************************************\n";
			//		char* str1 = "* Size mismatch in GPU=GPU matrix assignment *\n";
			//		char* str2 = "Left operand size: ";
			//		char* str3 = "Right operand size: ";
			//		char* catString = (char*) malloc(2*strlen(str0)+strlen(str1)+strlen(str1)+strlen(str2)+strlen(str3)+40*sizeof(char));
			//		sprintf(catString, "%s%s%s\n%s%i x %i\n%s%i x %i",str0,str1,str0,str2,Rows_,Columns_,str3,e.GetRows(),e.GetColumns());
			//		throw  GenericError(catString,__FILE__,__LINE__);
			//	} 
			//	return *this;
			//};




		//template<class A, class B, class OutType>
	//Dmatrix<OutType>::Dmatrix(const Expr<A,B>&e) {
	//	Rows_ = e.GetRows();
	//	Columns_ = e.GetColumns();
	//	if (e.IsDevice()) {
	//		CudaSafeCall(cudaMalloc((void **)&data_, Rows_*Columns_*sizeof(OutType)));
	//		evaluation_matrix_function(data_,e,GetNumElements()); }
	//	else {
	//		Hmatrix<OutType> temp(e);
	//		CudaSafeCall(cudaMalloc((void **)&data_, Rows_*Columns_*sizeof(OutType)));
	//		CudaSafeCall(cudaMemcpy(data_,temp.GetDataPointer(),GetNumElements()*sizeof(OutType),cudaMemcpyHostToDevice));
	//	}	
	//	//else {char* str0 = "*******************************************************\n"; 
	//	//	  char* str1 = "* Cannot construct a GPU matrix from a CPU expression *\n"; 
	//	//	  char* catString = (char*) malloc(2*strlen(str0)+strlen(str1)); 
	//	//	  sprintf(catString, "%s%s%s\n",str0,str1,str0); 
	//	//	  throw  GenericError(catString,__FILE__,__LINE__); }	
	//}

} // namespace

#endif