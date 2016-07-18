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

#ifndef __GRID_CUH__
#define __GRID_CUH__

#include <typeinfo>
#include <tuple>
//#include <utility>

namespace BB
{

	/************************************/
	/* CUDA MESHGRID-X EXPRESSION CLASS */
	/************************************/
	// A		- Expression to meshgrid along x
	// OutType	- Type of expression
	template <typename A, typename OutType>
	class GridXExpr
	{
		private:
			A a_;
			int Rows_;
			int Columns_;

		public:
			GridXExpr(const A& a, const int c, const int d): a_(a), Rows_(c), Columns_(d) { }
 
			inline	int GetRows()			const { return Rows_; }
			inline	int GetColumns()		const { return Columns_; }
			//inline int GetNumElements() const { return GetRows()*GetColumns(); }
			//bool IsAvector()		const { return (GetRows() == 1 || GetColumns() == 1)?true:false;}
      
			__host__ __device__ __forceinline __forceinline__	OutType operator[](int i)	const { return a_[i%Columns_]; }
	};

	// A		- Expression to meshgrid along x
	// OutType	- Type of expression
	template <typename A, typename OutType>
	class GridYExpr
	{
		private:
			A a_;
			int Rows_;
			int Columns_;

		public:
			GridYExpr(const A& a, const int c, const int d): a_(a), Rows_(c), Columns_(d) { }
 
			inline	int GetRows()								const { return Rows_; }
			inline	int GetColumns()							const { return Columns_; }
			//inline int GetNumElements() const { return GetRows()*GetColumns(); }
			__host__ __device__ __forceinline __forceinline__	bool IsAvector()		const { return (GetRows() == 1 || GetColumns() == 1)?true:false;}
      
			__host__ __device__ __forceinline __forceinline__	OutType operator[](int i)	const { return a_[i/Columns_]; }
	};

	MeshGridX_Matrix_Matrix_Matrix(T1,T2,GridXExpr,GridX);
	MeshGridX_Matrix_Matrix_CudaMatrix(T1,T2,GridXExpr,GridX);
	MeshGridX_Expression_Matrix_Matrix(T1,Q1,T2,GridXExpr,GridX);
	MeshGridX_Expression_Matrix_CudaMatrix(T1,Q1,T2,GridXExpr,GridX);
	MeshGridX_Matrix_Expression_Matrix(T1,T2,Q2,GridXExpr,GridX);
	MeshGridX_Matrix_Expression_CudaMatrix(T1,T2,Q2,GridXExpr,GridX);
	MeshGridX_Expression_Expression(T1,Q1,T2,Q2,GridXExpr,GridX);

	MeshGridY_Matrix_Matrix_Matrix(T1,T2,GridYExpr,GridY);
	MeshGridY_Matrix_Matrix_CudaMatrix(T1,T2,GridYExpr,GridY);
	MeshGridY_Expression_Matrix_Matrix(T1,Q1,T2,GridYExpr,GridY);
	MeshGridY_Expression_Matrix_CudaMatrix(T1,Q1,T2,GridYExpr,GridY);
	MeshGridY_Matrix_Expression_Matrix(T1,T2,Q2,GridYExpr,GridY);
	MeshGridY_Matrix_Expression_CudaMatrix(T1,T2,Q2,GridYExpr,GridY);
	MeshGridY_Expression_Expression(T1,Q1,T2,Q2,GridYExpr,GridY);

	MeshGrid_Matrix_Matrix_Matrix(T1,T2,GridXExpr,GridYExpr);
	MeshGrid_Matrix_Matrix_CudaMatrix(T1,T2,GridXExpr,GridYExpr);
	MeshGrid_Expression_Matrix_Matrix(T1,Q1,T2,GridXExpr,GridYExpr);
	MeshGrid_Expression_Matrix_CudaMatrix(T1,Q1,T2,GridXExpr,GridYExpr);
	MeshGrid_Matrix_Expression_Matrix(T1,T2,Q2,GridXExpr,GridYExpr);
	MeshGrid_Matrix_Expression_CudaMatrix(T1,T2,Q2,GridXExpr,GridYExpr);
	MeshGrid_Expression_Expression(T1,Q1,T2,Q2,GridXExpr,GridYExpr);

	//   template<class T1,class T2> 
	//std::tuple<CudaExpr<GridXExpr<const T1*,T1>,T1>, CudaExpr<GridYExpr<const T2*,T2>,T2>> Grid(const Matrix<T1>&a,const Matrix<T2>&b) 
	//{ if((a.IsAvector()) && (b.IsAvector())) 
	//	{ 
	//		typedef GridXExpr<const T1*,T1> MGExprX; 
	//		typedef GridYExpr<const T2*,T2> MGExprY; 
	//		int Rows	= (b.GetColumns()!=1)?b.GetColumns():b.GetRows(); 
	//		int Columns = (a.GetColumns()!=1)?a.GetColumns():a.GetRows(); 
	//		return  std::make_tuple(CudaExpr<MGExprX,T1>(MGExprX(a.GetDataPointer(),Rows,Columns),Rows,Columns),CudaExpr<MGExprY,T2>(MGExprY(a.GetDataPointer(),Rows,Columns),Rows,Columns)); 
	//	}  else {	char* str0 = "********************************************************************\n"; 
	//				char* str1 = "* CPU Grid of non-vector elements (matrix-matrix) not possible *\n"; 
	//				char* str2 = "First operand size: "; 
	//				char* str3 = "Second operand size: "; 
	//				char* catString = (char*) malloc(2*strlen(str0)+strlen(str1)+strlen(str2)+strlen(str3)+50*sizeof(char)); 
	//				sprintf(catString, "%s%s%s\n%s%i x %i\n%s%i x %i\n",str0,str1,str0,str2,a.GetRows(),a.GetColumns(),str3,b.GetRows(),b.GetColumns()); 
	//				throw  GenericError(catString,__FILE__,__LINE__); 
	//			} 
	//}

//// GridX on Expression-Expression -- TESTED
//	template<class T1,class Q1,class T2,class Q2> 
//	//std::tuple<CudaExpr<GridXExpr<const CudaExpr<Q1,T1>,T1>,T1>,CudaExpr<GridYExpr<const CudaExpr<Q2,T2>,T2>,T2>> Grid(const CudaExpr<Q1,T1>&a,const CudaExpr<Q2,T2>&b) 
//	std::pair<CudaExpr<GridXExpr<const CudaExpr<Q1,T1>,T1>,T1>,CudaExpr<GridYExpr<const CudaExpr<Q2,T2>,T2>,T2>> Grid(const CudaExpr<Q1,T1>&a,const CudaExpr<Q2,T2>&b) 
//	{ if((a.IsAvector()) && (b.IsAvector())) 
//		{ 
//			typedef GridXExpr<const CudaExpr<Q1,T1>,T1> MGExprX; 
//			typedef GridYExpr<const CudaExpr<Q2,T2>,T2> MGExprY; 
//			int Rows	= (b.GetColumns()!=1)?b.GetColumns():b.GetRows(); \
//			int Columns = (a.GetColumns()!=1)?a.GetColumns():a.GetRows(); \
//			//return std::make_tuple(CudaExpr<MGExprX,T1>(MGExprX(a,Rows,Columns),Rows,Columns),CudaExpr<MGExprY,T1>(MGExprY(a,Rows,Columns),Rows,Columns)); 
//			return std::make_pair(CudaExpr<MGExprX,T1>(MGExprX(a,Rows,Columns),Rows,Columns),CudaExpr<MGExprY,T1>(MGExprY(a,Rows,Columns),Rows,Columns)); 
//		}  else {	char* str0 = "************************************************************************\n"; 
//					char* str1 = "* Grid of non-vector elements (expression-expression) not possible *\n"; 
//					char* str2 = "First operand size: "; 
//					char* str3 = "Second operand size: "; 
//					char* catString = (char*) malloc(2*strlen(str0)+strlen(str1)+strlen(str2)+strlen(str3)+50*sizeof(char)); 
//					sprintf(catString, "%s%s%s\n%s%i x %i\n%s%i x %i\n",str0,str1,str0,str2,a.GetRows(),a.GetColumns(),str3,b.GetRows(),b.GetColumns()); 
//					throw  GenericError(catString,__FILE__,__LINE__); 
//				} 
//	}     


} // namespace
		
#endif