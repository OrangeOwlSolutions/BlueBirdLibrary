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

#ifndef __SUBMATRIXEXPRESSION_H__
#define __SUBMATRIXEXPRESSION_H__

namespace BB
{

	/******************************/
	/* SUBMATRIX EXPRESSION CLASS */
	/******************************/
	// A		- Type of input
	// Type		- Type of output
	template <class A, class Type>
	class SubMatrixExpr
	{
		private:
			int Rows_;						// Rows of the SubMatrix
			int Columns_;					// Columns of the SubMatrix
			int Rows_up_;					// Rows of the original Matrix
			int Columns_up_;				// Columns of the original Matrix
			int a_, c_;						// Starting indices of the SubMatrix as evaluated in the original Matrix
			int rowstep_, columnstep_;		// Undersampling step along rows and columns for the original matrix
			A M_;

		public:
			__forceinline SubMatrixExpr(A &M, int Rows_up, int Columns_up, int Rows, int Columns, int a, int rowstep, int c, int columnstep) : 
					  a_(a), c_(c), M_(M), 
					  Rows_(Rows), 
					  Columns_(Columns), 
					  Rows_up_(Rows_up), Columns_up_(Columns_up), 
					  rowstep_(rowstep), columnstep_(columnstep) { }

			__forceinline int GetRows()			const	{ return Rows_; }
			__forceinline int GetColumns()		const	{ return Columns_; }
			__forceinline int GetNumElements()	const	{ return GetRows()*GetColumns(); }

			__forceinline bool IsVector()		const	{ return (GetRows() == 1 || GetColumns() == 1)?true:false; }
    
			__forceinline __forceinline__ __host__ __device__ const Type& operator[](const int i) const
			{
				const int LocalRow = i/Columns_;			
				const int LocalColumn = i%Columns_;		
				const int GlobalRow = a_+rowstep_*LocalRow;			
				const int GlobalColumn = c_+columnstep_*LocalColumn;
				return M_[IDX2R(GlobalRow,GlobalColumn,Columns_up_)];
			}

			__forceinline __forceinline__ __host__ __device__		Type& operator[](const int i) 
			{
				const int LocalRow = i/Columns_;			
				const int LocalColumn = i%Columns_;		
				const int GlobalRow = a_+rowstep_*LocalRow;			
				const int GlobalColumn = c_+columnstep_*LocalColumn;
				return M_[IDX2R(GlobalRow,GlobalColumn,Columns_up_)];
			}

	};

	/**********************************************/
	/* SUBMATRIX EXPRESSION ROW-COLUMN STEP CLASS */
	/**********************************************/
	// A		- Type of input
	// Type		- Type of output
	template <class A, class Type>
	class SubMatrixExprRowColumnStep
	{
		private:
			int Rows_;						// Rows of the SubMatrix
			int Columns_;					// Columns of the SubMatrix
			int Rows_up_;					// Rows of the original Matrix
			int Columns_up_;				// Columns of the original Matrix
			int a_, c_;						// Starting indices of the SubMatrix as evaluated in the original Matrix
			int columnstep_;				// Undersampling step along columns for the original matrix
			A M_;

		public:
			__forceinline SubMatrixExprRowColumnStep(A &M, int Rows_up, int Columns_up, int Rows, int Columns, int a, int c, int columnstep) : 
					  a_(a), c_(c), M_(M), 
					  Rows_(Rows), 
					  Columns_(Columns), 
					  Rows_up_(Rows_up), Columns_up_(Columns_up), 
					  columnstep_(columnstep) { }

			__forceinline int GetRows()			const	{ return Rows_; }
			__forceinline int GetColumns()		const	{ return Columns_; }
			__forceinline int GetNumElements()	const	{ return GetRows()*GetColumns(); }

			__forceinline bool IsVector()		const	{ return (GetRows() == 1 || GetColumns() == 1)?true:false; }
    
			__forceinline __forceinline__ __host__ __device__ const Type& operator[](const int i) const
			{
				const int LocalRow = i/Columns_;			
				const int LocalColumn = i%Columns_;		
				const int GlobalRow = a_+LocalRow;			
				const int GlobalColumn = c_+columnstep_*LocalColumn;
				return M_[IDX2R(GlobalRow,GlobalColumn,Columns_up_)];
			}

			__forceinline __forceinline__ __host__ __device__		Type& operator[](const int i) 
			{
				const int LocalRow = i/Columns_;			
				const int LocalColumn = i%Columns_;		
				const int GlobalRow = a_+LocalRow;			
				const int GlobalColumn = c_+columnstep_*LocalColumn;
				return M_[IDX2R(GlobalRow,GlobalColumn,Columns_up_)];
			}

	};

	/**********************************************/
	/* SUBMATRIX EXPRESSION ROW STEP-COLUMN CLASS */
	/**********************************************/
	// A		- Type of input
	// Type		- Type of output
	template <class A, class Type>
	class SubMatrixExprRowStepColumn
	{
		private:
			int Rows_;						// Rows of the SubMatrix
			int Columns_;					// Columns of the SubMatrix
			int Rows_up_;					// Rows of the original Matrix
			int Columns_up_;				// Columns of the original Matrix
			int a_, c_;						// Starting indices of the SubMatrix as evaluated in the original Matrix
			int rowstep_;					// Undersampling step along rows for the original matrix
			A M_;

		public:
			__forceinline SubMatrixExprRowStepColumn(A &M, int Rows_up, int Columns_up, int Rows, int Columns, int a, int rowstep, int c) : 
					  a_(a), c_(c), M_(M), 
					  Rows_(Rows), 
					  Columns_(Columns), 
					  Rows_up_(Rows_up), Columns_up_(Columns_up), 
					  rowstep_(rowstep) { }

			__forceinline int GetRows()			const	{ return Rows_; }
			__forceinline int GetColumns()		const	{ return Columns_; }
			__forceinline int GetNumElements()	const	{ return GetRows()*GetColumns(); }

			__forceinline bool IsVector()		const	{ return (GetRows() == 1 || GetColumns() == 1)?true:false; }
    
			__forceinline __forceinline__ __host__ __device__ const Type& operator[](const int i) const
			{
				const int LocalRow = i/Columns_;			
				const int LocalColumn = i%Columns_;		
				const int GlobalRow = a_+rowstep_*LocalRow;			
				const int GlobalColumn = c_+LocalColumn;
				return M_[IDX2R(GlobalRow,GlobalColumn,Columns_up_)];
			}

			__forceinline __forceinline__ __host__ __device__		Type& operator[](const int i) 
			{
				const int LocalRow = i/Columns_;			
				const int LocalColumn = i%Columns_;		
				const int GlobalRow = a_+rowstep_*LocalRow;			
				const int GlobalColumn = c_+LocalColumn;
				return M_[IDX2R(GlobalRow,GlobalColumn,Columns_up_)];
			}

	};

	/**********************************/
	/* SUBMATRIX EXPRESSION ROW CLASS */
	/**********************************/

	// A		- Type of input
	// Type		- Type of output
	template <class A, class Type>
	class  SubMatrixExprRow
	{
		private:
			A M_;
			int IsPinned_;	
			int Offset_;

		public:
			__forceinline SubMatrixExprRow(A M, const int Offset, const int IsPinned = 0) : M_(M), Offset_(Offset), IsPinned_(IsPinned) { }

			__forceinline A GetDataPointer()	const { return M_; };
			__forceinline bool IsPinned()		const { return (IsPinned_ == PINNED) ? true : false; };

			__host__ __device__ __forceinline __forceinline__ const	Type& operator[](const int i) const { return M_[i+Offset_]; }
			__host__ __device__ __forceinline __forceinline__		Type& operator[](const int i)		{ return M_[i+Offset_]; }

	};

	/***************************************/
	/* SUBMATRIX EXPRESSION ROW STEP CLASS */
	/***************************************/
	// A		- Type of input
	// Type		- Type of output
	template <class A, class Type>
	class  SubMatrixExprRowStep
	{
		private:
			int columnstep_;				// Undersampling step along columns for the original matrix
			int Offset_;
			A M_;

		public:
			__forceinline __forceinline__ SubMatrixExprRowStep(A M, const int Offset, int columnstep) : M_(M), Offset_(Offset), columnstep_(columnstep) { }

			__forceinline __forceinline__ __host__ __device__ const	Type& operator[](const int i) const { return M_[Offset_+columnstep_*i]; }
			__forceinline __forceinline__ __host__ __device__ 		Type& operator[](const int i)		{ return M_[Offset_+columnstep_*i]; }

	};

	/*****************************************/
	/* SUBMATRIX EXPRESSION COLUMN-ROW CLASS */
	/*****************************************/
	// A		- Type of input
	// Type		- Type of output
	template <class A, class Type>
	class  SubMatrixExprColumnRow
	{
		private:
			int Rows_;						// Rows of the SubMatrix
			int Columns_;					// Columns of the SubMatrix
			int Rows_up_;					// Rows of the original Matrix
			int Columns_up_;				// Columns of the original Matrix
			int a_, c_;						// Starting indices of the SubMatrix as evaluated in the original Matrix
			A M_;

		public:
			__forceinline SubMatrixExprColumnRow(A &M, int Rows_up, int Columns_up, int Rows, int Columns, int a, int c) : 
					  a_(a), c_(c), M_(M), 
					  Rows_(Rows), 
					  Columns_(Columns), 
					  Rows_up_(Rows_up), Columns_up_(Columns_up) { }

			__forceinline int GetRows()			const	{ return Rows_; }
			__forceinline int GetColumns()		const	{ return Columns_; }
			__forceinline int GetNumElements()	const	{ return GetRows()*GetColumns(); }

			__forceinline bool IsVector()		const	{ return (GetRows() == 1 || GetColumns() == 1)?true:false; }
    
			__host__ __device__ __forceinline __forceinline__ const Type& operator[](const int i) const
			{
				const int LocalRow = i/Columns_;			
				const int LocalColumn = i%Columns_;		
				const int GlobalRow = a_+LocalRow;			
				const int GlobalColumn = c_+LocalColumn;
				return M_[IDX2R(GlobalRow,GlobalColumn,Columns_up_)];
			}

			__host__ __device__ __forceinline __forceinline__		Type& operator[](const int i) 
			{
				const int LocalRow = i/Columns_;			
				const int LocalColumn = i%Columns_;		
				const int GlobalRow = a_+LocalRow;			
				const int GlobalColumn = c_+LocalColumn;
				return M_[IDX2R(GlobalRow,GlobalColumn,Columns_up_)];
			}

	};

	/***************************************************/
	/* CUDA SUBMATRIX ROW PERMUTATION EXPRESSION CLASS */
	/***************************************************/
	// A		- Type of input
	// Type		- Type of output
	//template <class A, class Type>
	//class  CudaSubMatrixExprRowPerm
	//{
	//	private:
	//		A M_;
	//		const Hmatrix<int> B_;
	//		int IsPinned_;	

	//	public:
	//		CudaSubMatrixExprRowPerm(A M, const Hmatrix<int> &B, const int IsPinned = 0) : M_(M), B_(B), IsPinned_(IsPinned) { }

	//		A GetDataPointer()	const { return M_; };
	//		bool IsPinned()		const { return (IsPinned_ == PINNED) ? true : false; };

	//		__host__ __device__ inline const	Type& operator[](const int i) const { return M_[B_(i)]; }
	//		__host__ __device__ inline			Type& operator[](const int i)		{ return M_[B_(i)]; }

	//};

	/*************************************/
	/* SUBMATRIX EXPRESSION COLUMN CLASS */
	/*************************************/
	// A		- Type of input
	// Type		- Type of output
	template <class A, class Type>
	class  SubMatrixExprColumn
	{
		private:
			A M_;
			int step_;
			int offset_;
			int IsPinned_;	

		public:
			__forceinline SubMatrixExprColumn(A M, const int offset, const int step, const int IsPinned = 0) : M_(M), offset_(offset), step_(step), IsPinned_(IsPinned) { }

			__forceinline A GetDataPointer()	const { return M_; };
			__forceinline bool IsPinned()		const { return (IsPinned_ == PINNED) ? true : false; };

			__forceinline __forceinline__ __host__ __device__ const	Type& operator[](const int i) const { return M_[offset_+i*step_]; }
			__forceinline __forceinline__ __host__ __device__		Type& operator[](const int i)		{ return M_[offset_+i*step_]; }

	};

	/******************************************/
	/* SUBMATRIX EXPRESSION COLUMN STEP CLASS */
	/******************************************/
	// A		- Type of input
	// Type		- Type of output
	template <class A, class Type>
	class SubMatrixExprColumnStep
	{
		private:
			A M_;
			int offset_;
			int step_;
			int IsPinned_;	

		public:
			__forceinline SubMatrixExprColumnStep(A M, const int offset, const int step1, const int step2, const int IsPinned = 0) : M_(M), offset_(offset), step_(step1*step2), IsPinned_(IsPinned) { }

			__forceinline A GetDataPointer()	const { return M_; };
			__forceinline bool IsPinned()		const { return (IsPinned_ == PINNED) ? true : false; };

			__forceinline __forceinline__ __host__ __device__ const	Type& operator[](const int i) const { return M_[offset_+i*step_]; }
			__forceinline __forceinline__ __host__ __device__ 		Type& operator[](const int i)		{ return M_[offset_+i*step_]; }

	};

} // namespace

#endif
