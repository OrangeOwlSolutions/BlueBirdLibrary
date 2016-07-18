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


#ifndef __DEXPRESSION_CUH__
#define __DEXPRESSION_CUH__

//#include <typeinfo>
#include <iostream>
////#include "CudaSubMatrixExpression.cuh"
//
//#include "Constants.h"				// Needed for BLOCKSIZE

#include "SubMatrixExpression.h"
namespace BB
{

	template <class OutType> class Hmatrix;
	template <class OutType> class Dmatrix;
	//template <class A, class B> class Expr;
	////template <class A, class Type> class CudaSubMatrixExpr;
	//template <class A, class Type> class SubMatrixExprRow;
	//template <class A, class Type> class CudaSubMatrixExprRowCPU;
	//template <class A, class Type> class CudaSubMatrixExprRowStep;
	//template <class A, class Type> class CudaSubMatrixExprRowStepCPU;
	//template <class A, class Type> class CudaSubMatrixExprColumn;
	//template <class A> class Generator;

	/********************/
	/* EXPRESSION CLASS */
	/********************/

	//template <class T> class Scalar;
	//template <class T> Expr<Scalar<T>,T> SetOnes(int Rows, int Columns);

	// A	- Expression
	// B	- Type of expression
	template <class A, class B>
	class Expr
	{
		private:
			A		a_;
			int		Rows_;
			int		Columns_;
			int		IsDevice_;
 
		public:
			//Expr(const A &a, const int b, const int c, const int IsDevice) : a_(a), Rows_(b), Columns_(c), IsDevice_(IsDevice) { printf("Expression constructor\n"); }
			__forceinline Expr(const A &a, const int b, const int c, const int IsDevice) : a_(a), Rows_(b), Columns_(c), IsDevice_(IsDevice) { }
			//__forceinline Expr(const A &a, const int b, const int c, const int IsDevice) : a_(a), Rows_(b), Columns_(c), IsDevice_(IsDevice) { }
			//__forceinline Expr(const A &a, const int b, const int c, const int IsDevice) : a_(a), Rows_(b), Columns_(c), IsDevice_(IsDevice) { }
       
			__forceinline int GetRows()			const { return Rows_; }
			__forceinline int GetColumns()		const { return Columns_; }
			__forceinline int GetNumElements()	const { return Rows_*Columns_; }

			__forceinline A GetExpression()		const { return a_; }

			__forceinline int IsDevice()			const { return IsDevice_; }

			__forceinline bool IsVector()			const { return (GetRows() == 1 || GetColumns() == 1) ? true : false; };

			////const Expr<A,B>& operator=(Expr<A,B> ob)
			////{   
			////	Expr<A,B> e = *this;
			////	for (int i=0; i<GetNumElements(); i++) e[i] = ob[i]; 
			////	return *this;
			////}

			// --- Range
			__forceinline __forceinline__ Expr<SubMatrixExprRow<Expr<A,B>,B>,B> operator()(Range range)
			{	if((range.GetStart() >= 0) && (range.GetNumElements() > 0) && (range.GetEnd() < GetRows()*GetColumns())) {	
					typedef SubMatrixExprRow<Expr<A,B>,B> SExpr; 
					return Expr<SExpr,B>(SExpr(*this,range.GetStart(),0),1,range.GetNumElements(),(*this).IsDevice()); 
				} else	{	char* str0 = "****************************************\n"; 
				char* str1 = "* Invalid SubExpression access attempt *\n"; 
				char* str2 = "Expression size: "; 
				char* str3 = "SubExpression indices (a,b,c,d): "; 
				char* str4 = "SubExpression size: "; 
				char* catString = (char*) malloc(2*strlen(str0)+strlen(str1)+strlen(str2)+strlen(str3)+strlen(str4)+15*sizeof(char)); 
				sprintf(catString, "%s%s%s\n%s%i x %i\n%s(%i,%i,%i,%i)\n%s%i x %i\n",str0,str1,str0,str2,GetRows(),GetColumns(),str3,0,0,range.GetStart(),range.GetEnd(),str4,1,range.GetNumElements()); 
				throw  GenericError(catString,__FILE__,__LINE__); }
			}

			// --- int Range
			__forceinline __forceinline__ Expr<SubMatrixExprRow<Expr<A,B>,B>,B> operator()(int index, Range range2)
			{	if((index >= 0) && (index < GetRows()) && (range2.GetStart() >= 0) && (range2.GetNumElements() > 0) && (range2.GetEnd() < GetColumns())) {	
					typedef SubMatrixExprRow<Expr<A,B>,B> SExpr; 
					return Expr<SExpr,B>(SExpr(*this,index*GetColumns()+range2.GetStart(),0),1,range2.GetNumElements(),(*this).IsDevice()); 
				} else	{	char* str0 = "****************************************\n"; 
				char* str1 = "* Invalid SubExpression access attempt *\n"; 
				char* str2 = "Expression size: "; 
				char* str3 = "SubExpression indices (a,b,c,d): "; 
				char* str4 = "SubExpression size: "; 
				char* catString = (char*) malloc(2*strlen(str0)+strlen(str1)+strlen(str2)+strlen(str3)+strlen(str4)+15*sizeof(char)); 
				sprintf(catString, "%s%s%s\n%s%i x %i\n%s(%i,%i,%i,%i)\n%s%i x %i\n",str0,str1,str0,str2,GetRows(),GetColumns(),str3,0,0,range2.GetStart(),range2.GetEnd(),str4,1,range2.GetNumElements()); 
				throw  GenericError(catString,__FILE__,__LINE__); }
			}

			// --- Range int
			__forceinline __forceinline__ Expr<SubMatrixExprColumn<Expr<A,B>,B>,B> operator()(Range range1, int index)
			{	if((range1.GetStart() >= 0) && (range1.GetNumElements() > 0) && (range1.GetEnd() < GetRows()) && (index >= 0) && (index < GetColumns())) {	
					typedef SubMatrixExprColumn<Expr<A,B>,B> SExpr; 
					return Expr<SExpr,B>(SExpr(*this,range1.GetStart()*GetColumns()+index,GetColumns(),0),range1.GetNumElements(),1,(*this).IsDevice()); 
				} else	{char* str0 = "****************************************\n"; 
						char* str1 = "* Invalid SubExpression access attempt *\n"; 
						char* str2 = "Expression size: "; 
						char* str3 = "SubMatrix indices (a,b,c,d): "; 
						char* str4 = "SubMatrix size: "; 
						char* catString = (char*) malloc(2*strlen(str0)+strlen(str1)+strlen(str2)+strlen(str3)+strlen(str4)+15*sizeof(char)); 
						sprintf(catString, "%s%s%s\n%s%i x %i\n%s(%i,%i,%i,%i)\n%s%i x %i\n",str0,str1,str0,str2,GetRows(),GetColumns(),str3,range1.GetStart(),range1.GetEnd(),index,index,str4,range1.GetNumElements(),1); 
						throw  GenericError(catString,__FILE__,__LINE__); }
			}

			// --- SubExpressions - Range Range
			__forceinline __forceinline__ Expr<SubMatrixExprColumnRow<Expr<A,B>,B>,B> operator()(Range range1, Range range2)
			{	if((range1.GetStart() >= 0) && (range1.GetNumElements() > 0) && (range1.GetEnd() < Rows_) && 
				(range2.GetStart() >= 0) && (range2.GetNumElements() > 0) && (range2.GetEnd() < Rows_)) {	
					typedef SubMatrixExprColumnRow<Expr<A,B>,B> SExpr; 
					return Expr<SExpr,B>(SExpr(*this,GetRows(),GetColumns(),range1.GetNumElements(),range2.GetNumElements(),range1.GetStart(),range2.GetStart()),range1.GetNumElements(),range2.GetNumElements(),(*this).IsDevice()); 
				} else	{char* str0 = "****************************************\n"; 
						char* str1 = "* Invalid SubExpression access attempt *\n"; 
						char* str2 = "Expression size: "; 
						char* str3 = "SubMatrix indices (a,b,c,d): "; 
						char* str4 = "SubMatrix size: "; 
						char* catString = (char*) malloc(2*strlen(str0)+strlen(str1)+strlen(str2)+strlen(str3)+strlen(str4)+15*sizeof(char)); 
						sprintf(catString, "%s%s%s\n%s%i x %i\n%s(%i,%i,%i,%i)\n%s%i x %i\n",str0,str1,str0,str2,GetRows(),GetColumns(),str3,range1.GetStart(),range1.GetEnd(),range1.GetEnd(),range2.GetEnd(),str4,range1.GetNumElements(),range2.GetNumElements()); 
						throw  GenericError(catString,__FILE__,__LINE__); }
			}

			// --- SubExpressions - Span int
			__forceinline __forceinline__ Expr<SubMatrixExprColumn<Expr<A,B>,B>,B> operator()(SpanClass span1, int index)
			{	if((index >= 0) && (index < Columns_)) {	
					typedef SubMatrixExprColumn<Expr<A,B>,B> SExpr; 
					return Expr<SExpr,B>(SExpr(*this,index,GetColumns(),0),GetRows(),1,(*this).IsDevice()); 
			} else	{	char* str0 = "****************************************\n"; 
						char* str1 = "* Invalid SubExpression access attempt *\n"; 
						char* str2 = "Expression size: "; 
						char* str3 = "SubMatrix indices (a,b,c,d): "; 
						char* str4 = "SubMatrix size: "; 
						char* catString = (char*) malloc(2*strlen(str0)+strlen(str1)+strlen(str2)+strlen(str3)+strlen(str4)+15*sizeof(char)); 
						sprintf(catString, "%s%s%s\n%s%i x %i\n%s(%i,%i,%i,%i)\n%s%i x %i\n",str0,str1,str0,str2,GetRows(),GetColumns(),str3,0,GetRows()-1,index,index,str4,GetRows(),1); 
						throw  GenericError(catString,__FILE__,__LINE__); }
			}
			
			// --- SubExpressions - int Span
			__forceinline __forceinline__ Expr<SubMatrixExprRow<Expr<A,B>,B>,B> operator()(int index, SpanClass span2)
			{	if((index >= 0) && (index < Rows_)) {	
					typedef SubMatrixExprRow<Expr<A,B>,B> SExpr; 
					return Expr<SExpr,B>(SExpr(*this,index*GetColumns(),0),1,GetColumns(),(*this).IsDevice()); 
				} else	{char* str0 = "****************************************\n"; 
						char* str1 = "* Invalid SubExpression access attempt *\n"; 
						char* str2 = "Expression size: "; 
						char* str3 = "SubMatrix indices (a,b,c,d): "; 
						char* str4 = "SubMatrix size: "; 
						char* catString = (char*) malloc(2*strlen(str0)+strlen(str1)+strlen(str2)+strlen(str3)+strlen(str4)+15*sizeof(char)); 
						sprintf(catString, "%s%s%s\n%s%i x %i\n%s(%i,%i,%i,%i)\n%s%i x %i\n",str0,str1,str0,str2,GetRows(),GetColumns(),str3,index,index,0,GetColumns()-1,str4,1,GetColumns()); 
						throw  GenericError(catString,__FILE__,__LINE__); }
			}
			
			// --- SubExpressions - int RangeStep
			__forceinline __forceinline__ Expr<SubMatrixExprRowStep<Expr<A,B>,B>,B> operator()(int index, RangeStep range2)
			{	if((index >= 0) && (index < Rows_) && (range2.GetStart() >= 0) && (range2.GetNumElements() > 0) && (range2.GetEnd() < Columns_)) {	
					typedef SubMatrixExprRowStep<Expr<A,B>,B> SExpr;
					return Expr<SExpr,B>(SExpr(*this,index*Columns_+range2.GetStart(),range2.GetStep()),1,range2.GetNumElements(),(*this).IsDevice()); 
				} else	{char* str0 = "****************************************\n"; 
						char* str1 = "* Invalid SubExpression access attempt *\n"; 
						char* str2 = "Expression size: "; 
						char* str3 = "SubMatrix indices (a,b,c,d): "; 
						char* str4 = "SubMatrix size: "; 
						char* catString = (char*) malloc(2*strlen(str0)+strlen(str1)+strlen(str2)+strlen(str3)+strlen(str4)+15*sizeof(char)); 
						sprintf(catString, "%s%s%s\n%s%i x %i\n%s(%i,%i,%i,%i)\n%s%i x %i\n",str0,str1,str0,str2,GetRows(),GetColumns(),str3,index,index,range2.GetStart(),range2.GetEnd(),str4,1,range2.GetNumElements()); 
						throw  GenericError(catString,__FILE__,__LINE__); }
			}

			// --- SubExpressions - RangeStep int
			__forceinline __forceinline__ Expr<SubMatrixExprColumnStep<Expr<A,B>,B>,B> operator()(RangeStep range1, int index)
			{	if((range1.GetStart() >= 0) && (range1.GetNumElements() > 0) && (range1.GetEnd() < Rows_) && 
				(index >= 0) && (index < Columns_)) {	
					typedef SubMatrixExprColumnStep<Expr<A,B>,B> SExpr; 
					return Expr<SExpr,B>(SExpr(*this,range1.GetStart()*Columns_+index,Columns_,range1.GetStep(),0),range1.GetNumElements(),1,(*this).IsDevice()); 
				} else	{char* str0 = "****************************************\n"; 
						char* str1 = "* Invalid SubExpression access attempt *\n"; 
						char* str2 = "Expression size: "; 
						char* str3 = "SubMatrix indices (a,b,c,d): "; 
						char* str4 = "SubMatrix size: "; 
						char* catString = (char*) malloc(2*strlen(str0)+strlen(str1)+strlen(str2)+strlen(str3)+strlen(str4)+15*sizeof(char)); 
						sprintf(catString, "%s%s%s\n%s%i x %i\n%s(%i,%i,%i,%i)\n%s%i x %i\n",str0,str1,str0,str2,GetRows(),GetColumns(),str3,range1.GetStart(),range1.GetEnd(),index,index,str4,range1.GetNumElements(),1); 
						throw  GenericError(catString,__FILE__,__LINE__); }
			}

			// --- SubExpressions - Range RangeStep
			__forceinline __forceinline__ Expr<SubMatrixExprRowColumnStep<Expr<A,B>,B>,B> operator()(Range range1, RangeStep range2)
			{	if((range1.GetStart() >= 0) && (range1.GetNumElements() > 0) && (range1.GetEnd() < Rows_) && 
				   (range2.GetStart() >= 0) && (range2.GetNumElements() > 0) && (range2.GetEnd() < Columns_)) {	
					typedef SubMatrixExprRowColumnStep<Expr<A,B>,B> SExpr; 
					return Expr<SExpr,B>(SExpr(*this,Rows_,Columns_,range1.GetNumElements(),range2.GetNumElements(),range1.GetStart(),range2.GetStart(),range2.GetStep()),range1.GetNumElements(),range2.GetNumElements(),(*this).IsDevice()); 
				} else	{char* str0 = "****************************************\n"; 
						 char* str1 = "* Invalid SubExpression access attempt *\n"; 
						 char* str2 = "Expression size: ";	 
						 char* str3 = "SubMatrix indices (a,b,c,d): "; 
						 char* str4 = "SubMatrix size: "; 
						 char* catString = (char*) malloc(2*strlen(str0)+strlen(str1)+strlen(str2)+strlen(str3)+strlen(str4)+15*sizeof(char)); 
						 sprintf(catString, "%s%s%s\n%s%i x %i\n%s(%i,%i,%i,%i)\n%s%i x %i\n",str0,str1,str0,str2,GetRows(),GetColumns(),str3,range1.GetStart(),range1.GetEnd(),range1.GetEnd(),range2.GetEnd(),str4,range1.GetNumElements(),range2.GetNumElements()); 
						 throw  GenericError(catString,__FILE__,__LINE__); }
			}

			// --- SubExpressions - RangeStep Range
			__forceinline __forceinline__ Expr<SubMatrixExprRowStepColumn<Expr<A,B>,B>,B> operator()(RangeStep range1, Range range2)
			{	if((range1.GetStart() >= 0) && (range1.GetNumElements() > 0) && (range1.GetEnd() < Rows_) && 
				   (range2.GetStart() >= 0) && (range2.GetNumElements() > 0) && (range2.GetEnd() < Columns_)) {	
					typedef SubMatrixExprRowStepColumn<Expr<A,B>,B> SExpr; 
					return Expr<SExpr,B>(SExpr(*this,Rows_,Columns_,range1.GetNumElements(),range2.GetNumElements(),range1.GetStart(),range1.GetStep(),range2.GetStart()),range1.GetNumElements(),range2.GetNumElements(),(*this).IsDevice()); 
				} else	{char* str0 = "****************************************\n"; 
						 char* str1 = "* Invalid SubExpression access attempt *\n"; 
						 char* str2 = "Expression size: "; 
						 char* str3 = "SubMatrix indices (a,b,c,d): "; 
						 char* str4 = "SubMatrix size: "; 
						 char* catString = (char*) malloc(2*strlen(str0)+strlen(str1)+strlen(str2)+strlen(str3)+strlen(str4)+15*sizeof(char)); 
						 sprintf(catString, "%s%s%s\n%s%i x %i\n%s(%i,%i,%i,%i)\n%s%i x %i\n",str0,str1,str0,str2,Rows_,Columns_,str3,range1.GetStart(),range1.GetEnd(),range1.GetEnd(),range2.GetEnd(),str4,range1.GetNumElements(),range2.GetNumElements()); 
						 throw  GenericError(catString,__FILE__,__LINE__); }
			}

			// --- SubExpressions - RangeStep RangeStep
			__forceinline __forceinline__ Expr<SubMatrixExpr<Expr<A,B>,B>,B> operator()(RangeStep range1, RangeStep range2)
			{	if((range1.GetStart() >= 0) && (range1.GetNumElements() > 0) && (range1.GetEnd() < Rows_) && 
				   (range2.GetStart() >= 0) && (range2.GetNumElements() > 0) && (range2.GetEnd() < Columns_)) {	
					typedef SubMatrixExpr<Expr<A,B>,B> SExpr; 
					return Expr<SExpr,B>(SExpr(*this,Rows_,Columns_,range1.GetNumElements(),range2.GetNumElements(),range1.GetStart(),range1.GetStep(),range2.GetStart(),range2.GetStep()),
							range1.GetNumElements(),range2.GetNumElements(),(*this).IsDevice()); 
				} else	{char* str0 = "****************************************\n"; 
						 char* str1 = "* Invalid SubExpression access attempt *\n"; 
						 char* str2 = "Expression size: "; 
						 char* str3 = "SubMatrix indices (a,b,c,d): "; 
						 char* str4 = "SubMatrix size: "; 
						 char* catString = (char*) malloc(2*strlen(str0)+strlen(str1)+strlen(str2)+strlen(str3)+strlen(str4)+15*sizeof(char)); 
						 sprintf(catString, "%s%s%s\n%s%i x %i\n%s(%i,%i,%i,%i)\n%s%i x %i\n",str0,str1,str0,str2,Rows_,Columns_,str3,range1.GetStart(),range1.GetEnd(),range2.GetStart(),range2.GetEnd(),str4,range1.GetNumElements(),range2.GetNumElements()); 
						 throw  GenericError(catString,__FILE__,__LINE__); }
			}
			
			// --- SubExpressions - Range Span
			__forceinline __forceinline__ Expr<SubMatrixExprColumnRow<Expr<A,B>,B>,B> operator()(Range range1, SpanClass span2)
			{	if((range1.GetStart() >= 0) && (range1.GetNumElements() > 0) && (range1.GetEnd() < Rows_)) {	
					typedef SubMatrixExprColumnRow<Expr<A,B>,B> SExpr; 
					return Expr<SExpr,B>(SExpr(*this,Rows_,Columns_,range1.GetNumElements(),Columns_,range1.GetStart(),0),range1.GetNumElements(),Columns_,(*this).IsDevice()); 
				} else	{char* str0 = "****************************************\n"; 
						char* str1 = "* Invalid SubExpression access attempt *\n"; 
						char* str2 = "Expression size: "; 
						char* str3 = "SubMatrix indices (a,b,c,d): "; 
						char* str4 = "SubMatrix size: "; 
						char* catString = (char*) malloc(2*strlen(str0)+strlen(str1)+strlen(str2)+strlen(str3)+strlen(str4)+15*sizeof(char)); 
						sprintf(catString, "%s%s%s\n%s%i x %i\n%s(%i,%i,%i,%i)\n%s%i x %i\n",str0,str1,str0,str2,Rows_,Columns_,str3,range1.GetStart(),range1.GetEnd(),range1.GetEnd(),Columns_-1,str4,range1.GetNumElements(),Columns_-1); 
						throw  GenericError(catString,__FILE__,__LINE__); }
			}
			
			// --- SubExpressions - Span Range
			__forceinline __forceinline__ Expr<SubMatrixExprColumnRow<Expr<A,B>,B>,B> operator()(SpanClass span1, Range range2)
			{	if((range2.GetStart() >= 0) && (range2.GetNumElements() > 0) && (range2.GetEnd() < Rows_)) {	
					typedef SubMatrixExprColumnRow<Expr<A,B>,B> SExpr; 
					return Expr<SExpr,B>(SExpr(*this,Rows_,Columns_,Rows_,range2.GetNumElements(),0,range2.GetStart()),Rows_,range2.GetNumElements(),(*this).IsDevice()); 
				} else	{char* str0 = "****************************************\n"; 
						char* str1 = "* Invalid SubExpression access attempt *\n"; 
						char* str2 = "Expression size: "; 
						char* str3 = "SubMatrix indices (a,b,c,d): "; 
						char* str4 = "SubMatrix size: "; 
						char* catString = (char*) malloc(2*strlen(str0)+strlen(str1)+strlen(str2)+strlen(str3)+strlen(str4)+15*sizeof(char)); 
						sprintf(catString, "%s%s%s\n%s%i x %i\n%s(%i,%i,%i,%i)\n%s%i x %i\n",str0,str1,str0,str2,Rows_,Columns_,str3,0,Rows_-1,Rows_-1,range2.GetEnd(),str4,Rows_,range2.GetNumElements()); 
						throw  GenericError(catString,__FILE__,__LINE__); }
			}

// --- SubExpressions - Span RangeStep
__forceinline __forceinline__ Expr<SubMatrixExprRowColumnStep<Expr<A,B>,B>,B> operator()(SpanClass span1, RangeStep range2)
{	if((range2.GetStart() >= 0) && (range2.GetNumElements() > 0) && (range2.GetEnd() < Columns_)) {	
		typedef SubMatrixExprRowColumnStep<Expr<A,B>,B> SExpr; 
		return Expr<SExpr,B>(SExpr(*this,Rows_,Columns_,Rows_,range2.GetNumElements(),0,range2.GetStart(),range2.GetStep()),Rows_,range2.GetNumElements(),(*this).IsDevice()); 
	} else	{char* str0 = "****************************************\n"; 
			 char* str1 = "* Invalid SubExpression access attempt *\n"; 
			 char* str2 = "Expression size: "; 
			 char* str3 = "SubMatrix indices (a,b,c,d): "; 
			 char* str4 = "SubMatrix size: "; 
			 char* catString = (char*) malloc(2*strlen(str0)+strlen(str1)+strlen(str2)+strlen(str3)+strlen(str4)+15*sizeof(char)); 
			 sprintf(catString, "%s%s%s\n%s%i x %i\n%s(%i,%i,%i,%i)\n%s%i x %i\n",str0,str1,str0,str2,Rows_,Columns_,str3,0,Rows_-1,Rows_-1,range2.GetEnd(),str4,Rows_,range2.GetNumElements()); 
			 throw  GenericError(catString,__FILE__,__LINE__); }
}

// --- SubExpressions - RangeStep Span
__forceinline __forceinline__ Expr<SubMatrixExprRowStepColumn<Expr<A,B>,B>,B> operator()(RangeStep range1, SpanClass span2)
{	if((range1.GetStart() >= 0) && (range1.GetNumElements() > 0) && (range1.GetEnd() < Rows_)) {	
		typedef SubMatrixExprRowStepColumn<Expr<A,B>,B> SExpr; 
		return Expr<SExpr,B>(SExpr(*this,Rows_,Columns_,range1.GetNumElements(),Columns_,range1.GetStart(),range1.GetStep(),0),range1.GetNumElements(),Columns_,(*this).IsDevice()); 
	} else	{char* str0 = "****************************************\n"; 
			 char* str1 = "* Invalid SubExpression access attempt *\n"; 
			 char* str2 = "Expression size: "; 
			 char* str3 = "SubMatrix indices (a,b,c,d): "; 
			 char* str4 = "SubMatrix size: "; 
			 char* catString = (char*) malloc(2*strlen(str0)+strlen(str1)+strlen(str2)+strlen(str3)+strlen(str4)+15*sizeof(char)); 
			 sprintf(catString, "%s%s%s\n%s%i x %i\n%s(%i,%i,%i,%i)\n%s%i x %i\n",str0,str1,str0,str2,Rows_,Columns_,str3,range1.GetStart(),range1.GetEnd(),range1.GetEnd(),Columns_-1,str4,range1.GetNumElements(),Columns_); 
			 throw  GenericError(catString,__FILE__,__LINE__); }
}

			//// --- SubExpressions - Span Span
			//inline Expr<CudaSubMatrixExpr<const Expr<A,B>,B>,B> operator()(SpanClass span1, SpanClass span2)
			////inline Expr<CudaSubMatrixExpr<Expr<A,B>,B>,B> operator()(SpanClass span1, SpanClass span2)
			//{	Expr<A,B> e = *this;
			//	typedef CudaSubMatrixExpr<const Expr<A,B>,B> SExpr;  
			//	//typedef CudaSubMatrixExpr<Expr<A,B>,B> SExpr;  
			//	return Expr<SExpr,B>(SExpr(e,e.GetRows(),e.GetColumns(),Rows_,Columns_,0,1,0,1),Rows_,Columns_,e.IsDevice()); 
			//}

			//// SubExpression assignment
			//Expr<A,B> operator=(const Hmatrix<B> &ob)
			//{
			//	// NON SI PUò LEVARE QUESTA ASSEGNAZIONE??
			//	Expr<A,B> e = *this;
			//	if (IsDevice() == ISHOST) { 
			//		for (int i=0; i<GetNumElements(); i++) { a_[i] = ob.GetDataPointer()[i]; } }
			//	else if (IsDevice() == ISDEVICE) { }
			//		////Dmatrix<B> temp(ob);
			//		////evaluation_submatrix_function(a_,temp.data_,GetNumElements()); }
			//		//assign_cpu_matrix_to_gpu_expression(a_,ob); }
			//	return *this;
			//}

			//Expr<A,B> operator=(const Dmatrix<B> &ob)
			//{
			//	Expr<A,B> e = *this;
			//	evaluation_submatrix_function(a_,ob.data_,GetNumElements());
			//	return *this;
			//}

			// --- Assignment Expression = Scalar
			const Expr<A,B> operator=(const B c);

			//template <class C>
			//Expr<A,B> operator=(const Expr<C,B> ob)
			//{
			//	Expr<A,B> e = *this;
			//	// Controllo d'errore se le due espressioni non sono entrambe GPU o CPU
			//	if ((e.IsDevice() == ISHOST)&&(ob.IsDevice() == ISHOST)) for (int i=0; i<GetNumElements(); i++) a_[i] = ob[i]; 
			//	else 
			//		if ((e.IsDevice() == ISDEVICE)&&(ob.IsDevice() == ISDEVICE)) evaluation_submatrix_function_expression(e,ob,GetNumElements()); 
			//		//else
			//		//	if ((e.IsDevice() == ISHOST)&&(ob.IsDevice() == ISDEVICE)) { assign_gpu_expression_to_cpu_expression(e.GetExpression(),ob.GetExpression(),GetNumElements()); }
			//	return *this;
			//}

			//Expr<CudaSubMatrixExprRow<B*,B>,B> operator=(const Expr<CudaSubMatrixExprRow<B*,B>,B> ob)
			//{ 
			//	switch (IsDevice()) {
			//		// lhs is host
			//		case ISHOST:
			//			switch (ob.IsDevice()) {
			//				// rhs is host
			//				case ISHOST:
			//					memcpy(&a_[0],&ob[0],GetNumElements()*sizeof(B));
			//					//if (typeid(*this)==typeid(Expr<CudaSubMatrixExprRow<B*,B>,B>)) memcpy(&a_[0],&ob[0],GetNumElements()*sizeof(B));
			//					//else { for (int i=0; i<GetNumElements(); i++) a_[i] = ob[i]; }
			//				break;
			//				case ISDEVICE:
			//					//assign_gpu_expression_to_cpu_expression(e.GetExpression(),ob.GetExpression(),GetNumElements());
			//					std::cout << "Not yet implemented\n";
			//				break;
			//			}
			//		break;
			//		// lhs is device
			//		case ISDEVICE:
			//			switch (ob.IsDevice()) {
			//				// rhs is device
			//				case ISDEVICE:
			//					Expr<A,B> e = *this;
			//					evaluation_submatrix_function_expression(e,ob,GetNumElements());
			//				break;
			//			}
			//		break;
			//	}
			//	return *this;
			//}

			//// Submatrix - Submatrix
			////Expr<CudaSubMatrixExprRow<B*,B>,B> operator=(const Expr<CudaSubMatrixExprRow<B*,B>,B> ob);

			//// Submatrix - Hmatrix
			//// NON SI PUò DEFINIRE PERCHè VA IN CONFLITTO CON LA STESSA FUNZIONE CON USCITA Expr<A,B> E IL COMPILATORE DICE CHE NON E' POSSIBILE
			//// SOVRACCARICARE FUNZIONI DISTINTE PER IL SOLO TIPO DI USCITA
			////Expr<CudaSubMatrixExprRow<B*,B>,B> operator=(const Hmatrix<B> &ob)
			////{
			////	std::cout << "PIPPONE\n";
			////	Expr<A,B> e = *this;
			////	if (e.IsDevice() == ISHOST) for (int i=0; i<GetNumElements(); i++) a_[i] = ob.GetDataPointer()[i];
			////	//else if (e.IsDevice() == ISDEVICE) { 
			////	//	 if (ob.IsPinned())	CudaSafeCall(cudaMemcpyAsync(e.a_.M_,ob.GetDataPointer(),GetNumElements()*sizeof(B),cudaMemcpyHostToDevice,streams[p2p.active_GPU].GetActiveStream()));
			////	//	 else				CudaSafeCall(cudaMemcpy(e.a_.M_,ob.GetDataPointer(),GetNumElements()*sizeof(B),cudaMemcpyHostToDevice)); }
			////	return *this;
			////}

			//Expr<A,B> operator=(const Expr<Generator<B>,B> &ob)
			//{
			//	Expr<A,B> e = *this;
			//	// Controllo d'errore se le due espressioni non sono entrambe GPU o CPU
			//	if (e.IsDevice() == ISHOST) { 
			//		for (int i=0; i<GetNumElements(); i++) { a_[i] = ob[i]; } 
			//	}
			//	else if (e.IsDevice() == ISDEVICE) { 
			//		Dmatrix<B> temp(ob);
			//		evaluation_submatrix_function(a_,temp.data_,GetNumElements()); }
			//	return *this;
			//}

			//__host__ __device__ __forceinline__ const	B& operator[](const int i) const	{ printf("Expression []\n"); return a_[i]; }
			__host__ __device__ __forceinline __forceinline__ const	B& operator[](const int i) const	{ return a_[i]; }
			//__forceinline const	B& operator[](const int i) const	{ return a_[i]; }
			__host__ __device__ __forceinline __forceinline__ B& operator[](const int i) 			{ const Expr& constThis = *this; return const_cast<B&>(constThis[i]); }
	};

	// Submatrix - Submatrix
	//template<class A,class B>
	//Expr<CudaSubMatrixExprRow<B*,B>,B> Expr<A,B>::operator=(const Expr<CudaSubMatrixExprRow<B*,B>,B> ob)
	//{
	//	std::cout << (typeid(ob) == typeid(Expr<CudaSubMatrixExprRow<B*,B>,B>)) << "\n";
	//	//for (int i=0; i<GetNumElements(); i++) a_[i] = ob[i]; 
	//	// Non si può levare questa assegnazione?? Vedere parte GPU
	//	Expr<CudaSubMatrixExprRow<B*,B>,B> e = *this;
	//	// Controllo d'errore se le due espressioni non sono entrambe GPU o CPU
	//	//if ((e.IsDevice() == ISHOST)&&(ob.IsDevice() == ISHOST)) for (int i=0; i<GetNumElements(); i++) a_[i] = ob[i]; 
	//	if ((IsDevice() == ISHOST)&&(ob.IsDevice() == ISHOST)) memcpy((void*)&a_[0],(void*)&ob[0],GetNumElements()*sizeof(B)); 
	//	else 
	//		if ((IsDevice() == ISDEVICE)&&(ob.IsDevice() == ISDEVICE)) evaluation_submatrix_function_expression(e,ob,GetNumElements()); 
	//		else
	//			if ((IsDevice() == ISHOST)&&(ob.IsDevice() == ISDEVICE)) { assign_gpu_expression_to_cpu_expression(e.GetExpression(),ob.GetExpression(),GetNumElements()); }
	//			else assign_cpu_expression_to_gpu_expression(e.GetExpression(),ob.GetExpression(),GetNumElements());
	//		return *this;
	//}

			//Expr<CudaSubMatrixExprRowCPU<B*,B>,B> operator=(const Expr<CudaSubMatrixExprRowCPU<B*,B>,B> ob)
			////{	if((ob.Rows_ == Rows_) && (ob.Columns_ == Columns_)) for (int i=0; i<GetNumElements(); i++) a_[i] = ob[i]; 
			//{	if((ob.Rows_ == Rows_) && (ob.Columns_ == Columns_)) memcpy((void*)&a_[0],(void*)&ob[0],GetNumElements()*sizeof(B)); 
			//	else  {  
			//		char* str0 = "*************************************************\n";
			//		char* str1 = "* Size mismatch in CPU=CPU submatrix assignment *\n";
			//		char* str2 = "Left operand size: ";
			//		char* str3 = "Right operand size: ";
			//		char* catString = (char*) malloc(2*strlen(str0)+strlen(str1)+strlen(str1)+strlen(str2)+strlen(str3)+40*sizeof(char));
			//		sprintf(catString, "%s%s%s\n%s%i x %i\n%s%i x %i",str0,str1,str0,str2,Rows_,Columns_,str3,ob.GetRows(),ob.GetColumns());
			//		throw  GenericError(catString,__FILE__,__LINE__); } 
			//	return *this;
			//}

			//Expr<CudaSubMatrixExprRowStepCPU<B*,B>,B> operator=(const Expr<CudaSubMatrixExprRowStepCPU<B*,B>,B> ob)
			//{	std::cout << "pippo\n"; if((ob.Rows_ == Rows_) && (ob.Columns_ == Columns_)) for (int i=0; i<GetNumElements(); i++) a_[i] = ob[i]; 
			//	else  {  
			//		char* str0 = "*************************************************\n";
			//		char* str1 = "* Size mismatch in CPU=CPU submatrix assignment *\n";
			//		char* str2 = "Left operand size: ";
			//		char* str3 = "Right operand size: ";
			//		char* catString = (char*) malloc(2*strlen(str0)+strlen(str1)+strlen(str1)+strlen(str2)+strlen(str3)+40*sizeof(char));
			//		sprintf(catString, "%s%s%s\n%s%i x %i\n%s%i x %i",str0,str1,str0,str2,Rows_,Columns_,str3,ob.GetRows(),ob.GetColumns());
			//		throw  GenericError(catString,__FILE__,__LINE__); } 
			//	return *this;
			//}

	// --- Assignment SubMatrixExprRow = Scalar
	template <class B>
	__global__ __forceinline__ void evaluation_submatrix_constant(SubMatrixExprRow<B*,B> e, B c, int NumElements)
	{ 
		const int i = blockDim.x * blockIdx.x + threadIdx.x;
		if(i < NumElements) e[i] = c;
	}

	// --- Assignment SubMatrixExprRow = Scalar
	template <class B>		
	inline void eval_submatrix_constant_wrap(SubMatrixExprRow<B*,B> e, B c, int NumElements)
	{
		dim3 dimGrid(iDivUp(NumElements,dimBlock.x));
		evaluation_submatrix_constant<<<dimGrid,dimBlock,0,streams[p2p.active_GPU].GetActiveStream()>>>(e,c,NumElements);
		CudaCheckError();
	}			

	// --- Assignment SubMatrixExprRow = Scalar
	template <class B>
	void evaluation_submatrix_function_constant(SubMatrixExprRow<B*,B> e, B c, int NumElements) { eval_submatrix_constant_wrap(e, c, NumElements); }

	// --- Assignment Expression = Scalar
	template <class A, class B>
	const Expr<A,B> Expr<A,B>::operator=(const B c)
	{
		Expr<A,B> e = *this;
		if (e.IsDevice() == ISHOST) { 
			for (int i=0; i<GetNumElements(); i++) { a_[i] = c; } 
		}
		else if (e.IsDevice() == ISDEVICE) { 
			evaluation_submatrix_function_constant(a_,c,GetNumElements()); }
		return *this;
	}

	/********************************/
	/* CUDA BINARY EXPRESSION CLASS */
	/********************************/
	// A		- Type of first operand
	// B		- Type of second operand
	// op		- Operation
	// OutType	- Type of result of expression
	template <class A, class B, class op, class OutType>
	class BinExpr
	{
		private:
			A		a_;
			B		b_;
         
		public:
			__forceinline BinExpr(const A &a, const B &b): a_(a), b_(b) { }
    
			__host__ __device__ __forceinline __forceinline__ OutType operator[](int i) const { return op::eval(a_[i], b_[i]); }
	};

	/***********************************************/
	/* CUDA ONE-ARGUMENT FUNCTION EXPRESSION CLASS */
	/***********************************************/
	// Type		- Type of input
	// fun		- Function
	// OutType	- Type of result

	template <typename Type, typename fun, typename OutType>
	class FunExpr
	{
		private:
			Type v_;
			int		Rows_;
			int		Columns_;

		public:
			FunExpr(const Type &v, const int c, const int d) : v_(v), Rows_(c), Columns_(d) { }

			int GetRows()			const { return Rows_; }
			int GetColumns()		const { return Columns_; }      
 			int GetNumElements()	const { return Rows_*Columns_; }
			
			bool IsVector()		const { return (GetRows() == 1 || GetColumns() == 1) ? true : false; };

			__host__ __device__ inline  OutType operator[](int i) const { OutType c; c = fun::eval(v_[i]); return c; }
   
	};
	
	/************************************************/
	/* CUDA TWO-ARGUMENTS FUNCTION EXPRESSION CLASS */
	/************************************************/
	// Type1	- Type of input #1
	// Type2	- Type of input #2
	// fun		- Function
	// OutType	- Type of result
	template <typename Type1, typename Type2, typename fun, typename OutType>
	class FunExpr2
	{
		private:
			Type1	v_;
			Type2	w_;
			int		Rows_;
			int		Columns_;

		public:
			FunExpr2(const Type1 &v, const Type2 w, const int c, const int d) : v_(v), w_(w), Rows_(c), Columns_(d) {}

			int GetRows()			const { return Rows_; }
			int GetColumns()		const { return Columns_; }      
 			int GetNumElements()	const { return Rows_*Columns_; }
			
			bool IsVector()		const { return (GetRows() == 1 || GetColumns() == 1) ? true : false; };

			__host__ __device__ inline  OutType operator[](int i) const { OutType c; c = fun::eval(v_[i],w_); return c; }
    
	};

	template <typename A,typename B>
	class FFTExpr
	{
		private:
			A a_;
			int M_;
			int N_;
		public:
			FFTExpr(const A &a) : a_(a), M_((a.GetRows()%2) ? (GetRows()+1)/2:(GetRows())/2), N_((a.GetColumns()%2) ? (GetColumns()+1)/2:(GetColumns())/2)  {}
    
			int GetRows() const { return a_.GetRows(); }
			int GetColumns() const { return a_.GetColumns(); }
    
			int GetNumElements() const { return a_.GetNumElements(); }
    
			bool IsVector()const  { return (GetRows() == 1 || GetColumns() == 1)?true:false;}
    
			inline B operator()(const int i, const int j)const { return (*this)[IDX2R(i,j,GetColumns())]; }
		
			inline B operator[](const int i) const
			{
				const int row    = i/GetColumns();
				const int column = i%GetColumns();
				const int newRow = (row+M_)%GetRows();
				const int newColumn = (column+N_)%GetColumns();
				return a_[IDX2R(newRow,newColumn,GetColumns())];
			}
	};

}

#endif