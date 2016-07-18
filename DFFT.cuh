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

#ifndef __DFFT_CUH__
#define __DFFT_CUH__

namespace BB
{
	#include <cufft.h>
	
	/**********************/
	/* COMPLEX TO COMPLEX */
	/**********************/

	// --- Plan Complex to Complex (1D)
	cufftHandle DEVICE_FFT_1D_PLAN_C2C(const int NX, const int BATCH)
	{
		cufftHandle plan;
		cufftSafeCall(cufftPlan1d(&plan, NX, CUFFT_C2C, BATCH));
		return plan;
	}

	// --- Plan Complex to Complex (2D)
	cufftHandle DEVICE_FFT_2D_PLAN_C2C(const int NX, const int NY)
	{
		cufftHandle plan;
		cufftSafeCall(cufftPlan2d(&plan, NX, NY, CUFFT_C2C));
		return plan;
	}

	// --- Exec Complex to Complex --- Direct
	Dmatrix<float2_> FFT(const Dmatrix<float2_> &in, cufftHandle plan = NULL)
	{
		Dmatrix<float2_> out(in.GetRows(),in.GetColumns());

		if (plan == NULL)
		{
			if (in.IsVector()) cufftSafeCall(cufftPlan1d(&plan,in.GetNumElements(),CUFFT_C2C,1));    
			else cufftSafeCall(cufftPlan2d(&plan,in.GetRows(),in.GetColumns(),CUFFT_C2C));    
			cufftSafeCall(cufftSetStream(plan,streams[p2p.active_GPU].GetActiveStream()));
			cufftSafeCall(cufftExecC2C(plan,(cufftComplex*)(in.GetDataPointer()),(cufftComplex*)(out.GetDataPointer()),CUFFT_FORWARD)); 
			cufftSafeCall(cufftDestroy(plan));
		}
		else
		{	
			cufftSafeCall(cufftSetStream(plan,streams[p2p.active_GPU].GetActiveStream()));
			cufftSafeCall(cufftExecC2C(plan,(cufftComplex*)(in.GetDataPointer()),(cufftComplex*)(out.GetDataPointer()),CUFFT_FORWARD)); 
		}
		return out;
	}

	// --- Exec Complex to Complex --- Inverse
	Dmatrix<float2_> IFFT(const Dmatrix<float2_> &in, cufftHandle plan = NULL)
	{
		Dmatrix<float2_> out(in.GetRows(),in.GetColumns());

		if (plan == NULL)
		{
			if (in.IsVector()) cufftSafeCall(cufftPlan1d(&plan,in.GetNumElements(),CUFFT_C2C,1));    
			else cufftSafeCall(cufftPlan2d(&plan,in.GetRows(),in.GetColumns(),CUFFT_C2C));    
			cufftSafeCall(cufftSetStream(plan,streams[p2p.active_GPU].GetActiveStream()));
			cufftSafeCall(cufftExecC2C(plan,(cufftComplex*)(in.GetDataPointer()),(cufftComplex*)(out.GetDataPointer()),CUFFT_INVERSE)); 
			cufftSafeCall(cufftDestroy(plan));
		}
		else
		{	
			cufftSafeCall(cufftSetStream(plan,streams[p2p.active_GPU].GetActiveStream()));
			cufftSafeCall(cufftExecC2C(plan,(cufftComplex*)(in.GetDataPointer()),(cufftComplex*)(out.GetDataPointer()),CUFFT_INVERSE)); 
		}
		return out;
	}

	// --- Exec Complex Expression to Complex --- Direct
	template <class Q>
	Dmatrix<float2_> FFT(const Expr<Q,float2_> &e, cufftHandle plan = NULL)
	{
		Dmatrix<float2_> in(e.GetRows(),e.GetColumns());
		Dmatrix<float2_> out(in.GetRows(),in.GetColumns());

		in = e;

		if (plan == NULL)
		{
			if (in.IsVector()) cufftSafeCall(cufftPlan1d(&plan,in.GetNumElements(),CUFFT_C2C,1));    
			else cufftSafeCall(cufftPlan2d(&plan,in.GetRows(),in.GetColumns(),CUFFT_C2C));    
			cufftSafeCall(cufftSetStream(plan,streams[p2p.active_GPU].GetActiveStream()));
			cufftSafeCall(cufftExecC2C(plan,(cufftComplex*)(in.GetDataPointer()),(cufftComplex*)(out.GetDataPointer()),CUFFT_FORWARD)); 
			cufftSafeCall(cufftDestroy(plan));
		}
		else 
		{
			cufftSafeCall(cufftSetStream(plan,streams[p2p.active_GPU].GetActiveStream()));
			cufftSafeCall(cufftExecC2C(plan,(cufftComplex*)(in.GetDataPointer()),(cufftComplex*)(out.GetDataPointer()),CUFFT_FORWARD)); 
		}
		return out;
	}

	// --- Exec Complex Expression to Complex --- Inverse
	template <class Q>
	Dmatrix<float2_> IFFT(const Expr<Q,float2_> &e, cufftHandle plan = NULL)
	{
		Dmatrix<float2_> in(e.GetRows(),e.GetColumns());
		Dmatrix<float2_> out(in.GetRows(),in.GetColumns());

		in = e;

		if (plan == NULL)
		{
			if (in.IsVector()) cufftSafeCall(cufftPlan1d(&plan,in.GetNumElements(),CUFFT_C2C,1));    
			else cufftSafeCall(cufftPlan2d(&plan,in.GetRows(),in.GetColumns(),CUFFT_C2C));    
			cufftSafeCall(cufftSetStream(plan,streams[p2p.active_GPU].GetActiveStream()));
			cufftSafeCall(cufftExecC2C(plan,(cufftComplex*)(in.GetDataPointer()),(cufftComplex*)(out.GetDataPointer()),CUFFT_INVERSE)); 
			cufftSafeCall(cufftDestroy(plan));
		}
		else 
		{
			cufftSafeCall(cufftSetStream(plan,streams[p2p.active_GPU].GetActiveStream()));
			cufftSafeCall(cufftExecC2C(plan,(cufftComplex*)(in.GetDataPointer()),(cufftComplex*)(out.GetDataPointer()),CUFFT_INVERSE)); 
		}
		return out;
	}

	/************************************/
	/* DOUBLE COMPLEX TO DOUBLE COMPLEX */
	/************************************/

	// --- Plan DoubleComplex to DoubleComplex
	cufftHandle DEVICE_FFT_1D_PLAN_Z2Z(const int NX, const int BATCH = 1)
	{
		cufftHandle plan;
		cufftSafeCall(cufftPlan1d(&plan, NX, CUFFT_Z2Z, BATCH));
		return plan;
	}

	// --- Plan DoubleComplex to DoubleComplex (2D)
	cufftHandle DEVICE_FFT_2D_PLAN_Z2Z(const int NX, const int NY)
	{
		cufftHandle plan;
		cufftSafeCall(cufftPlan2d(&plan, NX, NY, CUFFT_Z2Z));
		return plan;
	}

	// --- Exec DoubleComplex to DoubleComplex --- Direct
	Dmatrix<double2_> FFT(const Dmatrix<double2_> &in, cufftHandle plan = NULL)
	{
		Dmatrix<double2_> out(in.GetRows(),in.GetColumns());

		if (plan == NULL)
		{
			if (in.IsVector()) cufftSafeCall(cufftPlan1d(&plan,in.GetNumElements(),CUFFT_Z2Z,1));    
			else cufftSafeCall(cufftPlan2d(&plan,in.GetRows(),in.GetColumns(),CUFFT_Z2Z));    
			cufftSafeCall(cufftSetStream(plan,streams[p2p.active_GPU].GetActiveStream()));
			cufftSafeCall(cufftExecZ2Z(plan,(cufftDoubleComplex*)(in.GetDataPointer()),(cufftDoubleComplex*)(out.GetDataPointer()),CUFFT_FORWARD)); 
			cufftSafeCall(cufftDestroy(plan));
		}
		else
		{
			cufftSafeCall(cufftSetStream(plan,streams[p2p.active_GPU].GetActiveStream()));
			cufftSafeCall(cufftExecZ2Z(plan,(cufftDoubleComplex*)(in.GetDataPointer()),(cufftDoubleComplex*)(out.GetDataPointer()),CUFFT_FORWARD)); 
		}
		return out;
	}

	// --- Exec DoubleComplex to DoubleComplex --- Inverse
	Dmatrix<double2_> IFFT(const Dmatrix<double2_> &in, cufftHandle plan = NULL)
	{
		Dmatrix<double2_> out(in.GetRows(),in.GetColumns());

		if (plan == NULL)
		{
			if (in.IsVector()) cufftSafeCall(cufftPlan1d(&plan,in.GetNumElements(),CUFFT_Z2Z,1));    
			else cufftSafeCall(cufftPlan2d(&plan,in.GetRows(),in.GetColumns(),CUFFT_Z2Z));    
			cufftSafeCall(cufftSetStream(plan,streams[p2p.active_GPU].GetActiveStream()));
			cufftSafeCall(cufftExecZ2Z(plan,(cufftDoubleComplex*)(in.GetDataPointer()),(cufftDoubleComplex*)(out.GetDataPointer()),CUFFT_INVERSE)); 
			cufftSafeCall(cufftDestroy(plan));
		}
		else
		{
			cufftSafeCall(cufftSetStream(plan,streams[p2p.active_GPU].GetActiveStream()));
			cufftSafeCall(cufftExecZ2Z(plan,(cufftDoubleComplex*)(in.GetDataPointer()),(cufftDoubleComplex*)(out.GetDataPointer()),CUFFT_INVERSE)); 
		}
		return out;
	}

	// --- Exec DoubleComplex Expression to DoubleComplex --- Direct
	template <class Q>
	Dmatrix<double2_> FFT(const Expr<Q,double2_> &e, cufftHandle plan = NULL)
	{
		Dmatrix<double2_> in(e.GetRows(),e.GetColumns());
		Dmatrix<double2_> out(e.GetRows(),e.GetColumns());

		in = e;
		
		if (plan == NULL)
		{
			if (in.IsVector()) cufftSafeCall(cufftPlan1d(&plan,in.GetNumElements(),CUFFT_Z2Z,1));    
			else cufftSafeCall(cufftPlan2d(&plan,in.GetRows(),in.GetColumns(),CUFFT_Z2Z));    
			cufftSafeCall(cufftSetStream(plan,streams[p2p.active_GPU].GetActiveStream()));
			cufftSafeCall(cufftExecZ2Z(plan,(cufftDoubleComplex*)(in.GetDataPointer()),(cufftDoubleComplex*)(out.GetDataPointer()),CUFFT_FORWARD)); 
			cufftSafeCall(cufftDestroy(plan));
		}
		else 
		{	
			cufftSafeCall(cufftSetStream(plan,streams[p2p.active_GPU].GetActiveStream()));
			cufftSafeCall(cufftExecZ2Z(plan,(cufftDoubleComplex*)(in.GetDataPointer()),(cufftDoubleComplex*)(out.GetDataPointer()),CUFFT_FORWARD)); 
		}
		return out;
	}

	// --- Exec DoubleComplex Expression to DoubleComplex --- Inverse
	template <class Q>
	Dmatrix<double2_> IFFT(const Expr<Q,double2_> &e, cufftHandle plan = NULL)
	{
		Dmatrix<double2_> in(e.GetRows(),e.GetColumns());
		Dmatrix<double2_> out(e.GetRows(),e.GetColumns());

		in = e;
		
		if (plan == NULL)
		{
			if (in.IsVector()) cufftSafeCall(cufftPlan1d(&plan,in.GetNumElements(),CUFFT_Z2Z,1));    
			else cufftSafeCall(cufftPlan2d(&plan,in.GetRows(),in.GetColumns(),CUFFT_Z2Z));    
			cufftSafeCall(cufftSetStream(plan,streams[p2p.active_GPU].GetActiveStream()));
			cufftSafeCall(cufftExecZ2Z(plan,(cufftDoubleComplex*)(in.GetDataPointer()),(cufftDoubleComplex*)(out.GetDataPointer()),CUFFT_INVERSE)); 
			cufftSafeCall(cufftDestroy(plan));
		}
		else 
		{	
			cufftSafeCall(cufftSetStream(plan,streams[p2p.active_GPU].GetActiveStream()));
			cufftSafeCall(cufftExecZ2Z(plan,(cufftDoubleComplex*)(in.GetDataPointer()),(cufftDoubleComplex*)(out.GetDataPointer()),CUFFT_INVERSE)); 
		}
		return out;
	}

	/********************/
	/* PLAN DESTRUCTION */
	/********************/

	// --- Plan Destruction
	void DEVICE_FFT_DESTROY_PLAN(cufftHandle plan)
	{
		cufftSafeCall(cufftDestroy(plan));
	}
}
	
//	#include <fftw3.h>
//
//	template <class T>
//	void FFT(const Matrix<T> &in_d, Matrix<double2> &out)
//	{
//		Matrix<double2> in(in_d.GetRows(),in_d.GetColumns());
//		in = in_d +0*J;
//    
//		//cuda double2 type seems to be  bit compatible to fftw_complex
//		if(in.GetNumElements() == out.GetNumElements())
//		{    
//			//two vector
//			if(in.IsAvector() && out.IsAvector())
//			{
//				fftw_plan plan_forward;
//				plan_forward = fftw_plan_dft_1d ( in.GetNumElements(), (fftw_complex*)(in.data_),(fftw_complex*)(out.data_), FFTW_FORWARD, FFTW_ESTIMATE );
//				fftw_execute ( plan_forward );
//				fftw_destroy_plan(plan_forward);
//			}else if(in.GetRows() ==out.GetRows() && in.GetColumns() == out.GetColumns() )
//			{
//				int rank = 1;
//				int howmany =  in.GetRows();
//				int n[] = {in.GetColumns() };
//				int *inembed = NULL;
//				int *onembed = NULL;
//				int idist = in.GetColumns();
//				int odist = in.GetColumns();
//				int istride = 1;
//				int ostride = 1;
//				fftw_plan plan_forward;
//				plan_forward = fftw_plan_many_dft( rank,n,howmany,
//                                               (fftw_complex*)(in.data_),inembed,istride,idist,
//                                               (fftw_complex*)(out.data_),onembed,ostride,odist,
//                                                FFTW_FORWARD, FFTW_ESTIMATE );
//				fftw_execute ( plan_forward );
//				fftw_destroy_plan(plan_forward);
//            
//        }else
//            throw DifferentSizeFFT();
//        
//    }else
//        throw DifferentSizeFFT();
//        
//}
//void FFT(const Matrix<double2> &in,Matrix<double2> &out)
//{
//    //cuda double2 type seems to be  bit compatible to fftw_complex
//    if(in.GetNumElements() == out.GetNumElements())
//    {    
//        //two vector
//        if(in.IsAvector() && out.IsAvector())
//        {
//            fftw_plan plan_forward;
//            plan_forward = fftw_plan_dft_1d ( in.GetNumElements(), (fftw_complex*)(in.data_),(fftw_complex*)(out.data_), FFTW_FORWARD, FFTW_ESTIMATE );
//            fftw_execute ( plan_forward );
//            fftw_destroy_plan(plan_forward);
//        }else if(in.GetRows() ==out.GetRows() && in.GetColumns() == out.GetColumns() )
//        {
//            int rank = 1;
//            int howmany =  in.GetRows();
//            int n[] = {in.GetColumns() };
//            int *inembed = NULL;
//            int *onembed = NULL;
//            int idist = in.GetColumns();
//            int odist = in.GetColumns();
//            int istride = 1;
//            int ostride = 1;
//            fftw_plan plan_forward;
//            plan_forward = fftw_plan_many_dft( rank,n,howmany,
//                                               (fftw_complex*)(in.data_),inembed,istride,idist,
//                                               (fftw_complex*)(out.data_),onembed,ostride,odist,
//                                                FFTW_FORWARD, FFTW_ESTIMATE );
//            fftw_execute ( plan_forward );
//            fftw_destroy_plan(plan_forward);
//            
//        }else
//            throw DifferentSizeFFT();
//        
//    }else
//        throw DifferentSizeFFT();
//        
//}
//
//
//
//
//void FFT2(const Matrix<double2> &in,Matrix<double2> &out)
//{
//    //cuda double2 type seems to be  bit compatible to fftw_complex
//    if(in.GetRows() == out.GetRows()  && in.GetColumns() == out.GetColumns())
//    {    
//        
//            fftw_plan plan_forward;
//            plan_forward = fftw_plan_dft_2d ( in.GetRows(),in.GetColumns(), (fftw_complex*)(in.data_),(fftw_complex*)(out.data_), FFTW_FORWARD, FFTW_ESTIMATE );
//            fftw_execute ( plan_forward );
//            fftw_destroy_plan(plan_forward);        
//        
//    }else
//        throw DifferentSizeFFT();
//        
//}
//
//
//template <class T>
//void FFT2(const Matrix<T> &in_d, Matrix<double2> &out)
//{
//    Matrix<double2> in(in_d.GetRows(),in_d.GetColumns());
//    in = in_d +0*J;
//    //cuda double2 type seems to be  bit compatible to fftw_complex
//    if(in.GetRows() == out.GetRows()  && in.GetColumns() == out.GetColumns())
//    {    
//        
//            fftw_plan plan_forward;
//            plan_forward = fftw_plan_dft_2d ( in.GetRows(),in.GetColumns(), (fftw_complex*)(in.data_),(fftw_complex*)(out.data_), FFTW_FORWARD, FFTW_ESTIMATE );
//            fftw_execute ( plan_forward );
//            fftw_destroy_plan(plan_forward);        
//        
//    }else
//        throw DifferentSizeFFT();
//    
//        
//}
//
//template <class Q,class T >
//void FFT2(const Expr<Q,T> &e,Matrix<double2> &out)
//{
//    Matrix<double2> in(e.GetRows(),e.GetColumns());
//    in = e +0*J;
//    
//    //cuda double2 type seems to be  bit compatible to fftw_complex
//    if(in.GetRows() == out.GetRows()  && in.GetColumns() == out.GetColumns())
//    {    
//        
//            fftw_plan plan_forward;
//            plan_forward = fftw_plan_dft_2d ( in.GetRows(),in.GetColumns(), (fftw_complex*)(in.data_),(fftw_complex*)(out.data_), FFTW_FORWARD, FFTW_ESTIMATE );
//            fftw_execute ( plan_forward );
//            fftw_destroy_plan(plan_forward);        
//        
//    }else
//        throw DifferentSizeFFT();
//        
//}
//
//////FIXME: this function should perform r2c fft to save memory and time
////void FFT(const Matrix<double> &in_d,Matrix<double2> &out)
////{
////    Matrix<double2> in(in_d.GetRows(),in_d.GetColumns());
////    in = in_d +0*J;
////    
////    //cuda double2 type seems to be  bit compatible to fftw_complex
////    if(in.GetNumElements() == out.GetNumElements())
////    {    
////        //two vector
////        if(in.IsAvector() && out.IsAvector())
////        {
////            fftw_plan plan_forward;
////            plan_forward = fftw_plan_dft_1d ( in.GetNumElements(), (fftw_complex*)(in.data_),(fftw_complex*)(out.data_), FFTW_FORWARD, FFTW_ESTIMATE );
////            fftw_execute ( plan_forward );
////            fftw_destroy_plan(plan_forward);
////        }else if(in.GetRows() ==out.GetRows() && in.GetColumns() == out.GetColumns() )
////        {
////            int rank = 1;
////            int howmany =  in.GetRows();
////            int n[] = {in.GetColumns() };
////            int *inembed = NULL;
////            int *onembed = NULL;
////            int idist = in.GetColumns();
////            int odist = in.GetColumns();
////            int istride = 1;
////            int ostride = 1;
////            fftw_plan plan_forward;
////            plan_forward = fftw_plan_many_dft( rank,n,howmany,
////                                               (fftw_complex*)(in.data_),inembed,istride,idist,
////                                               (fftw_complex*)(out.data_),onembed,ostride,odist,
////                                                FFTW_FORWARD, FFTW_ESTIMATE );
////            fftw_execute ( plan_forward );
////            fftw_destroy_plan(plan_forward);
////            
////        }else
////            throw DifferentSizeFFT();
////        
////    }else
////        throw DifferentSizeFFT();
////        
////}
//
//template <class Q,class T >
//void FFT(const Expr<Q,T> &e,Matrix<double2> &out)
//{
//    Matrix<double2> in(e.GetRows(),e.GetColumns());
//    in = e +0*J;
//    
//    //cuda double2 type seems to be  bit compatible to fftw_complex
//    if(in.GetNumElements() == out.GetNumElements())
//    {    
//        //two vector
//        if(in.IsAvector() && out.IsAvector())
//        {
//            fftw_plan plan_forward;
//            plan_forward = fftw_plan_dft_1d ( in.GetNumElements(), (fftw_complex*)(in.data_),(fftw_complex*)(out.data_), FFTW_FORWARD, FFTW_ESTIMATE );
//            fftw_execute ( plan_forward );
//            fftw_destroy_plan(plan_forward);
//        }else if(in.GetRows() ==out.GetRows() && in.GetColumns() == out.GetColumns() )
//        {
//            int rank = 1;
//            int howmany =  in.GetRows();
//            int n[] = {in.GetColumns() };
//            int *inembed = NULL;
//            int *onembed = NULL;
//            int idist = in.GetColumns();
//            int odist = in.GetColumns();
//            int istride = 1;
//            int ostride = 1;
//            fftw_plan plan_forward;
//            plan_forward = fftw_plan_many_dft( rank,n,howmany,
//                                               (fftw_complex*)(in.data_),inembed,istride,idist,
//                                               (fftw_complex*)(out.data_),onembed,ostride,odist,
//                                                FFTW_FORWARD, FFTW_ESTIMATE );
//            fftw_execute ( plan_forward );
//            fftw_destroy_plan(plan_forward);
//            
//        }else
//            throw DifferentSizeFFT();
//        
//    }else
//        throw DifferentSizeFFT();
//        
//}
//
////for Matrix
//template <template<class>class A,class T>
//Expr<FFTExpr<const class A<T> &,T>,T> FFTShift(const class A<T> &a)
//{
//    typedef FFTExpr<const class A<T> &,T> FEx;
//    return Expr<FEx,T>(FEx(a));
//}
//
//
//
//
////for Expr
//template <template<class,class>class A,class Q,class T>
//Expr<FFTExpr<const class A<Q,T> &,T>,T> FFTShift(const class A<Q,T> &a)
//{
//    typedef FFTExpr<const class A<Q,T> &,T> FEx;
//    return Expr<FEx,T>(FEx(a));
//}
//
////for Matrix
//template <template<class>class A,class T>
//Expr<IFFTExpr<const class A<T> &,T>,T> IFFTShift(const class A<T> &a)
//{
//    typedef IFFTExpr<const class A<T> &,T> FEx;
//    return Expr<FEx,T>(FEx(a));
//}
//
//
//
//
////for Expr
//template <template<class,class>class A,class Q,class T>
//Expr<IFFTExpr<const class A<Q,T> &,T>,T> IFFTShift(const class A<Q,T> &a)
//{
//    typedef IFFTExpr<const class A<Q,T> &,T> FEx;
//    return Expr<FEx,T>(FEx(a));
//}
//
//}
//
//} // namespace
////
////template <typename A,typename B>
////class IFFTExpr
////{
////private:
////    A a_;
////    int M_;
////    int N_;
////public:
////    IFFTExpr(const A &a) :a_(a),
////                        M_( (a.GetRows()%2) ? (GetRows())/2: (GetRows())/2),
////                        N_( (a.GetColumns()%2) ? (GetColumns())/2: (GetColumns())/2)  {}
////    int GetRows() const
////    {
////        return a_.GetRows();
////    }
////    int GetColumns() const
////    {
////        return a_.GetColumns();
////    }
////    int GetNumElements() const 
////    {
////        return a_.GetNumElements();
////    }
////    bool IsAvector()const  { return (GetRows() == 1 || GetColumns() == 1)?true:false;}
////    inline B operator()(const int i, const int j)const
////    {
////        return (*this)[IDX2R(i,j,GetColumns())];
////    }
////    inline B operator[](const int i) const
////    {
////        const int row    = i/GetColumns();
////        const int column = i%GetColumns();
////        const int newRow = (row+M_)%GetRows();
////        const int newColumn = (column+N_)%GetColumns();
////
////        
////        return a_[IDX2R(newRow,newColumn,GetColumns())];
////    }
////};
//
#endif