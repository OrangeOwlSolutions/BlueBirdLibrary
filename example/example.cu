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

using namespace std;

// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <conio.h>

#include <typeinfo>
#include <iostream>

// includes CUDA
#include <cuda.h>
#include <cuda_runtime.h>

#include "ComplexTypes.cuh"										
#include "ComplexTypes.cu"										
using namespace BB;

// includes Expression Templates library
#include "Macros.h"											
#include "HExceptions.h"										
#include "HExceptions.cpp"										
#include "DExceptions.cuh"										
#include "Constants.h"											
#include "DStreams.cuh"											
#include "DStreams.cu"											
#include "P2P.cu"
#include "SubIndicesAccessHandling.h"							
#include "SubIndicesAccessHandling.cpp"							
#include "DGlobals.cuh"											
#include "CudaExpressionMacros.cuh"								
////#include "CudaExpressionKernels.cuh"						
#include "Expression.cuh"										
//
//// includes CUDA Expression Templates library
#include "Scalar.cuh"											
#include "Promotion.cuh"									
#include "Hmatrix.h"												
#include "Hmatrix.cpp"												
#include "DMatrixExpressionKernels.cuh"						
#include "Dmatrix.cuh"										
#include "Dmatrix.cu"										
////#include "CudaMatrixExpression.cuh"
#include "Addition.cuh"
#include "Addition.cu"
#include "Subtraction.cuh"
#include "Subtraction.cu"
#include "Multiplication.cuh"
#include "Multiplication.cu"
#include "Division.cuh"
#include "Division.cu"
#include "Functions.cuh"
#include "Functions.cu"
#include "Utility.cuh"
#include "Grid.cuh"
#include "SubMatrixExpression.h"
#include "DFFT.cuh"
#include "DBLAS.cuh"
#include "TimingCPU.h"
#include "TimingCPU.cpp"
#include "TimingGPU.cuh"
#include "TimingGPU.cu"
#include "DReduction.cuh"
#include "HReduction.h"
#include "InputOutPut.h"
#include "Utility.h"

//#pragma comment(lib, "winmm.lib")

// Input/Output
// --- Overload of << for int2_
std::ostream & operator << (std::ostream&, const int2_&);
// --- Overload of << for float2_
std::ostream & operator << (std::ostream&, const float2_&);
// --- Overload of << for double2_
std::ostream & operator << (std::ostream&, const double2_&);
// --- Overload of << for type T Hmatrix (int, float, double, int2_, float2_, double2_)
template <class T> std::ostream& operator << (std::ostream&, const Hmatrix<T>&);
// --- Overload of << for type T Dmatrix (int, float, double, int2_, float2_, double2_)
template <class T> std::ostream & operator << (std::ostream&, const Dmatrix<T>&);

/**********************************/
/* OVERLOAD OF << FOR EXPRESSIONS */
/**********************************/
template <class Q, class T>
ostream & operator << (ostream & output, const Expr<Q,T> v)
{
	Hmatrix<T> a(v.GetRows(),v.GetColumns());
	if (v.IsDevice()) {
		Dmatrix<T> b(v.GetRows(),v.GetColumns());
		b = v;
		a = b; }
	else {
		a = v; }
	output << a;
	return output;   
}

// Constant to CPU Hmatrix assignment
template <class OutType>
const BB::Hmatrix<OutType>& BB::Hmatrix<OutType>::operator=(const OutType c)
{
	*this = BB::Expr<BB::Scalar<OutType>,OutType>(BB::Scalar<OutType>(c),Rows_,Columns_,ISHOST);
	return *this;
}

// Constant to GPU Hmatrix assignment
template <class OutType>
const BB::Dmatrix<OutType>& BB::Dmatrix<OutType>::operator=(const OutType c)
{
	*this = BB::Expr<BB::Scalar<OutType>,OutType>(BB::Scalar<OutType>(c),Rows_,Columns_,ISDEVICE);
	return *this;
}

void main( int argc, char** argv) 
{
	// --- You should always initialize the streams and set the active stream before doing anything else
	streams[0].InitStreams(1);
	streams[0].SetStream(0);

	// --- Introduction of this scoping is needed to avoid issues with class destructors when using cudaDeviceReset();
	{
		int NumRows		= 4;
		int NumColumns	= 4;
		int NumElements = NumRows*NumColumns;

		// --- The types the library can deal with are int, float, double, int2_, float2_, double2_ (the latter three being complex types)
		
		// --- Defining GPU matrices is simple...
		Dmatrix<double2_>		A_D(NumRows,NumColumns);
		Dmatrix<double2_>		B_D(NumRows,NumColumns);
		Dmatrix<double2_>		C_D(NumRows,NumColumns);

		// --- Defining CPU matrices is as simple as for GPU matrices...
		Hmatrix<double2_>		A_H(NumRows,NumColumns);
		Hmatrix<double2_>		B_H(NumRows,NumColumns);
		Hmatrix<double2_>		C_H(NumRows,NumColumns);

		// --- The default type DefaultType is set in Constant.h. You can change it. Useful to simplify the syntax. If you want to define a matrix of
		//     type DefaultType, then just use, for example, Dmatrix<> A_H(NumRows,NumColumns);
		
		// --- Setting matrices to 0s or 1s is simple and uses the same syntax for CPU and GPU matrices
		A_H = SetOnes(NumRows,NumColumns);  // SetOnes is Matlab's ones equivalent. If SetOnes does not return the DefaultType, use for instance SetOnes<float>(NumRows,NumColumns);
		B_D = SetZeros(NumRows,NumColumns); // SetZeros is Matlab's zeros equivalent. If SetZeros does not return the DefaultType, use for instance SetOnes<float>(NumRows,NumColumns);

		// --- You can use cout to display the results
		cout << "Displaying A_H " << endl << endl;
		cout << A_H << endl;
		cout << "Displaying B_H " << endl << endl;
		cout << B_D << endl;

		// --- Moving matrices from CPU to GPU and viceversa...
		B_H = B_D;
		A_D = A_H;
	
		// --- Assigning matrices from CPU to CPU and from GPU to GPU. All the possible assignments from real to real (e.g., float = int) are defined. 
		//     Assignments from real to complex (double2_ = float) are also defined.
		C_H = B_H;
		C_D = B_H;

		// --- You can initialize CPU or GPU matrices on expressions or on other CPU or GPU matrices (CPU on GPU and GPU on CPU initializations are possible)
		Hmatrix<float>		D_H(EqSpace<float>(3,NumElements+3,NumElements),PINNED);   // uses PINNED memory for D_H. This is important for async
																					   // global memory transactions.
		                                                                               // EqSpace is Matlab's linspace equivalent.
		Dmatrix<float>		D_D(D_H);
		
		// --- You can mimic the Matlab's ":" operator 
		Dmatrix<float>		E_D(Colon<float>(1,4));  // Equivalent of Matlab's 1:4;
		Hmatrix<float>		E_H(Colon<float>(1,2,4));  // Equivalent of Matlab's 1:2:4;

		// --- You can read individual elements for CPU and GPU matrices ...
		cout << E_D(0) << endl;
		cout << E_H(0) << endl;

		// --- ... similarly, you can write individual elements for CPU matrices ...
		E_H(0)=3;

		// --- ... and for GPU matrices.
		cout << "Before the assignment: E_D(0) = " << E_D(0) << "\n\n";
		E_D(0)=3.;
		cout << "After the assignment: E_D(0) = " << E_D(0) << "\n\n";
		cout << "Before the assignment: A_D(0) = " << A_D(0) << "\n\n";
		A_D(0)=double2_(33.,0.);
		cout << "After the assignment: A_D(0) = " << A_D(0) << "\n\n";

		// --- You can resize both CPU and GPU matrices, for example as
		cout << "Original number of rows = " << A_D.GetRows() << "\n\n";
		cout << "Original number of columns = " << A_D.GetColumns() << "\n\n";
		A_D.Resize(1,NumElements);
		cout << "New number of rows = " << A_D.GetRows() << "\n\n";
		cout << "New number of columns = " << A_D.GetColumns() << "\n\n";
	
		// --- You can create grids on CPU or GPU with a syntax similar to Matlab's meshgrid.
		//     H_D=GridX(F_D,G_D); requires both F_D and G_D to be vectors. Then it fills a matrix H_D of size length(G_D)xlength(F_D) that replicates
		//                         the vector F_D along the rows.
		Dmatrix<float>		F_D(EqSpace<float>(1,4,4));
		Dmatrix<float>		G_D(EqSpace<float>(1,3,3));
		Dmatrix<float>		H_D(3,4);
		H_D = GridX(F_D,G_D);
		cout << H_D << "\n";
		//     I_D=GridX(F_D,G_D); requires both F_D and G_D to be vectors. Then it fills a matrix I_D of size length(G_D)xlength(F_D) that replicates
		//                         the vector G_D along the columns.
		Dmatrix<float>		I_D(3,4);
		I_D = GridY(F_D,G_D);
		cout << I_D << "\n";

		tie(H_D,I_D)=Grid(F_D,G_D);
		cout << H_D << "\n";
		cout << I_D << "\n";
	
		// --- You can easily time your applications by CPU and GPU timers.
		TimingCPU	timerCPU;
		TimingGPU	timerGPU;

		timerCPU.StartCounter();
		A_H = B_H * C_H;
		cout << "CPU timing = " << timerCPU.GetCounter() << " ms\n";
	
		timerGPU.StartCounter();
		A_D.Resize(NumRows,NumColumns);
		try { A_D = B_D * C_D; } catch(exception &e) { cout << e.what() << endl; getch(); return; }
		cout << "GPU timing = " << timerGPU.GetCounter() << " ms\n";

		// --- You can perform FFTs as (FFT is currently limited only to GPU) - the FFT is executed in a proper stream
		A_D = SetOnes(NumRows,NumColumns);
		B_D = FFT(A_D);		// in this case, the plan is calculated internally to the FFT routine
		cout << B_D << "\n";

		cufftHandle plan = DEVICE_FFT_2D_PLAN_Z2Z(A_D.GetRows(),A_D.GetColumns()); // in this case, you explicitly calculate the plan and possibly reuse it
		B_D = FFT(A_D,plan);
		cout << B_D << "\n";

		// --- Invere FFTs are also possible
		B_D = (1./(NumRows*NumColumns))*IFFT(B_D,plan);
		cout << B_D << "\n";

		// --- You can also calculate FFTs of expressions
		B_D = FFT(sin(3.*B_D));
		cout << B_D << "\n";
		
		DEVICE_FFT_DESTROY_PLAN(plan);

		// --- You can easily perform matrix-matrix multiplications as (handle automatically created and destroyed) ...
		Dmatrix<float2_>  L_D(3,5);
		Dmatrix<float2_>  M_D(5,1);
		Dmatrix<float2_>  N_D(3,1);

		L_D = SetOnes<float2_>(3,5);
		M_D = SetOnes<float2_>(5,1);
		
		try { N_D = MatMul(L_D,M_D); } catch(exception &e) { cout << e.what() << endl; getch(); return; }
		cout << N_D << "\n";
	
		// --- ... or (handle manually created and destroyed)
		cublasHandle_t handle = DEVICE_BLAS_CREATE_HANDLE();
		N_D = MatMul(sin(L_D),cos(M_D),handle);
		DEVICE_BLAS_DESTROY_HANDLE(handle);

		cout << N_D << "\n";

		// --- You can output expressions with cout
		cout << sin(L_D) << endl;
		Hmatrix<float2_>	L_H(L_D);
		cout << sin(L_H) << endl;

		// --- Extracting Sub-Expressions (Range)
		Hmatrix<float2_> N_H(N_D);
		Hmatrix<float2_> M_H(M_D);
		N_H.Resize(1,3);
		N_H = M_H(Range(0,2));

		N_D.Resize(1,3);
		N_D = M_D(Range(0,2));

		cout << sin(N_D)(Range(0,2)) << endl;

		// --- Extracting Sub-Expressions (int-Range)
		M_H.Resize(1,5);
		M_D.Resize(1,5);
		N_H = M_H(0,Range(0,2));
		N_D = M_D(0,Range(0,2));

		cout << sin(N_D)(0,Range(0,2)) << endl;

		// --- Extracting Sub-Expressions (Range-Int)
		Hmatrix<float2_> O_H(5,4);
		Hmatrix<float2_> P_H(3,1);

		O_H(0,0) = 0.; O_H(1,0) = 1.; O_H(2,0) = 2.; O_H(3,0) = 3.; O_H(4,0) = 4.;
		O_H(0,1) = 5.; O_H(1,1) = 6.; O_H(2,1) = 7.; O_H(3,1) = 8.; O_H(4,1) = 9.;
		O_H(0,2) = 10.; O_H(1,2) = 11.; O_H(2,2) = 12.; O_H(3,2) = 13.; O_H(4,2) = 14.;
		O_H(0,3) = 15.; O_H(1,3) = 16.; O_H(2,3) = 17.; O_H(3,3) = 18.; O_H(4,3) = 19.;
		P_H = O_H(Range(1,3),2);
		Dmatrix<float2_> O_D(O_H);
		Dmatrix<float2_> P_D(P_H);
		P_D = O_D(Range(1,3),0);

		cout << sin(O_D)(Range(0,2),0) << "\n"; 

		// --- Extracting Sub-Expressions (Range-Range)
		Hmatrix <float2_> Q_H(3,2);
		Dmatrix <float2_> Q_D(3,2);
		Q_H = O_H(Range(2,4),Range(1,2));
		Q_D = O_D(Range(2,4),Range(1,2));
		cout << sin(O_H)(Range(2,4),Range(1,2)) << "\n"; 

		// --- Extracting Sub-Expressions (Span-int)
		Hmatrix <float2_> R_H(5,1);
		Dmatrix <float2_> R_D(5,1);
		R_H = O_H(Span,2);
		R_D = O_D(Span,1);
		cout << sin(O_D)(Span,2) << endl;

		// --- Extracting Sub-Expressions (int-Span)
		Hmatrix <float2_> S_H(1,4);
		Dmatrix <float2_> S_D(1,4);
		S_H = O_H(1,Span);
		S_D = O_D(2,Span);
		cout << sin(O_D)(2,Span) << endl;

		// --- Extracting Sub-Expressions (int-RangeStep)
		Hmatrix <float2_> T_H(1,2);
		Dmatrix <float2_> T_D(1,2);
		T_H = O_H(1,RangeStep(0,2,3));
		T_D = O_D(1,RangeStep(0,2,3));
		cout << sin(O_H)(1,RangeStep(0,2,3)) << endl;

		// --- Extracting Sub-Expressions (RangeStep-int)
		T_H.Resize(2,1);
		T_D.Resize(2,1);
		T_H = O_H(RangeStep(1,2,4),1);
		T_D = O_D(RangeStep(1,2,4),1);
		cout << sin(O_D)(RangeStep(1,2,4),1) << endl;

		// --- Extracting Sub-Expressions (Range-RangeStep)
		Q_H = O_H(Range(1,3),RangeStep(0,2,3));
		Q_D = O_D(Range(1,3),RangeStep(0,2,3));
		cout << cos(O_H)(Range(1,3),RangeStep(0,2,3)) << endl;

		// --- Extracting Sub-Expressions (RangeStep-Range)
		Q_H = O_H(RangeStep(0,2,4),Range(1,2));
		Q_D = O_D(RangeStep(0,2,4),Range(1,2));
		cout << cos(O_D)(RangeStep(0,2,4),Range(1,2)) << endl;

		// --- Extracting Sub-Expressions (RangeStep-RangeStep)
		Q_H = O_H(RangeStep(0,2,4),RangeStep(0,2,3));
		Q_D = O_D(RangeStep(0,2,4),RangeStep(0,2,3));
		cout << cos(O_D)(RangeStep(0,2,4),RangeStep(0,2,3)) << endl;

		// --- Extracting Sub-Expressions (RangeStep-RangeStep)
		Q_H = O_H(RangeStep(0,2,4),RangeStep(0,2,3));
		Q_D = O_D(RangeStep(0,2,4),RangeStep(0,2,3));
		cout << cos(O_D)(RangeStep(0,2,4),RangeStep(0,2,3)) << endl;

		// --- Extracting Sub-Expressions (Range-Span)
		Hmatrix<float2_> U_H(3,4);
		Dmatrix<float2_> U_D(3,4);
		U_H = O_H(Range(0,2),Span);
		U_D = O_D(Range(0,2),Span);
		cout << cos(O_H)(Range(0,2),Span) << endl;

		// --- Extracting Sub-Expressions (Span-Range)
		Hmatrix<float2_> V_H(5,3);
		Dmatrix<float2_> V_D(5,3);
		V_H = O_H(Span,Range(0,2));
		V_D = O_D(Span,Range(0,2));
		cout << sin(O_D)(Span,Range(0,2)) << endl;

		// --- Extracting Sub-Expressions (Span-RangeStep)
		Hmatrix<float2_> W_H(5,2);
		Dmatrix<float2_> W_D(5,2);
		W_H = O_H(Span,RangeStep(0,2,3));
		W_D = O_D(Span,RangeStep(0,2,3));
		cout << sin(O_H)(Span,RangeStep(0,2,3)) << endl;

		// --- Extracting Sub-Expressions (RangeStep-Span)
		Hmatrix<float2_> X_H(2,4);
		Dmatrix<float2_> X_D(2,4);
		X_H = O_H(RangeStep(0,2,3),Span);
		X_D = O_D(RangeStep(0,2,3),Span);
		cout << sin(O_H)(RangeStep(0,2,3),Span) << endl;
		
		//Hmatrix<float2_> W_H(5,2);
		//Dmatrix<float2_> W_D(5,2);
		//W_H = O_H(Span,RangeStep(0,2,3));
		//W_D = O_D(Span,RangeStep(0,2,3));

		// --- Reduction (+) - real case - Dmatrix
		Hmatrix<double> ar(1,20);
		for (unsigned i=0; i<20; ++i) {
			ar(i) = 1;
		}
		Dmatrix<double> br(ar);
		double sumr = SumAll(sin(br));
		double sumrCPU = SumAll(sin(ar));

		cout << "CPU reduction result: " << sumrCPU << endl;
		cout << "GPU reduction result: " << sumr << endl;

		// --- Reduction (+) - complex case - Dmatrix
		Hmatrix<double2_> ac(1,20);
		for (unsigned i=0; i<20; ++i) {
			ac(i).c.x = 1;
			ac(i).c.y = 2;
		}
		Dmatrix<double2_> bc(ac);
		double2_ sumc = SumAll(bc);
		double2_ sumcCPU = SumAll(ac);

		cout << "CPU reduction result: real part = " << sumcCPU.c.x << "; imaginary part = " << sumcCPU.c.y << endl;
		cout << "GPU reduction result: real part = " << sumcCPU.c.x << "; imaginary part = " << sumcCPU.c.y << endl;

	}
	
	cudaDeviceReset();

	std::cout << "Going to sleep" << std::endl;
	getch();
}

