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


#ifndef __CUDABESSEL_CUH__
#define __CUDABESSEL_CUH__

namespace BB
{
	__host__ __device__ double fma2(double x, double y, double z) { return x*y+z; }

	/********************************************************/
	/* MODIFIED BESSEL FUNCTION CALCULATION DEVICE FUNCTION */
	/********************************************************/
	//__device__ double bessi0(double x)
	//{
	//// -- See paper
	//// J.M. Blair, "Rational Chebyshev approximations for the modified Bessel functions I_0(x) and I_1(x)", Math. Comput., vol. 28, n. 126, pp. 581-583, Apr. 1974.   

	//	double num, den, x2;

	//	x2 = CudaOpAbs::eval(x*x);
	//	x  = CudaOpAbs::eval(x);
 //  
	//	if (x > 15.0) 
	//	{
	//		den = 1.0 / x;
	//		num =                -4.4979236558557991E+006;
	//		num = fma (num, den,  2.7472555659426521E+006);
	//		num = fma (num, den, -6.4572046640793153E+005);
	//		num = fma (num, den,  8.5476214845610564E+004);
	//		num = fma (num, den, -7.1127665397362362E+003);
	//		num = fma (num, den,  4.1710918140001479E+002);
	//		num = fma (num, den, -1.3787683843558749E+001);
	//		num = fma (num, den,  1.1452802345029696E+000);
	//		num = fma (num, den,  2.1935487807470277E-001);
	//		num = fma (num, den,  9.0727240339987830E-002);
	//		num = fma (num, den,  4.4741066428061006E-002);
	//		num = fma (num, den,  2.9219412078729436E-002);
	//		num = fma (num, den,  2.8050629067165909E-002);
	//		num = fma (num, den,  4.9867785050221047E-002);
	//		num = fma (num, den,  3.9894228040143265E-001);
	//		num = num * den;
	//		den = sqrt (x);
	//		num = num * den;
	//		den = exp (0.5 * x);  /* prevent premature overflow */
	//		num = num * den;
	//		num = num * den;
	//		return num;
	//	}
	//	else
	//	{
	//		num = -0.27288446572737951578789523409E+010;
	//		num = fma (num, x2, -0.6768549084673824894340380223E+009);
	//		num = fma (num, x2, -0.4130296432630476829274339869E+008);
	//		num = fma (num, x2, -0.11016595146164611763171787004E+007);
	//		num = fma (num, x2, -0.1624100026427837007503320319E+005);
	//		num = fma (num, x2, -0.1503841142335444405893518061E+003);
	//		num = fma (num, x2, -0.947449149975326604416967031E+000);
	//		num = fma (num, x2, -0.4287350374762007105516581810E-002);
	//		num = fma (num, x2, -0.1447896113298369009581404138E-004);
	//		num = fma (num, x2, -0.375114023744978945259642850E-007);
	//		num = fma (num, x2, -0.760147559624348256501094832E-010);
	//		num = fma (num, x2, -0.121992831543841162565677055E-012);
	//		num = fma (num, x2, -0.15587387207852991014838679E-015);
	//		num = fma (num, x2, -0.15795544211478823152992269E-018);
	//		num = fma (num, x2, -0.1247819710175804058844059E-021);
	//		num = fma (num, x2, -0.72585406935875957424755E-025);
	//		num = fma (num, x2, -0.28840544803647313855232E-028);      
	//  
	//		den = -0.2728844657273795156746641315E+010;
	//		den = fma (den, x2, 0.5356255851066290475987259E+007);
	//		den = fma (den, x2, -0.38305191682802536272760E+004);
	//		den = fma (den, x2, 0.1E+001);

	//		return num/den;
	//	}
	//}

	/********************************************************/
	/* MODIFIED BESSEL FUNCTION CALCULATION DEVICE FUNCTION */
	/********************************************************/
	__host__ __device__ double bessi0(double x)
	{
	// -- See paper
	// J.M. Blair, "Rational Chebyshev approximations for the modified Bessel functions I_0(x) and I_1(x)", Math. Comput., vol. 28, n. 126, pp. 581-583, Apr. 1974.   

		double num, den, x2;

		x2 = x*x;
		x  = fabs(x);
   
		if (x > 15.0) 
		{
			den = 1.0 / x;
			num =				  -4.4979236558557991E+006;
			num = fma2 (num, den,  2.7472555659426521E+006);
			num = fma2 (num, den, -6.4572046640793153E+005);
			num = fma2 (num, den,  8.5476214845610564E+004);
			num = fma2 (num, den, -7.1127665397362362E+003);
			num = fma2 (num, den,  4.1710918140001479E+002);
			num = fma2 (num, den, -1.3787683843558749E+001);
			num = fma2 (num, den,  1.1452802345029696E+000);
			num = fma2 (num, den,  2.1935487807470277E-001);
			num = fma2 (num, den,  9.0727240339987830E-002);
			num = fma2 (num, den,  4.4741066428061006E-002);
			num = fma2 (num, den,  2.9219412078729436E-002);
			num = fma2 (num, den,  2.8050629067165909E-002);
			num = fma2 (num, den,  4.9867785050221047E-002);
			num = fma2 (num, den,  3.9894228040143265E-001);
			num = num * den;
			den = sqrt (x);
			num = num * den;
			den = exp (0.5 * x);  /* prevent premature overflow */
			num = num * den;
			num = num * den;
			return num;
		}
		else
		{
			num = -0.27288446572737951578789523409E+010;
			num = fma2 (num, x2, -0.6768549084673824894340380223E+009);
			num = fma2 (num, x2, -0.4130296432630476829274339869E+008);
			num = fma2 (num, x2, -0.11016595146164611763171787004E+007);
			num = fma2 (num, x2, -0.1624100026427837007503320319E+005);
			num = fma2 (num, x2, -0.1503841142335444405893518061E+003);
			num = fma2 (num, x2, -0.947449149975326604416967031E+000);
			num = fma2 (num, x2, -0.4287350374762007105516581810E-002);
			num = fma2 (num, x2, -0.1447896113298369009581404138E-004);
			num = fma2 (num, x2, -0.375114023744978945259642850E-007);
			num = fma2 (num, x2, -0.760147559624348256501094832E-010);
			num = fma2 (num, x2, -0.121992831543841162565677055E-012);
			num = fma2 (num, x2, -0.15587387207852991014838679E-015);
			num = fma2 (num, x2, -0.15795544211478823152992269E-018);
			num = fma2 (num, x2, -0.1247819710175804058844059E-021);
			num = fma2 (num, x2, -0.72585406935875957424755E-025);
			num = fma2 (num, x2, -0.28840544803647313855232E-028);      
	  
			den = -0.2728844657273795156746641315E+010;
			den = fma2 (den, x2, 0.5356255851066290475987259E+007);
			den = fma2 (den, x2, -0.38305191682802536272760E+004);
			den = fma2 (den, x2, 0.1E+001);

			return num/den;
		}
	}

	//const double p[17]	= {	-0.27288446572737951578789523409*1e10, 
	//						-0.6768549084673824894340380223*1e9, 
	//						-0.4130296432630476829274339869*1e8, 
	//						-0.11016595146164611763171787004*1e7,
	//						-0.1624100026427837007503320319*1e5,
	//						-0.1503841142335444405893518061*1e3,
	//						-0.947449149975326604416967031*1e0,
	//						-0.4287350374762007105516581810*1e-2,
	//						-0.1447896113298369009581404138*1e-4,
	//						-0.375114023744978945259642850*1e-7,
	//						-0.760147559624348256501094832*1e-10,
	//						-0.121992831543841162565677055*1e-12,
	//						-0.15587387207852991014838679*1e-15,
	//						-0.15795544211478823152992269*1e-18,
	//						-0.1247819710175804058844059*1e-21,
	//						-0.72585406935875957424755*1e-25,
	//						-0.28840544803647313855232*1e-28};

	//const double q[4]	= { -0.2728844657273795156746641315*1e10,
	//						0.5356255851066290475987259*1e7,
	//						-0.38305191682802536272760*1e4,
 //							0.1*1e1};

	//const double cp[26]	= { 0.8013085461969871612106457692*1e0/2, 
	//					    0.17290977661213460475126976*1e-2, 
	//						0.174344303731276665678151*1e-4, 
	//						0.3391002544196612353807*1e-6,
	//						0.101386676244446816514*1e-7,
	//						0.4212987673401844259*1e-9,
	//						0.230587178626748963*1e-10,
	//						0.16147708215256485*1e-11,
	//						0.1430354372503558*1e-12,
	//						0.160493342152869*1e-13,
	//						0.22704188628639*1e-14,
	//						0.3839318408111*1e-15,
	//						0.675456929962*1e-16,
	//						0.91151381855*1e-17,
	//						-0.3729687231*1e-18,
	//						-0.8619437387*1e-18,
	//						-0.3466400954*1e-18,
	//						-0.706240530*1e-19,
	//						0.30556730*1e-20,
 //   						0.76603496*1e-20,
	//						0.23745789*1e-20,
	//						0.119838*1e-22,
	//						-0.2438946*1e-21,
	//						-0.720868*1e-22,
	//						0.69870*1e-23,
	//						0.96880*1e-23};

	///********************************************************/
	///* MODIFIED BESSEL FUNCTION CALCULATION DEVICE FUNCTION */
	///********************************************************/
	//__host__ __device__ double bessi0(double x)
	//{
	//	// Valid only for |x|<=15.0 -- See paper
	//	// J.M. Blair, "Rational Chebyshev approximations for the modified Bessel functions I_0(x) and I_1(x)", Math. Comput., vol. 28, n. 126, pp. 581-583, Apr. 1974.   

	//	double num, den, x2, y, tn_2, tn_1, tn, tn_1_old;

	//	x2 = fabs(x*x);

	//	if (fabs(x) <= 15.0) 
	//	{
	//		num=p[0]+x2*(p[1]+x2*(p[2]+x2*(p[3]+x2*(p[4]+x2*(p[5]+x2*(p[6]+x2*(p[7]+x2*(p[8]+x2*(p[9]+x2*(p[10]+x2*(p[11]+x2*(p[12]+x2*(p[13]+x2*(p[14]+x2*(p[15]+p[16]*x2)))))))))))))));
	//		den=q[0]+x2*(q[1]+x2*(q[2]+q[3]*x2));
	//		return num/den;
	//	}
	//	else
	//	{
	//		y=30./x-1.;
	//		num=0.;
	//		tn_2=1.;
	//		num=cp[0]*tn_2;
	//		tn_1=y;
	//		num=num+cp[1]*tn_1;
	//		for (int k=2; k<=25; k++)
	//		{
	//			tn=2.*y*tn_1-tn_2;
	//			num=num+cp[k]*tn;
	//			tn_1_old=tn_1;
	//			tn_1=tn;
	//			tn_2=tn_1_old;
	//		}
	//		//return num*exp(x)/sqrt(x);
	//		return num*exp(x)*rsqrt(x);
	//	}
	//}

	//class CudaOpBessI0
	//{
	//	public:
	//		__device__ __host__ static inline float		eval(int a)		{ return bessi0((float)a); }
	//		__device__ __host__ static inline float		eval(float a)	{ return bessi0(a); }
	//		__device__ __host__ static inline double	eval(double a)	{ return bessi0(a); }
	//};

	//Function_on_Scalar_Promotion(T,CudaOpBessI0,bessi0);
	//Function_on_Matrix_Promotion(T,CudaOpBessI0,bessi0);
	//Function_on_Expression_Promotion(T,Q1,CudaOpBessI0,bessi0);

}

#endif