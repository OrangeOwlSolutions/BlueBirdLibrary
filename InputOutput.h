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


#ifndef __INPUTOUTPUT_H__
#define __INPUTOUTPUT_H__

#include <fstream>

namespace BB
{
	// --- Load individual real matrix from txt file
	template <class T>
	Hmatrix<T> loadrealtxt(const char* filename, int Rows, int Columns)
	{
		Hmatrix<T> out(Rows,Columns);

		ifstream infile;
		infile.open(filename);
		for(int i=0; i<Rows*Columns; i++) infile >> out.GetDataPointer()[i]; 
		infile.close();

		return out;
	}

	// --- Load individual complex matrix from txt file
	template <class T>
	Hmatrix<T> loadcomplextxt(const char* filename, int Rows, int Columns)
	{
		Hmatrix<T> out(Rows,Columns);

		ifstream infile;
		infile.open(filename);
		for(int i=0; i<Rows*Columns; i++) { infile >> out.GetDataPointer()[i].c.x; infile >> out.GetDataPointer()[i].c.y; }
		infile.close();

		return out;
	}

	// --- Load individual complex matrix from txt file - DefaultType
	Hmatrix<DefaultType> loadcomplextxt(const char* filename, int Rows, int Columns)
	{
		Hmatrix<DefaultType> out(Rows,Columns);

		ifstream infile;
		infile.open(filename);
		for(int i=0; i<Rows*Columns; i++) { infile >> out.GetDataPointer()[i].c.x; infile >> out.GetDataPointer()[i].c.y; }
		infile.close();

		return out;
	}

	// --- Save individual real CPU matrix to txt file
	template <class T>
	void saverealtxt(const Hmatrix<T>& ob, const char* filename)
	{
		ofstream outfile;
		outfile.open(filename);
		for(int i=0; i<ob.GetNumElements(); i++) outfile << std::setprecision(prec_save) << ob.GetDataPointer()[i] << "\n"; 
		outfile.close();
	}

	// --- Save individual real CPU matrix to txt file - DefaultType
	void saverealtxt(const Hmatrix<DefaultType>& ob, const char* filename)
	{
		ofstream outfile;
		outfile.open(filename);
		for(int i=0; i<ob.GetNumElements(); i++) outfile << std::setprecision(prec_save) << ob.GetDataPointer()[i] << "\n"; 
		outfile.close();
	}

	// --- Save individual real GPU matrix to txt file
	template <class T>
	void saverealtxt(const Dmatrix<T>& obGPU, const char* filename)
	{
		Hmatrix<T> ob(obGPU);
		ofstream outfile;
		outfile.open(filename);
		for(int i=0; i<ob.GetNumElements(); i++) outfile << std::setprecision(prec_save) << ob.GetDataPointer()[i] << "\n"; 
		outfile.close();
	}

	// --- Save individual real GPU matrix to txt file - DefaultType
	void saverealtxt(const Dmatrix<DefaultType>& obGPU, const char* filename)
	{
		Hmatrix<DefaultType> ob(obGPU);
		ofstream outfile;
		outfile.open(filename);
		for(int i=0; i<ob.GetNumElements(); i++) outfile << std::setprecision(prec_save) << ob.GetDataPointer()[i] << "\n"; 
		outfile.close();
	}

	// --- Save individual real matrix to txt file - Expressions
	template <class Q, class T>
	void saverealtxt(const Expr<Q,T> e, const char* filename)
	{
		Hmatrix<T> ob(e.GetRows(),e.GetColumns());
		if (e.IsDevice()) {
			Dmatrix<T> b(e.GetRows(),e.GetColumns());
			b = e;
			ob = b; }
		else {
			ob = e; }
		ofstream outfile;
		outfile.open(filename);
		for(int i=0; i<ob.GetNumElements(); i++) outfile << std::setprecision(prec_save) << ob.GetDataPointer()[i] << "\n"; 
		outfile.close();
	}

	// --- Save individual complex CPU matrix to txt file
	template <class T>
	void savecomplextxt(const Hmatrix<T>& ob, const char* filename)
	{
		ofstream outfile;
		outfile.open(filename);
		for(int i=0; i<ob.GetNumElements(); i++) { outfile << std::setprecision(prec_save) << ob.GetDataPointer()[i].c.x << "\n"; outfile << std::setprecision(prec_save) << ob.GetDataPointer()[i].c.y << "\n"; }
		outfile.close();
	}

	// --- Save individual complex CPU matrix to txt file - DefaultType
	void savecomplextxt(const Hmatrix<DefaultType>& ob, const char* filename)
	{
		ofstream outfile;
		outfile.open(filename);
		for(int i=0; i<ob.GetNumElements(); i++) { outfile << std::setprecision(prec_save) << ob.GetDataPointer()[i].c.x << "\n"; outfile << std::setprecision(prec_save) << ob.GetDataPointer()[i].c.y << "\n"; }
		outfile.close();
	}

	// --- Save individual complex GPU matrix to txt file 
	template <class T>
	void savecomplextxt(const Dmatrix<T>& obGPU, const char* filename)
	{
		Hmatrix<T> ob(obGPU);
		ofstream outfile;
		outfile.open(filename);
		for(int i=0; i<ob.GetNumElements(); i++) { outfile << std::setprecision(prec_save) << ob.GetDataPointer()[i].c.x << "\n"; outfile << std::setprecision(prec_save) << ob.GetDataPointer()[i].c.y << "\n"; }
		outfile.close();
	}

	// --- Save individual complex GPU matrix to txt file - DefaultType
	void savecomplextxt(const Dmatrix<DefaultType>& obGPU, const char* filename)
	{
		Hmatrix<DefaultType> ob(obGPU);
		ofstream outfile;
		outfile.open(filename);
		for(int i=0; i<ob.GetNumElements(); i++) { outfile << std::setprecision(prec_save) << ob.GetDataPointer()[i].c.x << "\n"; outfile << std::setprecision(prec_save) << ob.GetDataPointer()[i].c.y << "\n"; }
		outfile.close();
	}

	// --- Save individual complex matrix to txt file - Expressions
	template <class Q, class T>
	void savecomplextxt(const Expr<Q,T> e, const char* filename)
	{
		Hmatrix<T> ob(e.GetRows(),e.GetColumns());
		if (e.IsDevice()) {
			Dmatrix<T> b(e.GetRows(),e.GetColumns());
			b = e;
			ob = b; }
		else {
			ob = e; }
		ofstream outfile;
		outfile.open(filename);
		for(int i=0; i<ob.GetNumElements(); i++) { outfile << std::setprecision(prec_save) << ob.GetDataPointer()[i].c.x << "\n"; outfile << std::setprecision(prec_save) << ob.GetDataPointer()[i].c.y << "\n"; }
		outfile.close();
	}

}

#endif