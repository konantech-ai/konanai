/*K
This file is part of Konantech AI Framework project
It is subject to the license terms in the LICENSE file found in the top-level directory of this distribution
*/
// path_generator.h
#pragma once

// Inclusion of additional headers
#include <string>		// for std::string
#include <vector>		// for std::vector

/*
	Caution! This generator can only be used in the following environments.
	  - Windows system
	  - Multi-byte string (std::string, char*)
*/
typedef class CPathGenerator {
public:
	CPathGenerator();
	virtual ~CPathGenerator();

	// Get the solution path using C11 STL (thread safe)
	static std::string GetSolutionPathA(unsigned int uiMaxLength = 4096);

	// Get the solution path using std::strtok() (Don't support multi-threading)
	static void GetSolutionPathA(char* szDstBuffer, unsigned int uiMaxLength = 4096);

protected:
	// For GetSolutionPath()
	static bool IsSeparatorA(char c);
	static bool IsNotSeparatorA(char c);
	static void ExtractTokensA(const char* szSrcString, std::vector<std::string>& tokens);

} PathGenerator;
