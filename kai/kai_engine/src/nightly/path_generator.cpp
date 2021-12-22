/*K
This file is part of Konantech AI Framework project
It is subject to the license terms in the LICENSE file found in the top-level directory of this distribution
*/
#include "path_generator.h"
#include <memory>		// for std::allocator<T>, std::uninitialized_fill(), std::uninitialized_copy()
#include <Windows.h>

CPathGenerator::CPathGenerator() {
	// TODO :
}

CPathGenerator::~CPathGenerator() {
	// TODO :
}

std::string CPathGenerator::GetSolutionPathA(unsigned int uiMaxLength) {
	// Declare the type of allocator
	std::allocator<char> alloc;

	// Initialize buffers
	char* szCurrentDir = alloc.allocate(uiMaxLength);
	char* szModuleFilename = alloc.allocate(uiMaxLength);
	std::uninitialized_fill(szCurrentDir, szCurrentDir + uiMaxLength, NULL);
	std::uninitialized_fill(szModuleFilename, szModuleFilename + uiMaxLength, NULL);

	// Get default paths
	GetCurrentDirectoryA(4096, szCurrentDir);
	GetModuleFileNameA(GetModuleHandleA(NULL), szModuleFilename, uiMaxLength);

	std::vector<std::string> tokens_current_dir, tokens_module_filename;
	ExtractTokensA(szCurrentDir, tokens_current_dir);
	ExtractTokensA(szModuleFilename, tokens_module_filename);

	// Finalize buffers
	alloc.deallocate(szCurrentDir, uiMaxLength);
	alloc.deallocate(szModuleFilename, uiMaxLength);

	// Reset an output string
	std::string rs = "";

	std::vector<std::string>::const_iterator i = tokens_current_dir.begin();
	std::vector<std::string>::const_iterator j = tokens_module_filename.begin();

	for (; i != tokens_current_dir.end() && j != tokens_module_filename.end() && i->compare(*j) == 0; ++i, ++j)
		rs += (*i + "/");

	if (rs.size() == 0)
		rs = "./";

	return rs;
}

void CPathGenerator::GetSolutionPathA(char* szDstBuffer, unsigned int uiMaxLength) {
	// Declare the type of allocator
	std::allocator<char> alloc;

	// Initialize buffers
	char* szCurrentDir = alloc.allocate(uiMaxLength);
	char* szModuleFilename = alloc.allocate(uiMaxLength);
	std::uninitialized_fill(szCurrentDir, szCurrentDir + uiMaxLength, NULL);
	std::uninitialized_fill(szModuleFilename, szModuleFilename + uiMaxLength, NULL);

	// Get default paths
	GetCurrentDirectoryA(4096, szCurrentDir);
	GetModuleFileNameA(GetModuleHandleA(NULL), szModuleFilename, uiMaxLength);

	// Define separators
	char separator[8] = "\\,/";

	// Declare a result buffer of tokenization
	std::vector<char*> tokens;

	// This tokenizer don't support multi-threading. (No thread safe)
	// Also, all of separator characters in original string "_String" will be changed to 'NULL'.
	char* token = std::strtok(szCurrentDir, separator);

	while (token) {
		tokens.push_back(token);
		token = std::strtok(NULL, separator);
	}

	// Reset an output string
	std::strcpy(szDstBuffer, "");

	std::vector<char*>::iterator it = tokens.begin();
	token = std::strtok(szModuleFilename, separator);

	while (it != tokens.end() && token != NULL && std::strcmp(*it, token) == 0) {
		// Update an output string
		std::strcat(szDstBuffer, *it);
		std::strcat(szDstBuffer, "/");

		// Re-tokenization
		token = std::strtok(NULL, separator);

		// Update an iterator
		++it;
	}

	//if(std::strlen(szDstBuffer)!=0 && szDstBuffer[std::strlen(szDstBuffer)-1] == '/')
	//	szDstBuffer[std::strlen(szDstBuffer)-1] = NULL;

	if (szDstBuffer[0] == NULL)
		std::strcpy(szDstBuffer, "./");

	// Finalize buffers
	alloc.deallocate(szCurrentDir, uiMaxLength);
	alloc.deallocate(szModuleFilename, uiMaxLength);
}

bool CPathGenerator::IsSeparatorA(char c) {
	switch (c) {
	case '\\':
		return true;
	case '/':
		return true;
	default:
		break;
	}

	return false;
}

bool CPathGenerator::IsNotSeparatorA(char c) {
	return !IsSeparatorA(c);
}

void CPathGenerator::ExtractTokensA(const char* szSrcString, std::vector<std::string>& tokens) {
	std::string str = szSrcString;

	std::string::iterator i = str.begin();

	while (i != str.end()) {
		// Skip parsers
		i = std::find_if(i, str.end(), IsNotSeparatorA);

		// Find the end of the token
		std::string::iterator j = std::find_if(i, str.end(), IsSeparatorA);

		// Get the token
		if (i != str.end())
			tokens.push_back(std::string(i, j));

		// Update an iterator
		i = j;
	}
}
