/*
 * The Apache Software License, Version 1.1
 *
 * Copyright (c) 1999-2002 The Apache Software Foundation.  All rights
 * reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *
 * 1. Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in
 *    the documentation and/or other materials provided with the
 *    distribution.
 *
 * 3. The end-user documentation included with the redistribution,
 *    if any, must include the following acknowledgment:
 *       "This product includes software developed by the
 *        Apache Software Foundation (http://www.apache.org/)."
 *    Alternately, this acknowledgment may appear in the software itself,
 *    if and wherever such third-party acknowledgments normally appear.
 *
 * 4. The names "Xerces" and "Apache Software Foundation" must
 *    not be used to endorse or promote products derived from this
 *    software without prior written permission. For written
 *    permission, please contact apache\@apache.org.
 *
 * 5. Products derived from this software may not be called "Apache",
 *    nor may "Apache" appear in their name, without prior written
 *    permission of the Apache Software Foundation.
 *
 * THIS SOFTWARE IS PROVIDED ``AS IS'' AND ANY EXPRESSED OR IMPLIED
 * WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES
 * OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED.  IN NO EVENT SHALL THE APACHE SOFTWARE FOUNDATION OR
 * ITS CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
 * SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
 * LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF
 * USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT
 * OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF
 * SUCH DAMAGE.
 * ====================================================================
 *
 * This software consists of voluntary contributions made by many
 * individuals on behalf of the Apache Software Foundation, and was
 * originally based on software copyright (c) 1999, International
 * Business Machines, Inc., http://www.ibm.com .  For more information
 * on the Apache Software Foundation, please see
 * <http://www.apache.org/>.
 */

/*
 * $Log: VCPPDefs.hpp,v $
 * Revision 1.9  2003/05/29 13:52:36  gareth
 * fixed typo for version number
 *
 * Revision 1.8  2003/05/29 11:18:37  gareth
 * Added macros in so we can determine whether to do things like iostream as opposed to iostream.h and whether to use std:: or not.
 *
 * Revision 1.7  2002/11/04 14:45:20  tng
 * C++ Namespace Support.
 *
 * Revision 1.6  2002/06/25 16:05:24  tng
 * DOM L3: move the operator delete to DOMDocumentImpl.hpp
 *
 * Revision 1.5  2002/05/28 12:57:17  tng
 * Fix typo.
 *
 * Revision 1.4  2002/05/27 18:02:40  tng
 * define XMLSize_t, XMLSSize_t and their associate MAX
 *
 * Revision 1.3  2002/05/21 19:45:53  tng
 * Define DOMSize_t and XMLSize_t
 *
 * Revision 1.2  2002/04/17 20:30:01  tng
 * [Bug 7583] Build warnings with MS Visual Studio .NET.
 *
 * Revision 1.1.1.1  2002/02/01 22:22:19  peiyongz
 * sane_include
 *
 * Revision 1.13  2001/06/04 20:11:54  tng
 * IDOM: Complete IDNodeIterator, IDTreeWalker, IDNodeFilter.
 *
 * Revision 1.12  2001/06/04 13:45:06  tng
 * The "hash" argument clashes with STL hash.  Fixed by Pei Yong Zhang.
 *
 * Revision 1.11  2001/05/29 18:50:24  tng
 * IDOM: call allocate directly for array allocation to avoid overloading operator new[] which leads to compilation error on SUN CC 4.2
 *
 * Revision 1.10  2001/05/28 20:59:21  tng
 * IDOM: move operator new[] to VCPPDefs as only Windows VCPP requires its presense
 *
 * Revision 1.9  2001/05/23 20:35:03  tng
 * IDOM: Move operator delete to VCPPDefs.hpp as only VCPP needs a matching delete operator.
 *
 * Revision 1.8  2001/03/02 20:53:08  knoaman
 * Schema: Regular expression - misc. updates for error messages,
 * and additions of new functions to XMLString class.
 *
 * Revision 1.7  2000/06/16 21:13:23  rahulj
 * Add 'D' suffix to the library name for the 'DEBUG' build
 * configuration.
 *
 * Revision 1.6  2000/03/02 19:55:09  roddey
 * This checkin includes many changes done while waiting for the
 * 1.1.0 code to be finished. I can't list them all here, but a list is
 * available elsewhere.
 *
 * Revision 1.5  2000/02/06 07:48:18  rahulj
 * Year 2K copyright swat.
 *
 * Revision 1.4  2000/01/14 01:19:22  roddey
 * Added a define of XML_LSTRSUPPORT to indicate supoprt of L"" type
 * prefixes on this compiler.
 *
 * Revision 1.3  2000/01/14 00:51:30  roddey
 * Added the requested XMLStrL() macro to support some portable
 * optimization of DOM code. This still needs to be added to the other
 * per-compiler files.
 *
 * Revision 1.2  1999/11/10 21:26:14  abagchi
 * Changed the DLL name
 *
 * Revision 1.1.1.1  1999/11/09 01:07:41  twl
 * Initial checkin
 *
 * Revision 1.3  1999/11/08 20:45:25  rahul
 * Swat for adding in Product name and CVS comment log variable.
 *
 */

#if !defined(PGCPPDEFS_HPP)
#define PGCPPDEFS_HPP

// ---------------------------------------------------------------------------
//  Include some runtime files that will be needed product wide
// ---------------------------------------------------------------------------
#include <sys/types.h>  // for size_t and ssize_t
#include <limits.h>  // for MAX of size_t and ssize_t

// ---------------------------------------------------------------------------
//  A define in the build for each project is also used to control whether
//  the export keyword is from the project's viewpoint or the client's.
//  These defines provide the platform specific keywords that they need
//  to do this.
// ---------------------------------------------------------------------------
// PGI : this requires COMDAT support.  EDG makes static functions of
// DLLEXPORT classes : global , expecting Microsoft C COMDAT support
// for these functions.   We get multiple defines.
//define PLATFORM_EXPORT     __declspec(dllexport)
//define PLATFORM_IMPORT     __declspec(dllimport)
#define PLATFORM_EXPORT
#define PLATFORM_IMPORT

//define XALAN_PLATFORM_EXPORT 
//define XALAN_PLATFORM_IMPORT
//define XALAN_PLATFORM_EXPORT_FUNCTION(T) T XALAN_PLATFORM_EXPORT
//define XALAN_PLATFORM_IMPORT_FUNCTION(T) T XALAN_PLATFORM_IMPORT


// ---------------------------------------------------------------------------
//  Indicate that we do not support native bools
//  If the compiler can handle boolean itself, do not define it
// ---------------------------------------------------------------------------
// #define NO_NATIVE_BOOL

// ---------------------------------------------------------------------------
//  Each compiler might support L"" prefixed constants. There are places
//  where it is advantageous to use the L"" where it supported, to avoid
//  unnecessary transcoding.
//  If your compiler does not support it, don't define this.
// ---------------------------------------------------------------------------
#define XML_LSTRSUPPORT

// ---------------------------------------------------------------------------
//  Indicate that we support C++ namespace
//  Do not define it if the compile cannot handle C++ namespace
// ---------------------------------------------------------------------------
#define XERCES_HAS_CPP_NAMESPACE

// ---------------------------------------------------------------------------
//  Define our version of the XML character
// ---------------------------------------------------------------------------
typedef unsigned short XMLCh;

// ---------------------------------------------------------------------------
//  Define unsigned 16 and 32 bits integers
// ---------------------------------------------------------------------------
typedef unsigned short  XMLUInt16;
typedef unsigned int    XMLUInt32;

// ---------------------------------------------------------------------------
//  Define signed 32 bits integers
// ---------------------------------------------------------------------------
typedef int             XMLInt32;

// ---------------------------------------------------------------------------
//  XMLSize_t is the unsigned integral type.
// ---------------------------------------------------------------------------
#if defined(_SIZE_T) && defined(SIZE_MAX) && defined(_SSIZE_T) && defined(SSIZE_MAX)
    typedef size_t              XMLSize_t;
    #define XML_SIZE_MAX        SIZE_MAX
    typedef ssize_t             XMLSSize_t;
    #define XML_SSIZE_MAX       SSIZE_MAX
#else
    typedef unsigned long       XMLSize_t;
    #define XML_SIZE_MAX        ULONG_MAX
    typedef long                XMLSSize_t;
    #define XML_SSIZE_MAX       LONG_MAX
#endif

// ---------------------------------------------------------------------------
//  Force on the Xerces debug token if it was on in the build environment
// ---------------------------------------------------------------------------
#if defined(_DEBUG)
#define XERCES_DEBUG
#endif

#if _MSC_VER > 1300
#define XERCES_NEW_IOSTREAMS
#define XERCES_STD_NAMESPACE
#endif


// ---------------------------------------------------------------------------
//  The name of the DLL that is built by the Visual C++ version of the
//  system. We append a previously defined token which holds the DLL
//  versioning string. This is defined in XercesDefs.hpp which is what this
//  file is included into.
// ---------------------------------------------------------------------------
#if defined(XERCES_DEBUG)
const char* const Xerces_DLLName = "xerces-c_" Xerces_DLLVersionStr "D";
#else
const char* const Xerces_DLLName = "xerces-c_" Xerces_DLLVersionStr;
#endif


#endif //VCPPDEFS_HPP

