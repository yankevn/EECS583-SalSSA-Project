//----------------------------  sparse_matrix_ez.h  ---------------------------
//    $Id: sparse_matrix_ez.h,v 1.1 2004/09/14 00:53:35 wolf Exp $
//    Version: $Name:  $
//
//    Copyright (C) 2002, 2003, 2004 by the deal.II authors
//
//    This file is subject to QPL and may not be  distributed
//    without copyright and license information. Please refer
//    to the file deal.II/doc/license.html for the  text  and
//    further information on this license.
//
//----------------------------  sparse_matrix_ez.h  ---------------------------
#ifndef __deal2__sparse_matrix_ez_h
#define __deal2__sparse_matrix_ez_h


#include <base/config.h>
#include <base/exceptions.h>
#include <base/subscriptor.h>
#include <base/smartpointer.h>

#include <vector>

template<typename number> class Vector;
template<typename number> class FullMatrix;

/*! @addtogroup Matrix1
 *@{
 */

/**
 * Sparse matrix without sparsity pattern.
 *
 * Instead of using a pre-assembled sparsity pattern, this matrix
 * builds the pattern on the fly. Filling the matrix may consume more
 * time as with @p SparseMatrix, since large memory movements may be
 * involved. To help optimizing things, an expected row-length may be
 * provided to the constructor, as well as an increment size
 * for rows.
 *
 * The storage structure: like with the usual sparse matrix, it is
 * attempted to store only non-zero elements. these are stored in a
 * single data array @p data. They are ordered by row and inside each
 * row by column number. Each row is described by its starting point
 * in the data array and its length. These are stored in the
 * @p row_info array, together with additional useful information.
 *
 * Due to the structure, gaps may occur between rows. Whenever a new
 * entry must be created, an attempt is made to use the gap in its
 * row. If the gap is full, the row must be extended and all
 * subsequent rows must be shifted backwards. This is a very expensive
 * operation and should be avoided as much as possible.
 *
 * This is, where the optimization parameters, provided to the
 * constructor or to the function @p reinit come
 * in. @p default_row_length is the amount of entries that will be
 * allocated for each row on initialization (the actual length of the
 * rows is still zero). This means, that @p default_row_length
 * entries can be added to this row without shifting other rows. If
 * less entries are added, the additional memory will be wasted.
 *
 * If the space for a row is not sufficient, then it is enlarged by
 * @p default_increment entries. This way, the subsequent rows are
 * not shifted by single entries very often.
 *
 * Finally, the @p default_reserve allocates extra space at the end
 * of the data array. This space is used whenever a row must be
 * enlarged. Since @p std::vector doubles the capacity everytime it
 * must increase it, this value should allow for all the growth needed.
 *
 * Suggested settings: @p default_row_length should be the length of
 * a typical row, for instance the size of the stencil in regular
 * parts of the grid. Then, @p default_increment may be the expected
 * amount of entries added to the row by having one hanging node. This
 * way, a good compromise between memory consumption and speed should
 * be achieved. @p default_reserve should then be an estimate for the
 * number of hanging nodes times @p default_increment.
 *
 * Letting @p default_increment zero causes an exception whenever a
 * row overflows.
 *
 * If the rows are expected to be filled more or less from first to
 * last, using a @p default_row_length of zero may not be such a bad
 * idea.
 *
 * The name of this matrix is in reverence to a publication of the
 * Internal Revenue Service of the United States of America. I hope
 * some other aliens will appreciate it. By the way, the suffix makes
 * sense by pronouncing it the American way.
 *
 * @author Guido Kanschat, 2002
 */
template <typename number>
class SparseMatrixEZ : public Subscriptor
{
  public:
				     /**
				      * The class for storing the
				      * column number of an entry
				      * together with its value.
				      */
    struct Entry
    {
					 /**
					  * Standard constructor. Sets
					  * @p column to
					  * @p invalid.
					  */
	Entry();

					 /**
					  * Constructor. Fills column
					  * and value.
					  */
	Entry(unsigned int column,
	      const number& value);
	
					 /**
					  * The column number.
					  */
	unsigned int column;
					 /**
					  * The value there.
					  */
	number value;
					 /**
					  * Comparison operator for finding.
					  */
//	bool operator==(const Entry&) const;

					 /**
					  * Less than operator for sorting.
					  */
//	bool operator < (const Entry&) const;
					 /**
					  * Non-existent column number.
					  */
	static const unsigned int invalid = deal_II_numbers::invalid_unsigned_int;
    };

				     /**
				      * Structure for storing
				      * information on a matrix
				      * row. One object for each row
				      * will be stored in the matrix.
				      */
    struct RowInfo
    {
					 /**
					  * Constructor.
					  */
	RowInfo (unsigned int start = Entry::invalid);
	
					 /**
					  * Index of first entry of
					  * the row in the data field.
					  */
	unsigned int start;
					 /**
					  * Number of entries in this
					  * row.
					  */
	unsigned short length;
					 /**
					  * Position of the diagonal
					  * element relative tor the
					  * start index.
					  */
	unsigned short diagonal;
					 /**
					  * Value for non-existing diagonal.
					  */
	static const unsigned short
	invalid_diagonal = static_cast<unsigned short>(-1);
    };

  public:

				     /**
				      * STL conforming iterator.
				      */
    class const_iterator
    {
      private:
                                         /**
                                          * Accessor class for iterators
                                          */
        class Accessor
        {
          public:
                                             /**
                                              * Constructor. Since we use
                                              * accessors only for read
                                              * access, a const matrix
                                              * pointer is sufficient.
                                              */
            Accessor (const SparseMatrixEZ<number> *matrix,
                      const unsigned int            row,
                      const unsigned short          index);

                                             /**
                                              * Row number of the element
                                              * represented by this
                                              * object.
                                              */
            unsigned int row() const;

                                             /**
                                              * Index in row of the element
                                              * represented by this
                                              * object.
                                              */
            unsigned short index() const;

                                             /**
                                              * Column number of the
                                              * element represented by
                                              * this object.
                                              */
            unsigned int column() const;

                                             /**
                                              * Value of this matrix entry.
                                              */
            number value() const;
	
          protected:
                                             /**
                                              * The matrix accessed.
                                              */
            const SparseMatrixEZ<number>* matrix;

                                             /**
                                              * Current row number.
                                              */
            unsigned int a_row;

                                             /**
                                              * Current index in row.
                                              */
            unsigned short a_index;

                                             /**
                                              * Make enclosing class a
                                              * friend.
                                              */
            friend class const_iterator;
        };
          
      public:
                                         /**
                                          * Constructor.
                                          */ 
	const_iterator(const SparseMatrixEZ<number> *matrix,
		       const unsigned int            row,
		       const unsigned short          index);
	  
                                         /**
                                          * Prefix increment. This
                                          * always returns a valid
                                          * entry or <tt>end()</tt>.
                                          */
	const_iterator& operator++ ();

                                         /**
                                          * Postfix increment. This
                                          * always returns a valid
                                          * entry or <tt>end()</tt>.
                                          */
	const_iterator& operator++ (int);

                                         /**
                                          * Dereferencing operator.
                                          */
	const Accessor& operator* () const;

                                         /**
                                          * Dereferencing operator.
                                          */
	const Accessor* operator-> () const;

                                         /**
                                          * Comparison. True, if
                                          * both iterators point to
                                          * the same matrix
                                          * position.
                                          */
	bool operator == (const const_iterator&) const;
                                         /**
                                          * Inverse of <tt>==</tt>.
                                          */
	bool operator != (const const_iterator&) const;

                                         /**
                                          * Comparison
                                          * operator. Result is true
                                          * if either the first row
                                          * number is smaller or if
                                          * the row numbers are
                                          * equal and the first
                                          * index is smaller.
                                          */
	bool operator < (const const_iterator&) const;

      private:
                                         /**
                                          * Store an object of the
                                          * accessor class.
                                          */
        Accessor accessor;

					 /**
					  * Make the enclosing class a
					  * friend. This is only
					  * necessary since icc7
					  * otherwise wouldn't allow
					  * us to make
					  * const_iterator::Accessor a
					  * friend, stating that it
					  * can't access this class --
					  * this is of course bogus,
					  * since granting friendship
					  * doesn't need access to the
					  * class being granted
					  * friendship...
					  */
#ifdef DEAL_II_NESTED_CLASS_FRIEND_BUG	
	template <typename> friend class SparseMatrixEZ;
#endif
    };
    
				     /**
				      * Type of matrix entries. In analogy to
				      * the STL container classes.
				      */
    typedef number value_type;
    
				     /**
				      * Constructor. Initializes an
				      * empty matrix of dimension zero
				      * times zero.
				      */
    SparseMatrixEZ ();

				     /**
				      * Dummy copy constructor. This
				      * is here for use in
				      * containers. It may only be
				      * called for empty objects.
				      *
				      * If you really want to copy a whole
				      * matrix, you can do so by using the
				      * @p copy_from function.
				      */
    SparseMatrixEZ (const SparseMatrixEZ &);

				     /**
				      * Constructor. Generates a
				      * matrix of the given size,
				      * ready to be filled. The
				      * optional parameters
				      * @p default_row_length and
				      * @p default_increment allow
				      * for preallocating
				      * memory. Providing these
				      * properly is essential for an
				      * efficient assembling of the
				      * matrix.
				      */
    explicit SparseMatrixEZ (const unsigned int n_rows,
			     const unsigned int n_columns,
			     const unsigned int default_row_length = 0,
			     const unsigned int default_increment = 1);
    
				     /**
				      * Destructor. Free all memory, but do not
				      * release the memory of the sparsity
				      * structure.
				      */
    ~SparseMatrixEZ ();

				     /** 
				      * Pseudo operator only copying
				      * empty objects.
				      */
    SparseMatrixEZ<number>& operator = (const SparseMatrixEZ<number> &);

                                     /**
                                      * This operator assigns a scalar to
                                      * a matrix. Since this does usually
                                      * not make much sense (should we set
                                      * all matrix entries to this value?
                                      * Only the nonzero entries of the
                                      * sparsity pattern?), this operation
                                      * is only allowed if the actual
                                      * value to be assigned is zero. This
                                      * operator only exists to allow for
                                      * the obvious notation
                                      * <tt>matrix=0</tt>, which sets all
                                      * elements of the matrix to zero,
                                      * but keep the sparsity pattern
                                      * previously used.
                                      */
    SparseMatrixEZ<number>& operator = (const double d);

				     /**
				      * Reinitialize the sparse matrix
				      * to the dimensions provided.
				      * The matrix will have no
				      * entries at this point. The
				      * optional parameters
				      * @p default_row_length,
				      * @p default_increment and
				      * @p reserve allow
				      * for preallocating
				      * memory. Providing these
				      * properly is essential for an
				      * efficient assembling of the
				      * matrix.
				      */
    void reinit (const unsigned int n_rows,
		 const unsigned int n_columns,
		 unsigned int default_row_length = 0,
		 unsigned int default_increment = 1,
		 unsigned int reserve = 0);

				     /**
				      * Release all memory and return
				      * to a state just like after
				      * having called the default
				      * constructor. It also forgets
				      * its sparsity pattern.
				      */
    void clear ();
    
				     /**
				      * Return whether the object is
				      * empty. It is empty if
				      * both dimensions are zero.
				      */
    bool empty () const;

				     /**
				      * Return the dimension of the
				      * image space.  To remember: the
				      * matrix is of dimension
				      * $m \times n$.
				      */
    unsigned int m () const;
    
				     /**
				      * Return the dimension of the
				      * range space.  To remember: the
				      * matrix is of dimension
				      * $m \times n$.
				      */
    unsigned int n () const;

				     /**
				      * Set the element <tt>(i,j)</tt> to
				      * @p value. Allocates the entry,
				      * if it does not exist and
				      * @p value is non-zero.
				      */
    void set (const unsigned int i, const unsigned int j,
	      const number value);
    
				     /**
				      * Add @p value to the element
				      * <tt>(i,j)</tt>. Allocates the entry
				      * if it does not exist. Filters
				      * out zeroes automatically.
				      */
    void add (const unsigned int i, const unsigned int j,
	      const number value);

				     /**
				      * Symmetrize the matrix by
				      * forming the mean value between
				      * the existing matrix and its
				      * transpose, $A = \frac 12(A+A^T)$.
				      *
				      * This operation assumes that
				      * the underlying sparsity
				      * pattern represents a symmetric
				      * object. If this is not the
				      * case, then the result of this
				      * operation will not be a
				      * symmetric matrix, since it
				      * only explicitly symmetrizes
				      * by looping over the lower left
				      * triangular part for efficiency
				      * reasons; if there are entries
				      * in the upper right triangle,
				      * then these elements are missed
				      * in the
				      * symmetrization. Symmetrization
				      * of the sparsity pattern can be
				      * obtain by the
				      * SparsityPattern@p ::symmetrize
				      * function.
				      */
//    void symmetrize ();
    
				     /**
				      * Copy the given matrix to this
				      * one.  The operation throws an
				      * error if the sparsity patterns
				      * of the two involved matrices
				      * do not point to the same
				      * object, since in this case the
				      * copy operation is
				      * cheaper. Since this operation
				      * is notheless not for free, we
				      * do not make it available
				      * through <tt>operator =</tt>, since
				      * this may lead to unwanted
				      * usage, e.g. in copy arguments
				      * to functions, which should
				      * really be arguments by
				      * reference.
				      *
				      * The source matrix may be a matrix
				      * of arbitrary type, as long as its
				      * data type is convertible to the
				      * data type of this matrix.
				      *
				      * The function returns a reference to
				      * @p this.
				      */
    template <class MATRIX>
    SparseMatrixEZ<number> &
    copy_from (const MATRIX &source);

				     /**
				      * This function is complete
				      * analogous to the
				      * SparsityPattern@p ::copy_from
				      * function in that it allows to
				      * initialize a whole matrix in
				      * one step. See there for more
				      * information on argument types
				      * and their meaning. You can
				      * also find a small example on
				      * how to use this function
				      * there.
				      *
				      * The only difference to the
				      * cited function is that the
				      * objects which the inner
				      * iterator points to need to be
				      * of type <tt>std::pair<unsigned int, value</tt>,
				      * where @p value
				      * needs to be convertible to the
				      * element type of this class, as
				      * specified by the @p number
				      * template argument.
				      *
				      * Previous content of the matrix
				      * is overwritten. Note that the
				      * entries specified by the input
				      * parameters need not
				      * necessarily cover all elements
				      * of the matrix. Elements not
				      * covered remain untouched.
				      */
//    template <typename ForwardIterator>
//    void copy_from (const ForwardIterator begin,
//		    const ForwardIterator end);    

				     /**
				      * Copy the nonzero entries of a
				      * full matrix into this
				      * object. Previous content is
				      * deleted. Note that the
				      * underlying sparsity pattern
				      * must be appropriate to hold
				      * the nonzero entries of the
				      * full matrix.
				      */
//    template <typename somenumber>
//    void copy_from (const FullMatrix<somenumber> &matrix);
    
				     /**
				      * Add @p matrix scaled by
				      * @p factor to this matrix.
				      *
				      * The source matrix may be a
				      * matrix of arbitrary type, as
				      * long as its data type is
				      * convertible to the data type
				      * of this matrix and it has the
				      * standard @p const_iterator.
				      */
    template <class MATRIX>
    void add_scaled (const number factor,
		     const MATRIX &matrix);
    
				     /**
				      * Return the value of the entry
				      * (i,j).  This may be an
				      * expensive operation and you
				      * should always take care where
				      * to call this function.  In
				      * order to avoid abuse, this
				      * function throws an exception
				      * if the required element does
				      * not exist in the matrix.
				      *
				      * In case you want a function
				      * that returns zero instead (for
				      * entries that are not in the
				      * sparsity pattern of the
				      * matrix), use the @p el
				      * function.
				      */
    number operator () (const unsigned int i,
			const unsigned int j) const;

				     /**
				      * Return the value of the entry
				      * (i,j). Returns zero for all
				      * non-existing entries.
				      */
    number el (const unsigned int i,
	       const unsigned int j) const;

				     /**
				      * Return the main diagonal element in
				      * the @p ith row. This function throws an
				      * error if the matrix is not square.
				      *
				      * This function is considerably
				      * faster than the <tt>operator()</tt>,
				      * since for square matrices, the
				      * diagonal entry is always the
				      * first to be stored in each row
				      * and access therefore does not
				      * involve searching for the
				      * right column number.
				      */
//    number diag_element (const unsigned int i) const;

				     /**
				      * Same as above, but return a
				      * writeable reference. You're
				      * sure you know what you do?
				      */
//    number & diag_element (const unsigned int i);
    
				     /**
				      * Matrix-vector multiplication:
				      * let $dst = M*src$ with $M$
				      * being this matrix.
				      */
    template <typename somenumber>
    void vmult (Vector<somenumber>       &dst,
		const Vector<somenumber> &src) const;
    
				     /**
				      * Matrix-vector multiplication:
				      * let $dst = M^T*src$ with $M$
				      * being this matrix. This
				      * function does the same as
				      * @p vmult but takes the
				      * transposed matrix.
				      */
    template <typename somenumber>
    void Tvmult (Vector<somenumber>       &dst,
		 const Vector<somenumber> &src) const;
  
				     /**
				      * Adding Matrix-vector
				      * multiplication. Add $M*src$ on
				      * $dst$ with $M$ being this
				      * matrix.
				      */
    template <typename somenumber>
    void vmult_add (Vector<somenumber>       &dst,
		    const Vector<somenumber> &src) const;
    
				     /**
				      * Adding Matrix-vector
				      * multiplication. Add $M^T*src$
				      * to $dst$ with $M$ being this
				      * matrix. This function does the
				      * same as @p vmult_add but takes
				      * the transposed matrix.
				      */
    template <typename somenumber>
    void Tvmult_add (Vector<somenumber>       &dst,
		     const Vector<somenumber> &src) const;
  
				     /**
				      * Return the square of the norm
				      * of the vector $v$ with respect
				      * to the norm induced by this
				      * matrix,
				      * i.e. $\left(v,Mv\right)$. This
				      * is useful, e.g. in the finite
				      * element context, where the
				      * $L_2$ norm of a function
				      * equals the matrix norm with
				      * respect to the mass matrix of
				      * the vector representing the
				      * nodal values of the finite
				      * element function.
				      *
				      * Obviously, the matrix needs to
				      * be square for this operation.
				      */
//    template <typename somenumber>
//    somenumber matrix_norm_square (const Vector<somenumber> &v) const;

				     /**
				      * Compute the matrix scalar
				      * product $\left(u,Mv\right)$.
				      */
//    template <typename somenumber>
//    somenumber matrix_scalar_product (const Vector<somenumber> &u,
//				      const Vector<somenumber> &v) const;

				     /**
				      * Frobenius-norm of the matrix.
				      */
    number l2_norm () const;
    
    				     /**
				      * Return the l1-norm of the matrix, that is
				      * $|M|_1=max_{all columns j}\sum_{all 
				      * rows i} |M_ij|$,
				      * (max. sum of columns).
				      * This is the
				      * natural matrix norm that is compatible
				      * to the l1-norm for vectors, i.e.
				      * $|Mv|_1\leq |M|_1 |v|_1$.
				      * (cf. Haemmerlin-Hoffmann : Numerische Mathematik)
				      */
//    number l1_norm () const;

    				     /**
				      * Return the linfty-norm of the
				      * matrix, that is
				      * $|M|_infty=max_{all rows i}\sum_{all 
				      * columns j} |M_ij|$,
				      * (max. sum of rows).
				      * This is the
				      * natural matrix norm that is compatible
				      * to the linfty-norm of vectors, i.e.
				      * $|Mv|_infty \leq |M|_infty |v|_infty$.
				      * (cf. Haemmerlin-Hoffmann : Numerische Mathematik)
				      */
//    number linfty_norm () const;

				     /**
				      * Apply the Jacobi
				      * preconditioner, which
				      * multiplies every element of
				      * the @p src vector by the
				      * inverse of the respective
				      * diagonal element and
				      * multiplies the result with the
				      * damping factor @p omega.
				      */
    template <typename somenumber>
    void precondition_Jacobi (Vector<somenumber>       &dst,
			      const Vector<somenumber> &src,
			      const number              omega = 1.) const;

				     /**
				      * Apply SSOR preconditioning to
				      * @p src.
				      */
    template <typename somenumber>
    void precondition_SSOR (Vector<somenumber>       &dst,
			    const Vector<somenumber> &src,
			    const number              om = 1.) const;

				     /**
				      * Apply SOR preconditioning matrix to @p src.
				      * The result of this method is
				      * $dst = (om D - L)^{-1} src$.
				      */
    template <typename somenumber>
    void precondition_SOR (Vector<somenumber>       &dst,
			   const Vector<somenumber> &src,
 			   const number              om = 1.) const;
    
				     /**
				      * Apply transpose SOR preconditioning matrix to @p src.
				      * The result of this method is
				      * $dst = (om D - U)^{-1} src$.
				      */
    template <typename somenumber>
    void precondition_TSOR (Vector<somenumber>       &dst,
			    const Vector<somenumber> &src,
			    const number              om = 1.) const;

				     /**
				      * Add the matrix @p A
				      * conjugated by @p B, that is,
				      * $B A B^T$ to this object. If
				      * the parameter @p transpose is
				      * true, compute $B^T A B$.
				      *
				      * This function requires that
				      * @p B has a @p const_iterator
				      * traversing all matrix entries
				      * and that @p A has a function
				      * <tt>el(i,j)</tt> for access to a
				      * specific entry.
				      */
    template <class MATRIXA, class MATRIXB>
    void conjugate_add (const MATRIXA& A,
			const MATRIXB& B,
			const bool transpose = false);
    
				     /**
				      * STL-like iterator with the
				      * first existing entry.
				      */
    const_iterator begin () const;

				     /**
				      * Final iterator.
				      */
    const_iterator end () const;
    
				     /**
				      * STL-like iterator with the
				      * first entry of row @p r. If
				      * this row is empty, the result
				      * is <tt>end(r)</tt>, which does NOT
				      * point into row @p r..
				      */
    const_iterator begin (const unsigned int r) const;

				     /**
				      * Final iterator of row
				      * @p r. The result may be
				      * different from <tt>end()</tt>!
				      */
    const_iterator end (const unsigned int r) const;
    
				     /**
				      * Return the number of nonzero
				      * elements of this
				      * matrix.
				      */
//    unsigned int n_nonzero_elements () const;

				     /**
				      * Return the number of actually
				      * nonzero elements of this
				      * matrix.
				      */
//    unsigned int n_actually_nonzero_elements () const;
    
				     /**
				      * Print the matrix to the given
				      * stream, using the format
				      * <tt>(line,col) value</tt>, i.e. one
				      * nonzero entry of the matrix
				      * per line.
				      */
    void print (std::ostream &out) const;

				     /**
				      * Print the matrix in the usual
				      * format, i.e. as a matrix and
				      * not as a list of nonzero
				      * elements. For better
				      * readability, elements not in
				      * the matrix are displayed as
				      * empty space, while matrix
				      * elements which are explicitly
				      * set to zero are displayed as
				      * such.
				      *
				      * The parameters allow for a
				      * flexible setting of the output
				      * format: @p precision and
				      * @p scientific are used to
				      * determine the number format,
				      * where @p scientific = @p false
				      * means fixed point notation.  A
				      * zero entry for @p width makes
				      * the function compute a width,
				      * but it may be changed to a
				      * positive value, if output is
				      * crude.
				      *
				      * Additionally, a character for
				      * an empty value may be
				      * specified.
				      *
				      * Finally, the whole matrix can
				      * be multiplied with a common
				      * denominator to produce more
				      * readable output, even
				      * integers.
				      *
				      * This function
				      * may produce @em large amounts of
				      * output if applied to a large matrix!
				      */
//      void print_formatted (std::ostream       &out,
//  			  const unsigned int  precision   = 3,
//  			  const bool          scientific  = true,
//  			  const unsigned int  width       = 0,
//  			  const char         *zero_string = " ",
//  			  const double        denominator = 1.) const;

    				     /**
				      * Write the data of this object
				      * in binary mode to a file.
				      *
				      * Note that this binary format
				      * is platform dependent.
				      */
    void block_write (std::ostream &out) const;

				     /**
				      * Read data that has previously
				      * been written by
				      * @p block_write.
				      *
				      * The object is resized on this
				      * operation, and all previous
				      * contents are lost.
				      *
				      * A primitive form of error
				      * checking is performed which
				      * will recognize the bluntest
				      * attempts to interpret some
				      * data as a vector stored
				      * bitwise to a file, but not
				      * more.
				      */
    void block_read (std::istream &in);


				     /**
				      * Determine an estimate for the
				      * memory consumption (in bytes)
				      * of this object.
				      */
    unsigned int memory_consumption () const;

				     /**
				      * Print statistics. If @p full
				      * is @p true, prints a
				      * histogram of all existing row
				      * lengths and allocated row
				      * lengths. Otherwise, just the
				      * relation of allocated and used
				      * entries is shown.
				      */
    template <class STREAM>
    void print_statistics (STREAM& s, bool full = false);

				     /**
				      * Compute numbers of entries.
				      *
				      * In the first three arguments,
				      * this function returns the
				      * number of entries used,
				      * allocated and reserved by this
				      * matrix.
				      *
				      * If the final argument is true,
				      * the number of entries in each
				      * line is printed as well.
				      */
    void compute_statistics (unsigned int& used,
			     unsigned int& allocated,
			     unsigned int& reserved,
			     std::vector<unsigned int>& used_by_line,
			     const bool compute_by_line) const;
			     
				     /**
				      * Exception for applying
				      * inverse-type operators to
				      * rectangular matrices.
				      */
    DeclException0(ExcNoSquareMatrix);
    
				     /**
				      * Exception for missing diagonal entry.
				      */
    DeclException0(ExcNoDiagonal);
    
    				     /**
				      * Exception
				      */
    DeclException2 (ExcInvalidEntry,
		    int, int,
		    << "The entry with index (" << arg1 << ',' << arg2
		    << ") does not exist.");

    DeclException2(ExcEntryAllocationFailure,
		   int, int,
		   << "An entry with index (" << arg1 << ',' << arg2
		   << ") cannot be allocated.");
  private:
				     /**
				      * Find an entry and return a
				      * const pointer. Return a
				      * zero-pointer if the entry does
				      * not exist.
				      */
    const Entry* locate (const unsigned int row,
			 const unsigned int col) const;

				     /**
				      * Find an entry and return a
				      * writable pointer. Return a
				      * zero-pointer if the entry does
				      * not exist.
				      */
    Entry* locate (const unsigned int row,
		   const unsigned int col);

				     /**
				      * Find an entry or generate it.
				      */
    Entry* allocate (const unsigned int row,
		     const unsigned int col);
    
				     /**
				      * Version of @p vmult which only
				      * performs its actions on the
				      * region defined by
				      * <tt>[begin_row,end_row)</tt>. This
				      * function is called by @p vmult
				      * in the case of enabled
				      * multithreading.
				      */
    template <typename somenumber>
    void threaded_vmult (Vector<somenumber>       &dst,
			 const Vector<somenumber> &src,
			 const unsigned int        begin_row,
			 const unsigned int        end_row) const;

				     /**
				      * Version of
				      * @p matrix_norm_square which
				      * only performs its actions on
				      * the region defined by
				      * <tt>[begin_row,end_row)</tt>. This
				      * function is called by
				      * @p matrix_norm_square in the
				      * case of enabled
				      * multithreading.
				      */
    template <typename somenumber>
    void threaded_matrix_norm_square (const Vector<somenumber> &v,
				      const unsigned int        begin_row,
				      const unsigned int        end_row,
				      somenumber               *partial_sum) const;

    				     /**
				      * Version of
				      * @p matrix_scalar_product which
				      * only performs its actions on
				      * the region defined by
				      * <tt>[begin_row,end_row)</tt>. This
				      * function is called by
				      * @p matrix_scalar_product in the
				      * case of enabled
				      * multithreading.
				      */
    template <typename somenumber>
    void threaded_matrix_scalar_product (const Vector<somenumber> &u,
					 const Vector<somenumber> &v,
					 const unsigned int        begin_row,
					 const unsigned int        end_row,
					 somenumber               *partial_sum) const;

				     /**
				      * Number of columns. This is
				      * used to check vector
				      * dimensions only.
				      */
    unsigned int n_columns;

				     /**
				      * Info structure for each row.
				      */
    std::vector<RowInfo> row_info;
    
				     /**
				      * Data storage.
				      */
    std::vector<Entry> data;

				     /**
				      * Increment when a row grows.
				      */
    unsigned int increment;

                                     /**
                                      * Make member classes
                                      * friends. Not strictly
                                      * necessary according to the
                                      * standard, but some compilers
                                      * require this...
                                      */
#ifdef DEAL_II_NESTED_CLASS_FRIEND_BUG    
    friend class const_iterator;
    friend class const_iterator::Accessor;
#endif
};

/*@}*/
/*---------------------- Inline functions -----------------------------------*/

template <typename number>
inline
SparseMatrixEZ<number>::Entry::Entry(unsigned int column,
				     const number& value)
		:
		column(column),
  value(value)
{}



template <typename number>
inline
SparseMatrixEZ<number>::Entry::Entry()
		:
		column(invalid),
  value(0)
{}


template <typename number>
inline
SparseMatrixEZ<number>::RowInfo::RowInfo(unsigned int start)
		:
		start(start), length(0), diagonal(invalid_diagonal)
{}


//----------------------------------------------------------------------//
template <typename number>
inline
SparseMatrixEZ<number>::const_iterator::Accessor::
Accessor (const SparseMatrixEZ<number> *matrix,
          const unsigned int            r,
          const unsigned short          i)
		:
		matrix(matrix),
		a_row(r),
		a_index(i)
{}


template <typename number>
inline
unsigned int
SparseMatrixEZ<number>::const_iterator::Accessor::row() const
{
  return a_row;
}


template <typename number>
inline
unsigned int
SparseMatrixEZ<number>::const_iterator::Accessor::column() const
{
  return matrix->data[matrix->row_info[a_row].start+a_index].column;
}


template <typename number>
inline
unsigned short
SparseMatrixEZ<number>::const_iterator::Accessor::index() const
{
  return a_index;
}



template <typename number>
inline
number
SparseMatrixEZ<number>::const_iterator::Accessor::value() const
{
  return matrix->data[matrix->row_info[a_row].start+a_index].value;
}


template <typename number>
inline
SparseMatrixEZ<number>::const_iterator::
const_iterator(const SparseMatrixEZ<number> *matrix,
               const unsigned int            r,
               const unsigned short          i)
		:
		accessor(matrix, r, i)
{
				   // Finish if this is the end()
  if (r==accessor.matrix->m() && i==0) return;

				   // Make sure we never construct an
				   // iterator pointing to a
				   // non-existing entry

				   // If the index points beyond the
				   // end of the row, try the next
				   // row.
  if (accessor.a_index >= accessor.matrix->row_info[accessor.a_row].length)
    {
     do
	{
	  ++accessor.a_row;
	}
				      // Beware! If the next row is
				      // empty, iterate until a
				      // non-empty row is found or we
				      // hit the end of the matrix.
      while (accessor.a_row < accessor.matrix->m()
	     && accessor.matrix->row_info[accessor.a_row].length == 0);
    }
}


template <typename number>
inline
typename SparseMatrixEZ<number>::const_iterator&
SparseMatrixEZ<number>::const_iterator::operator++ ()
{
  Assert (accessor.a_row < accessor.matrix->m(), ExcIteratorPastEnd());

				   // Increment column index
  ++(accessor.a_index);
				   // If index exceeds number of
				   // entries in this row, proceed
				   // with next row.
  if (accessor.a_index >= accessor.matrix->row_info[accessor.a_row].length)
    {
      accessor.a_index = 0;
				       // Do this loop to avoid
				       // elements in empty rows
      do
	{
	  ++accessor.a_row;
	}
      while (accessor.a_row < accessor.matrix->m()
	     && accessor.matrix->row_info[accessor.a_row].length == 0);
    }
  return *this;
}


template <typename number>
inline
const typename SparseMatrixEZ<number>::const_iterator::Accessor&
SparseMatrixEZ<number>::const_iterator::operator* () const
{
  return accessor;
}


template <typename number>
inline
const typename SparseMatrixEZ<number>::const_iterator::Accessor*
SparseMatrixEZ<number>::const_iterator::operator-> () const
{
  return &accessor;
}


template <typename number>
inline
bool
SparseMatrixEZ<number>::const_iterator::operator == (
  const const_iterator& other) const
{
  return (accessor.row() == other.accessor.row() &&
          accessor.index() == other.accessor.index());
}


template <typename number>
inline
bool
SparseMatrixEZ<number>::const_iterator::
operator != (const const_iterator& other) const
{
  return ! (*this == other);
}


template <typename number>
inline
bool
SparseMatrixEZ<number>::const_iterator::
operator < (const const_iterator& other) const
{
  return (accessor.row() < other.accessor.row() ||
	  (accessor.row() == other.accessor.row() &&
           accessor.index() < other.accessor.index()));
}


//----------------------------------------------------------------------//
template <typename number>
inline
unsigned int SparseMatrixEZ<number>::m () const
{
  return row_info.size();
}


template <typename number>
inline
unsigned int SparseMatrixEZ<number>::n () const
{
  return n_columns;
}


template <typename number>
inline
typename SparseMatrixEZ<number>::Entry*
SparseMatrixEZ<number>::locate (const unsigned int row,
				const unsigned int col)
{
  Assert (row<m(), ExcIndexRange(row,0,m()));
  Assert (col<n(), ExcIndexRange(col,0,n()));

  const RowInfo& r = row_info[row];
  const unsigned int end = r.start + r.length;
  for (unsigned int i=r.start;i<end;++i)
    {
      Entry * const entry = &data[i];
      if (entry->column == col)
	return entry;
      if (entry->column == Entry::invalid)
	return 0;
    }
  return 0;
}



template <typename number>
inline
const typename SparseMatrixEZ<number>::Entry*
SparseMatrixEZ<number>::locate (const unsigned int row,
				const unsigned int col) const
{
  SparseMatrixEZ<number>* t = const_cast<SparseMatrixEZ<number>*> (this);
  return t->locate(row,col);
}


template <typename number>
inline
typename SparseMatrixEZ<number>::Entry*
SparseMatrixEZ<number>::allocate (const unsigned int row,
				  const unsigned int col)
{
  Assert (row<m(), ExcIndexRange(row,0,m()));
  Assert (col<n(), ExcIndexRange(col,0,n()));

  RowInfo& r = row_info[row];
  const unsigned int end = r.start + r.length;

  unsigned int i = r.start;
  if (r.diagonal != RowInfo::invalid_diagonal && col >= row)
    i += r.diagonal;
				   // Find position of entry
  while (i<end && data[i].column < col) ++i;
  
				   // entry found
  if (i != end && data[i].column == col)
    return &data[i];

				   // Now, we must insert the new
				   // entry and move all successive
				   // entries back.
  
				   // If no more space is available
				   // for this row, insert new
				   // elements into the vector.
  if (row != row_info.size()-1)
    {
      if (end >= row_info[row+1].start)
	{
	  // Failure if increment 0
	  Assert(increment!=0,ExcEntryAllocationFailure(row,col));
	  
					   // Insert new entries
	  data.insert(data.begin()+end, increment, Entry());
					   // Update starts of
					   // following rows
	  for (unsigned int rn=row+1;rn<row_info.size();++rn)
	    row_info[rn].start += increment;
	}
    } else {
      if (end >= data.size())
	{
					   // Here, appending a block
					   // does not increase
					   // performance.
	  data.push_back(Entry());
	}
    }
  Entry* entry = &data[i];
				   // Save original entry
  Entry temp = *entry;
				   // Insert new entry here to
				   // make sure all entries
				   // are ordered by column
				   // index
  entry->column = col;
  entry->value = 0;
				   // Update row_info
  ++r.length;
  if (col == row)
    r.diagonal = i - r.start;
  else if (col<row && r.diagonal!= RowInfo::invalid_diagonal)
    ++r.diagonal;

  if (i == end)
      return entry;
  
				   // Move all entries in this
				   // row up by one
  for (unsigned int j = i+1; j < end; ++j)
    {
				       // There should be no invalid
				       // entry below end
      Assert (data[j].column != Entry::invalid, ExcInternalError());
      Entry temp2 = data[j];
      data[j] = temp;
      temp = temp2;
    }
  Assert (data[end].column == Entry::invalid, ExcInternalError());
  data[end] = temp;

  return entry;
}



template <typename number>
inline
void SparseMatrixEZ<number>::set (const unsigned int i,
				  const unsigned int j,
				  const number value)
{
  Assert (i<m(), ExcIndexRange(i,0,m()));
  Assert (j<n(), ExcIndexRange(j,0,n()));

  if (value == 0.)
    {
      Entry* entry = locate(i,j);
      if (entry != 0)
	{
	  entry->value = 0.;
	}
    }
  else
    {
      Entry* entry = allocate(i,j);
      entry->value = value;
    }
}



template <typename number>
inline
void SparseMatrixEZ<number>::add (const unsigned int i,
				  const unsigned int j,
				  const number value)
{
  Assert (i<m(), ExcIndexRange(i,0,m()));
  Assert (j<n(), ExcIndexRange(j,0,n()));

				   // ignore zero additions
  if (value == 0)
    return;
  
  Entry* entry = allocate(i,j);
  entry->value += value;
}



template <typename number>
inline
number SparseMatrixEZ<number>::el (const unsigned int i,
				   const unsigned int j) const
{
  const Entry* entry = locate(i,j);
  if (entry)
    return entry->value;
  return 0.;
}



template <typename number>
inline
number SparseMatrixEZ<number>::operator() (const unsigned int i,
					   const unsigned int j) const
{
  const Entry* entry = locate(i,j);
  if (entry)
    return entry->value;
  Assert(false, ExcInvalidEntry(i,j));
  return 0.;
}


template <typename number>
inline
typename SparseMatrixEZ<number>::const_iterator
SparseMatrixEZ<number>::begin () const
{
  const_iterator result(this, 0, 0);
  return result;
}

template <typename number>
inline
typename SparseMatrixEZ<number>::const_iterator
SparseMatrixEZ<number>::end () const
{
  return const_iterator(this, m(), 0);
}

template <typename number>
inline
typename SparseMatrixEZ<number>::const_iterator
SparseMatrixEZ<number>::begin (const unsigned int r) const
{
  Assert (r<m(), ExcIndexRange(r,0,m()));
  const_iterator result (this, r, 0);
  return result;
}

template <typename number>
inline
typename SparseMatrixEZ<number>::const_iterator
SparseMatrixEZ<number>::end (const unsigned int r) const
{
  Assert (r<m(), ExcIndexRange(r,0,m()));
  const_iterator result(this, r+1, 0);
  return result;
}

template<typename number>
template <class MATRIX>
inline
SparseMatrixEZ<number>&
SparseMatrixEZ<number>::copy_from (const MATRIX& M)
{
  reinit(M.m(), M.n());
  
  typename MATRIX::const_iterator start = M.begin();
  const typename MATRIX::const_iterator final = M.end();

  while (start != final)
    {
      if (start->value() != 0.)
	set(start->row(), start->column(), start->value());
      ++start;
    }
  return *this;
}

template<typename number>
template <class MATRIX>
inline
void
SparseMatrixEZ<number>::add_scaled (const number factor,
				    const MATRIX& M)
{
  Assert (M.m() == m(), ExcDimensionMismatch(M.m(), m()));
  Assert (M.n() == n(), ExcDimensionMismatch(M.n(), n()));

  if (factor == 0.)
    return;
  
  typename MATRIX::const_iterator start = M.begin();
  const typename MATRIX::const_iterator final = M.end();

  while (start != final)
    {
      if (start->value() != 0.)
	add(start->row(), start->column(), factor * start->value());
      ++start;
    }
}


template<typename number>
template <class MATRIXA, class MATRIXB>
inline void
SparseMatrixEZ<number>::conjugate_add (const MATRIXA& A,
				       const MATRIXB& B,
				       const bool transpose)
{
// Compute the result
// r_ij = \sum_kl b_ik b_jl a_kl

//    Assert (n() == B.m(), ExcDimensionMismatch(n(), B.m()));
//    Assert (m() == B.m(), ExcDimensionMismatch(m(), B.m()));
//    Assert (A.n() == B.n(), ExcDimensionMismatch(A.n(), B.n()));
//    Assert (A.m() == B.n(), ExcDimensionMismatch(A.m(), B.n()));

				   // Somehow, we have to avoid making
				   // this an operation of complexity
				   // n^2. For the transpose case, we
				   // can go through the non-zero
				   // elements of A^-1 and use the
				   // corresponding rows of B only.
				   // For the non-transpose case, we
				   // must find a trick.
  typename MATRIXB::const_iterator b1 = B.begin();
  const typename MATRIXB::const_iterator b_final = B.end();
  if (transpose)
    while (b1 != b_final)
      {
	const unsigned int i = b1->column();
	const unsigned int k = b1->row();
	typename MATRIXB::const_iterator b2 = B.begin();
	while (b2 != b_final)
	  {
	    const unsigned int j = b2->column();
	    const unsigned int l = b2->row();
	    
	    const typename MATRIXA::value_type a = A.el(k,l);
	    
	    if (a != 0.)
	      add (i, j, a * b1->value() * b2->value());
	    ++b2;
	  }
	++b1;
      }
  else
    {
				       // Determine minimal and
				       // maximal row for a column in
				       // advance.

      std::vector<unsigned int> minrow(B.n(), B.m());
      std::vector<unsigned int> maxrow(B.n(), 0);
      while (b1 != b_final)
	{
	  const unsigned int r = b1->row();
	  if (r < minrow[b1->column()])
	    minrow[b1->column()] = r;
	  if (r > maxrow[b1->column()])
	    maxrow[b1->column()] = r;
	  ++b1;
	}

      typename MATRIXA::const_iterator ai = A.begin();
      const typename MATRIXA::const_iterator ae = A.end();

      while (ai != ae)
	{
	  const typename MATRIXA::value_type a = ai->value();
					   // Don't do anything if
					   // this entry is zero.
	  if (a == 0.) continue;

					   // Now, loop over all rows
					   // having possibly a
					   // nonzero entry in column
					   // ai->row()
	  b1 = B.begin(minrow[ai->row()]);
	  const typename MATRIXB::const_iterator
	    be1 = B.end(maxrow[ai->row()]);
	  const typename MATRIXB::const_iterator
	    be2 = B.end(maxrow[ai->column()]);

	  while (b1 != be1)
	    {
	      const double b1v = b1->value();
					       // We need the product
					       // of both. If it is
					       // zero, we can save
					       // the work
	      if (b1->column() == ai->row() && (b1v != 0.))
		{
		  const unsigned int i = b1->row();
		  
		  typename MATRIXB::const_iterator
		    b2 = B.begin(minrow[ai->column()]);
		  while (b2 != be2)
		    {
		      if (b2->column() == ai->column())
			{
			  const unsigned int j = b2->row();
			  add (i, j, a * b1v * b2->value());
			}
		      ++b2;
		    }
		}
	      ++b1;
	    }
	  ++ai;
	}
    }
}


template <typename number>
template <class STREAM>
inline
void
SparseMatrixEZ<number>::print_statistics(STREAM& out, bool full)
{
  unsigned int used;
  unsigned int allocated;
  unsigned int reserved;
  std::vector<unsigned int> used_by_line;

  compute_statistics (used, allocated, reserved, used_by_line, full);

  out << "SparseMatrixEZ:used      entries:" << used << std::endl
      << "SparseMatrixEZ:allocated entries:" << allocated << std::endl
      << "SparseMatrixEZ:reserved  entries:" << reserved << std::endl;

  if (full)
    {
      for (unsigned int i=0; i< used_by_line.size();++i)
	if (used_by_line[i] != 0)
	  out << "SparseMatrixEZ:entries\t" << i
	      << "\trows\t" << used_by_line[i]
	      << std::endl;
      
    }
}


#endif
/*----------------------------   sparse_matrix.h     ---------------------------*/
