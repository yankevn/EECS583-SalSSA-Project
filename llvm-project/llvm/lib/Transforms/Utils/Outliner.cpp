//===- Outliner.cpp - Generic outliner interface -----------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements an interface for outlining congruent sequences.
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Utils/Outliner.h"
#include "llvm/ADT/BitVector.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/PriorityQueue.h"
#include "llvm/Support/Allocator.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;

bool OutlinerImpl::run() {
  std::vector<OutlineCandidate> CandidateList;
  // 1) Find candidates.
  findOutliningOccurrences(CandidateList);
  errs() << "Canditates Found: " << CandidateList.size() << "\n";
  if (CandidateList.empty())
    return false;
  errs() << "Analizing benefits\n";
  // 2) Analyze for benefit.
  analyzeCandidateList(CandidateList);

  errs() << "Running outliner on candidates\n";
  // 3) Prune and Outline.
  return pruneAndOutline(CandidateList);
}

// Suffix tree implementation.
namespace {

/// Represents an undefined index in the suffix tree.
const size_t EmptyIdx = -1;

/// A node in a suffix tree which represents a substring or suffix.
///
/// Each node has either no children or at least two children, with the root
/// being a exception in the empty tree.
///
/// Children are represented as a map between unsigned integers and nodes. If
/// a node N has a child M on unsigned integer k, then the mapping represented
/// by N is a proper prefix of the mapping represented by M. Note that this,
/// although similar to a trie is somewhat different: each node stores a full
/// substring of the full mapping rather than a single character state.
///
/// Each internal node contains a pointer to the internal node representing
/// the same string, but with the first character chopped off. This is stored
/// in \p Link. Each leaf node stores the start index of its respective
/// suffix in \p SuffixIdx.
struct SuffixTreeNode {

  /// The children of this node.
  ///
  /// A child existing on an unsigned integer implies that from the mapping
  /// represented by the current node, there is a way to reach another
  /// mapping by tacking that character on the end of the current string.
  DenseMap<unsigned, SuffixTreeNode *> Children;

  /// A flag set to false if the node has been pruned from the tree.
  bool IsInTree = true;

  /// The start index of this node's substring in the main string.
  size_t StartIdx = EmptyIdx;

  /// The end index of this node's substring in the main string.
  ///
  /// Every leaf node must have its \p EndIdx incremented at the end of every
  /// step in the construction algorithm. To avoid having to update O(N)
  /// nodes individually at the end of every step, the end index is stored
  /// as a pointer.
  size_t *EndIdx = nullptr;

  /// For leaves, the start index of the suffix represented by this node.
  ///
  /// For all other nodes, this is ignored.
  size_t SuffixIdx = EmptyIdx;

  /// \brief For internal nodes, a pointer to the internal node representing
  /// the same sequence with the first character chopped off.
  ///
  /// This acts as a shortcut in Ukkonen's algorithm. One of the things that
  /// Ukkonen's algorithm does to achieve linear-time construction is
  /// keep track of which node the next insert should be at. This makes each
  /// insert O(1), and there are a total of O(N) inserts. The suffix link
  /// helps with inserting children of internal nodes.
  ///
  /// Say we add a child to an internal node with associated mapping S. The
  /// next insertion must be at the node representing S - its first character.
  /// This is given by the way that we iteratively build the tree in Ukkonen's
  /// algorithm. The main idea is to look at the suffixes of each prefix in the
  /// string, starting with the longest suffix of the prefix, and ending with
  /// the shortest. Therefore, if we keep pointers between such nodes, we can
  /// move to the next insertion point in O(1) time. If we don't, then we'd
  /// have to query from the root, which takes O(N) time. This would make the
  /// construction algorithm O(N^2) rather than O(N).
  SuffixTreeNode *Link = nullptr;

  /// The parent of this node. Every node except for the root has a parent.
  SuffixTreeNode *Parent = nullptr;

  /// The number of times this node's string appears in the tree.
  ///
  /// This is equal to the number of leaf children of the string. It represents
  /// the number of suffixes that the node's string is a prefix of.
  size_t OccurrenceCount = 0;

  /// The length of the string formed by concatenating the edge labels from the
  /// root to this node.
  size_t ConcatLen = 0;

  /// Returns true if this node is a leaf.
  bool isLeaf() const { return SuffixIdx != EmptyIdx; }

  /// Returns true if this node is the root of its owning \p SuffixTree.
  bool isRoot() const { return StartIdx == EmptyIdx; }

  /// Return the number of elements in the substring associated with this node.
  size_t size() const {

    // Is it the root? If so, it's the empty string so return 0.
    if (isRoot())
      return 0;

    assert(*EndIdx != EmptyIdx && "EndIdx is undefined!");

    // Size = the number of elements in the string.
    // For example, [0 1 2 3] has length 4, not 3. 3-0 = 3, so we have 3-0+1.
    return *EndIdx - StartIdx + 1;
  }

  SuffixTreeNode(size_t StartIdx, size_t *EndIdx, SuffixTreeNode *Link,
                 SuffixTreeNode *Parent)
      : StartIdx(StartIdx), EndIdx(EndIdx), Link(Link), Parent(Parent) {}

  SuffixTreeNode() {}
};

/// A data structure for fast substring queries.
///
/// Suffix trees represent the suffixes of their input strings in their leaves.
/// A suffix tree is a type of compressed trie structure where each node
/// represents an entire substring rather than a single character. Each leaf
/// of the tree is a suffix.
///
/// A suffix tree can be seen as a type of state machine where each state is a
/// substring of the full string. The tree is structured so that, for a string
/// of length N, there are exactly N leaves in the tree. This structure allows
/// us to quickly find repeated substrings of the input string.
///
/// In this implementation, a "string" is a vector of unsigned integers.
/// These integers may result from hashing some data type. A suffix tree can
/// contain 1 or many strings, which can then be queried as one large string.
///
/// The suffix tree is implemented using Ukkonen's algorithm for linear-time
/// suffix tree construction. Ukkonen's algorithm is explained in more detail
/// in the paper by Esko Ukkonen "On-line construction of suffix trees. The
/// paper is available at
///
/// https://www.cs.helsinki.fi/u/ukkonen/SuffixT1withFigs.pdf
class SuffixTree {
public:
  /// Stores each leaf node in the tree.
  ///
  /// This is used for finding outlining candidates.
  std::vector<SuffixTreeNode *> LeafVector;

  /// Each element is an integer representing an instruction in the module.
  ArrayRef<unsigned> Str;

private:
  /// Maintains each node in the tree.
  SpecificBumpPtrAllocator<SuffixTreeNode> NodeAllocator;

  /// The root of the suffix tree.
  ///
  /// The root represents the empty string. It is maintained by the
  /// \p NodeAllocator like every other node in the tree.
  SuffixTreeNode *Root = nullptr;

  /// Maintains the end indices of the internal nodes in the tree.
  ///
  /// Each internal node is guaranteed to never have its end index change
  /// during the construction algorithm; however, leaves must be updated at
  /// every step. Therefore, we need to store leaf end indices by reference
  /// to avoid updating O(N) leaves at every step of construction. Thus,
  /// every internal node must be allocated its own end index.
  BumpPtrAllocator InternalEndIdxAllocator;

  /// The end index of each leaf in the tree.
  size_t LeafEndIdx = -1;

  /// \brief Helper struct which keeps track of the next insertion point in
  /// Ukkonen's algorithm.
  struct ActiveState {
    /// The next node to insert at.
    SuffixTreeNode *Node;

    /// The index of the first character in the substring currently being added.
    size_t Idx = EmptyIdx;

    /// The length of the substring we have to add at the current step.
    size_t Len = 0;
  };

  /// \brief The point the next insertion will take place at in the
  /// construction algorithm.
  ActiveState Active;

  /// Allocate a leaf node and add it to the tree.
  ///
  /// \param Parent The parent of this node.
  /// \param StartIdx The start index of this node's associated string.
  /// \param Edge The label on the edge leaving \p Parent to this node.
  ///
  /// \returns A pointer to the allocated leaf node.
  SuffixTreeNode *insertLeaf(SuffixTreeNode &Parent, size_t StartIdx,
                             unsigned Edge) {

    assert(StartIdx <= LeafEndIdx && "String can't start after it ends!");

    SuffixTreeNode *N = new (NodeAllocator.Allocate())
        SuffixTreeNode(StartIdx, &LeafEndIdx, nullptr, &Parent);
    Parent.Children[Edge] = N;

    return N;
  }

  /// Allocate an internal node and add it to the tree.
  ///
  /// \param Parent The parent of this node. Only null when allocating the root.
  /// \param StartIdx The start index of this node's associated string.
  /// \param EndIdx The end index of this node's associated string.
  /// \param Edge The label on the edge leaving \p Parent to this node.
  ///
  /// \returns A pointer to the allocated internal node.
  SuffixTreeNode *insertInternalNode(SuffixTreeNode *Parent, size_t StartIdx,
                                     size_t EndIdx, unsigned Edge) {

    assert(StartIdx <= EndIdx && "String can't start after it ends!");
    assert(!(!Parent && StartIdx != EmptyIdx) &&
           "Non-root internal nodes must have parents!");

    size_t *E = new (InternalEndIdxAllocator) size_t(EndIdx);
    SuffixTreeNode *N = new (NodeAllocator.Allocate())
        SuffixTreeNode(StartIdx, E, Root, Parent);
    if (Parent)
      Parent->Children[Edge] = N;

    return N;
  }

  /// \brief Set the suffix indices of the leaves to the start indices of their
  /// respective suffixes. Also stores each leaf in \p LeafVector at its
  /// respective suffix index.
  ///
  /// \param[in] CurrNode The node currently being visited.
  /// \param CurrIdx The current index of the string being visited.
  void setSuffixIndices(SuffixTreeNode &CurrNode, size_t CurrIdx) {

    bool IsLeaf = CurrNode.Children.size() == 0 && !CurrNode.isRoot();

    // Store the length of the concatenation of all strings from the root to
    // this node.
    if (!CurrNode.isRoot()) {
      if (CurrNode.ConcatLen == 0)
        CurrNode.ConcatLen = CurrNode.size();

      if (CurrNode.Parent)
        CurrNode.ConcatLen += CurrNode.Parent->ConcatLen;
    }

    // Traverse the tree depth-first.
    for (auto &ChildPair : CurrNode.Children) {
      assert(ChildPair.second && "Node had a null child!");
      setSuffixIndices(*ChildPair.second, CurrIdx + ChildPair.second->size());
    }

    // Is this node a leaf?
    if (IsLeaf) {
      // If yes, give it a suffix index and bump its parent's occurrence count.
      CurrNode.SuffixIdx = Str.size() - CurrIdx;
      assert(CurrNode.Parent && "CurrNode had no parent!");
      CurrNode.Parent->OccurrenceCount++;

      // Store the leaf in the leaf vector for pruning later.
      LeafVector[CurrNode.SuffixIdx] = &CurrNode;
    }
  }

  /// \brief Construct the suffix tree for the prefix of the input ending at
  /// \p EndIdx.
  ///
  /// Used to construct the full suffix tree iteratively. At the end of each
  /// step, the constructed suffix tree is either a valid suffix tree, or a
  /// suffix tree with implicit suffixes. At the end of the final step, the
  /// suffix tree is a valid tree.
  ///
  /// \param EndIdx The end index of the current prefix in the main string.
  /// \param SuffixesToAdd The number of suffixes that must be added
  /// to complete the suffix tree at the current phase.
  ///
  /// \returns The number of suffixes that have not been added at the end of
  /// this step.
  unsigned extend(size_t EndIdx, size_t SuffixesToAdd) {
    SuffixTreeNode *NeedsLink = nullptr;

    while (SuffixesToAdd > 0) {

      // Are we waiting to add anything other than just the last character?
      if (Active.Len == 0) {
        // If not, then say the active index is the end index.
        Active.Idx = EndIdx;
      }

      assert(Active.Idx <= EndIdx && "Start index can't be after end index!");

      // The first character in the current substring we're looking at.
      unsigned FirstChar = Str[Active.Idx];

      // Have we inserted anything starting with FirstChar at the current node?
      if (Active.Node->Children.count(FirstChar) == 0) {
        // If not, then we can just insert a leaf and move too the next step.
        insertLeaf(*Active.Node, EndIdx, FirstChar);

        // The active node is an internal node, and we visited it, so it must
        // need a link if it doesn't have one.
        if (NeedsLink) {
          NeedsLink->Link = Active.Node;
          NeedsLink = nullptr;
        }
      } else {
        // There's a match with FirstChar, so look for the point in the tree to
        // insert a new node.
        SuffixTreeNode *NextNode = Active.Node->Children[FirstChar];

        size_t SubstringLen = NextNode->size();

        // Is the current suffix we're trying to insert longer than the size of
        // the child we want to move to?
        if (Active.Len >= SubstringLen) {
          // If yes, then consume the characters we've seen and move to the next
          // node.
          Active.Idx += SubstringLen;
          Active.Len -= SubstringLen;
          Active.Node = NextNode;
          continue;
        }

        // Otherwise, the suffix we're trying to insert must be contained in the
        // next node we want to move to.
        unsigned LastChar = Str[EndIdx];

        // Is the string we're trying to insert a substring of the next node?
        if (Str[NextNode->StartIdx + Active.Len] == LastChar) {
          // If yes, then we're done for this step. Remember our insertion point
          // and move to the next end index. At this point, we have an implicit
          // suffix tree.
          if (NeedsLink && !Active.Node->isRoot()) {
            NeedsLink->Link = Active.Node;
            NeedsLink = nullptr;
          }

          Active.Len++;
          break;
        }

        // The string we're trying to insert isn't a substring of the next node,
        // but matches up to a point. Split the node.
        //
        // For example, say we ended our search at a node n and we're trying to
        // insert ABD. Then we'll create a new node s for AB, reduce n to just
        // representing C, and insert a new leaf node l to represent d. This
        // allows us to ensure that if n was a leaf, it remains a leaf.
        //
        //   | ABC  ---split--->  | AB
        //   n                    s
        //                     C / \ D
        //                      n   l

        // The node s from the diagram
        SuffixTreeNode *SplitNode =
            insertInternalNode(Active.Node, NextNode->StartIdx,
                               NextNode->StartIdx + Active.Len - 1, FirstChar);

        // Insert the new node representing the new substring into the tree as
        // a child of the split node. This is the node l from the diagram.
        insertLeaf(*SplitNode, EndIdx, LastChar);

        // Make the old node a child of the split node and update its start
        // index. This is the node n from the diagram.
        NextNode->StartIdx += Active.Len;
        NextNode->Parent = SplitNode;
        SplitNode->Children[Str[NextNode->StartIdx]] = NextNode;

        // SplitNode is an internal node, update the suffix link.
        if (NeedsLink)
          NeedsLink->Link = SplitNode;

        NeedsLink = SplitNode;
      }

      // We've added something new to the tree, so there's one less suffix to
      // add.
      SuffixesToAdd--;

      if (Active.Node->isRoot()) {
        if (Active.Len > 0) {
          Active.Len--;
          Active.Idx = EndIdx - SuffixesToAdd + 1;
        }
      } else {
        // Start the next phase at the next smallest suffix.
        Active.Node = Active.Node->Link;
      }
    }

    return SuffixesToAdd;
  }

public:
  /// Construct a suffix tree from a sequence of unsigned integers.
  ///
  /// \param Str The string to construct the suffix tree for.
  SuffixTree(ArrayRef<unsigned> Str) : Str(Str) {
    Root = insertInternalNode(nullptr, EmptyIdx, EmptyIdx, 0);
    Root->IsInTree = true;
    Active.Node = Root;
    LeafVector = std::vector<SuffixTreeNode *>(Str.size());

    // Keep track of the number of suffixes we have to add of the current
    // prefix.
    size_t SuffixesToAdd = 0;
    Active.Node = Root;

    // Construct the suffix tree iteratively on each prefix of the string.
    // PfxEndIdx is the end index of the current prefix.
    // End is one past the last element in the string.
    for (size_t PfxEndIdx = 0, End = Str.size(); PfxEndIdx < End; PfxEndIdx++) {
      SuffixesToAdd++;
      LeafEndIdx = PfxEndIdx; // Extend each of the leaves.
      SuffixesToAdd = extend(PfxEndIdx, SuffixesToAdd);
    }

    // Set the suffix indices of each leaf.
    assert(Root && "Root node can't be nullptr!");
    setSuffixIndices(*Root, 0);
  }
};
} // namespace

// Suffix array implementation.
namespace {
/// \brief Compute the suffix array.
//   Basic adapted implementation of SA-IS algorithm.
class SuffixArray {
public:
  // Compute the suffix array of /p S with given alphabet size /p AlphabetSize
  // and store the result in /p SA
  static void compute(ArrayRef<unsigned> S, std::vector<int> &SA,
                      unsigned AlphabetSize) {
    SuffixArray SACtr(S.size(), SA);
    SACtr.computeSAIS(S, S.size(), AlphabetSize);
  }

private:
  SuffixArray(size_t ArraySize, std::vector<int> &SA) : SA(SA) {
    SA.resize(ArraySize);
  }

  template <typename T>
  void computeSAIS(ArrayRef<T> S, int N, unsigned AlphabetSize) {
    // Bitvector for LS-type array.
    BitVector LTypeArray(N);

    // Classify each character from S as either LType or SType.
    LTypeArray.set(N - 1);
    for (int i = N - 3, e = 0; i >= e; --i) {
      // S(i) is type S iff: S(i) < S(i+1) or S(i)==S(i+1) and S(i+1) is type
      // S
      if (S[i] < S[i + 1] || (S[i] == S[i + 1] && LTypeArray.test(i + 1)))
        LTypeArray.set(i);
    }

    // Stage 1: Reduce the problem and bucket sort all S-substrings.
    Bucket.resize(AlphabetSize + 1);
    /// Get the bucket ends.
    getBuckets(S, true, N, AlphabetSize);
    for (int i = 0; i < N; ++i)
      SA[i] = -1;
    for (int i = 1; i < N; ++i)
      if (isLMS(i, LTypeArray))
        SA[--Bucket[S[i]]] = i;
    induceSA(S, LTypeArray, N, AlphabetSize);
    Bucket.clear();

    /// Compact the sorted substrings into the first N1 items of the suffix
    /// array.
    int N1 = 0;
    for (int i = 0; i < N; ++i)
      if (isLMS(SA[i], LTypeArray))
        SA[N1++] = SA[i];

    /// Find the lexicographic names of the substrings.
    for (int i = N1; i < N; ++i)
      SA[i] = -1;
    int Name = 0, Prev = -1;
    for (int i = 0; i < N1; ++i) {
      int Pos = SA[i];
      for (int d = 0; d < N; ++d) {
        if (Prev == -1 || S[Pos + d] != S[Prev + d] ||
            LTypeArray.test(Pos + d) != LTypeArray.test(Prev + d)) {
          ++Name;
          Prev = Pos;
          break;
        }
        if (d > 0 &&
            (isLMS(Pos + d, LTypeArray) || isLMS(Prev + d, LTypeArray)))
          break;
      }
      Pos = (Pos % 2 == 0) ? Pos / 2 : (Pos - 1) / 2;
      SA[N1 + Pos] = Name - 1;
    }
    for (int i = N - 1, j = i; i >= N1; --i)
      if (SA[i] >= 0)
        SA[j--] = SA[i];

    // Stage 2: Solve the reduced problem.
    /// If the names aren't unique enough yet, we recurse until they are.
    size_t S1Start = N - N1;
    int *S1 = SA.data() + S1Start;
    if (Name < N1)
      computeSAIS(ArrayRef<int>(S1, N1), N1, Name - 1);
    // Otherwise we can compute the suffix array directly.
    else {
      for (int i = 0; i < N1; ++i)
        SA[S1[i]] = i;
    }

    // Stage 3: Induce the result from the reduced solution.
    Bucket.resize(AlphabetSize + 1);
    /// Place the LMS characters into their respective buckets.
    getBuckets(S, true, N, AlphabetSize);
    /// Get P1.
    for (int i = 1, j = 0; i < N; ++i)
      if (isLMS(i, LTypeArray))
        S1[j++] = i;
    /// Get the index in S.
    for (int i = 0; i < N1; ++i)
      SA[i] = S1[SA[i]];
    /// Initialize the suffix array from N1 to N - 1.
    for (int i = N1; i < N; ++i)
      SA[i] = -1;
    for (int i = N1 - 1; i >= 0; --i) {
      int j = SA[i];
      SA[i] = -1;
      SA[--Bucket[S[j]]] = j;
    }
    induceSA(S, LTypeArray, N, AlphabetSize);
  }

  // Check to see if S(Idx) is a left most S-type character.
  bool isLMS(int Idx, BitVector &LTypeArray) {
    return Idx > 0 && LTypeArray.test(Idx) && !LTypeArray.test(Idx - 1);
  }
  template <typename T>
  void getBuckets(ArrayRef<T> S, bool End, unsigned N, unsigned AlphabetSize) {
    /// Clear buckets.
    Bucket.assign(AlphabetSize + 1, 0);
    /// Compute the size of each bucket.
    for (size_t i = 0, e = S.size(); i < e; ++i)
      ++Bucket[S[i]];
    int Sum = 0;
    if (!End) {
      for (size_t i = 0, e = AlphabetSize + 1; i < e; ++i) {
        Sum += Bucket[i];
        Bucket[i] = Sum - Bucket[i];
      }
    } else
      for (size_t i = 0; i <= AlphabetSize; ++i)
        Bucket[i] = Sum += Bucket[i];
  }

  // Compute SA1
  template <typename T>
  void induceSA(ArrayRef<T> S, BitVector &LTypeArray, unsigned N,
                unsigned AlphabetSize) {
    // Induce SA1
    getBuckets(S, false, N, AlphabetSize);
    for (size_t i = 0; i < N; ++i) {
      int j = SA[i] - 1;
      if (j >= 0 && !LTypeArray.test(j))
        SA[Bucket[S[j]]++] = j;
    }
    // Induce Sas
    getBuckets(S, true, N, AlphabetSize);
    for (ssize_t i = N - 1; i >= 0; --i) {
      int j = SA[i] - 1;
      if (j >= 0 && LTypeArray.test(j))
        SA[--Bucket[S[j]]] = j;
    }
  }
  std::vector<int> &SA;
  std::vector<int> Bucket;
};
// Construct the LCP array for a given suffix array /p SA and string /p S.
static std::vector<int> computeLCP(ArrayRef<unsigned> S, ArrayRef<int> SA) {
  int N = S.size();
  std::vector<int> LCP(N), Rank(N);
  for (int i = 0; i < N; ++i)
    Rank[SA[i]] = i;
  for (int i = 0, k = 0; i < N; ++i) {
    if (Rank[i] == N - 1) {
      k = 0;
      continue;
    }
    int j = SA[Rank[i] + 1];
    while (i + k < N && j + k < N && S[i + k] == S[j + k])
      ++k;
    LCP[Rank[i]] = k;
    if (k > 0)
      --k;
  }
  return LCP;
}
} // namespace

void SequencedOutlinerImpl::OutlinerMapper::prepareForCandidateSelection() {
  // We regroup the illegal indices so that our alphabet is of a defined size.
  unsigned Diff = (UINT_MAX - IllegalID);
  for (unsigned &InstId : CCVec) {
    if (InstId > IllegalID)
      InstId = 1 + (UINT_MAX - InstId);
    else
      InstId += Diff;
  }
  CCID += Diff;

  // REQUIRED: N-1 must be 0 to act as a sentinel for the suffix array
  // algorithm.
  CCVec.push_back(0);
}

void SequencedOutlinerImpl::findSTCandidates(
    std::vector<OutlineCandidate> &CL) {
  ArrayRef<unsigned> CongruencyVec = OM.getCongruencyVector();
  SuffixTree ST(CongruencyVec);

  // An interval tree of our current candidates.
  BitVector CandidateInterval(CongruencyVec.size());

  std::vector<unsigned> Occurrences;
  // FIXME: Visit internal nodes instead of leaves.
  for (SuffixTreeNode *Leaf : ST.LeafVector) {
    assert(Leaf && "Leaves in LeafVector cannot be null!");
    if (!Leaf->IsInTree)
      continue;

    assert(Leaf->Parent && "All leaves must have parents!");
    SuffixTreeNode &Parent = *(Leaf->Parent);

    // If it doesn't appear enough, or we already outlined from it, skip it.
    if (Parent.OccurrenceCount < MinOccurrences || Parent.isRoot() ||
        !Parent.IsInTree)
      continue;

    // How many instructions would outlining this string save?
    size_t StringLen = Leaf->ConcatLen - Leaf->size();

    Occurrences.clear();
    for (auto &ChildPair : Parent.Children) {
      SuffixTreeNode *M = ChildPair.second;
      unsigned Idx = M->SuffixIdx;

      // Is it a leaf? If so, we have an occurrence of this candidate.
      if (M && M->IsInTree && M->isLeaf() &&
          CandidateInterval.find_first_in(Idx, Idx + StringLen) == -1) {
        CandidateInterval.set(Idx, Idx + StringLen);
        Occurrences.push_back(Idx);
        M->IsInTree = false;
      }
    }
    for (unsigned Idx : CandidateInterval.set_bits())
      CandidateInterval.reset(Idx, Idx + StringLen);

    // If it's not beneficial, skip it.
    if (prePrune(Occurrences, StringLen))
      continue;

    CL.emplace_back(StringLen, Occurrences);

    // Move to the next function.
    Parent.IsInTree = false;
  }
}
void SequencedOutlinerImpl::findSACandidates(
    std::vector<OutlineCandidate> &CL) {
  // Build the suffix array and longest common prefix array.
  ArrayRef<unsigned> CongruencyVec = OM.getCongruencyVector();
  std::vector<int> SuffixArr, LcpArr;
  SuffixArray::compute(CongruencyVec, SuffixArr, OM.getNumCongruencyClasses());
  LcpArr = computeLCP(CongruencyVec, SuffixArr);

  // An interval tree of our current candidates.
  BitVector CandidateInterval(CongruencyVec.size());

  // Try to guess the amount of candidates we could have for this module.
  //  * Tuned via clang build *
  size_t NumPotentialOccurrences = 0, CurrentSizeSeq = 1;
  std::for_each(LcpArr.begin(), LcpArr.end(), [&](unsigned Size) {
    if (Size >= MinInstructionLen)
      NumPotentialOccurrences += ++CurrentSizeSeq;
    else
      CurrentSizeSeq = 1;
  });
  CL.reserve(NumPotentialOccurrences * 0.025);

  // Walk the suffix array to build potential candidates.
  SmallDenseSet<size_t, 16> FailedOccurrences;
  size_t PrevSize = 0;
  std::vector<unsigned> Occurrences;
  for (size_t i = 1, e = SuffixArr.size(); i < e; ++i) {
    size_t Size = LcpArr[i];

    // Preskip invalid size.
    if (Size < MinInstructionLen) {
      PrevSize = 0;
      continue;
    }

    size_t OccurIdx = SuffixArr[i];

    // We have already matched against this size.
    if (PrevSize >= Size) {
      PrevSize = Size;
      continue;
    }

    // Create a new interval tree with our current candidate to pre prune
    //   overlaps.
    Occurrences.clear();
    CandidateInterval.set(OccurIdx, OccurIdx + Size);
    Occurrences.push_back(OccurIdx);
    FailedOccurrences.clear();
    bool HasPreviousSharedOccurrence = false;

    // Continuously consider potentital chain sizes for this candidate until
    // they are no longer profitable.
    size_t OrigSize = Size, LastValidSize = 0;
    for (size_t SizeFromIdx = i, AugmentAmount = 0;
         Size >= MinInstructionLen;) {
      bool AddedNewOccurrence = false;

      // Augment the candidate set by the change in size from the
      // last iteration.
      if (AugmentAmount > 0)
        for (size_t Idx : Occurrences)
          CandidateInterval.reset(Idx + Size, Idx + Size + AugmentAmount);
      LastValidSize = Size;

      // After augmenting the candidate set, there may be new candidates that
      // no longer overlap with any of the others currently being considered.
      for (auto It = FailedOccurrences.begin(), E = FailedOccurrences.end();
           It != E;) {
        size_t Idx = *It;
        ++It;
        if (CandidateInterval.find_first_in(Idx, Idx + Size) != -1)
          continue;
        FailedOccurrences.erase(Idx);
        CandidateInterval.set(Idx, Idx + Size);
        Occurrences.push_back(Idx);
        AddedNewOccurrence = true;
      }

      // Count the number of occurrences.
      for (size_t j = i + Occurrences.size(); j < e; ++j) {
        // The longest common prefix must be able to hold our size.
        if ((size_t)LcpArr[j - 1] < Size)
          break;

        // Check to see if this candidate overlaps with any of our currently
        // considered candidates. If it doesn't we add it to our current set.
        size_t JIdx = SuffixArr[j];
        if (CandidateInterval.find_first_in(JIdx, JIdx + Size) == -1) {
          CandidateInterval.set(JIdx, JIdx + Size);
          Occurrences.push_back(JIdx);
          AddedNewOccurrence = true;
        } else
          FailedOccurrences.insert(JIdx);

        // If our next size is less than the current, we won't get any more
        //  candidates for this chain.
        if ((size_t)LcpArr[j] < Size)
          break;
      }

      // If we added a new candidate and we have enough to satisfy our
      // constraints then we build a new outline chain candidate.
      if (AddedNewOccurrence && Occurrences.size() >= MinOccurrences) {
        // Recheck the prune size each iteration.
        if (!prePrune(Occurrences, Size)) {
          /// Cache shared sizes between candidates chains to make analysis
          /// easier.
          if (HasPreviousSharedOccurrence)
            CL.back().SharedSizeWithNext = Size;
          else
            HasPreviousSharedOccurrence = true;
          /// Build new function with candidate sequence.
          CL.emplace_back(Size, Occurrences);
        }
      }

      // Find the next size to consider for this candidate.
      for (size_t NewSizeE = e - 1; ++SizeFromIdx < NewSizeE;) {
        size_t NewSize = static_cast<size_t>(LcpArr[SizeFromIdx]);
        if (NewSize < Size) {
          AugmentAmount = Size - NewSize;
          Size = NewSize;
          break;
        }
      }

      // If we have already encountered a greater size, then the new size
      //  was either invalid or we've already considered this size but
      //  with more candidates.
      if (Size == LastValidSize || Size <= PrevSize)
        break;
    }
    for (unsigned Idx : Occurrences)
      CandidateInterval.reset(Idx, Idx + LastValidSize);
    PrevSize = OrigSize;
  }
}
void SequencedOutlinerImpl::findOutliningOccurrences(
    std::vector<OutlineCandidate> &CL) {
  OM.prepareForCandidateSelection();
  CL.clear();
  if (CSM == CSM_SuffixArray)
    findSACandidates(CL);
  else if (CSM == CSM_SuffixTree)
    findSTCandidates(CL);
  else
    llvm_unreachable("Unknown outliner candidate selection method.");
}

bool SequencedOutlinerImpl::pruneAndOutline(std::vector<OutlineCandidate> &CL) {
  // Helper comparator for candidate indexes.
  struct Comparator {
    Comparator(ArrayRef<OutlineCandidate> CL) : CL(CL) {}
    bool operator()(unsigned L, unsigned R) {
      return CL[L].Benefit < CL[R].Benefit;
    }
    ArrayRef<OutlineCandidate> CL;
  };

  // Build a priority worklist for the valid candidates.
  std::vector<unsigned> OriginalSetOfValidCands;
  OriginalSetOfValidCands.reserve(CL.size());
  for (unsigned i = 0, e = CL.size(); i < e; ++i)
    if (CL[i].isValid())
      OriginalSetOfValidCands.push_back(i);

  Comparator C(CL);
  PriorityQueue<unsigned, std::vector<unsigned>, Comparator> MostBenefitial(
      C, OriginalSetOfValidCands);
  BitVector InsertedOccurrences(OM.getNumMappedInstructions());
  BitVector ValidOccurrencesPerCandidate;
  size_t OutlinedCandidates = 0;
  errs() << "Starting search for benefitial candidates\n";
  while (!MostBenefitial.empty()) {
    unsigned CandIdx = MostBenefitial.top();
    MostBenefitial.pop();

    OutlineCandidate &OC = CL[CandIdx];
    ValidOccurrencesPerCandidate.reset();
    ValidOccurrencesPerCandidate.resize(OC.size());

    // Check overlaps.
    for (ssize_t i = OC.size() - 1; i >= 0; --i) {
      unsigned Occur = OC.getOccurrence(i);
      if (InsertedOccurrences.find_first_in(Occur, Occur + OC.Len) == -1) {
        ValidOccurrencesPerCandidate.set(i);
        continue;
      }
      if (OC.Benefit < OC.BenefitPerOccur) {
        OC.invalidate();
        break;
      }
      OC.Benefit -= OC.BenefitPerOccur;
      OC.removeOccurrence(i);
    }

    errs() << "Testing Candidate Validation\n";
    // Add valid occurrences if this candidate is still profitable.
    if (!OC.isValid())
      continue;

    errs() << "Found Profitable Candidate\n";

    // If we have a cheap benefit function then we update the benefit
    //  to get the candidate that is actually the best.
    if (ValidOccurrencesPerCandidate.size() != OC.size()) {
      MostBenefitial.push(CandIdx);
      continue;
    }

    errs() << "Executing outline transformation\n";

    // Outline the remaining valid occurrences from this candidate.
    outlineCandidate(OC, OutlinedCandidates++);
    for (unsigned OccurIdx : ValidOccurrencesPerCandidate.set_bits()) {
      unsigned Occur = OC.getOccurrence(OccurIdx);
      InsertedOccurrences.set(Occur, Occur + OC.Len);
    }
  }
  return OutlinedCandidates > 0;
}