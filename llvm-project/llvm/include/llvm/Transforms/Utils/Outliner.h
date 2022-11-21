//===- Outliner.h - A generic outlining utility interface around the Utils lib
//------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file includes utilities for defining outliner functionality.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TRANSFORMS_UTILS_OUTLINER_H
#define LLVM_TRANSFORMS_UTILS_OUTLINER_H

#include "llvm/ADT/ArrayRef.h"
#include <vector>

namespace llvm {
class BitVector;

/// \brief A potential candidate to be outlined.
struct OutlineCandidate {
  struct AdditionalData {};
  /// The amount of instructions being saved.
  unsigned Len;

  /// The computed benefit of outlining this candidate.
  unsigned Benefit = 0;

  /// The estimated benefit we receive per occurrence.
  unsigned BenefitPerOccur = 0;

  /// Identifier for each occurrence.
  std::vector<unsigned> Occurrences;

  /// Instruction size that this candidate shares with the next.
  unsigned SharedSizeWithNext = 0;

  /// Additional info attached to this candidate.
  AdditionalData *Data = nullptr;

  // Accessors.
  using Iterator = std::vector<unsigned>::iterator;
  Iterator begin() { return Occurrences.begin(); }
  Iterator end() { return Occurrences.end(); }
  size_t size() const { return Occurrences.size(); }

  // Check to see if this chain is still profitable to outline.
  bool isValid() const { return Benefit != 0; }
  // Set this candidate as not profitable.
  void invalidate() { Benefit = 0; }
  // Get the candidate at index /p Idx.
  unsigned getOccurrence(size_t Idx) const {
    assert(Idx < size() && "Invalid occurrence index.");
    return Occurrences[Idx];
  }
  // Remove the occurrence at index /p Idx
  void removeOccurrence(size_t Idx) {
    Occurrences[Idx] = Occurrences.back();
    Occurrences.pop_back();
  }
  // Get the additional data attached.
  template <typename T> T &getData() {
    assert(Data && "Getting an invalid data pointer.");
    return *static_cast<T *>(Data);
  }

  OutlineCandidate(unsigned Len, std::vector<unsigned> &Occurrences)
      : Len(Len), Occurrences(Occurrences) {}
  OutlineCandidate(unsigned Len) : Len(Len) {}
};

/// \brief The base interface for outliner functionality.
class OutlinerImpl {
public:
  virtual ~OutlinerImpl() {}
  bool run();

protected:
  // Main interface functions.
  virtual void findOutliningOccurrences(std::vector<OutlineCandidate> &) = 0;
  virtual void analyzeCandidateList(std::vector<OutlineCandidate> &) = 0;
  virtual bool pruneAndOutline(std::vector<OutlineCandidate> &) = 0;
};

/// \brief Outliner impl for outlining sequences.
class SequencedOutlinerImpl : public OutlinerImpl {
public:
  /// \brief A base mapper that is used to map instructions to a congruency
  ///  vector.
  class OutlinerMapper {
  public:
    friend class SequencedOutlinerImpl;
    virtual ~OutlinerMapper() {}

    // Get the unsigned vector representation of the congruencies for the
    // module.
    ArrayRef<unsigned> getCongruencyVector() const { return CCVec; }
    // Get the number of different congruency classes found in the module.
    unsigned getNumCongruencyClasses() const { return CCID; }
    // Get the number of instructions mapped in this module.
    virtual unsigned getNumMappedInstructions() const = 0;

  protected:
    /// Internal vector representation of the instructions within the mapped
    /// module.
    std::vector<unsigned> CCVec;

    /// Current Congruency ID.
    unsigned CCID = 1;

    /// An id for illegal instructions.
    unsigned IllegalID = UINT_MAX;

  private:
    // Helper utility to finalize the congruency vector for use in candidate
    //  selection.
    void prepareForCandidateSelection();
  };

  // The method that the impl uses to find outlining candidates.
  enum CandidateSelectionMethod {
    CSM_SuffixTree,
    CSM_SuffixArray,
    CSM_None
  };

  SequencedOutlinerImpl(
      OutlinerMapper &OM, unsigned MinInstructionLen, unsigned MinOccurrences,
      CandidateSelectionMethod CSM)
      : OM(OM), MinInstructionLen(MinInstructionLen),
        MinOccurrences(MinOccurrences), CSM(CSM) {}
  virtual ~SequencedOutlinerImpl() {}

protected:
  // Checks to see whether we should prune an abstract candidate.
  virtual bool prePrune(ArrayRef<unsigned> Occurs, unsigned Size) {
    return false;
  };
  // Outline a profitable candidate.
  virtual void outlineCandidate(OutlineCandidate &Cand, size_t CandNum) = 0;

  // Helper for getting the mapper as a parent type.
  template<typename T>
  T &getMapperAs() {
    return static_cast<T &>(OM);
  }

  // A mapper that holds the state of the current module.
  OutlinerMapper &OM;

  // Metric utilities
  unsigned MinInstructionLen;
  unsigned MinOccurrences;
private:
  // Helpers for finding candidates in different ways.
  CandidateSelectionMethod CSM;

  void findSTCandidates(std::vector<OutlineCandidate> &CL);
  void findSACandidates(std::vector<OutlineCandidate> &CL);

  // Base overrides.
  virtual void
  findOutliningOccurrences(std::vector<OutlineCandidate> &) override;
  virtual bool pruneAndOutline(std::vector<OutlineCandidate> &) override;
};
} // namespace llvm

#endif // LLVM_TRANSFORMS_UTILS_OUTLINER_H