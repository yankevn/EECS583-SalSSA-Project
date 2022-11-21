//===-- CodeSizeOutliner.cpp - Propagate constants through calls -----===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This pass implements a simple algorithm for outlining congruent chains of
//  instructions from the current module.
//
//===----------------------------------------------------------------------===//
#include "llvm/Transforms/IPO/CodeSizeOutliner.h"
#include "llvm/ADT/BitVector.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/SparseBitVector.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/Analysis/BranchProbabilityInfo.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/OptimizationRemarkEmitter.h"
#include "llvm/Analysis/ProfileSummaryInfo.h"
#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/Bitcode/BitcodeWriterPass.h"
#include "llvm/IR/CallSite.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DIBuilder.h"
#include "llvm/IR/DiagnosticInfo.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/Pass.h"
#include "llvm/Transforms/IPO.h"
#include "llvm/Transforms/Utils/Local.h"
#include "llvm/Transforms/Utils/Outliner.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Alignment.h"
#include <sstream>

using namespace llvm;

static cl::opt<unsigned> MinOccurrences(
    "cso-min-occurrences", cl::init(2), cl::Hidden,
    cl::desc(
        "Min number of occurrences to consider a candidate for outlining."));
static cl::opt<unsigned> MinInstructionLength(
    "cso-min-instructions", cl::init(1), cl::Hidden,
    cl::desc(
        "Min number of instructions to consider a candidate for outlining."));
static cl::opt<unsigned>
    MinBenefit("cso-min-benefit", cl::init(1), cl::Hidden,
               cl::desc("Min estimated benefit to be considered profitable."));
static cl::opt<bool>
    EnableOutliner("enable-cso", cl::init(false), cl::Hidden,
                   cl::desc("Enable outlining for code size."));
static cl::opt<bool> DumpCC(
    "cso-dump-cc", cl::init(false), cl::Hidden,
    cl::desc("Dump information about the congruent between instructions."));

#define DEBUG_TYPE "codesizeoutliner"
STATISTIC(NumOccurrencesOutlined, "Number of occurrences outlined");
STATISTIC(NumCandidatesOutlined, "Number of outlined functions created");

namespace {
/// \brief The information necessary to create an outlined function from a set
/// of repeated instruction occurrences.
struct AdditionalCandidateData : public OutlineCandidate::AdditionalData {
  /// Inputs into this candidate : Vector<Instr Index, Op#>.
  std::vector<std::pair<size_t, size_t>> InputSeq;

  /// Outputs from this candidate.
  SparseBitVector<> Outputs;

  /// The index of the output to fold into a return.
  unsigned OutputToFold = -1;
};

/// \brief GVN Expression with relaxed equivalency constraints.
class RelaxedExpression {
public:
  /// A special state for special equivalence constraints.
  enum SpecialState { StructGep, ConstrainedCall, None };

  RelaxedExpression(Instruction &I)
    : Inst(&I), SS(SpecialState::None), HashVal(0) {
    // Check the special state.
    /// Struct geps require constant indices.
    if (GetElementPtrInst *GEP = dyn_cast<GetElementPtrInst>(&I)) {
      if (gepContainsStructType(GEP))
        SS = RelaxedExpression::StructGep;
    } else if (CallInst *CI = dyn_cast<CallInst>(&I)) {
      /// Don't take the address of inline asm calls.
      if (CI->isInlineAsm())
        SS = RelaxedExpression::ConstrainedCall;
      /// Intrinsics and functions without exact definitions can not
      ///  have their address taken.
      else if (Function *F = CI->getCalledFunction()) {
        if (!F->hasExactDefinition())
          SS = RelaxedExpression::ConstrainedCall;
      }
    }
  }
  bool equals(const RelaxedExpression &OE) const {
    if (SS != OE.SS)
      return false;
    // Handle calls separately to allow for mismatched tail calls.
    if (CallInst *CI = dyn_cast<CallInst>(Inst)) {
      const CallInst *RCI = cast<CallInst>(OE.Inst);
      if (CI->getFunctionType() != RCI->getFunctionType() ||
          CI->getNumOperands() != RCI->getNumOperands())
        return false;
      if (CI->getCallingConv() != RCI->getCallingConv() ||
          CI->getAttributes() != RCI->getAttributes() ||
          !CI->hasIdenticalOperandBundleSchema(*RCI))
        return false;
    } else if (!Inst->isSameOperationAs(OE.Inst,
                                        Instruction::CompareIgnoringAlignment))
      return false;
    return checkSpecialEquivalence(OE);
  }
  // Special checks for instructions that have non generic equivalency.
  bool checkSpecialEquivalence(const RelaxedExpression &Other) const {
    const Instruction *OE = Other.Inst;
    switch (Inst->getOpcode()) {
    case Instruction::ShuffleVector: {
      // LangRef : The shuffle mask operand is required to be a
      //  constant vector with either constant integer or undef values.
      return Inst->getOperand(2) == OE->getOperand(2);
    }
    case Instruction::Call: {
      const CallInst *CI = cast<CallInst>(Inst);
      const CallInst *OECI = cast<CallInst>(OE);
      if (SS == ConstrainedCall)
        return checkConstrainedCallEquivalence(CI, OECI);
      break;
    }
    case Instruction::GetElementPtr: {
      // Struct indices must be constant.
      if (SS == StructGep)
        return compareStructIndices(cast<GetElementPtrInst>(Inst),
                                    cast<GetElementPtrInst>(OE));
      break;
    }
    default:
      break;
    }
    return true;
  }
  static bool checkConstrainedCallEquivalence(const CallInst *CI,
                                              const CallInst *OECI) {
    const Value *CIVal = CI->getCalledValue();
    if (CIVal != OECI->getCalledValue())
      return false;
    if (const Function *CalledII = dyn_cast<Function>(CIVal)) {
      errs() << "Breaking here!\n";
      return false; //TODO: fix checking of intrinsics
      switch (CalledII->getIntrinsicID()) {
      case Intrinsic::memmove:
      case Intrinsic::memcpy:
      case Intrinsic::memset:
        /// Alignment.
        return CI->getArgOperand(3) == OECI->getArgOperand(3) &&
               /// Volatile flag.
               CI->getArgOperand(4) == OECI->getArgOperand(4);
      case Intrinsic::objectsize:
        /// Min.
        return CI->getArgOperand(1) == OECI->getArgOperand(1) &&
               /// Null unknown.
               CI->getArgOperand(2) == OECI->getArgOperand(2);
      case Intrinsic::expect:
        /// Expected value.
        return CI->getArgOperand(1) == OECI->getArgOperand(1);
      case Intrinsic::prefetch:
        /// RW.
        return CI->getArgOperand(1) == OECI->getArgOperand(1) &&
               /// Locality.
               CI->getArgOperand(2) == OECI->getArgOperand(2) &&
               /// Cache Type.
               CI->getArgOperand(3) == OECI->getArgOperand(3);
      case Intrinsic::ctlz:
      case Intrinsic::cttz:
        /// Is Zero Undef
        return CI->getArgOperand(1) == OECI->getArgOperand(1);
      default:
        break;
      }
      errs() << "Passed!\n";
    }
    return true;
  }

  static bool compareStructIndices(GetElementPtrInst *L,
                                   const GetElementPtrInst *R) {
    SmallVector<Value *, 8> GepIdxs(L->indices().begin(), L->indices().end());
    GepIdxs.pop_back();
    while (!GepIdxs.empty()) {
      Type *PtrTy = L->getGEPReturnType(dyn_cast<PointerType>(L->getPointerOperandType())->getElementType(), L->getPointerOperand(), ArrayRef<Value*>(GepIdxs));
      if (PtrTy->isPointerTy() &&
          PtrTy->getPointerElementType()->isStructTy()) {
        unsigned OpNo = GepIdxs.size() + 1;
        if (L->getOperand(OpNo) != R->getOperand(OpNo))
          return false;
      }
      GepIdxs.pop_back();
    }
    return true;
  }
  // Check to see if the provided gep /p GEP indexes a struct type.
  bool gepContainsStructType(GetElementPtrInst *GEP) {
    Type *PtrTy = GEP->getPointerOperandType();
    if (PtrTy->isPointerTy() && PtrTy->getPointerElementType()->isStructTy())
      return true;

    SmallVector<Value *, 8> GepIdxs(GEP->indices().begin(),
                                    GEP->indices().end());
    GepIdxs.pop_back();
    while (!GepIdxs.empty()) {
      PtrTy = GEP->getGEPReturnType(dyn_cast<PointerType>(GEP->getPointerOperandType())->getElementType(), GEP->getPointerOperand(), GepIdxs);
      if (PtrTy->isPointerTy() && PtrTy->getPointerElementType()->isStructTy())
        return true;
      GepIdxs.pop_back();
    }
    return false;
  }

  hash_code getComputedHash() const {
    if (static_cast<unsigned>(HashVal) == 0)
      HashVal = getHashValue();
    return HashVal;
  }
  hash_code getHashValue() const {
    SmallVector<size_t, 8> HashRange;
    HashRange.push_back(SS);
    HashRange.push_back(Inst->getNumOperands());
    HashRange.push_back(reinterpret_cast<size_t>(Inst->getType()));
    for (auto &Op : Inst->operands())
      HashRange.emplace_back(reinterpret_cast<size_t>(Op->getType()));
    return hash_combine_range(HashRange.begin(), HashRange.end());
  }

private:
  Instruction *Inst;
  SpecialState SS;
  mutable hash_code HashVal;
};
}; // namespace

namespace llvm {
// DenseMapInfo for the relaxed expression class.
template <> struct DenseMapInfo<const RelaxedExpression *> {
  static const RelaxedExpression *getEmptyKey() {
    auto Val = static_cast<uintptr_t>(-1);
    Val <<=
        PointerLikeTypeTraits<const RelaxedExpression *>::NumLowBitsAvailable;
    return reinterpret_cast<const RelaxedExpression *>(Val);
  }
  static const RelaxedExpression *getTombstoneKey() {
    auto Val = static_cast<uintptr_t>(~1U);
    Val <<=
        PointerLikeTypeTraits<const RelaxedExpression *>::NumLowBitsAvailable;
    return reinterpret_cast<const RelaxedExpression *>(Val);
  }
  static unsigned getHashValue(const RelaxedExpression *E) {
    return E->getComputedHash();
  }
  static bool isEqual(const RelaxedExpression *LHS,
                      const RelaxedExpression *RHS) {
    if (LHS == RHS)
      return true;
    if (LHS == getTombstoneKey() || RHS == getTombstoneKey() ||
        LHS == getEmptyKey() || RHS == getEmptyKey())
      return false;
    // Compare hashes before equality.  This is *not* what the hashtable does,
    // since it is computing it modulo the number of buckets, whereas we are
    // using the full hash keyspace.  Since the hashes are precomputed, this
    // check is *much* faster than equality.
    if (LHS->getComputedHash() != RHS->getComputedHash())
      return false;
    return LHS->equals(*RHS);
  }
};
} // namespace llvm

namespace {
/// \brief Helper struct containing mapping information for a module.
class IRInstructionMapper : public SequencedOutlinerImpl::OutlinerMapper {
public:
  virtual ~IRInstructionMapper() { ExpressionAllocator.Reset(); }

  ///
  ///  Instruction Information Utilities.
  ///

  /// \brief Holds information about a particular instruction.
  struct InstructionInfo {
    /// The inputs/operands going into this instruction.
    SmallVector<unsigned, 4> InputIndexes;

    /// The index of the farthest use of this instruction in the same block
    /// parent.
    unsigned FarthestInSameBlockOutput = 0;

    /// The size cost of this instruction.
    unsigned Cost = -1;
  };

  // Get the instruction at index /p Idx.
  Instruction *getInstr(unsigned Idx) { return InstrVec[Idx]; }
  // Get the index of /p I inside of the internal vector.
  unsigned getInstrIdx(Instruction *I) {
    auto It = InstructionToIdxMap.find(I);
    return LLVM_UNLIKELY(It == InstructionToIdxMap.end()) ? -1 : It->second;
  }
  // Get the parent function of the instruction at /p Idx.
  Function *getInstrFunction(size_t Idx) {
    assert(Idx < InstrVec.size() && "Invalid instruction index");
    return InstrVec[Idx]->getFunction();
  }
  // Get or compute the cost of the instruction at /p InstrIdx.
  unsigned getInstrCost(TargetTransformInfo &TTI, size_t InstrIdx) {
    InstructionInfo *Info = InstrInfo[InstrIdx];
    if (Info->Cost == unsigned(-1)) {
      Instruction *I = InstrVec[InstrIdx];
      switch (I->getOpcode()) {
      case Instruction::FDiv:
      case Instruction::FRem:
      case Instruction::SDiv:
      case Instruction::SRem:
      case Instruction::UDiv:
      case Instruction::URem:
        Info->Cost = TargetTransformInfo::TCC_Basic * 2;
        break;
      default:
        Info->Cost = TTI.getUserCost(I);
        break;
      }
      // Be conservative about the cost of loads given they may be folded.
      //  Estimating a lower cost helps to prevent over estimating the
      //  benefit of this instruction.
      if (isa<LoadInst>(I))
        --Info->Cost;
      // We give a little bonus to tail calls as they are likely to be more
      // beneficial
      //  within our outlined function.
      if (CallInst *CI = dyn_cast<CallInst>(I))
        if (CI->isTailCall())
          ++Info->Cost;
    }
    return Info->Cost;
  }
  // Get the instruction info attached to index /p InstrIdx
  InstructionInfo &getInstrInfo(size_t InstrIdx) {
    InstructionInfo *Info = InstrInfo[InstrIdx];
    assert(Info && "Queried instruction has no info created.");
    return *Info;
  }
  // Create instruction info for the instruction at /p InstrIdx
  void createInstrInfo(size_t InstrIdx) {
    InstructionInfo *&Info = InstrInfo[InstrIdx];
    assert(!Info && "Instruction info already generated.");
    Info = new (InfoAllocator.Allocate()) InstructionInfo();
    Instruction *Inst = InstrVec[InstrIdx];

    /// Inputs.
    unsigned NumOperands = Inst->getNumOperands();
    Info->InputIndexes.reserve(NumOperands);
    for (unsigned InIt = 0; InIt < NumOperands; ++InIt) {
      unsigned IIdx = 0;
      Value *Op = Inst->getOperand(InIt);
      assert(!isa<BasicBlock>(Op) && "Basic block inputs can not be handled.");
      if (Instruction *I = dyn_cast<Instruction>(Op)) {
        unsigned FoundIIdx = getInstrIdx(I);
        if (FoundIIdx <= InstrIdx)
          IIdx = FoundIIdx;
      }
      Info->InputIndexes.emplace_back(IIdx);
    }

    /// Outputs.
    for (User *Usr : Inst->users()) {
      Instruction *I = dyn_cast<Instruction>(Usr);
      if (!I || I->getParent() != Inst->getParent()) {
        Info->FarthestInSameBlockOutput = -1;
        break;
      }
      unsigned IIdx = getInstrIdx(I);
      if (IIdx > Info->FarthestInSameBlockOutput)
        Info->FarthestInSameBlockOutput = IIdx;
    }
  }

  ///
  ///  Module Mapping Utilities.
  ///

  // Map the module /p M to prepare for outlining.
  void mapModule(Module &M, ProfileSummaryInfo *PSI,
                 function_ref<BlockFrequencyInfo &(Function &)> GetBFI) {
    bool HasProfileData = PSI->hasProfileSummary();

    // Insert illegal ID at the front to act as a sentinel.
    InstrVec.push_back(nullptr);
    CCVec.push_back(IllegalID--);

    for (Function &F : M) {
      if (!F.hasExactDefinition() || F.hasGC() || F.isDefTriviallyDead() || F.isDeclaration())
        continue;
      if (F.hasFnAttribute(Attribute::OptimizeNone))
        continue;
      // Don't outline non -Oz functions without profile data.
      bool FavorSize = true; // F.hasMinSize(); //F.hasOptSize();
      if (!FavorSize && !HasProfileData)
        continue;
      BlockFrequencyInfo *BFI = HasProfileData ? &GetBFI(F) : nullptr;
      for (BasicBlock &BB : F) {
        if (BB.isEHPad())
          continue;

        // Skip hot blocks if we have profile data.
        if (HasProfileData)
          if (FavorSize ? PSI->isHotBlock(&BB, BFI) : !PSI->isColdBlock(&BB, BFI))
            continue;

        // Try to map each instruction to a congruency id.
        for (Instruction &I : BB) {
          // Ignore debug info intrinsics.
          if (isa<DbgInfoIntrinsic>(&I))
            continue;
          InstrVec.push_back(&I);
          if (canMapInstruction(&I)) {
            InstructionToIdxMap.try_emplace(&I, CCVec.size());
            CCVec.push_back(mapInstruction(I));
          } else
            CCVec.push_back(IllegalID--);
        }
      }
    }
    InstrInfo.assign(InstrVec.size(), nullptr);
    if (DumpCC)
      dumpCC(M);
  }
  // Get the number of instructions mapped in this module.
  virtual unsigned getNumMappedInstructions() const override {
    return InstrVec.size();
  }

private:
  // Map the instruction /p I to a congruency class.
  unsigned mapInstruction(Instruction &I) {
    // We map each valid instruction to a Relaxed expression and use this for
    // detecting congruency.
    auto *E = new (ExpressionAllocator) RelaxedExpression(I);
    // Assign a CC id to this instruction.
    auto ItPair = GlobalCC[I.getOpcode()].try_emplace(E, CCID);
    if (ItPair.second)
      ++CCID;
    return ItPair.first->second;
  }
  // Checks to see if a provided instruction /p I can be mapped.
  bool canMapInstruction(Instruction *I) {
    if (I->getOpcode() == Instruction::Call) {
      CallInst *CI = cast<CallInst>(I);
      // Be very conservative about musttail because it has additional
      // guarantees that must be met.
      if (CI->isMustTailCall())
        return false;
      // Be conservative about return twice calls.
      if (CI->canReturnTwice())
        return false;
      CallSite CS(CI);
      switch (CS.getIntrinsicID()) {
      case Intrinsic::objectsize:
      case Intrinsic::expect:
      case Intrinsic::prefetch:
      // Lib C functions are fine.
      case Intrinsic::memcpy:
      case Intrinsic::memmove:
      case Intrinsic::memset:
      case Intrinsic::sqrt:
      case Intrinsic::pow:
      case Intrinsic::powi:
      case Intrinsic::sin:
      case Intrinsic::cos:
      case Intrinsic::exp:
      case Intrinsic::exp2:
      case Intrinsic::log:
      case Intrinsic::log2:
      case Intrinsic::log10:
      case Intrinsic::fma:
      case Intrinsic::fabs:
      case Intrinsic::minnum:
      case Intrinsic::maxnum:
      case Intrinsic::copysign:
      case Intrinsic::floor:
      case Intrinsic::ceil:
      case Intrinsic::trunc:
      case Intrinsic::rint:
      case Intrinsic::nearbyint:
      case Intrinsic::round:
      // Bit manipulation intrinsics.
      case Intrinsic::bitreverse:
      case Intrinsic::bswap:
      case Intrinsic::ctpop:
      case Intrinsic::ctlz:
      case Intrinsic::cttz:
        return true;
      // Non intrinsics are fine if they don't have an inalloca arg.
      case Intrinsic::not_intrinsic:
        return !CS.hasInAllocaArgument();
      default:
        return false;
      }
      return true;
    }
    return !(isa<AllocaInst>(I) || isa<PHINode>(I) || I->isTerminator());
  }
  // Dump the mapped congruencies found for the module /p M.
  void dumpCC(Module &M) {
    for (Function &F : M) {
      dbgs() << "function : " << F.getName() << "\n";
      for (auto &BB : F) {
        dbgs() << "block : " << BB.getName() << "\n";
        for (Instruction &I : BB) {
          auto It = InstructionToIdxMap.find(&I);
          size_t Idx = It == InstructionToIdxMap.end() ? -1 : CCVec[It->second];
          dbgs() << "-- " << Idx << " : " << I << '\n';
        }
      }
    }
    for (auto &CCIt : GlobalCC) {
      for (auto CC : CCIt) {
        size_t Idx = CC.second;
        dbgs() << "-- Examining CC ID : " << Idx << "\n";
        for (size_t i = 0, e = CCVec.size(); i < e; ++i)
          if (CCVec[i] == Idx)
            dbgs() << " - " << *InstrVec[i] << "\n";
        dbgs() << "\n";
      }
    }
  }

  /// Stores location of instructions mapped to the corresponding index in
  ///  the CCVec.
  std::vector<Instruction *> InstrVec;

  /// Stores information for parallel instruction in InstrVec.
  std::vector<InstructionInfo *> InstrInfo;
  SpecificBumpPtrAllocator<InstructionInfo> InfoAllocator;

  /// Map<Instruction, Index in CCVec>
  DenseMap<Instruction *, unsigned> InstructionToIdxMap;

  /// Mapping of expression to congruency id.
  std::array<DenseMap<const RelaxedExpression *, unsigned>,
             Instruction::OtherOpsEnd>
      GlobalCC;

  /// Memory management for the GVN expressions used for congruency.
  mutable BumpPtrAllocator ExpressionAllocator;
};

/// \brief Cache output allocas created during outlining and reuse them.
class OutputAllocaCache {
public:
  // Get an alloca for type /p Ty in function /p F that has an array
  //  count of atleast /p Count.
  AllocaInst *getFreeAlloca(Function *F, Type *Ty, unsigned Count) {
    if (AllocaInst *NewAI = findAndUpdate(F, Ty, Count))
      return NewAI;
    // Create the new alloca inst.
    Instruction *AllocaInsertPnt = &F->getEntryBlock().front();
    Value *AllocSize =
        ConstantInt::get(Type::getInt32Ty(Ty->getContext()), Count);
    unsigned AddrSpace = Ty->isPointerTy() ? Ty->getPointerAddressSpace() : 0;
    AllocaInst *NewAlloca =
        new AllocaInst(Ty, AddrSpace, AllocSize, "", AllocaInsertPnt);
    DenseMap<Type *, AllocaInst *> &TyMap = FunctionToAllocaCache[F];
    TyMap.try_emplace(Ty, NewAlloca);
    return NewAlloca;
  }

private:
  // Try to find an existing alloca an update the count if needed.
  AllocaInst *findAndUpdate(Function *F, Type *Ty, unsigned Count) {
    AllocaInst *AI = nullptr;
    DenseMap<Type *, AllocaInst *> &TyMap = FunctionToAllocaCache[F];
    auto It = TyMap.find(Ty);
    if (It != TyMap.end()) {
      AI = It->second;
      ConstantInt *CI = cast<ConstantInt>(AI->getArraySize());
      if (Count > CI->getZExtValue())
        AI->setOperand(0, ConstantInt::get(CI->getType(), Count));
    }
    return AI;
  }

  /// Map<Function, Map<Alloca Type, Allocation Inst>>
  DenseMap<Function *, DenseMap<Type *, AllocaInst *>> FunctionToAllocaCache;
};

/// \brief A specific instance of an outlined candidate.
struct FunctionSplicer {
  FunctionSplicer(bool EmitProfileData) : EmitProfileData(EmitProfileData) {
    // Prepare the function attributes that we don't want to inherit from any
    // parents.
    NonInheritAttrs.addAttribute(Attribute::AlwaysInline);
    NonInheritAttrs.addAttribute(Attribute::ArgMemOnly);
    NonInheritAttrs.addAttribute(Attribute::InaccessibleMemOnly);
    NonInheritAttrs.addAttribute(Attribute::InaccessibleMemOrArgMemOnly);
    NonInheritAttrs.addAttribute(Attribute::InlineHint);
    NonInheritAttrs.addAttribute(Attribute::Naked);
    NonInheritAttrs.addAttribute(Attribute::NoInline);
    NonInheritAttrs.addAttribute(Attribute::NoRecurse);
    NonInheritAttrs.addAttribute(Attribute::ReturnsTwice);
    NonInheritAttrs.addAttribute(Attribute::ReadOnly);
    NonInheritAttrs.addAttribute(Attribute::ReadNone);
    NonInheritAttrs.addAttribute(Attribute::NoReturn);
    NonInheritAttrs.addAttribute(Attribute::WriteOnly);
  }
  // Reset the outliner to prepare for a new instance.
  void prepareForNewInstance(OutlineCandidate &OC, IRInstructionMapper &OM) {
    OutlinedFn = nullptr;
    OutputTypeCount.clear();
    OutputGepIdx.clear();
    InitialStartIdx = OC.getOccurrence(0);
    AdditionalCandidateData &Data = OC.getData<AdditionalCandidateData>();
    if (Data.Outputs.count() < 2)
      return;
    // Collect information about the outputs of this new candidate.
    // Output parameters are condensed by type, so we need to build the
    //  offset from the base pointer for each output.
    for (size_t OutputIdx : Data.Outputs) {
      if (OutputIdx == Data.OutputToFold)
        continue;
      Type *OutTy = OM.getInstr(InitialStartIdx + OutputIdx)->getType();
      unsigned GepIdx, TyArgNo;
      auto TyCountIt = OutputTypeCount.find(OutTy);
      if (TyCountIt == OutputTypeCount.end()) {
        GepIdx = 0;
        TyArgNo = OutputTypeCount.size();
        OutputTypeCount.try_emplace(OutTy, 1, TyArgNo);
      } else {
        GepIdx = TyCountIt->second.first++;
        TyArgNo = TyCountIt->second.second;
      }
      OutputGepIdx.try_emplace(OutputIdx, TyArgNo, GepIdx);
    }
  }

  // Outline a new occurrence of an outline chain.
  void outlineOccurrence(OutlineCandidate &OC, size_t StartIdx,
                         IRInstructionMapper &OM) {
    ++NumOccurrencesOutlined;
    Instruction *Tail = OM.getInstr(StartIdx + (OC.Len - 1));
    Function *ParentFn = Tail->getFunction();
    bool InitialOccur = !OutlinedFn;
    AdditionalCandidateData &Data = OC.getData<AdditionalCandidateData>();

    /// Split the outline chain into its own block.
    Instruction *Head = OM.getInstr(StartIdx);
    BasicBlock *EntryBlock = Head->getParent();
    // Split our chain instance into a separate block for extraction.
    /// Split up to the head if we aren't at the front of our block.
    if (Head != &EntryBlock->front() ||
        EntryBlock == &ParentFn->getEntryBlock())
      EntryBlock = EntryBlock->splitBasicBlock(Head->getIterator());
    /// Split after the tail.
    auto SentinalIt = Tail->getNextNode()->getIterator();
    BasicBlock *Exit = EntryBlock->splitBasicBlock(SentinalIt);

    // Create a new block to patch the outlined section.
    BasicBlock *OutlineBlock =
        BasicBlock::Create(ParentFn->getContext(), "cso.patch", ParentFn);

    // Create parameter vec for the new call.
    unsigned NumOutputs = Data.Outputs.count();
    std::vector<Value *> Args;
    Args.reserve(Data.InputSeq.size() + NumOutputs);

    // Build inputs/outputs in order.
    for (size_t i = 0, e = Data.InputSeq.size(); i < e; ++i) {
      size_t InstrIdx, OpNo;
      std::tie(InstrIdx, OpNo) = Data.InputSeq[i];
      Args.push_back(OM.getInstr(StartIdx + InstrIdx)->getOperand(OpNo));
    }

    // Replace uses of outputs and create reloads.
    if (NumOutputs > 0) {
      SmallVector<Value *, 4> OutputArgs(OutputTypeCount.size());
      for (auto &ArgPair : OutputTypeCount) {
        unsigned ArgNo, Count;
        std::tie(Count, ArgNo) = ArgPair.second;
        OutputArgs[ArgNo] = OAC.getFreeAlloca(ParentFn, ArgPair.first, Count);
      }
      Args.insert(Args.end(), OutputArgs.begin(), OutputArgs.end());
      for (size_t OutputIdx : Data.Outputs) {
        if (OutputIdx == Data.OutputToFold)
          continue;
        Instruction *Out = OM.getInstr(StartIdx + OutputIdx);
        unsigned ArgNo, GepIdx;
        std::tie(ArgNo, GepIdx) = OutputGepIdx[OutputIdx];
        Value *OutputArg = OutputArgs[ArgNo];
        if (GepIdx != 0) {
          Value *Idx =
              ConstantInt::get(Type::getInt32Ty(Out->getContext()), GepIdx);
          OutputArg = GetElementPtrInst::CreateInBounds(
              Out->getType(), OutputArg, Idx, "", OutlineBlock);
        }
        Instruction *Reload = new LoadInst(OutputArg, "", OutlineBlock);
        Out->replaceUsesOutsideBlock(Reload, EntryBlock);
      }
      if (!InitialOccur) {
        Instruction *Out = OM.getInstr(StartIdx + Data.OutputToFold);
        Out->removeFromParent();
        Out->dropAllReferences();
      }
    }

    // Replace branches to entry block.
    EntryBlock->replaceAllUsesWith(OutlineBlock);
    BranchInst::Create(Exit, OutlineBlock);

    // Get the first valid debug info.
    DebugLoc CallLoc;
    if (ParentFn->getSubprogram())
      for (auto It = Head->getIterator(), E = EntryBlock->end();
           !CallLoc && It != E; ++It)
        CallLoc = It->getDebugLoc();

    // If this is the first occurrence then we outline the chain, otherwise we
    // erase the entry block because it's dead.
    if (OutlinedFn) {
      unsigned KnownIDs[] = {LLVMContext::MD_tbaa,
                             LLVMContext::MD_alias_scope,
                             LLVMContext::MD_noalias,
                             LLVMContext::MD_range,
                             LLVMContext::MD_invariant_load,
                             LLVMContext::MD_nonnull,
                             LLVMContext::MD_invariant_group,
                             LLVMContext::MD_align,
                             LLVMContext::MD_dereferenceable,
                             LLVMContext::MD_fpmath,
                             LLVMContext::MD_dereferenceable_or_null};

      // Merge special state.
      for (unsigned InitI = InitialStartIdx, CurI = StartIdx,
                    InitE = InitialStartIdx + OC.Len;
           InitI < InitE; ++InitI, ++CurI) {
        Instruction *InitII = OM.getInstr(InitI);
        Instruction *CurII = OM.getInstr(CurI);
        // Make sure the alignment is valid as we skip it during congruency
        // finding.
        if (LoadInst *LI = dyn_cast<LoadInst>(InitII))
          LI->setAlignment(Align(std::min(LI->getAlignment(),
                                    cast<LoadInst>(CurII)->getAlignment())));
        else if (StoreInst *SI = dyn_cast<StoreInst>(InitII))
          SI->setAlignment(Align(std::min(SI->getAlignment(),
                                    cast<StoreInst>(CurII)->getAlignment())));
        // Make sure that no tails are propagated properly.
        else if (CallInst *CI = dyn_cast<CallInst>(InitII)) {
          auto TCK = cast<CallInst>(CurII)->getTailCallKind();
          if (TCK == CallInst::TCK_NoTail)
            CI->setTailCallKind(CallInst::TCK_NoTail);
          else if (TCK != CallInst::TCK_None && !CI->isNoTailCall())
            CI->setTailCallKind(TCK);
        }
        // Be conservative about flags like nsw/nuw.
        else
          InitII->andIRFlags(OM.getInstr(CurI));

        // Merge metadata.
        DebugLoc DL = InitII->getDebugLoc();
        if (DL && DL != CurII->getDebugLoc())
          InitII->setDebugLoc(DebugLoc());
        combineMetadata(InitII, CurII, KnownIDs, false); //TODO: not sure if DoesKMove is false or true
      }
      EntryBlock->eraseFromParent();
      AttributeFuncs::mergeAttributesForInlining(*OutlinedFn, *ParentFn);
    } else
      outlineInitialOccurrence(OC, OM, StartIdx, Args, EntryBlock);

    // Create the patchup for the outline section.
    CallInst *CI =
        CallInst::Create(OutlinedFn, Args, "", &OutlineBlock->front());
    if (Data.OutputToFold != unsigned(-1)) {
      Instruction *FoldedOut = OM.getInstr(StartIdx + Data.OutputToFold);
      if (InitialOccur)
        FoldedOut->replaceUsesOutsideBlock(CI, FoldedOut->getParent());
      else {
        FoldedOut->replaceAllUsesWith(CI);
        FoldedOut->deleteValue();
      }
    }
    CI->setDebugLoc(CallLoc);
  }

  // \brief Finalize this function as an outline instance. In this step we
  // re-unique the inputs to the function.
  void finalize(OutlineCandidate &OC) {
    FunctionType *CurFnTy = OutlinedFn->getFunctionType();
    AdditionalCandidateData &Data = OC.getData<AdditionalCandidateData>();
    unsigned NumInputs = Data.InputSeq.size();

    // Map <ArgNo> to <Congruency Group Idx>
    DenseMap<unsigned, unsigned> ArgNoToCG, RhsArgNoToCG;
    std::vector<BitVector> ArgCongruencyGroups, RhsArgCGroups;

    // Helper function to collect the information on the use of inputs in the
    // outlined function. This identifies which arguments are actually congruent
    // to each other for a specific function call.
    auto CollectInputInfo = [&](CallInst *CI,
                                DenseMap<unsigned, unsigned> &ANoToCG,
                                std::vector<BitVector> &ArgCGroups) {
      ArgCGroups.reserve(NumInputs);
      for (unsigned i = 0, e = NumInputs; i < e; ++i) {
        // We already evaluated the equivalencies for this arg.
        if (ANoToCG.count(i))
          continue;
        Value *Op = CI->getOperand(i);
        BitVector CurrentGroup(NumInputs);
        CurrentGroup.set(i);
        ANoToCG.try_emplace(i, ArgCGroups.size());
        for (unsigned j = i + 1; j < e; ++j) {
          if (ANoToCG.count(j))
            continue;
          if (CI->getArgOperand(j) == Op) {
            CurrentGroup.set(j);
            ANoToCG.try_emplace(j, ArgCGroups.size());
          }
        }
        ArgCGroups.emplace_back(std::move(CurrentGroup));
      }
    };

    // Build initial equivalences from the first call.
    auto UserIt = OutlinedFn->user_begin();
    CallInst *FirstCI = cast<CallInst>(*UserIt);
    ++UserIt;
    CollectInputInfo(FirstCI, ArgNoToCG, ArgCongruencyGroups);

    // If we have the same amount of congruency groups as we do arguments,
    //   the they are already unique.
    if (NumInputs == ArgCongruencyGroups.size())
      return;
    // Check every other user to see if the equivalencies hold up.
    BitVector ResolvedInputs(NumInputs);
    // BitVector helper to hold non congruent matches between argument groups.
    BitVector NonCongruentLeft;
    for (auto UserE = OutlinedFn->user_end(); UserIt != UserE; ++UserIt) {
      CallInst *CI = cast<CallInst>(*UserIt);
      RhsArgCGroups.clear();
      RhsArgNoToCG.clear();
      CollectInputInfo(CI, RhsArgNoToCG, RhsArgCGroups);
      ResolvedInputs.reset();
      for (unsigned i = 0, e = NumInputs; i < e; ++i) {
        if (ResolvedInputs.test(i))
          continue;
        BitVector &LhsGroup = ArgCongruencyGroups[ArgNoToCG[i]];
        BitVector &RhsGroup = RhsArgCGroups[RhsArgNoToCG[i]];

        /// Build non congruent arguments between the groups.
        NonCongruentLeft = LhsGroup;
        /// Congruent matches.
        LhsGroup &= RhsGroup;
        assert(LhsGroup.count() > 0);

        // Non congruent sets on both sides still act as a congruency group.
        NonCongruentLeft ^= LhsGroup;

        // Mark arguments as handled.
        for (unsigned SetBit : LhsGroup.set_bits())
          ResolvedInputs.set(SetBit);
        if (NonCongruentLeft.count() == 0)
          continue;

        // Move the non congruent matches from the left group to a
        //   new congruency group.
        unsigned NewGroupId = ArgCongruencyGroups.size();
        ArgCongruencyGroups.emplace_back(std::move(NonCongruentLeft));

        // Move non congruent matches to a new congruency group
        //   and remove them from the top level mapping.
        for (unsigned SetBit : ArgCongruencyGroups.back().set_bits())
          ArgNoToCG[SetBit] = NewGroupId;
      }
    }

    // No inputs can be condensed.
    if (NumInputs == ArgCongruencyGroups.size())
      return;

    // Build new function from the condensed inputs.
    std::vector<Type *> NewFnTys;
    for (auto &It : ArgCongruencyGroups)
      NewFnTys.push_back(CurFnTy->getParamType(It.find_first()));
    for (unsigned i = NumInputs, e = CurFnTy->getNumParams(); i < e; ++i)
      NewFnTys.push_back(CurFnTy->getParamType(i));

    // Create the merged function.
    FunctionType *NewFnTy =
        FunctionType::get(CurFnTy->getReturnType(), NewFnTys, false);
    Function *MergedFn = Function::Create(NewFnTy, OutlinedFn->getLinkage(), "",
                                          OutlinedFn->getParent());
    MergedFn->takeName(OutlinedFn);
    MergedFn->copyAttributesFrom(OutlinedFn);
    DenseMap<Argument *, Value *> ValMap;

    // Remap the arguments.
    auto ArgI = OutlinedFn->arg_begin();
    for (size_t i = 0, e = NumInputs; i < e; ++i, ++ArgI)
      ValMap[&*ArgI] = std::next(MergedFn->arg_begin(), ArgNoToCG[i]);
    auto MergeArgI =
        std::next(MergedFn->arg_begin(), ArgCongruencyGroups.size());
    for (size_t i = NumInputs, e = CurFnTy->getNumParams(); i < e;
         ++i, ++ArgI, ++MergeArgI) {
      ValMap[&*ArgI] = &*MergeArgI;
    }
    /// Move Fn Body.
    MergedFn->getBasicBlockList().splice(MergedFn->begin(),
                                         OutlinedFn->getBasicBlockList());
    /// Remap arguments.
    for (Argument &A : OutlinedFn->args()) {
      auto NewArgIt = ValMap.find(&A);
      if (NewArgIt != ValMap.end())
        A.replaceAllUsesWith(NewArgIt->second);
    }

    // Rewrite the calls to this function with calls to the merged function.
    std::vector<Value *> CallArgs;
    for (auto U = OutlinedFn->use_begin(), E = OutlinedFn->use_end(); U != E;) {
      CallInst *CI = dyn_cast<CallInst>(U->getUser());
      ++U;
      CallArgs.clear();
      /// Map call args by their congruency group.
      for (auto &It : ArgCongruencyGroups)
        CallArgs.push_back(CI->getArgOperand(It.find_first()));
      /// Outputs are retained, except for the promoted value.
      for (unsigned i = NumInputs, e = CurFnTy->getNumParams(); i < e; ++i)
        CallArgs.push_back(CI->getArgOperand(i));
      CallInst *NewCall = CallInst::Create(MergedFn, CallArgs, "", CI);
      NewCall->setDebugLoc(CI->getDebugLoc());
      CI->replaceAllUsesWith(NewCall);
      CI->eraseFromParent();
    }
    OutlinedFn->eraseFromParent();
  }

private:
  // \brief Outline the initial occurrence of this chain.
  void outlineInitialOccurrence(OutlineCandidate &OC, IRInstructionMapper &OM,
                                size_t StartIdx, ArrayRef<Value *> Args,
                                BasicBlock *Entry) {
    /// Function type for outlined function.
    std::vector<Type *> Tys;
    Tys.reserve(Args.size());
    for (Value *Arg : Args)
      Tys.push_back(Arg->getType());
    LLVMContext &Ctx = Entry->getContext();
    AdditionalCandidateData &Data = OC.getData<AdditionalCandidateData>();
    Instruction *FoldedOut = nullptr;
    Type *OutputTy = Type::getVoidTy(Ctx);
    if (Data.OutputToFold != unsigned(-1)) {
      FoldedOut = OM.getInstr(StartIdx + Data.OutputToFold);
      OutputTy = FoldedOut->getType();
    }
    FunctionType *FTy = FunctionType::get(OutputTy, Tys, false);
    Function *ParentFn = Entry->getParent();

    /// Create function and move blocks.
    OutlinedFn = Function::Create(FTy, GlobalValue::PrivateLinkage, "cso",
                                  Entry->getModule());
    OutlinedFn->getBasicBlockList().splice(
        OutlinedFn->end(), Entry->getParent()->getBasicBlockList(),
        Entry->getIterator());
    OutlinedFn->setUnnamedAddr(GlobalValue::UnnamedAddr::Local);
    Entry->getTerminator()->eraseFromParent();
    ReturnInst::Create(Ctx, FoldedOut, Entry);

    // FIXME: Fix this if we can have merged debug info.
    if (ParentFn->getSubprogram()) {
      for (auto It = Entry->begin(), E = Entry->end(); It != E;) {
        Instruction &I = *It;
        ++It;
        if (isa<DbgInfoIntrinsic>(I))
          Entry->getInstList().erase(I.getIterator());
      }
    }

    // FIXME: Ideally we should compute the real count for this function but
    //  for now we just tag it as cold.
    if (EmitProfileData)
      OutlinedFn->addFnAttr(Attribute::Cold);

    /// Create stores for any output variables.
    auto OutputArgBegin = OutlinedFn->arg_begin();
    std::advance(OutputArgBegin, Data.InputSeq.size());
    for (size_t OutputIdx : Data.Outputs) {
      if (OutputIdx == Data.OutputToFold)
        continue;
      unsigned ArgNo, GepIdx;
      std::tie(ArgNo, GepIdx) = OutputGepIdx[OutputIdx];
      Value *OutputArg = &*std::next(OutputArgBegin, ArgNo);
      Instruction *Out = OM.getInstr(StartIdx + OutputIdx);
      Instruction *InstPtr = Out->getNextNode();
      if (GepIdx != 0) {
        Value *Idx =
            ConstantInt::get(Type::getInt32Ty(Out->getContext()), GepIdx);
        OutputArg = GetElementPtrInst::CreateInBounds(Out->getType(), OutputArg,
                                                      Idx, "", InstPtr);
      }
      new StoreInst(Out, OutputArg, InstPtr);
    }

    /// Replace input operands with function arguments.
    auto ArgI = OutlinedFn->arg_begin();
    for (auto &InputPair : Data.InputSeq) {
      size_t InputInstIdx, OpNo;
      std::tie(InputInstIdx, OpNo) = InputPair;
      Instruction *InputInst = OM.getInstr(StartIdx + InputInstIdx);
      InputInst->setOperand(OpNo, &*ArgI++);
    }

    /// Inherit attributes.
    OutlinedFn->addAttributes(
        AttributeList::FunctionIndex,
        ParentFn->getAttributes().getAttributes(AttributeList::FunctionIndex));
    OutlinedFn->removeAttributes(AttributeList::FunctionIndex, NonInheritAttrs);
    OutlinedFn->removeFnAttr(Attribute::AllocSize);
  }
  // The function created after outlining.
  Function *OutlinedFn = nullptr;
  // Output relocation information.
  DenseMap<Type *, std::pair<unsigned, unsigned>> OutputTypeCount;
  DenseMap<unsigned, std::pair<unsigned, unsigned>> OutputGepIdx;
  OutputAllocaCache OAC;
  // Set of attributes that are not to be inherited from parent functions.
  AttrBuilder NonInheritAttrs;
  // If we are emitting profile date during outlining.
  bool EmitProfileData;
  // The index of the initial occurrence for the current splice.
  unsigned InitialStartIdx;
};

/// \brief Perform analysis and verification for the found outline candidates.
struct OutlinerAnalysisImpl {
  OutlinerAnalysisImpl(
      IRInstructionMapper &OM, std::vector<OutlineCandidate> &CandidateList,
      function_ref<TargetTransformInfo &(Function &)> GetTTI,
      const DataLayout *Layout,
      SpecificBumpPtrAllocator<AdditionalCandidateData> &DataAlloc)
      : OM(OM), CandidateList(CandidateList), GetTTI(GetTTI), Layout(Layout),
        DataAlloc(DataAlloc) {}

  // Analyze our found candidates for benefit and correctness.
  void analyzeCandidateList() {
    // Generate instruction info for each instruction that is part
    //  of an occurrence.
    BitVector OccurInstrInfo(OM.getNumMappedInstructions());
    for (OutlineCandidate &OC : CandidateList) {
      OC.Data = new (DataAlloc.Allocate()) AdditionalCandidateData();
      for (unsigned Occur : OC)
        OccurInstrInfo.set(Occur, Occur + OC.Len);
    }
    for (unsigned Idx : OccurInstrInfo.set_bits())
      OM.createInstrInfo(Idx);

    // Estimate the function type and benefit of each chain.
    estimateFunctionType(0, CandidateList.size());

    // After initial benefit analysis, we verify the occurrences of each
    // candidate are compatible.
    verifyOccurrenceCompatability();
  }

private:
  // Helper struct for verifying the compatibility of occurrences within a
  // candidate.
  struct VerifyInst : public std::pair<Function *, std::vector<unsigned>> {
    void reset(Function *F) {
      first = F;
      second.clear();
    }
    bool compare(VerifyInst &R,
                 function_ref<TargetTransformInfo &(Function &)> GetTTI) {
      Function *LF = first, *RF = R.first;
      if (LF != RF) {
        if (!AttributeFuncs::areInlineCompatible(*LF, *RF))
          return false;
        TargetTransformInfo &TTI = GetTTI(*LF);
        if (!TTI.areInlineCompatible(LF, RF))
          return false;
      }
      return second == R.second;
    }
  };

  // Verify the compatibilities of the occurrences in each candidate.
  void verifyOccurrenceCompatability() {
    // Vector of different verification leaders.
    std::vector<VerifyInst> VerificationCands;
    // Count of occurrences to an verification instance.
    std::vector<size_t> Counts;
    // Maps occurrences to the input index they belong to.
    std::vector<size_t> OccurrenceInputIdx;
    // Verification instance of the current occurrence being considered.
    VerifyInst CurrentOccurVerifyInst;

    size_t CurrentCandidateListSize = CandidateList.size();
    for (size_t i = 0, e = CurrentCandidateListSize; i < e; ++i) {
      OutlineCandidate &OC = CandidateList[i];
      if (!OC.isValid())
        continue;
      // Compute the input sequence for this occurrence.
      for (size_t Oi = OccurrenceInputIdx.size(), Oe = OC.size(); Oi < Oe;
           ++Oi) {
        unsigned Occur = OC.getOccurrence(Oi);
        CurrentOccurVerifyInst.reset(OM.getInstrFunction(Occur));
        for (size_t InstrIdx = Occur, InstrE = Occur + OC.Len;
             InstrIdx < InstrE; ++InstrIdx) {
          IRInstructionMapper::InstructionInfo &II = OM.getInstrInfo(InstrIdx);
          for (unsigned InIdx : II.InputIndexes) {
            // We only really need to verify inputs coming from within the
            // sequence. Other inputs simply help to verify the ordering.
            unsigned InputIdx = InIdx < Occur ? -1 : InIdx - Occur;
            CurrentOccurVerifyInst.second.push_back(InputIdx);
          }
        }

        // Check for existing mapping of this instance.
        auto It =
            std::find_if(VerificationCands.begin(), VerificationCands.end(),
                         [&](VerifyInst &L) {
                           return L.compare(CurrentOccurVerifyInst, GetTTI);
                         });
        if (It != VerificationCands.end()) {
          size_t InternalInputIt = It - VerificationCands.begin();
          ++Counts[InternalInputIt];
          OccurrenceInputIdx.push_back(InternalInputIt);
        } else {
          unsigned OrigCapacity = CurrentOccurVerifyInst.second.capacity();
          OccurrenceInputIdx.push_back(VerificationCands.size());
          VerificationCands.emplace_back(std::move(CurrentOccurVerifyInst));
          CurrentOccurVerifyInst.second.reserve(OrigCapacity);
          Counts.push_back(1);
        }
      }

      OutlineCandidate &Cur = CandidateList[i];
      size_t SharedSizeWithNext = Cur.SharedSizeWithNext;
      size_t IdxOfNext = i + 1;
      while (SharedSizeWithNext > 0 && !CandidateList[IdxOfNext].isValid())
        SharedSizeWithNext = CandidateList[IdxOfNext++].SharedSizeWithNext;
      size_t FirstOccur = Cur.getOccurrence(0);

      // Only split if needed.
      if (Counts.size() > 1)
        splitOutlineChain(i, Counts, OccurrenceInputIdx);

      // If we share a size with the next chain then we do cleanup and set up to
      //  reduce the amount of work we need to do during the next iteration.
      if (SharedSizeWithNext > 0 && CandidateList[IdxOfNext].isValid()) {
        // Get the cut off point for moving to the next candidate.
        size_t SharedCutOffPoint = 0;
        for (size_t InstrIdx = FirstOccur,
                    InstrE = FirstOccur + SharedSizeWithNext;
             InstrIdx < InstrE; ++InstrIdx) {
          IRInstructionMapper::InstructionInfo &II = OM.getInstrInfo(InstrIdx);
          SharedCutOffPoint += II.InputIndexes.size();
        }

        // Update the size of the internal inputs vectors.
        for (size_t InIt = 0, InE = VerificationCands.size(); InIt < InE;
             ++InIt)
          VerificationCands[InIt].second.resize(SharedCutOffPoint);

        // Don't bother merging if there is only one instance.
        if (Counts.size() == 1)
          continue;

        // Set resolved occurrences that point to the first vector.
        BitVector UnResolved(OccurrenceInputIdx.size(), true);
        for (size_t i = 0, e = OccurrenceInputIdx.size(); i < e; ++i)
          if (OccurrenceInputIdx[i] == 0)
            UnResolved.reset(i);
        // Condense the internal inputs vector in the case where two vectors are
        //  now equivalent given the smaller size.
        size_t InsertIdx = 1;
        for (size_t i = 1, e = VerificationCands.size(); i < e; ++i) {
          VerifyInst &VI = VerificationCands[i];
          auto RemapOccurIdxs = [&](size_t Old, size_t New) {
            for (size_t OccurIdx : UnResolved.set_bits()) {
              if (OccurrenceInputIdx[OccurIdx] == Old) {
                OccurrenceInputIdx[OccurIdx] = New;
                UnResolved.reset(OccurIdx);
              }
            }
          };
          // Try to remap to an existing instance.
          auto Remap = [&]() -> bool {
            for (size_t j = 0; j < InsertIdx; ++j) {
              if (VerificationCands[j].compare(VI, GetTTI)) {
                Counts[j] += Counts[i];
                RemapOccurIdxs(i, j);
                return true;
              }
            }
            return false;
          };

          // Update mapping with new instance.
          if (!Remap()) {
            if (i != InsertIdx) {
              Counts[InsertIdx] = Counts[i];
              RemapOccurIdxs(i, InsertIdx);
              VerificationCands[InsertIdx++] = std::move(VerificationCands[i]);
            } else
              ++InsertIdx;
          }
        }
        VerificationCands.resize(InsertIdx);
        Counts.resize(InsertIdx);
        continue;
      }
      // Otherwise we just cleanup what we used.
      VerificationCands.clear();
      Counts.clear();
      OccurrenceInputIdx.clear();
    }
    estimateFunctionType(CurrentCandidateListSize, CandidateList.size());
  }

  // Split the OutlineChain at index /p CurrentChainIndex into N different
  // chains with the provided memberships.
  void splitOutlineChain(size_t CurrentChainIndex,
                         const std::vector<size_t> &MembershipCounts,
                         const std::vector<size_t> &OccurrenceInputIdx) {
    OutlineCandidate *OrigChain = &CandidateList[CurrentChainIndex];
    AdditionalCandidateData &OrigData =
        OrigChain->getData<AdditionalCandidateData>();
    SmallVector<size_t, 4> SplitChains(MembershipCounts.size(), -1);
    size_t FirstValid = 0;
    while (FirstValid < MembershipCounts.size() &&
           MembershipCounts[FirstValid] < MinOccurrences)
      ++FirstValid;
    if (FirstValid == MembershipCounts.size()) {
      OrigChain->invalidate();
      return;
    }

    SplitChains[FirstValid] = CurrentChainIndex;

    // Add new chains for each valid count after the first.
    unsigned OrigLen = OrigChain->Len;
    for (size_t i = FirstValid + 1, e = MembershipCounts.size(); i < e; ++i) {
      size_t Count = MembershipCounts[i];
      if (Count < MinOccurrences)
        continue;
      SplitChains[i] = CandidateList.size();
      CandidateList.emplace_back(OrigLen);
      OutlineCandidate &NewChain = CandidateList.back();
      NewChain.Occurrences.reserve(Count);
      NewChain.Data = new (DataAlloc.Allocate()) AdditionalCandidateData();
      AdditionalCandidateData &NCData =
          NewChain.getData<AdditionalCandidateData>();
      NCData.InputSeq.reserve(OrigData.InputSeq.capacity());
    }

    // Move occurrences to their new parents.
    OrigChain = &CandidateList[CurrentChainIndex];
    for (size_t i = 0, e = OrigChain->size(); i < e; ++i) {
      size_t NewParentIdx = SplitChains[OccurrenceInputIdx[i]];
      if (NewParentIdx != CurrentChainIndex && NewParentIdx != size_t(-1)) {
        OutlineCandidate &NewParent = CandidateList[NewParentIdx];
        NewParent.Occurrences.push_back(OrigChain->Occurrences[i]);
      }
    }
    bool RecomputeInputs = FirstValid != 0;
    for (ssize_t i = OrigChain->size() - 1; i >= 0; --i)
      if (OccurrenceInputIdx[i] != FirstValid)
        OrigChain->removeOccurrence(i);

    // Update the occurrences and recalculate the analysis for the split
    // candidate.
    computeInputOutputSequence(*OrigChain, RecomputeInputs);
    estimateFunctionType(CurrentChainIndex, CurrentChainIndex + 1);
  }

  // Estimate the function type of each outline chain in the given range.
  void estimateFunctionType(size_t StartIdx, size_t EndIdx) {
    std::vector<Value *> InputOperands;
    BitVector UnFoldableInputs;
    for (size_t i = StartIdx; i < EndIdx; ++i) {
      OutlineCandidate &OC = CandidateList[i];
      AdditionalCandidateData &OCData = OC.getData<AdditionalCandidateData>();
      unsigned FirstOccur = *OC.begin();

      // Compute the input sequence if needed.
      if (OCData.InputSeq.empty()) {
        computeInputOutputSequence(OC);

        // After computing we also compute the input sequences of chains
        //  that we share size with. We have already calculated the input
        //  sequence for this chain as it is a subset of our current
        //  sequence.
        OutlineCandidate *Cur = &OC;
        while (Cur->SharedSizeWithNext > 0) {
          OutlineCandidate *Next = Cur + 1;
          AdditionalCandidateData &CurData =
              Cur->getData<AdditionalCandidateData>();
          size_t NewSize = 0;
          for (size_t e = CurData.InputSeq.size(); NewSize < e; ++NewSize)
            if (CurData.InputSeq[NewSize].first >= Cur->SharedSizeWithNext)
              break;
          AdditionalCandidateData &NextData =
              Next->getData<AdditionalCandidateData>();
          NextData.InputSeq.assign(CurData.InputSeq.begin(),
                                   CurData.InputSeq.begin() + NewSize);
          computeInputOutputSequence(*Next, false);
          Cur = Next;
        }
      }

      // Check to see if we share candidates with our predecessor. If we
      //   do then we can avoid rechecking candidates that we already have
      //   information for.
      size_t SharedOccurrencesWithPrev = 1;
      if (i > StartIdx) {
        OutlineCandidate &Prev = CandidateList[i - 1];
        if (Prev.SharedSizeWithNext > 0)
          SharedOccurrencesWithPrev = Prev.size();
      }

      // Only recompute the input operands if we didn't share a size with the
      // previous chain.
      if (SharedOccurrencesWithPrev == 1) {
        // Get the operands for the first candidate.
        InputOperands.clear();
        UnFoldableInputs.reset();
        UnFoldableInputs.resize(OCData.InputSeq.size());
        for (size_t i = 0, e = OCData.InputSeq.size(); i < e; ++i) {
          auto &Seq = OCData.InputSeq[i];
          Value *Op =
              OM.getInstr(FirstOccur + Seq.first)->getOperand(Seq.second);
          InputOperands.push_back(Op);
          if (isa<Instruction>(Op) || isa<Argument>(Op))
            UnFoldableInputs.set(i);
        }
      } else {
        InputOperands.resize(OCData.InputSeq.size(), nullptr);
        UnFoldableInputs.resize(InputOperands.size());
      }

      // Check to see which inputs won't be folded.
      for (size_t Ci = SharedOccurrencesWithPrev, Ce = OC.size(); Ci != Ce;
           ++Ci) {
        unsigned Occur = OC.getOccurrence(Ci);
        size_t InputNo = 0;
        for (auto &Seq : OCData.InputSeq) {
          if (UnFoldableInputs.test(InputNo)) {
            ++InputNo;
            continue;
          }
          Value *InputOp =
              OM.getInstr(Occur + Seq.first)->getOperand(Seq.second);
          if (InputOp != InputOperands[InputNo])
            UnFoldableInputs.set(InputNo);
          ++InputNo;
        }
      }

      // Remove all of the inputs that will be folded.
      for (size_t i = 0, e = OCData.InputSeq.size(), InputNo = e - 1; i < e;
           ++i, --InputNo) {
        if (!UnFoldableInputs.test(InputNo))
          OCData.InputSeq.erase(OCData.InputSeq.begin() + InputNo);
      }
    }
    computeOutlineBenefit(CandidateList, StartIdx, EndIdx);
  }

  // Compute the external input sequence(Inst#+Op#) and outputs of a given
  // chain.
  void computeInputOutputSequence(OutlineCandidate &OC,
                                  bool ComputeInputs = true) {
    AdditionalCandidateData &Data = OC.getData<AdditionalCandidateData>();
    // Inputs are operands that come from outside of the chain range.
    if (ComputeInputs) {
      Data.InputSeq.clear();
      Data.InputSeq.reserve(OC.Len);
      unsigned FirstOccur = OC.getOccurrence(0);
      for (size_t InstrIdx = FirstOccur, InstrE = FirstOccur + OC.Len;
           InstrIdx < InstrE; ++InstrIdx) {
        IRInstructionMapper::InstructionInfo &II = OM.getInstrInfo(InstrIdx);
        for (size_t i = 0, e = II.InputIndexes.size(); i < e; ++i)
          if (II.InputIndexes[i] < FirstOccur)
            Data.InputSeq.emplace_back(InstrIdx - FirstOccur, i);
      }
    }

    // Outputs are internal instructions that have uses outside of the chain
    // range.
    Data.Outputs.clear();
    for (unsigned Occur : OC) {
      for (size_t InstrIdx = Occur, InstrE = Occur + OC.Len; InstrIdx < InstrE;
           ++InstrIdx) {
        IRInstructionMapper::InstructionInfo &II = OM.getInstrInfo(InstrIdx);
        if (II.FarthestInSameBlockOutput >= InstrE)
          Data.Outputs.set(InstrIdx - Occur);
      }
    }
  }

  // Computes the estimated benefit of a set of potential functions.
  void computeOutlineBenefit(std::vector<OutlineCandidate> &CandidateList,
                             size_t StartIdx, size_t EndIdx) {
    SmallDenseSet<Value *, 8> UniqueInputOperands;
    SmallPtrSet<Type *, 8> OutputTypes;
    for (size_t i = StartIdx; i < EndIdx; ++i) {
      OutlineCandidate &OC = CandidateList[i];
      AdditionalCandidateData &Data = OC.getData<AdditionalCandidateData>();

      /// Reset benefit metrics.
      OC.invalidate();

      // Sanity check.
      unsigned NumOccurences = OC.size();
      if (NumOccurences < MinOccurrences)
        continue;

      // Estimate the cost of this chain of instructions.
      /// For now we just use the first candidate.
      unsigned FirstOccur = OC.getOccurrence(0);
      Function *FirstOccurFn = OM.getInstrFunction(FirstOccur);
      TargetTransformInfo &TTI = GetTTI(*FirstOccurFn);
      unsigned ChainCost = 0;
      for (size_t i = 0, e = OC.Len, InstIdx = FirstOccur; i < e;
           ++i, ++InstIdx)
        ChainCost += OM.getInstrCost(TTI, InstIdx);

      unsigned WidestRegister = TTI.getRegisterBitWidth(false);
      auto EstimateIrregularTypeParamCost = [&](Type *Ty) {
        return Layout->getTypeSizeInBits(Ty) / WidestRegister;
      };

      /// Estimate inputs to this function.
      UniqueInputOperands.clear();
      for (auto &Seq : Data.InputSeq) {
        Value *IOp =
            OM.getInstr(FirstOccur + Seq.first)->getOperand(Seq.second);
        UniqueInputOperands.insert(IOp);
      }

      unsigned NumFunctionParams = UniqueInputOperands.size();
      unsigned NumOutputs = Data.Outputs.count();
      /// We penalize container parameters.
      for (Value *Op : UniqueInputOperands) {
        Type *OpTy = Op->getType();
        if (OpTy->isVectorTy() || OpTy->isStructTy())
          NumFunctionParams += EstimateIrregularTypeParamCost(OpTy);
      }

      // Compute the output to fold to a return as well as the total number of
      // outputs passed to the function.
      if (NumOutputs == 1) {
        Data.OutputToFold = Data.Outputs.find_first();
      } else if (NumOutputs > 0) {
        // Outputs are condensed by type, so the number of actual parameters
        //  going to the outlined function is the number of distinct types
        //  for the outputs. We try to reduce this by folding an output that
        //  is the single member of its type.
        OutputTypes.clear();
        SmallDenseMap<Type *, unsigned, 8> BeneficialToFold;
        for (size_t OutputIdx : Data.Outputs) {
          Type *ParamEleTy = OM.getInstr(FirstOccur + OutputIdx)->getType();
          if (OutputTypes.insert(ParamEleTy).second)
            BeneficialToFold.try_emplace(ParamEleTy, OutputIdx);
          else
            BeneficialToFold.erase(ParamEleTy);
        }
        NumFunctionParams += OutputTypes.size();
        if (!BeneficialToFold.empty()) {
          Data.OutputToFold = 0;
          for (auto &TyPair : BeneficialToFold)
            if (Data.OutputToFold < TyPair.second)
              Data.OutputToFold = TyPair.second;
          --NumFunctionParams;
        } else
          Data.OutputToFold = Data.Outputs.find_last();
      } else
        Data.OutputToFold = -1;

      /// The new function contains one instance of the chain of instructions,
      ///  has to prepare each parameter, and contains a return.
      unsigned NewFunctionCost =
          std::min(NumFunctionParams, TTI.getNumberOfRegisters(false)) +
          ChainCost + 1;
      unsigned CostFromReLoad = 0;

      /// Add the cost for each output.
      OutputTypes.clear();
      for (size_t OutputIdx : Data.Outputs) {
        Type *ParamEleTy = OM.getInstr(FirstOccur + OutputIdx)->getType();
        Type *ParamTy = ParamEleTy->getPointerTo();

        // Extremely basic cost estimate for loading and storing a struct type.
        // FIXME: This needs to be improved.
        if (ParamEleTy->isStructTy()) {
          unsigned EstCost = EstimateIrregularTypeParamCost(ParamEleTy);
          NewFunctionCost += EstCost;
          CostFromReLoad += EstCost;
          continue;
        }
        size_t StoreSize = Layout->getTypeStoreSize(ParamTy);
        size_t AddrSpace = ParamTy->getPointerAddressSpace();

        /// There will be a store into this variable in the outlined function.
        NewFunctionCost += TTI.getMemoryOpCost(Instruction::Store, ParamEleTy,
                                               MaybeAlign(StoreSize), AddrSpace,nullptr); //TODO: not sure which instruction to pass
        /// Account for a potential add to the base output pointer.
        if (OutputIdx != Data.OutputToFold)
          if (!OutputTypes.insert(ParamEleTy).second)
            NewFunctionCost += 1;

        /// Each output value has a reload in the parent function.
        /// NOTE: This isn't entirely true if a specific instance doesn't use
        /// the value.
        CostFromReLoad += TTI.getMemoryOpCost(Instruction::Load, ParamEleTy,
                                              MaybeAlign(StoreSize), AddrSpace,nullptr); //TODO: not sure which instruction to pass
      }

      /// A call is generated at each occurence.
      ///   = call instruction + prepare each parameter + reload outputs.
      unsigned CostPerOccurence = 1 + NumFunctionParams + CostFromReLoad;

      // No possibility of benefit.
      if (CostPerOccurence >= ChainCost)
        continue;

      // Compute the benefit of the chain and each occurrence.
      OC.BenefitPerOccur = ChainCost - CostPerOccurence;
      unsigned OutlineBenefit = OC.BenefitPerOccur * NumOccurences;
      if (OutlineBenefit > NewFunctionCost) {
        OC.Benefit = OutlineBenefit - NewFunctionCost;
        if(OC.Benefit < MinBenefit)
          OC.invalidate();
      }

      LLVM_DEBUG(dbgs() << "Num : " << NumOccurences << "; Len : " << OC.Len
                   << "\n");
      LLVM_DEBUG(dbgs() << "Inputs : " << Data.InputSeq.size()
                   << "; Outputs : " << NumOutputs << "\n");
      LLVM_DEBUG(dbgs() << "Chain Cost : " << ChainCost << "\n");
      LLVM_DEBUG(dbgs() << "CostPerOccur : " << CostPerOccurence << "\n");
      LLVM_DEBUG(dbgs() << "BenefitPerOccur : " << OC.BenefitPerOccur << "\n");
      LLVM_DEBUG(dbgs() << "NewFunctionCost : " << NewFunctionCost << "\n\n");
    }
  }

  // Internal members
  IRInstructionMapper &OM;
  std::vector<OutlineCandidate> &CandidateList;
  function_ref<TargetTransformInfo &(Function &)> GetTTI;
  const DataLayout *Layout;
  SpecificBumpPtrAllocator<AdditionalCandidateData> &DataAlloc;
};

/// \brief Outliner for sequenced IR instructions.
class SequencedIROutliner : public SequencedOutlinerImpl {
public:
  SequencedIROutliner(OutlinerMapper &OM, bool EmitProfileData,
                      function_ref<TargetTransformInfo &(Function &)> GetTTI,
                      Module &M)
      : SequencedOutlinerImpl(OM, ::MinInstructionLength, ::MinOccurrences,
                              SequencedOutlinerImpl::CSM_SuffixArray),
        GetTTI(GetTTI), DL(&M.getDataLayout()), FS(EmitProfileData),
        EmitRemarks(areRemarksEnabled(M)) {}
  virtual ~SequencedIROutliner() {}
  virtual void
  analyzeCandidateList(std::vector<OutlineCandidate> &CL) override {
    IRInstructionMapper &MIM = getMapperAs<IRInstructionMapper>();
    OutlinerAnalysisImpl OAI(MIM, CL, GetTTI, DL, DataAlloc);
    OAI.analyzeCandidateList();
  }
  virtual void outlineCandidate(OutlineCandidate &OC, size_t CandNum) override {
    IRInstructionMapper &IM = getMapperAs<IRInstructionMapper>();
    FS.prepareForNewInstance(OC, IM);
    for (unsigned i = 0, e = OC.size(); i < e; ++i) {
      unsigned Occur = OC.getOccurrence(i);
      if (EmitRemarks)
        emitOptRemark(OC, CandNum, i, Occur);
      FS.outlineOccurrence(OC, Occur, IM);
      errs() << "Block Outlined: " << CandNum << "\n";
    }
    FS.finalize(OC);

#ifndef NDEBUG
    Function *OF = IM.getInstrFunction(OC.getOccurrence(0));
    LLVM_DEBUG(dbgs() << "** Outlining : " << OF->getName() << "\n"
                 << " occurrences : " << OC.size() << "\n"
                 << " size : " << OC.Len << "\n"
                 << " benefit : " << OC.Benefit << "\n");
#endif
    ++NumCandidatesOutlined;
  }

  // \brief Emit a remark about a code block that was outlined.
  void emitOptRemark(OutlineCandidate &OC, unsigned CandidateIdx,
                     unsigned ValidOccurNum, unsigned Occur) {
    IRInstructionMapper &IM = getMapperAs<IRInstructionMapper>();
    Instruction *Head = IM.getInstr(Occur);
    Instruction *Tail = IM.getInstr(Occur + OC.Len - 1);
    Function *ParentFn = Head->getFunction();
    OptimizationRemarkEmitter ORE(ParentFn);

    std::ostringstream NameStream, AdditionalInfo;
    NameStream << "Outlining Candidate " << CandidateIdx << ": Occurrence "
               << ValidOccurNum;
    AdditionalInfo << ": " << OC.Len << " instructions : "
                   << " Estimated benefit=" << OC.Benefit;

    std::string Msg = NameStream.str();
    std::string Additional = AdditionalInfo.str();
    if (Tail->getDebugLoc() != Head->getDebugLoc()) {
      ore::NV BeginLoc("Begin location", Head);
      BeginLoc.Val = "Begin location";
      ore::NV TailLoc("End location", Tail);
      TailLoc.Val = "End location";
      ORE.emit(OptimizationRemark(DEBUG_TYPE, "Outlined", Head)
               << Msg << " Begin " << Additional << " : " << TailLoc);
      ORE.emit(OptimizationRemark(DEBUG_TYPE, "Outlined", Tail)
               << Msg << " End : " << BeginLoc);
    } else
      ORE.emit(OptimizationRemark(DEBUG_TYPE, "Outlined", Head)
               << Msg << Additional);
  }

  // Emitting optimization remarks creates a compile hit due to the string
  //   manipulation. Check if its enabled so that we can avoid it.
  bool areRemarksEnabled(Module &M) {
    bool IsEnabled = false;//OptimizationRemark::isEnabled(DEBUG_TYPE); //TODO: not sure what to replace it with
    if (!IsEnabled && !M.empty()) {
      OptimizationRemarkEmitter ORE(&M.getFunctionList().front());
      //IsEnabled = ORE.allowExtraAnalysis(); //TODO: fix this unkown call
    }
    return IsEnabled;
  }
  SpecificBumpPtrAllocator<AdditionalCandidateData> DataAlloc;
  function_ref<TargetTransformInfo &(Function &)> GetTTI;
  const DataLayout *DL;
  FunctionSplicer FS;
  bool EmitRemarks;
};

// Run the outliner over the provided module /p M.
bool runImpl(Module &M, ProfileSummaryInfo *PSI,
             function_ref<BlockFrequencyInfo &(Function &)> GetBFI,
             function_ref<TargetTransformInfo &(Function &)> GetTTI) {
  if (!EnableOutliner)
    return false;
  IRInstructionMapper OM;
  OM.mapModule(M, PSI, GetBFI);

  errs() << "Testing potential candidates\n";
  // No potential candidates.
  if (OM.getNumMappedInstructions() < MinOccurrences * MinInstructionLength)
    return false;

  errs() << "Running Sequenced IR Outliner\n";
  // Logic for outlining an individual candidate.
  bool EmitProfileData = PSI->hasProfileSummary();
  SequencedIROutliner SOI(OM, EmitProfileData, GetTTI, M);
  return SOI.run();
}
} // namespace

PreservedAnalyses CodeSizeOutlinerPass::run(Module &M,
                                            ModuleAnalysisManager &AM) {
  auto &FAM = AM.getResult<FunctionAnalysisManagerModuleProxy>(M).getManager();
  std::function<TargetTransformInfo &(Function &)> GetTTI =
      [&](Function &F) -> TargetTransformInfo & {
    return FAM.getResult<TargetIRAnalysis>(F);
  };
  std::function<BlockFrequencyInfo &(Function &)> GetBFI =
      [&](Function &F) -> BlockFrequencyInfo & {
    return FAM.getResult<BlockFrequencyAnalysis>(F);
  };
  ProfileSummaryInfo *PSI = &AM.getResult<ProfileSummaryAnalysis>(M);
  return runImpl(M, PSI, GetBFI, GetTTI) ? PreservedAnalyses::none()
                                         : PreservedAnalyses::all();
}

struct CodeSizeOutlinerLegacyPass : public ModulePass {
  static char ID; // Pass identification, replacement for typeid
  CodeSizeOutlinerLegacyPass() : ModulePass(ID) {
    initializeCodeSizeOutlinerLegacyPassPass(*PassRegistry::getPassRegistry());
  }
  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.addRequired<ProfileSummaryInfoWrapperPass>();
    AU.addRequired<TargetTransformInfoWrapperPass>();
  }
  bool runOnModule(Module &M) override {
    if (skipModule(M))
      return false;
    TargetTransformInfoWrapperPass *TTIWP =
        &getAnalysis<TargetTransformInfoWrapperPass>();
    DenseMap<Function *, TargetTransformInfo> TTIMap;
    std::function<TargetTransformInfo &(Function &)> GetTTI =
        [&](Function &F) -> TargetTransformInfo & {
      auto TTIIt = TTIMap.find(&F);
      if (TTIIt == TTIMap.end())
        TTIIt = TTIMap.try_emplace(&F, std::move(TTIWP->getTTI(F))).first;
      return TTIIt->second;
    };
    std::unique_ptr<BlockFrequencyInfo> CurBFI;
    std::function<BlockFrequencyInfo &(Function &)> GetBFI =
        [&](Function &F) -> BlockFrequencyInfo & {
      DominatorTree DT(F);
      LoopInfo LI(DT);
      BranchProbabilityInfo BPI(F, LI);
      CurBFI.reset(new BlockFrequencyInfo(F, BPI, LI));
      return *CurBFI.get();
    };
    ProfileSummaryInfo *PSI =
        &getAnalysis<ProfileSummaryInfoWrapperPass>().getPSI();
    return runImpl(M, PSI, GetBFI, GetTTI);
  }
};
char CodeSizeOutlinerLegacyPass::ID = 0;
INITIALIZE_PASS_BEGIN(CodeSizeOutlinerLegacyPass, DEBUG_TYPE,
                      "Code Size Outliner", false, false)
INITIALIZE_PASS_DEPENDENCY(ProfileSummaryInfoWrapperPass)
INITIALIZE_PASS_DEPENDENCY(TargetTransformInfoWrapperPass)
INITIALIZE_PASS_END(CodeSizeOutlinerLegacyPass, DEBUG_TYPE,
                    "Code Size Outliner", false, false)
ModulePass *llvm::createCodeSizeOutlinerPass() {
  return new CodeSizeOutlinerLegacyPass();
}
