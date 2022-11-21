
GlobalVariable* CreateGlobalString(Module &M, std::string RawString) {
  ArrayType *Ty = ArrayType::get(Type::getInt8Ty(M.getContext()),RawString.size()+1);
  /*
  GlobalVariable *GV = new GlobalVariable(M, Ty, true, GlobalValue::InternalLinkage,
    //ConstantArray::get(M.getContext(),RawString),
    ConstantDataArray::getString(M.getContext(),RawString),
    "tr.fname", 0, false, 0);
  */

  GlobalVariable *GV = new GlobalVariable(M, Ty, true, GlobalValue::InternalLinkage, ConstantDataArray::getString(M.getContext(),RawString),"tr.fname");
  return GV;
}

Value *GetStringPtr(IRBuilder<> &Builder, GlobalVariable *GV, LLVMContext &Context) {
  std::vector<Value*> GEPIdxs;
  GEPIdxs.push_back(ConstantInt::get(Type::getInt32Ty(Context), 0, false));
  GEPIdxs.push_back(ConstantInt::get(Type::getInt32Ty(Context), 0, false));
  //return (GetElementPtrInst::Create( GV, GEPIdxs.begin(), GEPIdxs.end(), "gepi", (BasicBlock*)0));
  return Builder.CreateInBoundsGEP(GV, ArrayRef<Value*>(GEPIdxs));
}


static Value *GetNullValue(Type *Ty) {
  switch (Ty->getTypeID()) {
  case Type::IntegerTyID:
  case Type::HalfTyID:
  case Type::FloatTyID:
  case Type::DoubleTyID:
  case Type::X86_FP80TyID:
  case Type::FP128TyID:
  case Type::PPC_FP128TyID:
  case Type::PointerTyID:
  case Type::StructTyID:
  case Type::ArrayTyID:
  case Type::VectorTyID:
  case Type::TokenTyID:
    return Constant::getNullValue(Ty);
  default:
    return UndefValue::get(Ty);
  }
}


class DynamicallyTraceFunction : public ModulePass {
public:
  static char ID;
  DynamicallyTraceFunction() : ModulePass(ID) {
     initializeDynamicallyTraceFunctionPass(*PassRegistry::getPassRegistry());
  }

  bool runOnModule(Module &M) override {

    GlobalVariable *GVRet = CreateGlobalString(M,std::string("Ret"));
    GlobalVariable *GVArg = CreateGlobalString(M,std::string("Arg"));

    for (Function &F : M) {
      if (F.isDeclaration() || F.getName().empty() || F.getName()[0]=='.')
        continue;
      
      GlobalVariable *GVName = CreateGlobalString(M,std::string(F.getName()));

      { ///report entry
        std::string Name = "__report_entry";
        std::vector<Type*> ArgsTy;
        ArgsTy.push_back( PointerType::get(Type::getInt8Ty(M.getContext()),0) );
        FunctionCallee ReportF = M.getOrInsertFunction(StringRef(Name), FunctionType::get(FunctionType::getVoidTy(F.getContext()), ArrayRef<Type*>(ArgsTy), false));
        IRBuilder<> Builder( &*F.getEntryBlock().getFirstInsertionPt() );
        std::vector<Value*> ArgVals;
        ArgVals.push_back( GetStringPtr(Builder, GVName, M.getContext()) );
        Builder.CreateCall(ReportF.getFunctionType(), ReportF.getCallee(),ArgVals);

        for (Argument &Arg : F.args()) {
          Type *ArgTy = Arg.getType();
          if (ArgTy->isPointerTy()) {
            continue;
          } else if (ArgTy->isIntegerTy() || ArgTy->isFloatTy() || ArgTy->isDoubleTy()) {
            std::string Name;
            if (ArgTy->isIntegerTy()) {
              unsigned IntTyWidth = ArgTy->getIntegerBitWidth();
              Name = std::string("__report_i")+std::to_string(IntTyWidth);
            } else if (ArgTy->isFloatTy()) {
              Name = "__report_float";
            } else {
              Name = "__report_double";
            }

            std::vector<Type*> ArgsTy;
            ArgsTy.push_back( PointerType::get(Type::getInt8Ty(M.getContext()),0) );
            ArgsTy.push_back(ArgTy);

            FunctionCallee ReportF = M.getOrInsertFunction(StringRef(Name), FunctionType::get(FunctionType::getVoidTy(F.getContext()), ArrayRef<Type*>(ArgsTy), false));

            std::vector<Value*> ArgVals;
            ArgVals.push_back( GetStringPtr(Builder, GVArg, M.getContext()) );
            ArgVals.push_back( &Arg );
            Builder.CreateCall(ReportF.getFunctionType(), ReportF.getCallee(),ArgVals);
          }
        }
      }

      Type *RetTy = F.getReturnType();

      if (RetTy->isPointerTy()) {
            
      } else if (RetTy->isIntegerTy() || RetTy->isFloatTy() || RetTy->isDoubleTy()) {
        std::string Name;
        if (RetTy->isIntegerTy()) {
          unsigned IntTyWidth = RetTy->getIntegerBitWidth();
          Name = std::string("__report_i")+std::to_string(IntTyWidth);
        } else if (RetTy->isFloatTy()) {
          Name = "__report_float";
        } else {
          Name = "__report_double";
        }
        
        std::vector<Type*> ArgsTy;
        ArgsTy.push_back( PointerType::get(Type::getInt8Ty(M.getContext()),0) );
        ArgsTy.push_back(RetTy);

        FunctionCallee ReportF = M.getOrInsertFunction(StringRef(Name), FunctionType::get(FunctionType::getVoidTy(F.getContext()), ArrayRef<Type*>(ArgsTy), false));

        for (BasicBlock &BB : F) {
          if (ReturnInst *RI = dyn_cast<ReturnInst>(BB.getTerminator())) {
            IRBuilder<> Builder(RI);

            std::vector<Value*> ArgVals;
            ArgVals.push_back( GetStringPtr(Builder, GVRet, M.getContext()) );
            ArgVals.push_back(RI->getReturnValue());
            Builder.CreateCall(ReportF.getFunctionType(), ReportF.getCallee(),ArgVals);
          }
        }
      }

      { ///report exit
        std::string Name = "__report_exit";
        std::vector<Type*> ArgsTy;
        ArgsTy.push_back( PointerType::get(Type::getInt8Ty(M.getContext()),0) );
        FunctionCallee ReportF = M.getOrInsertFunction(StringRef(Name), FunctionType::get(FunctionType::getVoidTy(F.getContext()), ArrayRef<Type*>(ArgsTy), false));

        for (BasicBlock &BB : F) {
          if (ReturnInst *RI = dyn_cast<ReturnInst>(BB.getTerminator())) {
            IRBuilder<> Builder(RI);

            std::vector<Value*> ArgVals;
            ArgVals.push_back( GetStringPtr(Builder, GVName, M.getContext()) );
            Builder.CreateCall(ReportF.getFunctionType(), ReportF.getCallee(),ArgVals);
          }
        }
      }
    }



    //replace all undefs to null
    for (Function &F : M) {
      if (F.isDeclaration()) continue;

      for (Instruction &I : instructions(&F)) {
        for (unsigned i = 0; i<I.getNumOperands(); i++) {
          if (I.getOperand(i)==UndefValue::get(I.getOperand(i)->getType())) {
            I.setOperand(i,GetNullValue(I.getOperand(i)->getType()));
          }
        }
      }
    }

    return true;
  }
  void getAnalysisUsage(AnalysisUsage &AU) const override {}
};

char DynamicallyTraceFunction::ID = 0;
INITIALIZE_PASS(DynamicallyTraceFunction, "dynamic-trace-func", "Trace Functions Dynamically", false,
                false)


ModulePass *llvm::createDynamicallyTraceFunctionPass() {
  return new DynamicallyTraceFunction();
}
