; NOTE: Assertions have been autogenerated by utils/update_test_checks.py UTC_ARGS: --function-signature
; RUN: opt -openmpopt -S < %s | FileCheck %s
; RUN: opt -passes=openmpopt -S < %s | FileCheck %s
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"

%struct.ident_t = type { i32, i32, i32, i32, i8* }

@0 = private unnamed_addr global %struct.ident_t { i32 0, i32 34, i32 0, i32 0, i8* getelementptr inbounds ([23 x i8], [23 x i8]* @.str, i32 0, i32 0) }, align 8
@.str = private unnamed_addr constant [23 x i8] c";unknown;unknown;0;0;;\00", align 1

declare i32 @__kmpc_global_thread_num(%struct.ident_t*)
declare void @useI32(i32)

define void @external(i1 %c) {
; CHECK-LABEL: define {{[^@]+}}@external
; CHECK-SAME: (i1 [[C:%.*]])
; CHECK-NEXT:  entry:
; CHECK-NEXT:    [[C2:%.*]] = tail call i32 @__kmpc_global_thread_num(%struct.ident_t* nonnull @0)
; CHECK-NEXT:    br i1 [[C]], label [[T:%.*]], label [[E:%.*]]
; CHECK:       t:
; CHECK-NEXT:    call void @internal(i32 [[C2]], i32 [[C2]])
; CHECK-NEXT:    call void @useI32(i32 [[C2]])
; CHECK-NEXT:    br label [[M:%.*]]
; CHECK:       e:
; CHECK-NEXT:    call void @internal(i32 [[C2]], i32 [[C2]])
; CHECK-NEXT:    call void @useI32(i32 [[C2]])
; CHECK-NEXT:    br label [[M]]
; CHECK:       m:
; CHECK-NEXT:    call void @internal(i32 0, i32 [[C2]])
; CHECK-NEXT:    call void @useI32(i32 [[C2]])
; CHECK-NEXT:    ret void
;
entry:
  br i1 %c, label %t, label %e
t:
  %c0 = tail call i32 @__kmpc_global_thread_num(%struct.ident_t* nonnull @0)
  call void @internal(i32 %c0, i32 %c0)
  call void @useI32(i32 %c0)
  br label %m
e:
  %c1 = tail call i32 @__kmpc_global_thread_num(%struct.ident_t* nonnull @0)
  call void @internal(i32 %c1, i32 %c1)
  call void @useI32(i32 %c1)
  br label %m
m:
  %c2 = tail call i32 @__kmpc_global_thread_num(%struct.ident_t* nonnull @0)
  call void @internal(i32 0, i32 %c2)
  call void @useI32(i32 %c2)
  ret void
}

define internal void @internal(i32 %not_gtid, i32 %gtid) {
; CHECK-LABEL: define {{[^@]+}}@internal
; CHECK-SAME: (i32 [[NOT_GTID:%.*]], i32 [[GTID:%.*]])
; CHECK-NEXT:  entry:
; CHECK-NEXT:    [[C:%.*]] = icmp eq i32 [[GTID]], [[GTID]]
; CHECK-NEXT:    br i1 [[C]], label [[T:%.*]], label [[E:%.*]]
; CHECK:       t:
; CHECK-NEXT:    call void @useI32(i32 [[GTID]])
; CHECK-NEXT:    call void @external(i1 [[C]])
; CHECK-NEXT:    br label [[M:%.*]]
; CHECK:       e:
; CHECK-NEXT:    call void @useI32(i32 [[GTID]])
; CHECK-NEXT:    br label [[M]]
; CHECK:       m:
; CHECK-NEXT:    call void @useI32(i32 [[GTID]])
; CHECK-NEXT:    ret void
;
entry:
  %cc = tail call i32 @__kmpc_global_thread_num(%struct.ident_t* nonnull @0)
  %c = icmp eq i32 %cc, %gtid
  br i1 %c, label %t, label %e
t:
  %c0 = tail call i32 @__kmpc_global_thread_num(%struct.ident_t* nonnull @0)
  call void @useI32(i32 %c0)
  call void @external(i1 %c)
  br label %m
e:
  %c1 = tail call i32 @__kmpc_global_thread_num(%struct.ident_t* nonnull @0)
  call void @useI32(i32 %c1)
  br label %m
m:
  %c2 = tail call i32 @__kmpc_global_thread_num(%struct.ident_t* nonnull @0)
  call void @useI32(i32 %c2)
  ret void
}
