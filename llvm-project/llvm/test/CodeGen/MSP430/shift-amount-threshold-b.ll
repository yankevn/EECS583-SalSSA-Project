; NOTE: Assertions have been autogenerated by utils/update_llc_test_checks.py
; RUN: llc -mtriple=msp430-- -msp430-no-legal-immediate=true < %s | FileCheck %s

; Test case for the following transformation in TargetLowering::SimplifySetCC
; (X & -256) == 256 -> (X >> 8) == 1
define i16 @testSimplifySetCC_2(i16 %x) {
; CHECK-LABEL: testSimplifySetCC_2:
; CHECK:       ; %bb.0: ; %entry
; CHECK-NEXT:    and #-64, r12
; CHECK-NEXT:    cmp #64, r12
; CHECK-NEXT:    mov r2, r12
; CHECK-NEXT:    rra r12
; CHECK-NEXT:    and #1, r12
; CHECK-NEXT:    ret
entry:
  %and = and i16 %x, -64
  %cmp = icmp eq i16 %and, 64
  %conv = zext i1 %cmp to i16
  ret i16 %conv
}

; Test case for the following transformation in TargetLowering::SimplifySetCC
; X >  0x0ffffffff -> (X >> 32) >= 1
define i16 @testSimplifySetCC_3(i16 %x) {
; CHECK-LABEL: testSimplifySetCC_3:
; CHECK:       ; %bb.0: ; %entry
; CHECK-NEXT:    cmp #64, r12
; CHECK-NEXT:    mov r2, r12
; CHECK-NEXT:    and #1, r12
; CHECK-NEXT:    ret
entry:
  %cmp = icmp ugt i16 %x, 63
  %conv = zext i1 %cmp to i16
  ret i16 %conv
}

; Test case for the following transformation in TargetLowering::SimplifySetCC
; X <  0x100000000 -> (X >> 32) <  1
define i16 @testSimplifySetCC_4(i16 %x) {
; CHECK-LABEL: testSimplifySetCC_4:
; CHECK:       ; %bb.0: ; %entry
; CHECK-NEXT:    cmp #64, r12
; CHECK-NEXT:    mov #1, r12
; CHECK-NEXT:    bic r2, r12
; CHECK-NEXT:    ret
entry:
  %cmp = icmp ult i16 %x, 64
  %conv = zext i1 %cmp to i16
  ret i16 %conv
}
