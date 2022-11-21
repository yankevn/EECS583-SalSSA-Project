; RUN: llc -mtriple armv7 %s -o - | FileCheck %s

define float @f(<4 x i16>* nocapture %in) {
; CHECK-LABEL: f:
; CHECK:       @ %bb.0:
; CHECK-NEXT:    vld1.16 {d16}, [r0:64]
; CHECK-NEXT:    vmovl.u16 q8, d16
; CHECK-NEXT:    vcvt.f32.u32 q0, q8
; CHECK-NEXT:    vadd.f32 s4, s0, s1
; CHECK-NEXT:    vadd.f32 s0, s4, s2
; CHECK-NEXT:    vmov r0, s0
; CHECK-NEXT:    bx lr
  %1 = load <4 x i16>, <4 x i16>* %in
  %2 = uitofp <4 x i16> %1 to <4 x float>
  %3 = extractelement <4 x float> %2, i32 0
  %4 = extractelement <4 x float> %2, i32 1
  %5 = extractelement <4 x float> %2, i32 2

  %6 = fadd float %3, %4
  %7 = fadd float %6, %5

  ret float %7
}

define float @g(<4 x i16>* nocapture %in) {
; CHECK-LABEL: g:
; CHECK:       @ %bb.0:
; CHECK-NEXT:    vldr d16, [r0]
; CHECK-NEXT:    vmov.u16 r0, d16[0]
; CHECK-NEXT:    vmov s0, r0
; CHECK-NEXT:    vcvt.f32.u32 s0, s0
; CHECK-NEXT:    vmov r0, s0
; CHECK-NEXT:    bx lr
  %1 = load <4 x i16>, <4 x i16>* %in
  %2 = extractelement <4 x i16> %1, i32 0
  %3 = uitofp i16 %2 to float
  ret float %3
}

; Make sure we generate zext from <4 x i8> to <4 x 32>.
define <4 x i32> @h(<4 x i8> *%in) {
; CHECK-LABEL: h:
; CHECK:       @ %bb.0:
; CHECK-NEXT:    vld1.32 {d16[0]}, [r0:32]
; CHECK-NEXT:    vmovl.u8 q8, d16
; CHECK-NEXT:    vmovl.u16 q8, d16
; CHECK-NEXT:    vmov r0, r1, d16
; CHECK-NEXT:    vmov r2, r3, d17
; CHECK-NEXT:    bx lr
  %1 = load <4 x i8>, <4 x i8>* %in, align 4
  %2 = extractelement <4 x i8> %1, i32 0
  %3 = zext i8 %2 to i32
  %4 = insertelement <4 x i32> undef, i32 %3, i32 0
  %5 = extractelement <4 x i8> %1, i32 1
  %6 = zext i8 %5 to i32
  %7 = insertelement <4 x i32> %4, i32 %6, i32 1
  %8 = extractelement <4 x i8> %1, i32 2
  %9 = zext i8 %8 to i32
  %10 = insertelement <4 x i32> %7, i32 %9, i32 2
  %11 = extractelement <4 x i8> %1, i32 3
  %12 = zext i8 %11 to i32
  %13 = insertelement <4 x i32> %10, i32 %12, i32 3
  ret <4 x i32> %13
}

define float @i(<4 x i16>* nocapture %in) {
  ; FIXME: The vmov.u + sxt can convert to a vmov.s
; CHECK-LABEL: i:
; CHECK:       @ %bb.0:
; CHECK-NEXT:    vldr d16, [r0]
; CHECK-NEXT:    vmov.u16 r0, d16[0]
; CHECK-NEXT:    sxth r0, r0
; CHECK-NEXT:    vmov s0, r0
; CHECK-NEXT:    vcvt.f32.s32 s0, s0
; CHECK-NEXT:    vmov r0, s0
; CHECK-NEXT:    bx lr
  %1 = load <4 x i16>, <4 x i16>* %in
  %2 = extractelement <4 x i16> %1, i32 0
  %3 = sitofp i16 %2 to float
  ret float %3
}

define float @j(<8 x i8>* nocapture %in) {
; CHECK-LABEL: j:
; CHECK:       @ %bb.0:
; CHECK-NEXT:    vldr d16, [r0]
; CHECK-NEXT:    vmov.u8 r0, d16[7]
; CHECK-NEXT:    vmov s0, r0
; CHECK-NEXT:    vcvt.f32.u32 s0, s0
; CHECK-NEXT:    vmov r0, s0
; CHECK-NEXT:    bx lr
  %1 = load <8 x i8>, <8 x i8>* %in
  %2 = extractelement <8 x i8> %1, i32 7
  %3 = uitofp i8 %2 to float
  ret float %3
}

define float @k(<8 x i8>* nocapture %in) {
; FIXME: The vmov.u + sxt can convert to a vmov.s
; CHECK-LABEL: k:
; CHECK:       @ %bb.0:
; CHECK-NEXT:    vldr d16, [r0]
; CHECK-NEXT:    vmov.u8 r0, d16[7]
; CHECK-NEXT:    sxtb r0, r0
; CHECK-NEXT:    vmov s0, r0
; CHECK-NEXT:    vcvt.f32.s32 s0, s0
; CHECK-NEXT:    vmov r0, s0
; CHECK-NEXT:    bx lr
  %1 = load <8 x i8>, <8 x i8>* %in
  %2 = extractelement <8 x i8> %1, i32 7
  %3 = sitofp i8 %2 to float
  ret float %3
}

define float @KnownUpperZero(<4 x i16> %v) {
; CHECK-LABEL: KnownUpperZero:
; CHECK:       @ %bb.0:
; CHECK-NEXT:    vmov d16, r0, r1
; CHECK-NEXT:    vmov.u16 r0, d16[0]
; CHECK-NEXT:    vmov.u16 r1, d16[3]
; CHECK-NEXT:    and r0, r0, #3
; CHECK-NEXT:    vmov s0, r0
; CHECK-NEXT:    and r0, r1, #3
; CHECK-NEXT:    vmov s2, r0
; CHECK-NEXT:    vcvt.f32.s32 s0, s0
; CHECK-NEXT:    vcvt.f32.s32 s2, s2
; CHECK-NEXT:    vadd.f32 s0, s2, s0
; CHECK-NEXT:    vmov r0, s0
; CHECK-NEXT:    bx lr
  %1 = and <4 x i16> %v, <i16 3,i16 3,i16 3,i16 3>
  %2 = extractelement <4 x i16> %1, i32 3
  %3 = extractelement <4 x i16> %1, i32 0
  %sinf1 = sitofp i16 %2 to float
  %sinf2 = sitofp i16 %3 to float
  %sum =   fadd float %sinf1, %sinf2
  ret float %sum
}
