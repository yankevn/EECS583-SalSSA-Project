; NOTE: Assertions have been autogenerated by utils/update_llc_test_checks.py
; RUN: llc -mtriple=riscv32 -mattr +d -target-abi ilp32d < %s | FileCheck %s

define void @test_load_store(half* %p, half* %q) nounwind {
; CHECK-LABEL: test_load_store:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lh a0, 0(a0)
; CHECK-NEXT:    sh a0, 0(a1)
; CHECK-NEXT:    ret
  %a = load half, half* %p
  store half %a, half* %q
  ret void
}

define float @test_fpextend_float(half* %p) nounwind {
; CHECK-LABEL: test_fpextend_float:
; CHECK:       # %bb.0:
; CHECK-NEXT:    addi sp, sp, -16
; CHECK-NEXT:    sw ra, 12(sp)
; CHECK-NEXT:    lhu a0, 0(a0)
; CHECK-NEXT:    call __gnu_h2f_ieee
; CHECK-NEXT:    lw ra, 12(sp)
; CHECK-NEXT:    addi sp, sp, 16
; CHECK-NEXT:    ret
  %a = load half, half* %p
  %r = fpext half %a to float
  ret float %r
}

define double @test_fpextend_double(half* %p) nounwind {
; CHECK-LABEL: test_fpextend_double:
; CHECK:       # %bb.0:
; CHECK-NEXT:    addi sp, sp, -16
; CHECK-NEXT:    sw ra, 12(sp)
; CHECK-NEXT:    lhu a0, 0(a0)
; CHECK-NEXT:    call __gnu_h2f_ieee
; CHECK-NEXT:    fcvt.d.s fa0, fa0
; CHECK-NEXT:    lw ra, 12(sp)
; CHECK-NEXT:    addi sp, sp, 16
; CHECK-NEXT:    ret
  %a = load half, half* %p
  %r = fpext half %a to double
  ret double %r
}

define void @test_fptrunc_float(float %f, half* %p) nounwind {
; CHECK-LABEL: test_fptrunc_float:
; CHECK:       # %bb.0:
; CHECK-NEXT:    addi sp, sp, -16
; CHECK-NEXT:    sw ra, 12(sp)
; CHECK-NEXT:    sw s0, 8(sp)
; CHECK-NEXT:    mv s0, a0
; CHECK-NEXT:    call __gnu_f2h_ieee
; CHECK-NEXT:    sh a0, 0(s0)
; CHECK-NEXT:    lw s0, 8(sp)
; CHECK-NEXT:    lw ra, 12(sp)
; CHECK-NEXT:    addi sp, sp, 16
; CHECK-NEXT:    ret
  %a = fptrunc float %f to half
  store half %a, half* %p
  ret void
}

define void @test_fptrunc_double(double %d, half* %p) nounwind {
; CHECK-LABEL: test_fptrunc_double:
; CHECK:       # %bb.0:
; CHECK-NEXT:    addi sp, sp, -16
; CHECK-NEXT:    sw ra, 12(sp)
; CHECK-NEXT:    sw s0, 8(sp)
; CHECK-NEXT:    mv s0, a0
; CHECK-NEXT:    call __truncdfhf2
; CHECK-NEXT:    sh a0, 0(s0)
; CHECK-NEXT:    lw s0, 8(sp)
; CHECK-NEXT:    lw ra, 12(sp)
; CHECK-NEXT:    addi sp, sp, 16
; CHECK-NEXT:    ret
  %a = fptrunc double %d to half
  store half %a, half* %p
  ret void
}

define void @test_fadd(half* %p, half* %q) nounwind {
; CHECK-LABEL: test_fadd:
; CHECK:       # %bb.0:
; CHECK-NEXT:    addi sp, sp, -32
; CHECK-NEXT:    sw ra, 28(sp)
; CHECK-NEXT:    sw s0, 24(sp)
; CHECK-NEXT:    sw s1, 20(sp)
; CHECK-NEXT:    fsd fs0, 8(sp)
; CHECK-NEXT:    mv s0, a1
; CHECK-NEXT:    mv s1, a0
; CHECK-NEXT:    lhu a0, 0(a0)
; CHECK-NEXT:    call __gnu_h2f_ieee
; CHECK-NEXT:    fmv.s fs0, fa0
; CHECK-NEXT:    lhu a0, 0(s0)
; CHECK-NEXT:    call __gnu_h2f_ieee
; CHECK-NEXT:    fadd.s fa0, fs0, fa0
; CHECK-NEXT:    call __gnu_f2h_ieee
; CHECK-NEXT:    sh a0, 0(s1)
; CHECK-NEXT:    fld fs0, 8(sp)
; CHECK-NEXT:    lw s1, 20(sp)
; CHECK-NEXT:    lw s0, 24(sp)
; CHECK-NEXT:    lw ra, 28(sp)
; CHECK-NEXT:    addi sp, sp, 32
; CHECK-NEXT:    ret
  %a = load half, half* %p
  %b = load half, half* %q
  %r = fadd half %a, %b
  store half %r, half* %p
  ret void
}

define void @test_fmul(half* %p, half* %q) nounwind {
; CHECK-LABEL: test_fmul:
; CHECK:       # %bb.0:
; CHECK-NEXT:    addi sp, sp, -32
; CHECK-NEXT:    sw ra, 28(sp)
; CHECK-NEXT:    sw s0, 24(sp)
; CHECK-NEXT:    sw s1, 20(sp)
; CHECK-NEXT:    fsd fs0, 8(sp)
; CHECK-NEXT:    mv s0, a1
; CHECK-NEXT:    mv s1, a0
; CHECK-NEXT:    lhu a0, 0(a0)
; CHECK-NEXT:    call __gnu_h2f_ieee
; CHECK-NEXT:    fmv.s fs0, fa0
; CHECK-NEXT:    lhu a0, 0(s0)
; CHECK-NEXT:    call __gnu_h2f_ieee
; CHECK-NEXT:    fmul.s fa0, fs0, fa0
; CHECK-NEXT:    call __gnu_f2h_ieee
; CHECK-NEXT:    sh a0, 0(s1)
; CHECK-NEXT:    fld fs0, 8(sp)
; CHECK-NEXT:    lw s1, 20(sp)
; CHECK-NEXT:    lw s0, 24(sp)
; CHECK-NEXT:    lw ra, 28(sp)
; CHECK-NEXT:    addi sp, sp, 32
; CHECK-NEXT:    ret
  %a = load half, half* %p
  %b = load half, half* %q
  %r = fmul half %a, %b
  store half %r, half* %p
  ret void
}
