---
title: Beating the Auto-Vectorizer
toc-title: Table of contents
date: Aug 10. 2025
---

*Full disclosure: I'm not not an expert programmer, this is purely for exploring different optimization techniques. Take
everything you read here with a pinch of salt and always benchmark your code before optimizing.*

# Introduction

Writing fast and efficient code is important for various reasons, and you know them, otherwise you wouldn't read this article.
If you want to write fast code you need to utilize the hardware that your code is running on. One feature of all CPUs created
in the last 25 years have [single instruction, multiple data (SIMD)](https://en.wikipedia.org/wiki/Single_instruction,_multiple_data)
instructions that allows the programmer to issue one instruction that operate on multiple elements at once.

**Disclaimer:** I won't be using intrinsics in the manually vectorized implementations, I will use Rust's
[portable SIMD](https://doc.rust-lang.org/stable/std/simd/index.html) module. What I'll present are code that you write once
and run everywhere.

**Benchmarks:** The benchmarks contains some acronyms I wan't to explain here. TSC: time stamp counter, on my x64 machine I used
`rdtsc` immediately before and after running the hot function, TSC will be the difference between the two samples.

All source code presented can be found [here](https://github.com/MAlba124/malba124.github.io/blob/main/beating-the-auto-vectorizer).

# Match 1

Let's start off with the "hello world" of SIMD. Summing floats.

```rust
const S: usize = 16;
macro_rules! fold_sum_f32 {
    ($chunk:expr) => {
        $chunk.iter().fold(0.0, |acc, b| acc + b)
    };
}
```

## Contestant nr. 1

```rust
fn sum_naive(nums: &[f32]) -> f32 {
    fold_sum_f32!(nums)
}
```

## Contestant nr. 2

```rust
fn sum_auto(nums: &[f32]) -> f32 {
    let mut accumulators = [0.0; S];
    let (chunks, rem) = nums.as_chunks::<S>();
    for chunk in chunks {
        chunk
            .iter()
            .zip(accumulators.iter_mut())
            .for_each(|(n, acc)| *acc += *n);
    }

    fold_sum_f32!(accumulators) + fold_sum_f32!(rem)
}
```

## Contestant nr. 3

```rust
fn sum_manual(nums: &[f32]) -> f32 {
    let mut accumulator = f32x16::splat(0.0);
    let (chunks, rem) = nums.as_chunks::<S>();
    for chunk in chunks {
        let chunk = f32x16::from_array(*chunk);
        accumulator.add_assign(chunk);
    }

    accumulator.reduce_sum() + fold_sum_f32!(rem)
}
```

## Benchmarks

The length of the `nums` array was `1024` for the following benchmarks.

### Debug (`cargo run`)

<!--
(defun compute-perf (baseline auto manual)
    (interactive)
    (end-of-line)
    (newline)
    (insert (format "Baseline: %.2f Auto: %.2f Manual %.2f"
        (/ baseline baseline) (/ baseline auto) (/ baseline manual))))

(compute-perf 19751.0 33218.0 5356.0)
Baseline: 1.00 Auto: 0.59 Manual 3.69
-->

| Implementation | TSC average | Time average | Baseline (Naive) / TSC |
|----------------|-------------|--------------|------------------------|
| Naive          | 19751       | 7.657µs      | 1.00                   |
| Auto           | 33218       | 12.833µs     | 0.59                   |
| Manual         | 5356        | 2.09µs       | 3.69                   |

### Release (`cargo run --release`)

<!--
(compute-perf 4129.0 230.0 230.0)
Baseline: 1.00 Auto: 17.95 Manual 17.95
-->

| Implementation | TSC average | Time average | Baseline (Naive) / TSC |
|----------------|-------------|--------------|------------------------|
| Naive          | 4129        | 1.621µs      | 1                      |
| Auto           | 230         | 130ns        | 17.95                  |
| Manual         | 230         | 130ns        | 17.95                  |

## Results

It shouldn't come to anyones surprise that the compiler is able to create code equally performant as the manually
vectorized implementation. We can take a look at the generated assembly code with [cargo-show-asm](https://github.com/pacak/cargo-show-asm)
to see if there are any differences.

Auto:

```asm
sum_auto:
        .cfi_startproc
        mov ecx, esi
        and ecx, 15
        mov rax, rsi
        shr rax, 4
        xorps xmm2, xmm2
        xorps xmm1, xmm1
        xorps xmm0, xmm0
        cmp rsi, 16
        jb .LBB23_4
        mov rdx, rax
        shl rdx, 6
        xorps xmm4, xmm4
        xor r8d, r8d
        xorps xmm3, xmm3
        xorps xmm0, xmm0
        xorps xmm1, xmm1
        .p2align        4
.LBB23_2:
        movups xmm5, xmmword ptr [rdi + r8]
        addps xmm4, xmm5
        movups xmm5, xmmword ptr [rdi + r8 + 16]
        addps xmm3, xmm5
        movups xmm5, xmmword ptr [rdi + r8 + 32]
        addps xmm0, xmm5
        movups xmm5, xmmword ptr [rdi + r8 + 48]
        addps xmm1, xmm5
        add r8, 64
        cmp rdx, r8
        jne .LBB23_2
        xorps xmm5, xmm5
        addss xmm5, xmm4
        movaps xmm6, xmm4
        shufps xmm6, xmm4, 85
        addss xmm6, xmm5
        movaps xmm5, xmm4
        unpckhpd xmm5, xmm4
        addss xmm5, xmm6
        shufps xmm4, xmm4, 255
        addss xmm4, xmm5
        addss xmm4, xmm3
        movaps xmm5, xmm3
        shufps xmm5, xmm3, 85
        addss xmm5, xmm4
        movaps xmm4, xmm3
        unpckhpd xmm4, xmm3
        addss xmm4, xmm5
        shufps xmm3, xmm3, 255
        addss xmm3, xmm4
        addss xmm3, xmm0
        movaps xmm4, xmm0
        shufps xmm4, xmm0, 85
        addss xmm4, xmm3
        movaps xmm3, xmm0
        unpckhpd xmm3, xmm0
        addss xmm3, xmm4
        shufps xmm0, xmm0, 255
        addss xmm0, xmm3
        addss xmm0, xmm1
        movaps xmm3, xmm1
        shufps xmm3, xmm1, 85
        addss xmm3, xmm0
        movaps xmm0, xmm1
        unpckhpd xmm0, xmm1
        addss xmm0, xmm3
        shufps xmm1, xmm1, 255
.LBB23_4:
        test rcx, rcx
        je .LBB23_12
        mov edx, esi
        and edx, 7
        cmp ecx, 8
        jae .LBB23_7
        xorps xmm2, xmm2
        xor eax, eax
        jmp .LBB23_9
.LBB23_7:
        mov ecx, esi
        and ecx, 8
        shl rax, 6
        lea r8, [rax + rdi]
        add r8, 28
        xorps xmm2, xmm2
        xor eax, eax
        .p2align        4
.LBB23_8:
        addss xmm2, dword ptr [r8 + 4*rax - 28]
        addss xmm2, dword ptr [r8 + 4*rax - 24]
        addss xmm2, dword ptr [r8 + 4*rax - 20]
        addss xmm2, dword ptr [r8 + 4*rax - 16]
        addss xmm2, dword ptr [r8 + 4*rax - 12]
        addss xmm2, dword ptr [r8 + 4*rax - 8]
        addss xmm2, dword ptr [r8 + 4*rax - 4]
        addss xmm2, dword ptr [r8 + 4*rax]
        add rax, 8
        cmp rcx, rax
        jne .LBB23_8
.LBB23_9:
        test rdx, rdx
        je .LBB23_12
        and rsi, -16
        shl rax, 2
        lea rax, [rax + 4*rsi]
        add rdi, rax
        xor eax, eax
        .p2align        4
.LBB23_11:
        addss xmm2, dword ptr [rdi + 4*rax]
        inc rax
        cmp rdx, rax
        jne .LBB23_11
.LBB23_12:
        addss xmm0, xmm1
        addss xmm0, xmm2
        ret
```

Manual:

```asm
sum_manual:
        .cfi_startproc
        mov edx, esi
        and edx, 15
        mov rcx, rsi
        shr rcx, 4
        cmp rsi, 16
        jae .LBB24_2
        xorps xmm3, xmm3
        xorps xmm2, xmm2
        xorps xmm1, xmm1
        xorps xmm0, xmm0
        jmp .LBB24_4
.LBB24_2:
        mov rax, rcx
        shl rax, 6
        xorps xmm3, xmm3
        xor r8d, r8d
        xorps xmm2, xmm2
        xorps xmm1, xmm1
        xorps xmm0, xmm0
        .p2align        4
.LBB24_3:
        movups xmm4, xmmword ptr [rdi + r8]
        addps xmm3, xmm4
        movups xmm4, xmmword ptr [rdi + r8 + 16]
        addps xmm2, xmm4
        movups xmm4, xmmword ptr [rdi + r8 + 32]
        addps xmm1, xmm4
        movups xmm4, xmmword ptr [rdi + r8 + 48]
        addps xmm0, xmm4
        add r8, 64
        cmp rax, r8
        jne .LBB24_3
.LBB24_4:
        test rdx, rdx
        je .LBB24_5
        mov eax, esi
        and eax, 7
        cmp edx, 8
        jae .LBB24_8
        xorps xmm4, xmm4
        xor ecx, ecx
        jmp .LBB24_10
.LBB24_5:
        xorps xmm4, xmm4
        jmp .LBB24_13
.LBB24_8:
        mov edx, esi
        and edx, 8
        shl rcx, 6
        lea r8, [rcx + rdi]
        add r8, 28
        xorps xmm4, xmm4
        xor ecx, ecx
        .p2align        4
.LBB24_9:
        addss xmm4, dword ptr [r8 + 4*rcx - 28]
        addss xmm4, dword ptr [r8 + 4*rcx - 24]
        addss xmm4, dword ptr [r8 + 4*rcx - 20]
        addss xmm4, dword ptr [r8 + 4*rcx - 16]
        addss xmm4, dword ptr [r8 + 4*rcx - 12]
        addss xmm4, dword ptr [r8 + 4*rcx - 8]
        addss xmm4, dword ptr [r8 + 4*rcx - 4]
        addss xmm4, dword ptr [r8 + 4*rcx]
        add rcx, 8
        cmp rdx, rcx
        jne .LBB24_9
.LBB24_10:
        test rax, rax
        je .LBB24_13
        and rsi, -16
        shl rcx, 2
        lea rcx, [rcx + 4*rsi]
        add rdi, rcx
        xor ecx, ecx
        .p2align        4
.LBB24_12:
        addss xmm4, dword ptr [rdi + 4*rcx]
        inc rcx
        cmp rax, rcx
        jne .LBB24_12
.LBB24_13:
        movaps xmm5, xmm3
        shufps xmm5, xmm3, 85
        addss xmm5, xmm3
        movaps xmm6, xmm3
        unpckhpd xmm6, xmm3
        addss xmm6, xmm5
        shufps xmm3, xmm3, 255
        addss xmm3, xmm6
        addss xmm3, xmm2
        movaps xmm5, xmm2
        shufps xmm5, xmm2, 85
        addss xmm5, xmm3
        movaps xmm3, xmm2
        unpckhpd xmm3, xmm2
        addss xmm3, xmm5
        shufps xmm2, xmm2, 255
        addss xmm2, xmm3
        addss xmm2, xmm1
        movaps xmm3, xmm1
        shufps xmm3, xmm1, 85
        addss xmm3, xmm2
        movaps xmm2, xmm1
        unpckhpd xmm2, xmm1
        addss xmm2, xmm3
        shufps xmm1, xmm1, 255
        addss xmm1, xmm2
        addss xmm1, xmm0
        movaps xmm2, xmm0
        shufps xmm2, xmm0, 85
        addss xmm2, xmm1
        movaps xmm1, xmm0
        unpckhpd xmm1, xmm0
        addss xmm1, xmm2
        shufps xmm0, xmm0, 255
        addss xmm0, xmm1
        addss xmm0, xmm4
        ret
```

We can see that the hot path of the loop produces the exact same code if we compare the labels `.LBB23_2` for auto and
`.LBB24_3` for the manual implementation. The results are not that interesting and we can't trivially squeeze out any
more performance.
Let's move on.

# Match 2

Let's imageine you need to count how many newlines a file has, how would you do that quickly?
This next match will take a closer look at that.

```rust
const N: usize = 32;
macro_rules! sum_zeros {
    ($chunk:expr) => {
        $chunk.iter().fold(0, |acc, b| acc + (*b == 0) as usize)
    };
}
```

## Contestant nr. 1

```rust
fn count_newlines_naive(buf: &[u8]) -> usize {
    buf.iter()
        .filter(|ch| **ch == b'\n')
        .map(|_| 1)
        .sum::<usize>()
}
```

## Contestant nr. 2

```rust
fn count_newlines_auto(buf: &[u8]) -> usize {
    let mut count = 0;
    let (chunks, rem) = buf.as_chunks::<N>();
    for chunk in chunks {
        let res = chunk.iter().map(|b| b.wrapping_sub(b'\n'));
        count += res.fold(0, |acc, b| acc + (b == 0) as usize);
    }

    count + sum_zeros!(rem)
}
```

## Contestant nr. 3

```rust
fn count_newlines_manual(buf: &[u8]) -> usize {
    let mut count = 0;
    let (chunks, rem) = buf.as_chunks::<N>();
    let newlines = u8x32::splat(b'\n');
    for chunk in chunks {
        let chunk = u8x32::from_array(*chunk);
        let res = chunk.simd_eq(newlines);
        count += res.to_bitmask().count_ones() as usize;
    }

    count + sum_zeros!(rem)
}
```

## Benchmarks

The length of the `buf` array was `4096` for the following benchmarks.

### Debug (`cargo run`)

<!--
(defun compute-perf (baseline auto manual)
    (interactive)
    (end-of-line)
    (newline)
    (insert (format "Baseline: %.2f Auto: %.2f Manual %.2f"
        (/ baseline baseline) (/ baseline auto) (/ baseline manual))))

(compute-perf 44603.0 90775.0 30707.0)
Baseline: 1.00 Auto: 0.49 Manual 1.45
-->

| Implementation | TSC average | Time average | Baseline (Naive) / TSC |
|----------------|-------------|--------------|------------------------|
| Naive          | 44603       | 17.102µs     | 1.00                   |
| Auto           | 90775       | 35.047µs     | 0.49                   |
| Manual         | 30707       | 25.827µs     | 1.45                   |

### Release (`cargo run --release`)

<!--
(compute-perf 4420.0 4500.0 555.0)
Baseline: 1.00 Auto: 0.98 Manual 7.96
-->

| Implementation | TSC average | Time average | Baseline (Naive) / TSC |
|----------------|-------------|--------------|------------------------|
| Naive          | 4420        | 1.70µs       | 1                      |
| Auto           | 4500        | 1.758µs      | 0.98                   |
| Manual         | 555         | 230ns        | 7.96                   |

## Results

These results are quite interesting. Firstly, for debug the build our auto vectorized implementation are
significantly slower than the naive sum. Secondly, for release the build our manual implementation yields an 8x speed-up
compared to the other contestants which for all intents and purposes perform equally well.

*On a side note, `rustc` has a special option `-C target-cpu=native` that dramatically changes the results of these benchmarks
(hint; baseline gets 3x faster). I will not include those here, as it's not portable, but it's worth knowing that it's available
in those cases when you need it.*

Why does the compiler generate such slow code for the auto version? We'll take a closer look.

Auto:

```asm
count_newlines_auto:
        .cfi_startproc
        mov r8, rsi
        and r8, -32
        mov ecx, esi
        and ecx, 31
        cmp rsi, 32
        jae .LBB23_2
        xor edx, edx
        jmp .LBB23_4
.LBB23_2:
        xor eax, eax
        movdqa xmm0, xmmword ptr [rip + .LCPI23_0]
        xor edx, edx
        .p2align        4
.LBB23_3:
        movdqu xmm1, xmmword ptr [rdi + rax]
        movdqu xmm2, xmmword ptr [rdi + rax + 16]
        pcmpeqb xmm1, xmm0
        pmovmskb r9d, xmm1
        pcmpeqb xmm2, xmm0
        pmovmskb r10d, xmm2
        shl r10d, 16
        or r10d, r9d
        mov r9d, r10d
        shr r9d
        and r9d, 1431655765
        sub r10d, r9d
        mov r9d, r10d
        and r9d, 858993459
        shr r10d, 2
        and r10d, 858993459
        add r10d, r9d
        mov r9d, r10d
        shr r9d, 4
        add r9d, r10d
        and r9d, 252645135
        imul r9d, r9d, 16843009
        shr r9d, 24
        add rdx, r9
        add rax, 32
        cmp r8, rax
        jne .LBB23_3
.LBB23_4:
        test rcx, rcx
        je .LBB23_5
        cmp ecx, 4
        jae .LBB23_8
        xor esi, esi
        xor eax, eax
        jmp .LBB23_11
.LBB23_5:
        xor eax, eax
        add rax, rdx
        ret
.LBB23_8:
        and esi, 28
        lea rax, [r8 + rdi]
        add rax, 2
        pxor xmm0, xmm0
        xor r9d, r9d
        movdqa xmm3, xmmword ptr [rip + .LCPI23_1]
        pxor xmm2, xmm2
        pxor xmm1, xmm1
        .p2align        4
.LBB23_9:
        movzx r10d, word ptr [rax + r9 - 2]
        movd xmm4, r10d
        movzx r10d, word ptr [rax + r9]
        movd xmm5, r10d
        pcmpeqb xmm4, xmm0
        punpcklbw xmm4, xmm4
        pshuflw xmm4, xmm4, 212
        pshufd xmm4, xmm4, 212
        pand xmm4, xmm3
        paddq xmm2, xmm4
        pcmpeqb xmm5, xmm0
        punpcklbw xmm5, xmm5
        pshuflw xmm4, xmm5, 212
        pshufd xmm4, xmm4, 212
        pand xmm4, xmm3
        paddq xmm1, xmm4
        add r9, 4
        cmp rsi, r9
        jne .LBB23_9
        paddq xmm1, xmm2
        pshufd xmm0, xmm1, 238
        paddq xmm0, xmm1
        movq rax, xmm0
        cmp ecx, esi
        je .LBB23_13
.LBB23_11:
        add rdi, r8
        .p2align        4
.LBB23_12:
        cmp byte ptr [rdi + rsi], 1
        adc rax, 0
        inc rsi
        cmp rcx, rsi
        jne .LBB23_12
.LBB23_13:
        add rax, rdx
        ret
```


Manual:

```asm
count_newlines_manual:
        .cfi_startproc
        mov rdx, rsi
        and rdx, -32
        mov ecx, esi
        and ecx, 31
        mov eax, 10
        cmp rsi, 32
        jb .LBB24_4
        xor r8d, r8d
        movdqa xmm0, xmmword ptr [rip + .LCPI24_0]
        xor eax, eax
        .p2align        4
.LBB24_2:
        movdqu xmm1, xmmword ptr [rdi + r8]
        movdqu xmm2, xmmword ptr [rdi + r8 + 16]
        pcmpeqb xmm1, xmm0
        pmovmskb r9d, xmm1
        pcmpeqb xmm2, xmm0
        pmovmskb r10d, xmm2
        shl r10d, 16
        or r10d, r9d
        mov r9d, r10d
        shr r9d
        and r9d, 1431655765
        sub r10d, r9d
        mov r9d, r10d
        and r9d, 858993459
        shr r10d, 2
        and r10d, 858993459
        add r10d, r9d
        mov r9d, r10d
        shr r9d, 4
        add r9d, r10d
        and r9d, 252645135
        imul r9d, r9d, 16843009
        shr r9d, 24
        add rax, r9
        add r8, 32
        cmp rdx, r8
        jne .LBB24_2
.LBB24_4:
        test rcx, rcx
        je .LBB24_5
        cmp ecx, 4
        jae .LBB24_8
        xor esi, esi
        xor r8d, r8d
        jmp .LBB24_11
.LBB24_5:
        xor r8d, r8d
        add rax, r8
        ret
.LBB24_8:
        and esi, 28
        lea r8, [rdx + rdi]
        add r8, 2
        pxor xmm0, xmm0
        xor r9d, r9d
        movdqa xmm3, xmmword ptr [rip + .LCPI24_1]
        pxor xmm2, xmm2
        pxor xmm1, xmm1
        .p2align        4
.LBB24_9:
        movzx r10d, word ptr [r8 + r9 - 2]
        movd xmm4, r10d
        movzx r10d, word ptr [r8 + r9]
        movd xmm5, r10d
        pcmpeqb xmm4, xmm0
        punpcklbw xmm4, xmm4
        pshuflw xmm4, xmm4, 212
        pshufd xmm4, xmm4, 212
        pand xmm4, xmm3
        paddq xmm2, xmm4
        pcmpeqb xmm5, xmm0
        punpcklbw xmm5, xmm5
        pshuflw xmm4, xmm5, 212
        pshufd xmm4, xmm4, 212
        pand xmm4, xmm3
        paddq xmm1, xmm4
        add r9, 4
        cmp rsi, r9
        jne .LBB24_9
        paddq xmm1, xmm2
        pshufd xmm0, xmm1, 238
        paddq xmm0, xmm1
        movq r8, xmm0
        cmp ecx, esi
        je .LBB24_13
.LBB24_11:
        add rdi, rdx
        .p2align        4
.LBB24_12:
        cmp byte ptr [rdi + rsi], 1
        adc r8, 0
        inc rsi
        cmp rcx, rsi
        jne .LBB24_12
.LBB24_13:
        add rax, r8
        ret
```

WTF!? They're pretty much the same. I guess we need a follow up where we dig deeper and profile the two
functions to be able to understand how one of them is almost eight times faster than the other, so stay tuned!
