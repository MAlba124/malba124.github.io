#![feature(portable_simd)]

use std::{
    arch::x86_64::_rdtsc,
    fs::File,
    io::{Read, Write},
    ops::AddAssign,
    simd::{cmp::SimdPartialEq, f32x16, num::{SimdFloat, SimdUint}, u8x32},
    time::{Duration, Instant},
};

const N: usize = 32;
macro_rules! sum_zeros {
    ($chunk:expr) => {
        $chunk.iter().fold(0, |acc, b| acc + (*b == 0) as usize)
    };
}

#[inline(never)]
fn count_newlines_naive(buf: &[u8]) -> usize {
    buf.iter()
        .filter(|ch| **ch == b'\n')
        .map(|_| 1)
        .sum::<usize>()
}

#[inline(never)]
fn count_newlines_auto(buf: &[u8]) -> usize {
    let mut count = 0;
    let (chunks, rem) = buf.as_chunks::<N>();
    for chunk in chunks {
        let res = chunk.iter().map(|b| b.wrapping_sub(b'\n'));
        count += res.fold(0, |acc, b| acc + (b == 0) as usize);
    }

    count + sum_zeros!(rem)
}

#[inline(never)]
fn count_newlines_manual(buf: &[u8]) -> usize {
    let mut count = 0;
    let (chunks, rem) = buf.as_chunks::<N>();
    let newlines = u8x32::splat(b'\n');
    for chunk in chunks {
        let chunk = u8x32::from_array(*chunk);
        let res = chunk.simd_eq(newlines);
        count += res.to_bitmask().count_ones() as usize;
    }

    // count += 10; // needed to make the function not inlined

    count + sum_zeros!(rem)
}

struct Io {
    pub rand: File,
    pub null: File,
}

const CLEAR: &str = "\x1b[0m";
const CYAN: &str = "\x1b[1m\x1b[36m";
const MAGENTA: &str = "\x1b[1m\x1b[35m";

fn bench<D, B, T, R, C>(is_warmup: bool, mut user: T, get_data: D, benchee: B, trash_data: C)
where
    D: Fn(&mut Io, &mut T),
    B: Fn(&T) -> R,
    C: Fn(&mut Io, R),
{
    let mut io = Io {
        rand: File::open("/dev/random").unwrap(),
        null: File::options().write(true).open("/dev/null").unwrap(),
    };

    const ITERS: usize = 25000;

    let mut cycles = Vec::with_capacity(ITERS);
    let mut times = Vec::with_capacity(ITERS);
    let mut res = None;
    get_data(&mut io, &mut user);
    for _ in 0..ITERS {
        let start_time = Instant::now();
        let start = unsafe { _rdtsc() };

        res = Some(benchee(&user));

        let end = unsafe { _rdtsc() };
        let end_time = start_time.elapsed();

        cycles.push(end - start);
        times.push(end_time);
    }
    trash_data(&mut io, res.unwrap());

    if is_warmup {
        return;
    }

    cycles.sort_unstable();
    let cycles = &cycles[100..ITERS - 100];

    let sum_instrs: u64 = cycles.iter().sum();
    let cycles_avg = sum_instrs / ITERS as u64;
    let cycles_min = cycles.iter().min().unwrap();
    let cycles_max = cycles.iter().max().unwrap();

    times.sort_unstable();
    let times = &times[100..ITERS - 100];

    let sum_times: Duration = times.iter().sum();
    let times_avg = sum_times / ITERS as u32;
    let times_min = times.iter().min().unwrap();
    let times_max = times.iter().max().unwrap();

    println!(
        "tsc={{ avg: {CYAN}{cycles_avg}{CLEAR} min: {CYAN}{cycles_min}{CLEAR} max: {CYAN}{cycles_max}{CLEAR} }} \
         times={{ avg: {MAGENTA}{times_avg:?}{CLEAR} min: {MAGENTA}{times_min:?}{CLEAR} max: {MAGENTA}{times_max:?}{CLEAR} }}"
    );
}

fn newlines() {
    macro_rules! bench_newlines {
        ($bench:expr) => {
            for _ in 0..10 {
                bench(
                    false,
                    vec![0; 4096],
                    |io, buf| {
                        io.rand.read_exact(buf).unwrap();
                    },
                    |data| $bench(&data),
                    |io, res| {
                        io.null.write(&res.to_be_bytes()).unwrap();
                    },
                );
            }
        };
    }

    println!("############### NAIVE ###############");
    bench_newlines!(count_newlines_naive);
    println!("############### AUTO ###############");
    bench_newlines!(count_newlines_auto);
    println!("############### MANUAL ###############");
    bench_newlines!(count_newlines_manual);
}

const S: usize = 16;
macro_rules! fold_sum_f32 {
    ($chunk:expr) => {
        $chunk.iter().fold(0.0, |acc, b| acc + b)
    };
}

#[inline(never)]
fn sum_naive(nums: &[f32]) -> f32 {
    fold_sum_f32!(nums)
}

#[inline(never)]
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

#[inline(never)]
fn sum_manual(nums: &[f32]) -> f32 {
    let mut accumulator = f32x16::splat(0.0);
    let (chunks, rem) = nums.as_chunks::<S>();
    for chunk in chunks {
        let chunk = f32x16::from_array(*chunk);
        accumulator.add_assign(chunk);
    }

    accumulator.reduce_sum() + fold_sum_f32!(rem)
}

fn sum() {
    const ELEMS: usize = 1024;
    macro_rules! bench_sum {
        ($bench:expr) => {
            for _ in 0..10 {
                bench(
                    false,
                    vec![0.0; ELEMS],
                    |io, buf| {
                        let buf = unsafe {
                            std::slice::from_raw_parts_mut(buf.as_mut_ptr() as *mut u8, ELEMS * 4)
                        };
                        io.rand.read_exact(buf).unwrap();
                    },
                    |data| $bench(&data),
                    |io, res| {
                        io.null.write(&res.to_be_bytes()).unwrap();
                    },
                );
            }
        };
    }

    println!("############### NAIVE ###############");
    bench_sum!(sum_naive);
    println!("############### AUTO ###############");
    bench_sum!(sum_auto);
    println!("############### MANUAL ###############");
    bench_sum!(sum_manual);
}

fn main() {
    newlines();
    // sum();
}
