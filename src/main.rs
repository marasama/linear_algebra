// src/main.rs

mod algebra;

//use algebra::matrix::Matrix;
use algebra::vector::linear_combination;
use algebra::vector::Vector;

fn section(t: &str) {
    println!("\n===== {} =====", t);
}

// use crate::algebra::{vector::Vector, lineer_comb::linear_combination};
use std::panic::{self, AssertUnwindSafe};

fn assert_vec_debug_eq(v: &Vector<f32>, expected: &[f32]) {
    // Your Vector<K>: Debug prints the inner Vec with Rust's list syntax.
    let got = format!("{:?}", v); // e.g. "[10.0, -2.0, 0.5]"
    let exp = format!("{:?}", expected.to_vec());
    assert!(got == exp, "\nexpected: {}\n     got: {}\n", exp, got);
}

fn expect_panic<F: FnOnce() -> R + std::panic::UnwindSafe, R>(label: &str, f: F) {
    let res = panic::catch_unwind(AssertUnwindSafe(f));
    assert!(res.is_err(), "expected panic in test: {label}");
    println!("✓ {label} panicked as expected");
}

fn main() {
    println!("== linear_combination checks ==");

    // 1) Subject example: basis vectors
    let e1 = Vector::from([1.0f32, 0.0, 0.0]);
    let e2 = Vector::from([0.0f32, 1.0, 0.0]);
    let e3 = Vector::from([0.0f32, 0.0, 1.0]);

    let r1 = linear_combination(&[e1.clone(), e2.clone(), e3.clone()], &[10.0, -2.0, 0.5]);
    assert_vec_debug_eq(&r1, &[10.0, -2.0, 0.5]);
    println!("✓ subject example (basis vectors)");

    // 2) Subject example: arbitrary vectors
    let v1 = Vector::from([1.0f32, 2.0, 3.0]);
    let v2 = Vector::from([0.0f32, 10.0, -100.0]);
    let r2 = linear_combination(&[v1.clone(), v2.clone()], &[10.0, -2.0]);
    assert_vec_debug_eq(&r2, &[10.0, 0.0, 230.0]);
    println!("✓ subject example (arbitrary vectors)");

    // 3) Single term: should equal scaled input
    let r3 = linear_combination(&[v1.clone()], &[3.0]);
    assert_vec_debug_eq(&r3, &[3.0, 6.0, 9.0]);
    println!("✓ single-term equals scale");

    // 4) Zero coefficients -> zero vector
    let r4 = linear_combination(&[e1.clone(), e2.clone(), e3.clone()], &[0.0, 0.0, 0.0]);
    assert_vec_debug_eq(&r4, &[0.0, 0.0, 0.0]);
    println!("✓ zero coefficients produce zero vector");

    // 5) Larger k
    let a = Vector::from([1.0, 1.0, 1.0]);
    let b = Vector::from([2.0, 0.0, -2.0]);
    let c = Vector::from([0.5, -1.0, 4.0]);
    // 7*a + (-3)*b + 0.25*c
    let r5 = linear_combination(&[a, b, c], &[7.0, -3.0, 0.25]);
    // coord-wise:
    // [7*1  + -3*2  + 0.25*0.5,  7*1 + -3*0  + 0.25*-1,  7*1 + -3*-2 + 0.25*4]
    // [7    + -6    + 0.125,     7    +  0   + -0.25,    7    +  6    + 1   ]
    // [1.125, 6.75, 14.0]
    assert_vec_debug_eq(&r5, &[1.125, 6.75, 14.0]);
    println!("✓ larger k case");

    // 6) BAD: mismatched counts should panic (your function uses assert!)
    expect_panic("mismatched u/coefs lengths", || {
        let _ = linear_combination(&[e1.clone(), e2.clone()], &[1.0]);
    });

    // 7) BAD: dimension mismatch inside `u` should panic
    // NOTE: this relies on your implementation doing `res.add(&tmp).unwrap()`
    // so the size error from `add` isn't silently ignored.
    expect_panic("dimension mismatch among vectors", || {
        let short = Vector::from([1.0f32, 2.0]);
        let long = Vector::from([3.0f32, 4.0, 5.0]);
        let _ = linear_combination(&[short, long], &[1.0, 2.0]);
    });

    // 8) BAD: empty input should panic (your function asserts !u.is_empty())
    expect_panic("empty input slice", || {
        let empty: [Vector<f32>; 0] = [];
        let _ = linear_combination(&empty, &[]);
    });

    println!("All checks passed ✅");
}
//fn main() {
//    // ---------- VECTORS ----------
//    section("Vectors: construction & printing");
//    let mut u = Vector::from([2.0f32, 3.0]); // From<[f32; N]>
//    let v = Vector::from([5.0f32, 7.0]);
//
//    println!("u (Debug):   {:?}", u);
//    println!("u (Display):\n{}", u);
//    println!("v (Display):\n{}", v);
//
//    section("Vectors: add / sub / scl");
//    u.add(&v).expect("sizes must match");
//    println!("u += v:\n{:.4}", u);
//    assert_eq!(format!("{}", u), "[  7.0 ]\n[ 10.0 ]\n");
//
//    u.sub(&v).expect("sizes must match");
//    println!("u -= v:\n{}", u);
//    assert_eq!(format!("{}", u), "[ 2.0 ]\n[ 3.0 ]\n");
//
//    u.scl(2.0);
//    println!("u *= 2.0:\n{}", u);
//    assert_eq!(format!("{}", u), "[ 4.0 ]\n[ 6.0 ]\n");
//
//    section("Vectors: size mismatch error");
//    let mut a = Vector::from([1.0f32, 2.0, 3.0]);
//    let b = Vector::from([4.0f32, 5.0]);
//    let res = a.add(&b);
//    println!("a.add(b) -> {:?}", res);
//    assert!(res.is_err());
//
//    // ---------- MATRICES ----------
//    section("Matrices: construction & printing");
//
//    let m_new = Matrix::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 3, 2).expect("bad dims");
//
//    println!("m_new (Debug): {:?}", m_new);
//    println!("m_new (Display):\n{}", m_new);
//
//    let mut m = Matrix::from([[2.0f32, 1.0], [3.0, 4.0]]);
//    let n = Matrix::from([[20.0f32, 10.0], [30.0, 40.0]]);
//    println!("m:\n{}", m);
//    println!("n:\n{}", n);
//
//    section("Matrices: add / sub / scl");
//    m.add(&n).expect("shapes must match");
//    println!("m += n:\n{}", m);
//    assert_eq!(format!("{}", m), "[ 22.0 11.0 ]\n[ 33.0 44.0 ]\n");
//
//    m.sub(&n).expect("shapes must match");
//    println!("(m += n) -= n:\n{}", m);
//    assert_eq!(format!("{}", m), "[ 2.0 1.0 ]\n[ 3.0 4.0 ]\n");
//
//    m.scl(3.0);
//    println!("m *= 3:\n{}", m);
//    assert_eq!(format!("{}", m), "[ 6.0  3.0 ]\n[ 9.0 12.0 ]\n");
//
//    section("Matrices: alignment check");
//    let pretty = Matrix::from([[2.0f32, 10.0], [300.0, 4.0]]);
//    println!("pretty:\n{}", pretty);
//
//    section("Matrices: size mismatch error");
//    let mut m22 = Matrix::from([[1.0f32, 2.0], [3.0, 4.0]]);
//    let m23 = Matrix::from([[9.0f32, 8.0, 7.0], [6.0, 5.0, 4.0]]);
//    let add_res = m22.add(&m23);
//    println!("2x2 += 2x3 -> {:?}", add_res);
//    assert!(add_res.is_err());
//
//    m23.print_size();
//
//    println!("\nAll sanity checks passed ✅");
//}
