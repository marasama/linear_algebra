// src/main.rs

mod algebra;

use algebra::matrix::Matrix;
use algebra::vector::Vector;

fn section(t: &str) {
    println!("\n===== {} =====", t);
}

fn main() {
    // ---------- VECTORS ----------
    section("Vectors: construction & printing");
    let mut u = Vector::from([2.0f32, 3.0]); // From<[f32; N]>
    let v = Vector::from([5.0f32, 7.0]);

    println!("u (Debug):   {:?}", u);
    println!("u (Display):\n{}", u);
    println!("v (Display):\n{}", v);

    section("Vectors: add / sub / scl");
    u.add(&v).expect("sizes must match");
    println!("u += v:\n{:.4}", u);
    assert_eq!(format!("{}", u), "[  7.0 ]\n[ 10.0 ]\n");

    u.sub(&v).expect("sizes must match");
    println!("u -= v:\n{}", u);
    assert_eq!(format!("{}", u), "[ 2.0 ]\n[ 3.0 ]\n");

    u.scl(2.0);
    println!("u *= 2.0:\n{}", u);
    assert_eq!(format!("{}", u), "[ 4.0 ]\n[ 6.0 ]\n");

    section("Vectors: size mismatch error");
    let mut a = Vector::from([1.0f32, 2.0, 3.0]);
    let b = Vector::from([4.0f32, 5.0]);
    let res = a.add(&b);
    println!("a.add(b) -> {:?}", res);
    assert!(res.is_err());

    // ---------- MATRICES ----------
    section("Matrices: construction & printing");

    let m_new = Matrix::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 3, 2).expect("bad dims");

    println!("m_new (Debug): {:?}", m_new);
    println!("m_new (Display):\n{}", m_new);

    let mut m = Matrix::from([[2.0f32, 1.0], [3.0, 4.0]]);
    let n = Matrix::from([[20.0f32, 10.0], [30.0, 40.0]]);
    println!("m:\n{}", m);
    println!("n:\n{}", n);

    section("Matrices: add / sub / scl");
    m.add(&n).expect("shapes must match");
    println!("m += n:\n{}", m);
    assert_eq!(format!("{}", m), "[ 22.0 11.0 ]\n[ 33.0 44.0 ]\n");

    m.sub(&n).expect("shapes must match");
    println!("(m += n) -= n:\n{}", m);
    assert_eq!(format!("{}", m), "[ 2.0 1.0 ]\n[ 3.0 4.0 ]\n");

    m.scl(3.0);
    println!("m *= 3:\n{}", m);
    assert_eq!(format!("{}", m), "[ 6.0  3.0 ]\n[ 9.0 12.0 ]\n");

    section("Matrices: alignment check");
    let pretty = Matrix::from([[2.0f32, 10.0], [300.0, 4.0]]);
    println!("pretty:\n{}", pretty);

    section("Matrices: size mismatch error");
    let mut m22 = Matrix::from([[1.0f32, 2.0], [3.0, 4.0]]);
    let m23 = Matrix::from([[9.0f32, 8.0, 7.0], [6.0, 5.0, 4.0]]);
    let add_res = m22.add(&m23);
    println!("2x2 += 2x3 -> {:?}", add_res);
    assert!(add_res.is_err());

    m23.print_size();

    println!("\nAll sanity checks passed ✅");
}
