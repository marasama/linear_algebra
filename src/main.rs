// src/main.rs

// Load your modules declared in src/algebra/mod.rs via src/lib.rs
mod algebra;

use algebra::matrix::Matrix;
use algebra::vector::Vector;

fn section(t: &str) {
    println!("\n===== {} =====", t);
}

fn main() {
    // ---------- VECTORS ----------
    section("Vectors: construction & printing");
    let mut u = Vector::from([2.0f32, 3.0]); // From<[K; N]>
    let v = Vector::from([5.0f32, 7.0]);

    println!("u (Debug):   {:?}", u); // {:?} -> Debug
    println!("u (Display):\n{}", u); // {}  -> Display (your column style)
    println!("v (Display):\n{}", v);

    section("Vectors: add / sub / scl");
    u.add(&v).expect("sizes must match");
    println!("u += v:\n{}", u);
    assert_eq!(format!("{}", u), "[7]\n[10]\n");

    u.sub(&v).expect("sizes must match");
    println!("u -= v:\n{}", u);
    assert_eq!(format!("{}", u), "[2]\n[3]\n");

    u.scl(2.0);
    println!("u *= 2.0:\n{}", u);
    assert_eq!(format!("{}", u), "[4]\n[6]\n");

    section("Vectors: size mismatch error");
    let mut a = Vector::from([1.0f32, 2.0, 3.0]);
    let b = Vector::from([4.0f32, 5.0]);
    let res = a.add(&b);
    println!("a.add(b) -> {:?}", res);
    assert!(res.is_err());

    // ---------- MATRICES ----------
    section("Matrices: construction & printing");
    // Your Matrix::new signature is new(Vec<K>, cols, rows)

    let m_new = Matrix::new(vec![1, 2, 3, 4, 5, 6], 3, 2).expect("bad dims");

    println!("m_new (Debug): {:?}", m_new);
    println!("m_new (Display):\n{}", m_new);

    // From<[[K; C]; R]> (const generics)
    let mut m = Matrix::from([[2, 1], [3, 4]]);
    let n = Matrix::from([[20, 10], [30, 40]]);
    println!("m:\n{}", m);
    println!("n:\n{}", n);

    section("Matrices: add / sub / scl");
    m.add(&n).expect("shapes must match");
    println!("m += n:\n{}", m);
    assert_eq!(format!("{}", m), "[22 11]\n[33 44]\n");

    m.sub(&n).expect("shapes must match");
    println!("(m += n) -= n:\n{}", m);
    assert_eq!(format!("{}", m), "[2 1]\n[3 4]\n");

    m.scl(3);
    println!("m *= 3:\n{}", m);
    assert_eq!(format!("{}", m), "[6  3]\n[9 12]\n");

    section("Matrices: alignment check");
    let pretty = Matrix::from([[2, 10], [300, 4]]);
    println!("pretty:\n{}", pretty); // visually verify columns align

    section("Matrices: size mismatch error");
    let mut m22 = Matrix::from([[1, 2], [3, 4]]);
    let m23 = Matrix::from([[9, 8, 7], [6, 5, 4]]);
    let add_res = m22.add(&m23);
    println!("2x2 += 2x3 -> {:?}", add_res);
    assert!(add_res.is_err());

    m23.print_size();

    println!("\nAll sanity checks passed ✅");
}
