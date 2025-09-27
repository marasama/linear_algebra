mod algebra;

use algebra::matrix::Matrix;
// bring your Matrix into scope (adjust the path if Matrix is in a module)
// --- paste your Matrix impls here ---
// Matrix::new, size/print_size, add, sub, scl,
// and From<[[K; C]; R]> + your Debug impl, etc.

fn main() {
    // 1) Build matrices (requires your From<[[K; C]; R]> impl)
    let mut a = Matrix::<f64>::from([[1.0, 2.0], [3.0, 4.0]]);
    let b = Matrix::<f64>::from([[10.0, 20.0], [30.0, 40.0]]);

    // 2) Show sizes
    println!("== sizes ==");
    println!("a size: {:?}  b size: {:?}", a.size(), b.size());
    println!();

    // 3) Print with your Debug (default precision = 0)
    println!("== Debug (default precision) ==");
    println!("{:?}", a);
    println!("{:?}", b);

    // 4) Print with precision (your Debug uses the formatter precision)
    println!("== Debug with precision = 2 ==");
    println!("{:.2?}", a);
    println!("{:.2?}", b);

    // 5) add
    println!("== a.add(&b) ==");
    a.add(&b);
    println!("{:?}", a);

    // 6) sub (bring back to original a)
    println!("== a.sub(&b) ==");
    a.sub(&b);
    println!("{:?}", a);

    // 7) scl (scalar multiply)
    println!("== a.scl(0.5) ==");
    a.scl(0.5);
    println!("{:?}", a);

    // 8) Optional: demonstrate a size-mismatch panic using catch_unwind
    // (comment out if you don’t want to depend on unwind behavior)
    use std::panic;
    let c = Matrix::<f64>::from([[5.0, 6.0, 7.0], [8.0, 9.0, 10.0]]);
    println!("== trying a.add(&c) (should panic) ==");
    let result = panic::catch_unwind(|| {
        let mut tmp = a.clone();
        tmp.add(&c); // mismatched sizes -> panic in your add()
    });
    match result {
        Ok(_) => println!("no panic (unexpected)"),
        Err(_) => println!("panicked as expected on size mismatch in add()"),
    }
}
