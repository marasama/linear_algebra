mod algebra;

use algebra::matrix::Matrix;
fn main() {
    let mut a = Matrix::from([[2., 0., 0.], [0., 2., 0.], [0., 0., 2.]]);

    let mut b = a.inverse().unwrap();
    println!("{}", b);
    // [1.0, 0.625, 0.0, 0.0, -12.1666667]
    // [0.0, 0.0, 1.0, 0.0, -3.6666667]
    // [0.0, 0.0, 0.0, 1.0, 29.5 ]
}
