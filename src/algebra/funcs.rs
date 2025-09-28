use super::matrix::Matrix;
use super::vector::Vector;
use num_traits::{Float, NumCast};
use std::ops::{Add, Mul, Sub};

pub fn linear_combination<K: Float>(u: &[Vector<K>], coefs: &[K]) -> Vector<K> {
    assert_eq!(
        u.len(),
        coefs.len(),
        "Size mismatch at Vector::linear_combination()!"
    );
    let n = u[0].size();
    for vec in u.iter() {
        assert_eq!(
            vec.size(),
            n,
            "Not all the vectors are in same size linear_combination()!"
        );
        assert!(
            !vec.data.is_empty(),
            "Empty function input in linear_combination()!"
        );
    }
    let mut sum = vec![K::zero(); n];
    for (vect, &coef) in u.iter().zip(coefs.iter()) {
        for (a, &b) in sum.iter_mut().zip(vect.data.iter()) {
            *a = b.mul_add(coef, *a);
        }
    }
    Vector { data: sum }
}

pub fn lerp<V>(u: V, v: V, t: f32) -> V
where
    V: LerpImp,
{
    V::lerp_imp(u, v, t)
}

pub trait LerpImp: Sized {
    fn lerp_imp(u: Self, v: Self, t: f32) -> Self;
}

impl LerpImp for f32 {
    fn lerp_imp(u: Self, v: Self, t: f32) -> Self {
        (v - u).mul_add(t, u)
    }
}

impl LerpImp for f64 {
    fn lerp_imp(u: Self, v: Self, t: f32) -> Self {
        (v - u).mul_add(t as f64, u)
    }
}

impl<K: Float> LerpImp for Vector<K> {
    fn lerp_imp(u: Self, v: Self, t: f32) -> Self {
        assert_eq!(u.size(), v.size(), "Size mismatch at lerp()!");
        let mut new_data = vec![K::zero(); u.size()];
        let tt: K = NumCast::from(t).expect("Can't cast f32 to vector scalar!");
        for i in 0..u.size() {
            new_data[i] = (v.data[i] - u.data[i]).mul_add(tt, u.data[i]);
        }
        Vector { data: new_data }
    }
}

impl<K: Float> LerpImp for Matrix<K> {
    fn lerp_imp(u: Self, v: Self, t: f32) -> Self {
        assert_eq!(u.size(), v.size(), "Shape mismatch at lerp()");
        let mut new_data = vec![K::zero(); u.rows * u.cols];
        let tt: K = NumCast::from(t).expect("Can't cast f32 to vector scalar!");
        for i in 0..u.rows * u.cols {
            new_data[i] = (v.data[i] - u.data[i]).mul_add(tt, u.data[i]);
        }
        Matrix {
            data: new_data,
            rows: u.rows,
            cols: u.cols,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    // Bring types into scope the same way funcs.rs does
    use crate::algebra::matrix::Matrix;
    use crate::algebra::vector::Vector;

    // Helpers: pretty string with fixed precision, so spacing is stable enough.
    fn svec(v: &Vector<f32>) -> String {
        format!("{:.3}", v) // 3 decimals to avoid rounding surprises
    }
    fn smat(m: &Matrix<f32>) -> String {
        format!("{:.3}", m)
    }

    // ---------------- linear_combination ----------------

    #[test]
    fn lincomb_basis_vectors() {
        let e1 = Vector::from([1.0f32, 0.0, 0.0]);
        let e2 = Vector::from([0.0f32, 1.0, 0.0]);
        let e3 = Vector::from([0.0f32, 0.0, 1.0]);

        let r = linear_combination(&[e1, e2, e3], &[10.0, -2.0, 0.5]);
        println!("lincomb(basis) =\n{}", svec(&r));

        // Compare to an expected Vector via string formatting
        let exp = Vector::from([10.0f32, -2.0, 0.5]);
        assert_eq!(svec(&r), svec(&exp));
    }

    #[test]
    fn lincomb_arbitrary() {
        let v1 = Vector::from([1.0f32, 2.0, 3.0]);
        let v2 = Vector::from([0.0f32, 10.0, -100.0]);

        let r = linear_combination(&[v1, v2], &[10.0, -2.0]);
        println!("lincomb(arb) =\n{}", svec(&r));

        let exp = Vector::from([10.0f32, 0.0, 230.0]);
        assert_eq!(svec(&r), svec(&exp));
    }

    #[test]
    fn lincomb_zeros() {
        let v1 = Vector::from([1.0f32, 2.0, 3.0]);
        let v2 = Vector::from([4.0f32, 5.0, 6.0]);

        let r = linear_combination(&[v1, v2], &[0.0, 0.0]);
        println!("lincomb(zeros) =\n{}", svec(&r));

        let exp = Vector::from([0.0f32, 0.0, 0.0]);
        assert_eq!(svec(&r), svec(&exp));
    }

    #[test]
    fn lincomb_empty_ok() {
        let u: [Vector<f32>; 0] = [];
        let r = linear_combination(&u, &[]);
        println!("lincomb(empty) =\n{}", svec(&r));
        // should be an empty vector => prints nothing but still valid
        assert_eq!(r.size(), 0);
    }

    #[test]
    #[should_panic(expected = "Size mismatch at Vector::linear_combination()!")]
    fn lincomb_mismatched_lengths_panics() {
        let v1 = Vector::from([1.0f32, 2.0, 3.0]);
        let v2 = Vector::from([4.0f32, 5.0, 6.0]);
        let _ = linear_combination(&[v1, v2], &[1.0]); // coefs too short
    }

    #[test]
    #[should_panic(expected = "Not all the vectors are in same size Vector::linear_combination()!")]
    fn lincomb_dimension_mismatch_panics() {
        let v1 = Vector::from([1.0f32, 2.0, 3.0]);
        let v2 = Vector::from([4.0f32, 5.0]); // different size
        let _ = linear_combination(&[v1, v2], &[1.0, 2.0]);
    }

    // ---------------- lerp: scalars ----------------

    #[test]
    fn lerp_scalar_f32() {
        assert_eq!(lerp(0.0f32, 1.0, 0.0), 0.0);
        assert_eq!(lerp(0.0f32, 1.0, 1.0), 1.0);

        let mid = lerp(0.0f32, 1.0, 0.5);
        println!("lerp f32 mid = {}", mid);
        assert!((mid - 0.5).abs() < 1e-6);

        let v = lerp(21.0f32, 42.0, 0.3);
        println!("lerp f32 21->42 @0.3 = {}", v);
        assert!((v - 27.3).abs() < 1e-5);
    }

    #[test]
    fn lerp_scalar_f64() {
        let a = lerp(0.0f64, 1.0, 0.5);
        println!("lerp f64 mid = {}", a);
        assert!((a - 0.5).abs() < 1e-12);

        let b = lerp(21.0f64, 42.0, 0.3);
        println!("lerp f64 21->42 @0.3 = {}", b);
        assert!((b - 27.3f64).abs() < 1e-12);
    }

    // ---------------- lerp: Vector ----------------

    #[test]
    fn lerp_vector_endpoints_and_mid() {
        let u = Vector::from([2.0f32, 1.0]);
        let v = Vector::from([4.0f32, 2.0]);

        let r0 = lerp(u.clone(), v.clone(), 0.0);
        println!("lerp vec t=0:\n{}", svec(&r0));
        assert_eq!(svec(&r0), svec(&u));

        let r1 = lerp(u.clone(), v.clone(), 1.0);
        println!("lerp vec t=1:\n{}", svec(&r1));
        assert_eq!(svec(&r1), svec(&v));

        let r05 = lerp(u.clone(), v.clone(), 0.5);
        println!("lerp vec t=0.5:\n{}", svec(&r05));
        let exp_mid = Vector::from([3.0f32, 1.5]);
        assert_eq!(svec(&r05), svec(&exp_mid));
    }

    #[test]
    fn lerp_vector_arbitrary_t() {
        let u = Vector::from([2.0f32, 1.0]);
        let v = Vector::from([4.0f32, 2.0]);
        let r = lerp(u, v, 0.3);
        println!("lerp vec t=0.3:\n{}", svec(&r));
        let exp = Vector::from([2.6f32, 1.3]);
        assert_eq!(svec(&r), svec(&exp));
    }

    // ---------------- lerp: Matrix ----------------
    // Assumes you have Matrix::from([[..]]) and a Display that prints rows in order.

    #[test]
    fn lerp_matrix_example_from_subject() {
        let u = Matrix::from([[2.0f32, 1.0], [3.0, 4.0]]);
        let v = Matrix::from([[20.0f32, 10.0], [30.0, 40.0]]);
        let r = lerp(u, v, 0.5);
        println!("lerp matrix t=0.5:\n{}", smat(&r));

        // Expected numeric values (format-based compare to avoid private fields)
        let exp = Matrix::from([[11.0f32, 5.5], [16.5, 22.0]]);
        assert_eq!(smat(&r), smat(&exp));
    }

    #[test]
    #[should_panic(expected = "Shape mismatch at lerp(Matrix)!")]
    fn lerp_matrix_shape_mismatch_panics() {
        // Adjust these constructors if your Matrix::from signature differs.
        let u = Matrix::from([[1.0f32, 2.0]]);
        let v = Matrix::from([[1.0f32, 2.0], [3.0, 4.0]]);
        let _ = lerp(u, v, 0.25);
    }
}
