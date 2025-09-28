use super::matrix::Matrix;
use super::vector::Vector;
use num_traits::Float;
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
            "Not all the vectors are in same size Vector::linear_combination()!"
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

pub fn lerp<V>(u: V, v: V, t: f32) -> V {
    V::lerp_imp(u, v, t)
}

pub trait LerpImp<K> {
    fn lerp_imp(u: Self, v: Self, t: K) -> Self;
}

impl<K: Float> LerpImp<K> for K {
    fn lerp_imp(u: Self, v: Self, t: K) -> Self {
        (v - u).mul_add(t, u)
    }
}

impl<K: Float> LerpImp<K> for Vector<K> {
    fn lerp_imp(u: Self, v: Self, t: K) -> Self {
        assert_eq!(u.size(), v.size(), "Size mismatch at lerp()");
        let mut new_data = vec![K::zero(); u.size()];
        for i in 0..u.size() {
            new_data[i] = (v.data[i] - u.data[i]).mul_add(t, u.data[i]);
        }
        Vector { data: new_data }
    }
}

impl<K: Float> LerpImp<K> for Matrix<K> {
    fn lerp_imp(u: Self, v: Self, t: K) -> Self {
        assert_eq!(u.size(), v.size(), "Shape mismatch at lerp()");
        let mut new_data = vec![K::zero(); u.rows * u.cols];
        for i in 0..u.rows * u.cols {
            new_data[i] = (v.data[i] - u.data[i]).mul_add(t, u.data[i]);
        }
        Matrix {
            data: new_data,
            rows: u.rows,
            cols: u.cols,
        }
    }
}
