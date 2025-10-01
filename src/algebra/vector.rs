use super::funcs::linear_combination;
use num_traits::Float;
use std::clone::Clone;
use std::cmp::PartialEq;
use std::fmt;
use std::marker::Copy;
use std::ops::{Add, AddAssign, Mul, MulAssign, Neg, Sub, SubAssign};
#[derive(Clone)]
pub struct Vector<K: Float> {
    pub data: Vec<K>,
}

impl<K: Float> Vector<K> {
    pub fn new(data: Vec<K>) -> Vector<K> {
        Vector { data }
    }

    pub fn size(&self) -> usize {
        if self.data.is_empty() {
            return 0;
        }
        self.data.len()
    }
}

impl<K: Float, const N: usize> From<[K; N]> for Vector<K> {
    fn from(value: [K; N]) -> Self {
        Vector {
            data: Vec::from(value),
        }
    }
}

impl<K: Float> fmt::Display for Vector<K>
where
    K: fmt::Display,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let precs = f.precision().unwrap_or(1);

        let strs: Vec<String> = self
            .data
            .iter()
            .map(|x| format!("{:.prec$}", x, prec = precs))
            .collect();

        let max_width = strs.iter().map(|s| s.len()).max().unwrap_or(0);

        for s in strs {
            writeln!(f, "[ {:>width$} ]", s, width = max_width)?;
        }
        Ok(())
    }
}

impl<K: Float> fmt::Debug for Vector<K>
where
    K: fmt::Display,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let precs = f.precision().unwrap_or(1);

        let strs: Vec<String> = self
            .data
            .iter()
            .map(|x| format!("{:.prec$}", x, prec = precs))
            .collect();

        let max_width = strs.iter().map(|s| s.len()).max().unwrap_or(0);

        for s in strs {
            writeln!(f, "[ {:>width$} ]", s, width = max_width)?;
        }
        Ok(())
    }
}

impl<K: Float> Vector<K>
where
    K: Copy + AddAssign + SubAssign + MulAssign,
{
    pub fn add(&mut self, v: &Vector<K>) {
        assert_eq!(self.size(), v.size(), "Dimension mismatch at vector.add()");
        for (a, b) in self.data.iter_mut().zip(&v.data) {
            *a += *b;
        }
    }
    pub fn sub(&mut self, v: &Vector<K>) {
        assert_eq!(self.size(), v.size(), "Dimension mismatch at vector.sub()");
        for (a, b) in self.data.iter_mut().zip(&v.data) {
            *a -= *b;
        }
    }
    pub fn scl(&mut self, a: K) {
        for b in self.data.iter_mut() {
            *b *= a;
        }
    }
}

// --------(+) (-) (*) (+=) (-=) (*=)-----------------

impl<K: Float> Add<&Vector<K>> for &Vector<K>
where
    K: Copy + Add<Output = K>,
{
    type Output = Vector<K>;

    fn add(self, rhs: &Vector<K>) -> Self::Output {
        assert_eq!(
            self.size(),
            rhs.size(),
            "Dimension mismatch at (+) operator"
        );
        Vector {
            data: self
                .data
                .iter()
                .zip(&rhs.data)
                .map(|(&a, &b)| a + b)
                .collect(),
        }
    }
}

impl<K: Float> Sub<&Vector<K>> for &Vector<K>
where
    K: Copy + Sub<Output = K>,
{
    type Output = Vector<K>;

    fn sub(self, rhs: &Vector<K>) -> Self::Output {
        assert_eq!(
            self.size(),
            rhs.size(),
            "Dimension mismatch at (-) operator"
        );
        Vector {
            data: self
                .data
                .iter()
                .zip(&rhs.data)
                .map(|(&a, &b)| a - b)
                .collect(),
        }
    }
}

impl<K: Float> Mul<&Vector<K>> for &Vector<K>
where
    K: Copy + Mul<Output = K>,
{
    type Output = Vector<K>;

    fn mul(self, rhs: &Vector<K>) -> Self::Output {
        assert_eq!(
            self.size(),
            rhs.size(),
            "Dimension mismatch at (*) operator"
        );
        Vector {
            data: self
                .data
                .iter()
                .zip(&rhs.data)
                .map(|(&a, &b)| a * b)
                .collect(),
        }
    }
}

impl<K: Float> Neg for &Vector<K>
where
    K: Copy + Neg<Output = K>,
{
    type Output = Vector<K>;

    fn neg(self) -> Self::Output {
        Vector {
            data: self.data.iter().map(|&a| -a).collect(),
        }
    }
}

impl<K: Float> AddAssign<&Vector<K>> for Vector<K>
where
    K: AddAssign + Copy,
{
    fn add_assign(&mut self, rhs: &Vector<K>) {
        assert_eq!(
            self.size(),
            rhs.size(),
            "Dimension mismatch at (+=) operator"
        );
        for (a, b) in self.data.iter_mut().zip(&rhs.data) {
            *a += *b;
        }
    }
}

impl<K: Float> SubAssign<&Vector<K>> for Vector<K>
where
    K: SubAssign + Copy + AddAssign,
{
    fn sub_assign(&mut self, rhs: &Vector<K>) {
        assert_eq!(
            self.size(),
            rhs.size(),
            "Dimension mismatch at (-=) operator"
        );
        for (a, b) in self.data.iter_mut().zip(&rhs.data) {
            *a -= *b;
        }
    }
}

impl<K: Float> MulAssign<&Vector<K>> for Vector<K>
where
    K: MulAssign + Copy + AddAssign + SubAssign,
{
    fn mul_assign(&mut self, rhs: &Vector<K>) {
        assert_eq!(
            self.size(),
            rhs.size(),
            "Dimension mismatch at (*=) operator"
        );
        for (a, b) in self.data.iter_mut().zip(&rhs.data) {
            *a *= *b;
        }
    }
}

impl<K: Float> PartialEq<Vector<K>> for Vector<K>
where
    K: Eq,
{
    fn eq(&self, other: &Vector<K>) -> bool {
        if self.size() != other.size() {
            return false;
        }
        for (a, b) in self.data.iter().zip(&other.data) {
            if a != b {
                return false;
            }
        }
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    type F = f32; // tweak to f64 if you prefer

    // Small helper so we can quickly build vectors
    fn v(data: &[F]) -> Vector<F> {
        Vector::new(data.to_vec())
    }

    // ---- constructors & basic getters ----

    #[test]
    fn new_and_size() {
        let a = v(&[1.0, 2.0, 3.0]);
        println!("a =\n{}", a);
        assert_eq!(a.size(), 3);

        let empty = Vector::<F>::new(vec![]);
        println!("empty =\n{}", empty);
        assert_eq!(empty.size(), 0);
    }

    #[test]
    fn from_array() {
        let a: Vector<F> = Vector::from([1.0, 2.5, -3.0]);
        println!("from array =\n{}", a);
        assert_eq!(a.size(), 3);
        assert_eq!(a.data, vec![1.0, 2.5, -3.0]);
    }

    // ---- Display / Debug formatting ----

    #[test]
    fn display_default_precision_aligns() {
        let a = v(&[1.0, 12.3]);
        // default precision is 1
        let s = format!("{}", a);
        println!("Display default:\n{}", s);
        // first column should be padded to match max width ("12.3" is wider than "1.0")
        assert_eq!(s, "[  1.0 ]\n[ 12.3 ]\n");
    }

    #[test]
    fn display_with_precision_runtime() {
        let a = v(&[1.234, 12.3456]);
        let s = format!("{:.2}", a); // runtime precision = 2
        println!("Display {:.2}:\n{}", 2, s);
        assert_eq!(s, "[  1.23 ]\n[ 12.35 ]\n"); // rounded, right-aligned
    }

    #[test]
    fn debug_matches_display_style() {
        let a = v(&[3.0, -4.5]);
        let d1 = format!("{}", a);
        let d2 = format!("{:?}", a);
        println!("Display:\n{}", d1);
        println!("Debug:\n{}", d2);
        assert_eq!(d1, d2);
    }

    // ---- in-place ops: add/sub/scl ----

    #[test]
    fn add_in_place_ok() {
        let mut a = v(&[1.0, 2.0, 3.0]);
        let b = v(&[0.5, -2.0, 1.0]);
        println!("before add:\na =\n{}\nb =\n{}", a, b);
        Vector::add(&mut a, &b);
        println!("after add:\na =\n{}", a);
        assert_eq!(a.data, vec![1.5, 0.0, 4.0]);
    }

    #[test]
    #[should_panic(expected = "Dimension mismatch at vector.add()")]
    fn add_in_place_mismatch_panics() {
        let mut a = v(&[1.0, 2.0]);
        let b = v(&[1.0, 2.0, 3.0]);
        Vector::add(&mut a, &b);
    }

    #[test]
    fn sub_in_place_ok() {
        let mut a = v(&[1.0, 2.0, 3.0]);
        let b = v(&[0.5, -2.0, 1.0]);
        println!("before sub:\na =\n{}\nb =\n{}", a, b);
        Vector::sub(&mut a, &b);
        println!("after sub:\na =\n{}", a);
        assert_eq!(a.data, vec![0.5, 4.0, 2.0]);
    }

    #[test]
    fn scl_in_place_ok() {
        let mut a = v(&[1.0, -2.0, 3.5]);
        println!("before scl:\na =\n{}", a);
        a.scl(2.0);
        println!("after scl(2):\na =\n{}", a);
        assert_eq!(a.data, vec![2.0, -4.0, 7.0]);
    }

    // ---- operator overloads (+, -, *, neg) and assigns ----

    #[test]
    fn add_operator_ok() {
        let a = v(&[1.0, 2.0, 3.0]);
        let b = v(&[4.0, 5.0, 6.0]);
        let c = &a + &b;
        println!("a+b =\n{}", c);
        assert_eq!(c.data, vec![5.0, 7.0, 9.0]);
    }

    #[test]
    #[should_panic(expected = "Dimension mismatch at (+) operator")]
    fn add_operator_mismatch_panics() {
        let a = v(&[1.0, 2.0]);
        let b = v(&[3.0]);
        let _ = &a + &b;
    }

    #[test]
    fn sub_operator_ok() {
        let a = v(&[1.0, 2.0, 3.0]);
        let b = v(&[0.5, 1.0, 2.5]);
        let c = &a - &b;
        println!("a-b =\n{}", c);
        assert_eq!(c.data, vec![0.5, 1.0, 0.5]);
    }

    #[test]
    fn mul_operator_element_wise_ok() {
        let a = v(&[1.0, 2.0, 3.0]);
        let b = v(&[4.0, 5.0, 6.0]);
        let c = &a * &b; // element-wise (not dot)
        println!("a*b (elem-wise) =\n{}", c);
        assert_eq!(c.data, vec![4.0, 10.0, 18.0]);
    }

    #[test]
    fn neg_operator_ok() {
        let a = v(&[1.5, -2.0]);
        let b = -&a;
        println!("-a =\n{}", b);
        assert_eq!(b.data, vec![-1.5, 2.0]);
    }

    #[test]
    fn add_assign_ok() {
        let mut a = v(&[1.0, 2.0]);
        let b = v(&[3.0, 4.0]);
        println!("before a += b:\na =\n{}\nb =\n{}", a, b);
        a += &b;
        println!("after a += b:\na =\n{}", a);
        assert_eq!(a.data, vec![4.0, 6.0]);
    }

    #[test]
    #[should_panic(expected = "Dimension mismatch at (+=) operator")]
    fn add_assign_mismatch_panics() {
        let mut a = v(&[1.0]);
        let b = v(&[1.0, 2.0]);
        a += &b;
    }

    #[test]
    fn sub_assign_ok() {
        let mut a = v(&[5.0, 7.0]);
        let b = v(&[3.0, 4.0]);
        println!("before a -= b:\na =\n{}\nb =\n{}", a, b);
        a -= &b;
        println!("after a -= b:\na =\n{}", a);
        assert_eq!(a.data, vec![2.0, 3.0]);
    }

    #[test]
    fn mul_assign_ok() {
        let mut a = v(&[2.0, 3.0, 4.0]);
        let b = v(&[10.0, 0.5, -1.0]);
        println!("before a *= b:\na =\n{}\nb =\n{}", a, b);
        a *= &b;
        println!("after a *= b:\na =\n{}", a);
        assert_eq!(a.data, vec![20.0, 1.5, -4.0]);
    }

    // ---- linear_combination ----

    #[test]
    fn linear_combination_basis_vectors() {
        let e1 = v(&[1.0, 0.0, 0.0]);
        let e2 = v(&[0.0, 1.0, 0.0]);
        let e3 = v(&[0.0, 0.0, 1.0]);
        let coefs = [10.0, -2.0, 0.5];
        let r = linear_combination(&[e1, e2, e3], &coefs);
        println!("lincomb(basis) =\n{}", r);
        assert_eq!(r.data, vec![10.0, -2.0, 0.5]);
    }

    #[test]
    fn linear_combination_arbitrary() {
        let v1 = v(&[1.0, 2.0, 3.0]);
        let v2 = v(&[0.0, 10.0, -100.0]);
        let r = linear_combination(&[v1, v2], &[10.0, -2.0]);
        println!("lincomb(arb) =\n{}", r);
        assert_eq!(r.data, vec![10.0, 0.0, 230.0]);
    }

    #[test]
    fn linear_combination_zeros() {
        let v1 = v(&[1.0, 2.0, 3.0]);
        let v2 = v(&[4.0, 5.0, 6.0]);
        let r = linear_combination(&[v1, v2], &[0.0, 0.0]);
        println!("lincomb(zeros) =\n{}", r);
        assert_eq!(r.data, vec![0.0, 0.0, 0.0]);
    }

    #[test]
    #[should_panic(expected = "Size mismatch at Vector::linear_combination()!")]
    fn linear_combination_bad_lengths_panics() {
        let v1 = v(&[1.0, 2.0, 3.0]);
        let v2 = v(&[4.0, 5.0, 6.0]);
        let _ = linear_combination(&[v1, v2], &[1.0]); // coefs too short
    }

    #[test]
    #[should_panic(expected = "Not all the vectors are in same size Vector::linear_combination()!")]
    fn linear_combination_dim_mismatch_panics() {
        let v1 = v(&[1.0, 2.0, 3.0]);
        let v2 = v(&[4.0, 5.0]); // different size
        let _ = linear_combination(&[v1, v2], &[1.0, 2.0]);
    }

    // ---- dot product (v is moved) ----

    #[inline]
    fn approx_eq(a: F, b: F) -> bool {
        (a - b).abs() <= 1e-6
    }

    #[test]
    fn dot_zero_result() {
        let a = v(&[0.0, 0.0]);
        let b = v(&[1.0, 1.0]);
        let d = a.dot(b); // b moved
        println!("dot([0,0],[1,1]) = {}", d);
        assert!(approx_eq(d, 0.0));
    }

    #[test]
    fn dot_basic_examples() {
        // [1,1] · [1,1] = 2
        let a = v(&[1.0, 1.0]);
        let b = v(&[1.0, 1.0]);
        let d1 = a.dot(b);
        println!("dot([1,1],[1,1]) = {}", d1);
        assert!(approx_eq(d1, 2.0));

        // [-1,6] · [3,2] = -1*3 + 6*2 = 9
        let c = v(&[-1.0, 6.0]);
        let d = v(&[3.0, 2.0]);
        let d2 = c.dot(d);
        println!("dot([-1,6],[3,2]) = {}", d2);
        assert!(approx_eq(d2, 9.0));
    }

    #[test]
    fn dot_orthogonal_is_zero() {
        // [1,0,0] · [0,1,0] = 0
        let ex = v(&[1.0, 0.0, 0.0]);
        let ey = v(&[0.0, 1.0, 0.0]);
        let d = ex.dot(ey);
        println!("dot(ex, ey) = {}", d);
        assert!(approx_eq(d, 0.0));
    }

    #[test]
    fn dot_sign_and_magnitude() {
        let a = v(&[2.0, -3.0, 4.0]);
        let b = v(&[-5.0, 6.0, -7.0]);
        // 2*-5 + (-3)*6 + 4*-7 = -10 -18 -28 = -56
        let d = a.dot(b);
        println!("dot([2,-3,4],[-5,6,-7]) = {}", d);
        assert!(approx_eq(d, -56.0));
    }

    #[test]
    #[should_panic(expected = "Size mismatch at Vector::dot()!")]
    fn dot_dim_mismatch_panics() {
        let a = v(&[1.0, 2.0]);
        let b = v(&[3.0]);
        let _ = a.dot(b); // b moved; mismatched sizes should panic
    }

    // include this only if your implementation intentionally panics on empty vectors
    #[test]
    #[should_panic(expected = "Empty vector input at Vector::dot()!")]
    fn dot_empty_vectors_panics() {
        let a = v(&[]);
        let b = v(&[]);
        let _ = a.dot(b);
    }
}
