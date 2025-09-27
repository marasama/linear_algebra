use num_traits::Float;
use std::clone::Clone;
use std::cmp::PartialEq;
use std::fmt;
use std::marker::Copy;
use std::ops::{Add, AddAssign, Mul, MulAssign, Neg, Sub, SubAssign};
#[derive(Clone)]
struct Vector<K: Float> {
    data: Vec<K>,
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
            "Dimension mismatch at (+=) operator"
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
            "Dimension mismatch at (+=) operator"
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
