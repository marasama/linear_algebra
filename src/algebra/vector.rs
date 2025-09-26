use std::clone::Clone;
use std::cmp::PartialEq;
use std::fmt;
use std::marker::Copy;
use std::ops::{Add, AddAssign, Mul, MulAssign, Neg, Sub, SubAssign};
use std::process::Output;

#[derive(Clone)]
struct Vector<K> {
    data: Vec<K>,
}

impl<K> Vector<K> {
    pub fn new(data: Vec<K>) -> Vector<K> {
        Vector { data }
    }

    pub fn size(&self) -> usize {
        self.data.len()
    }
}

impl<K> fmt::Display for Vector<K>
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

impl<K> fmt::Debug for Vector<K>
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

impl<K> Vector<K>
where
    K: Copy + AddAssign + SubAssign + MulAssign + Add + Mul + Sub + Neg,
{
    pub fn add(&mut self, v: &Vector<K>) {
        assert_eq!(self.size(), v.size(), "Dimension mismatch at vector.add()");
        for (a, b) in self.data.iter_mut().zip(&v.data) {
            *a += *b;
        }
    }
    pub fn sub(&mut self, v: &Vector<K>) {
        assert_eq!(self.size(), v.size(), "Dimension mismatch at vector.add()");
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

impl<K> Add<&Vector<K>> for &Vector<K>
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

impl<K> Sub<&Vector<K>> for &Vector<K>
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

impl<K> Mul<&Vector<K>> for &Vector<K>
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

impl<K> Neg for &Vector<K>
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

impl<K> AddAssign<&Vector<K>> for Vector<K>
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

impl<K> SubAssign<&Vector<K>> for Vector<K>
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

impl<K> MulAssign<&Vector<K>> for Vector<K>
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
