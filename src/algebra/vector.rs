use std::fmt;
use std::ops::{Add, Mul, Sub};
#[derive(Clone)]

pub struct Vector<K> {
    data: Vec<K>,
}

impl<K> Vector<K> {
    pub fn new(data: Vec<K>) -> Self {
        Vector { data }
    }

    pub fn size(&self) -> usize {
        self.data.len()
    }
}

impl<K: fmt::Debug> fmt::Debug for Vector<K> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:?}", &self.data)
    }
}

impl<K: fmt::Display> fmt::Display for Vector<K> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for num in &self.data {
            writeln!(f, "[{}]", num)?;
        }
        Ok(())
    }
}

impl<K: Clone, const N: usize> From<[K; N]> for Vector<K> {
    fn from(value: [K; N]) -> Self {
        Vector {
            data: value.to_vec(),
        }
    }
}

impl<K> Vector<K>
where
    K: Add<Output = K> + Sub<Output = K> + Mul<Output = K> + Copy,
{
    pub fn add(&mut self, v: &Vector<K>) -> Result<(), String> {
        if self.size() != v.size() {
            return Err("Vector sizes do not match!".to_string());
        }

        for (a, b) in self.data.iter_mut().zip(&v.data) {
            *a = *a + *b;
        }
        Ok(())
    }
    pub fn sub(&mut self, v: &Vector<K>) -> Result<(), String> {
        if self.size() != v.size() {
            return Err("Vector sizes do not match!".to_string());
        }

        for (a, b) in self.data.iter_mut().zip(&v.data) {
            *a = *a - *b;
        }
        Ok(())
    }
    pub fn scl(&mut self, scalar: K) {
        for a in self.data.iter_mut() {
            *a = *a * scalar;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_add_f32() {
        let mut u = Vector::from([2.0f32, 3.0]);
        let v = Vector::from([5.0, 7.0]);
        u.add(&v).unwrap();
        assert_eq!(format!("{}", u), "[7]\n[10]\n");
    }

    #[test]
    fn test_sub_f32() {
        let mut u = Vector::from([2.0f32, 3.0]);
        let v = Vector::from([5.0, 7.0]);
        u.sub(&v).unwrap();
        assert_eq!(format!("{}", u), "[-3]\n[-4]\n");
    }

    #[test]
    fn test_scl_f32() {
        let mut u = Vector::from([2.0f32, 3.0]);
        u.scl(2.0);
        assert_eq!(format!("{}", u), "[4]\n[6]\n");
    }

    #[test]
    fn test_size_mismatch_f32() {
        let mut u = Vector::from([1.0f32, 2.0, 3.0]);
        let v = Vector::from([4.0f32, 5.0]);
        let result = u.add(&v);
        assert!(result.is_err());
    }

    #[test]
    fn test_add_i32() {
        let mut u = Vector::from([2, 3]);
        let v = Vector::from([5, 7]);
        u.add(&v).unwrap();
        assert_eq!(format!("{}", u), "[7]\n[10]\n");
    }

    #[test]
    fn test_scl_i32() {
        let mut u = Vector::from([2, 3]);
        u.scl(3);
        assert_eq!(format!("{}", u), "[6]\n[9]\n");
    }

    #[test]
    fn test_add_f64() {
        let mut u = Vector::from([1.5f64, 2.5]);
        let v = Vector::from([3.5, 4.5]);
        u.add(&v).unwrap();
        assert_eq!(format!("{}", u), "[5]\n[7]\n");
    }
}
