use std::fmt;
use std::ops::{Add, Mul, Sub};
#[derive(Clone)]

pub struct Matrix<K> {
    cols: usize,
    rows: usize,
    data: Vec<K>,
}

impl<K> Matrix<K> {
    pub fn new(data: Vec<K>, col: usize, row: usize) -> Result<Self, String> {
        if row * col != data.len() {
            return Err(format!(
                "Matrix::new: data length {} does not match {}*{}",
                data.len(),
                row,
                col
            ));
        }
        Ok(Matrix {
            cols: col,
            rows: row,
            data,
        })
    }
    pub fn print_size(&self) {
        println!("{}x{}", self.rows, self.cols);
    }
}

impl<K> Matrix<K>
where
    K: Add<Output = K> + Sub<Output = K> + Mul<Output = K> + Copy,
{
    pub fn add(&mut self, m: &Matrix<K>) -> Result<(), String> {
        if self.cols != m.cols || self.rows != m.rows {
            return Err("Matrix sizes do not match".to_string());
        }
        for (a, b) in self.data.iter_mut().zip(&m.data) {
            *a = *a + *b;
        }
        Ok(())
    }

    pub fn sub(&mut self, m: &Matrix<K>) -> Result<(), String> {
        if self.cols != m.cols || self.rows != m.rows {
            return Err("Matrix sizes do not match".to_string());
        }
        for (a, b) in self.data.iter_mut().zip(&m.data) {
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

impl<K: fmt::Debug> fmt::Debug for Matrix<K> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for i in 0..self.rows {
            write!(f, "[")?;
            for j in 0..self.cols {
                write!(f, " {:?}", &self.data[j + self.cols * i])?;
            }
            writeln!(f, "]")?;
        }
        Ok(())
    }
}

impl<K: fmt::Display> fmt::Display for Matrix<K> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // 1) compute per-column widths
        let mut col_len = vec![0usize; self.cols];
        for i in 0..self.rows {
            for j in 0..self.cols {
                let s = format!("{}", self.data[j + i * self.cols]);
                if s.len() > col_len[j] {
                    col_len[j] = s.len(); // <-- index by column
                }
            }
        }

        // 2) print rows using those widths
        for i in 0..self.rows {
            write!(f, "[")?;
            for j in 0..self.cols {
                if j > 0 {
                    write!(f, " ")?;
                }
                let s = format!("{}", self.data[j + i * self.cols]);
                write!(f, "{:>width$}", s, width = col_len[j])?; // <-- width by column
            }
            writeln!(f, "]")?;
        }
        Ok(())
    }
}

impl<K: Clone, const R: usize, const C: usize> From<[[K; C]; R]> for Matrix<K> {
    fn from(value: [[K; C]; R]) -> Self {
        let mut data = Vec::with_capacity(C * R);
        for row in value {
            for val in row {
                data.push(val);
            }
        }
        Matrix {
            cols: C,
            rows: R,
            data,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn disp(m: &Matrix<f32>) -> String {
        format!("{}", m)
    }

    #[test]
    fn new_ok_and_shape_f32() {
        let m = Matrix::new(vec![1.0, 2.0, 3.0, 4.0], 2, 2).unwrap();
        assert_eq!(format!("{:?}", m), "[ 1.0 2.0 ]\n[ 3.0 4.0 ]\n");
        assert_eq!(disp(&m), "[1 2]\n[3 4]\n");
        assert_eq!(m.cols, 2);
        assert_eq!(m.rows, 2);
    }

    #[test]
    fn new_err_f32() {
        let bad = Matrix::new(vec![1.0, 2.0, 3.0, 4.0, 5.0], 3, 2);
        assert!(bad.is_err());
        assert!(bad
            .unwrap_err()
            .contains("data length 5 does not match 2*3"));
    }

    #[test]
    fn from_2d_array_f32() {
        let m = Matrix::from([[2.0f32, 1.0], [3.0, 4.0]]);
        assert_eq!(disp(&m), "[2 1]\n[3 4]\n");
    }

    #[test]
    fn add_ok_f32() {
        let mut a = Matrix::from([[2.0f32, 1.0], [3.0, 4.0]]);
        let b = Matrix::from([[20.0f32, 10.0], [30.0, 40.0]]);
        a.add(&b).unwrap();
        assert_eq!(disp(&a), "[22 11]\n[33 44]\n");
    }

    #[test]
    fn sub_ok_f32() {
        let mut a = Matrix::from([[2.0f32, 1.0], [3.0, 4.0]]);
        let b = Matrix::from([[20.0f32, 10.0], [30.0, 40.0]]);
        a.sub(&b).unwrap();
        assert_eq!(disp(&a), "[-18 -9]\n[-27 -36]\n");
    }

    #[test]
    fn scl_ok_f32() {
        let mut a = Matrix::from([[2.0f32, 1.0], [3.0, 4.0]]);
        a.scl(0.5);
        assert_eq!(disp(&a), "[1 0.5]\n[1.5 2]\n");
        a.scl(2.0);
        assert_eq!(disp(&a), "[2 1]\n[3 4]\n");
    }

    #[test]
    fn add_shape_mismatch_f32() {
        let mut a = Matrix::from([[1.0f32, 2.0], [3.0, 4.0]]);
        let b = Matrix::from([[9.0f32, 8.0, 7.0], [6.0, 5.0, 4.0]]);
        let res = a.add(&b);
        assert!(res.is_err());
        assert_eq!(res.unwrap_err(), "Matrix sizes do not match");
    }

    #[test]
    fn display_alignment_f32() {
        let m = Matrix::from([[2.0f32, 10.0], [300.0, 4.0]]);
        let expected = "[  2  10]\n[300   4]\n";
        assert_eq!(disp(&m), expected);
    }

    #[test]
    fn debug_matrix_f32() {
        let m = Matrix::from([[1.0f32, 2.0], [3.0, 4.0]]);
        assert_eq!(format!("{:?}", m), "[ 1.0 2.0 ]\n[ 3.0 4.0 ]\n");
    }
}
