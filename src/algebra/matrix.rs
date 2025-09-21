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
            writeln!(f, " ]")?;
        }
        Ok(())
    }
}

impl<K: fmt::Display> fmt::Display for Matrix<K> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let precision = f.precision().unwrap_or(1);
        let mut col_len = vec![0usize; self.cols];
        for i in 0..self.rows {
            for j in 0..self.cols {
                let s = format!("{:.*}", precision, self.data[j + i * self.cols]);
                col_len[j] = col_len[j].max(s.len());
            }
        }

        for i in 0..self.rows {
            write!(f, "[")?;
            for j in 0..self.cols {
                let s = format!("{:.*}", precision, self.data[j + i * self.cols]);
                write!(f, " {:>width$}", s, width = col_len[j])?;
            }
            writeln!(f, " ]")?;
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

    // default display (uses precision = f.precision().unwrap_or(1))
    fn disp(m: &Matrix<f32>) -> String {
        format!("{}", m)
    }

    // display with N decimals (runtime precision)
    fn disp_p(m: &Matrix<f32>, p: usize) -> String {
        format!("{:.prec$}", m, prec = p)
    }

    #[test]
    fn new_ok_and_shape_f32() {
        let m = Matrix::new(vec![1.0, 2.0, 3.0, 4.0], 2, 2).unwrap();
        println!("new_ok_and_shape_f32 (Display):\n{}", m);
        println!("new_ok_and_shape_f32 (Debug):\n{:?}", m);
        // Display: 1 decimal by default, leading space before each field, space before ']'
        assert_eq!(disp(&m), "[ 1.0 2.0 ]\n[ 3.0 4.0 ]\n");
        // Debug keeps a space before the closing bracket too
        assert_eq!(format!("{:?}", m), "[ 1.0 2.0 ]\n[ 3.0 4.0 ]\n");
        assert_eq!(m.cols, 2);
        assert_eq!(m.rows, 2);
    }

    #[test]
    fn new_err_f32() {
        let bad = Matrix::new(vec![1.0, 2.0, 3.0, 4.0, 5.0], 3, 2);
        println!("new_err_f32 (Debug): {:?}", bad);
        assert!(bad.is_err());
        assert!(bad
            .unwrap_err()
            .contains("data length 5 does not match 2*3"));
    }

    #[test]
    fn from_2d_array_f32() {
        let m = Matrix::from([[2.0f32, 1.0], [3.0, 4.0]]);
        println!("from_2d_array_f32:\n{}", m);
        assert_eq!(disp(&m), "[ 2.0 1.0 ]\n[ 3.0 4.0 ]\n");
    }

    #[test]
    fn add_ok_f32() {
        let mut a = Matrix::from([[2.0f32, 1.0], [3.0, 4.0]]);
        let b = Matrix::from([[20.0f32, 10.0], [30.0, 40.0]]);
        a.add(&b).unwrap();
        println!("add_ok_f32:\n{}", a);
        assert_eq!(disp(&a), "[ 22.0 11.0 ]\n[ 33.0 44.0 ]\n");
    }

    #[test]
    fn sub_ok_f32() {
        let mut a = Matrix::from([[2.0f32, 1.0], [3.0, 4.0]]);
        let b = Matrix::from([[20.0f32, 10.0], [30.0, 40.0]]);
        a.sub(&b).unwrap();
        println!("sub_ok_f32:\n{}", a);
        // second column right-aligns; "-9.0" is shorter than "-36.0" → extra padding appears
        assert_eq!(disp(&a), "[ -18.0  -9.0 ]\n[ -27.0 -36.0 ]\n");
    }

    #[test]
    fn scl_ok_f32() {
        let mut a = Matrix::from([[2.0f32, 1.0], [3.0, 4.0]]);
        a.scl(0.5);
        println!("scl_ok_f32 after *0.5:\n{}", a);
        assert_eq!(disp(&a), "[ 1.0 0.5 ]\n[ 1.5 2.0 ]\n");
        a.scl(2.0);
        println!("scl_ok_f32 after *2.0:\n{}", a);
        assert_eq!(disp(&a), "[ 2.0 1.0 ]\n[ 3.0 4.0 ]\n");
    }

    #[test]
    fn add_shape_mismatch_f32() {
        let mut a = Matrix::from([[1.0f32, 2.0], [3.0, 4.0]]);
        let b = Matrix::from([[9.0f32, 8.0, 7.0], [6.0, 5.0, 4.0]]);
        let res = a.add(&b);
        println!("add_shape_mismatch_f32 (Debug): {:?}", res);
        assert!(res.is_err());
        assert_eq!(res.unwrap_err(), "Matrix sizes do not match");
    }

    #[test]
    fn display_alignment_f32() {
        let m = Matrix::from([[2.0f32, 10.0], [300.0, 4.0]]);
        println!("display_alignment_f32:\n{}", m);
        // widths from precision=1 strings: col0=len("300.0")=5, col1=len("10.0")=4
        // non-folded separator (single space between columns):
        let expected = "[   2.0 10.0 ]\n[ 300.0  4.0 ]\n";
        assert_eq!(disp(&m), expected);
    }

    #[test]
    fn debug_matrix_f32() {
        let m = Matrix::from([[1.0f32, 2.0], [3.0, 4.0]]);
        println!("debug_matrix_f32 (Debug):\n{:?}", m);
        assert_eq!(format!("{:?}", m), "[ 1.0 2.0 ]\n[ 3.0 4.0 ]\n");
    }

    // ───── Precision tests (> 5 decimals) ──────────────────────────────────────

    #[test]
    fn display_precision_6_uniform() {
        let m = Matrix::from([[1.234567f32, 2.0], [300.0, 4.0]]);
        let out = disp_p(&m, 6);
        println!("display_precision_6_uniform (p=6):\n{}", out);
        let expected = "[   1.234567 2.000000 ]\n[ 300.000000 4.000000 ]\n";
        assert_eq!(out, expected);
    }

    #[test]
    fn display_precision_8_alignment() {
        let m = Matrix::from([[2.0f32, 10.0], [300.0, 4.0]]);
        let out = disp_p(&m, 8);
        println!("display_precision_8_alignment (p=8):\n{}", out);
        // non-folded separator: single space between columns
        let expected = "[   2.00000000 10.00000000 ]\n[ 300.00000000  4.00000000 ]\n";
        assert_eq!(out, expected);
    }

    // ───── "Bad outputs" / error display helpers ──────────────────────────────

    #[test]
    fn result_debug_prints() {
        // Ensure debug-printing Results compiles
        let bad = Matrix::new(vec![1.0], 2, 2);
        println!("result_debug_prints (Debug bad new): {:?}", bad);
        assert!(bad.is_err());

        let mut a = Matrix::from([[1.0f32, 2.0], [3.0, 4.0]]);
        let b = Matrix::from([[9.0f32, 8.0, 7.0], [6.0, 5.0, 4.0]]);
        let res = a.add(&b);
        println!("result_debug_prints (Debug add mismatch): {:?}", res);
        assert!(res.is_err());
    }
}
