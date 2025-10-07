use num_traits::{Float, NumCast};
use std::{
    fmt::{self, Display},
    ops::{AddAssign, MulAssign, SubAssign},
};

#[derive(Clone)]
pub struct Matrix<K> {
    pub data: Vec<K>,
    pub rows: usize,
    pub cols: usize,
}

impl<K: Float> Matrix<K>
where
    K: AddAssign + SubAssign + MulAssign + Copy,
{
    pub fn add(&mut self, v: &Matrix<K>) {
        assert_eq!(
            self.data.len(),
            v.data.len(),
            "Size mismatch at matrix.add()!"
        );
        assert_eq!(self.rows, v.rows, "Size of rows mismatch at matrix.add()!");
        assert_eq!(self.cols, v.cols, "Size of cols mismatch at matrix.add()!");
        for (a, b) in self.data.iter_mut().zip(&v.data) {
            *a += *b;
        }
    }
    pub fn sub(&mut self, v: &Matrix<K>) {
        assert_eq!(
            self.data.len(),
            v.data.len(),
            "Size mismatch at matrix.sub()!"
        );
        assert_eq!(self.rows, v.rows, "Size of rows mismatch at matrix.sub()!");
        assert_eq!(self.cols, v.cols, "Size of cols mismatch at matrix.sub()!");
        for (a, b) in self.data.iter_mut().zip(&v.data) {
            *a -= *b;
        }
    }
    pub fn scl(&mut self, a: K) {
        for b in self.data.iter_mut() {
            *b *= a
        }
    }
}

impl<K: Float> Matrix<K> {
    pub fn new(m: Vec<K>, r: usize, c: usize) -> Self {
        assert_eq!(
            m.len(),
            r * c,
            "data length {} != rows*cols {}*{}",
            m.len(),
            r,
            c
        );
        Self {
            data: m,
            rows: r,
            cols: c,
        }
    }
    pub fn zero(r: usize, c: usize) -> Self {
        Self {
            data: vec![K::zero(); r * c],
            rows: r,
            cols: c,
        }
    }
    pub fn identity(r: usize, c: usize) -> Self {
        assert_eq!(
            r, c,
            "An identity matrix's rows and columns must be equal at Matrix::identity()!"
        );
        let mut id = vec![K::zero(); r * c];
        for i in 0..r {
            id[i + i * c] = K::one();
        }
        Matrix {
            data: id,
            rows: r,
            cols: c,
        }
    }
    pub fn print_size(&self) {
        println!("Row:{}xCol:{}", self.rows, self.cols);
    }
    pub fn size(&self) -> (usize, usize) {
        (self.rows, self.cols)
    }
    fn index(&self, r: usize, c: usize) -> usize {
        r * self.cols + c
    }
    pub fn get_val(&self, r: usize, c: usize) -> &K {
        &self.data[r * self.cols + c]
    }
    pub fn get_val_mut(&mut self, r: usize, c: usize) -> &K {
        &mut self.data[r * self.cols + c]
    }
}

impl<K: Float, const R: usize, const C: usize> From<[[K; C]; R]> for Matrix<K> {
    fn from(value: [[K; C]; R]) -> Self {
        let mut new_data = Vec::with_capacity(R * C);
        for row in value {
            new_data.extend(row);
        }
        Matrix {
            data: new_data,
            rows: R,
            cols: C,
        }
    }
}

impl<K: Float + Display> fmt::Debug for Matrix<K> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut max_length = 0;
        let prec = f.precision().unwrap_or(1);
        for a in 0..self.rows {
            for b in 0..self.cols {
                let tmp_len = format!("{:.prec$}", self.data[a * self.cols + b], prec = prec).len();
                if tmp_len > max_length {
                    max_length = tmp_len
                }
            }
        }
        for r in 0..self.rows {
            write!(f, "[ ")?;
            for c in 0..self.cols {
                write!(
                    f,
                    "{:>width$.prec$}",
                    self.data[r * self.cols + c],
                    width = max_length,
                    prec = prec
                )?;
                if c + 1 < self.cols {
                    write!(f, " ")?;
                }
            }
            writeln!(f, " ]")?;
        }
        Ok(())
    }
}
impl<K: Float + Display> fmt::Display for Matrix<K> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut max_length = 0;
        let prec = f.precision().unwrap_or(1);
        for a in 0..self.rows {
            for b in 0..self.cols {
                let tmp_len = format!("{:.prec$}", self.data[a * self.cols + b], prec = prec).len();
                if tmp_len > max_length {
                    max_length = tmp_len
                }
            }
        }
        for r in 0..self.rows {
            write!(f, "[ ")?;
            for c in 0..self.cols {
                write!(
                    f,
                    "{:>width$.prec$}",
                    self.data[r * self.cols + c],
                    width = max_length,
                    prec = prec
                )?;
                if c + 1 < self.cols {
                    write!(f, " ")?;
                }
            }
            writeln!(f, " ]")?;
        }
        Ok(())
    }
}
#[cfg(test)]
mod tests {
    use super::*;

    // ---------- helpers ----------
    fn approx_eq_slice_f32(a: &[f32], b: &[f32], eps: f32) -> bool {
        a.len() == b.len() && a.iter().zip(b).all(|(x, y)| (*x - *y).abs() <= eps)
    }
    fn approx_eq_slice_f64(a: &[f64], b: &[f64], eps: f64) -> bool {
        a.len() == b.len() && a.iter().zip(b).all(|(x, y)| (*x - *y).abs() <= eps)
    }

    // Force the compiler to check the concrete type without needing turbofish everywhere.
    fn assert_is_matrix_f32(_: &Matrix<f32>) {}
    fn assert_is_matrix_f64(_: &Matrix<f64>) {}

    // Capture Display/Debug into a String
    fn disp<K: Float + std::fmt::Display>(m: &Matrix<K>) -> String {
        format!("{}", m)
    }
    fn disp_p<K: Float + std::fmt::Display>(m: &Matrix<K>, p: usize) -> String {
        format!("{:.prec$}", m, prec = p)
    }
    fn dbg_str<K: Float + std::fmt::Display>(m: &Matrix<K>) -> String {
        format!("{:?}", m)
    }
    fn dbg_str_p<K: std::fmt::Display + Float>(m: &Matrix<K>, p: usize) -> String {
        format!("{:.prec$?}", m, prec = p)
    }

    // ---------- type & construction ----------
    #[test]
    fn construct_from_2d_array_f32_and_f64() {
        // Unsuffixed float literals are f64 by default, so for f32 we must suffix or annotate.
        let m32 = Matrix::<f32>::from([[1.0f32, 2.0f32], [3.0f32, 4.0f32]]);
        let m64 = Matrix::<f64>::from([[1.0, 2.0], [3.0, 4.0]]);
        assert_is_matrix_f32(&m32);
        assert_is_matrix_f64(&m64);
        assert_eq!(m32.size(), (2, 2));
        assert_eq!(m64.size(), (2, 2));
        assert!(approx_eq_slice_f32(&m32.data, &[1.0, 2.0, 3.0, 4.0], 0.0));
        assert!(approx_eq_slice_f64(&m64.data, &[1.0, 2.0, 3.0, 4.0], 0.0));
    }

    #[test]
    fn construct_zero_by_zero_ok() {
        // valid: 0x0 with empty data
        let m: Matrix<f64> = Matrix::new(vec![], 0, 0);
        assert_eq!(m.size(), (0, 0));
        assert!(m.data.is_empty());
    }

    #[test]
    #[should_panic(expected = "data length")]
    fn new_panics_on_wrong_len() {
        // length 3 cannot make 2x2
        let _ = Matrix::new(vec![1.0f32, 2.0, 3.0], 2, 2);
    }

    // ---------- arithmetic ----------
    #[test]
    fn add_sub_scl_f32() {
        let mut a = Matrix::<f32>::from([[1.0f32, 2.0], [3.0, 4.0]]);
        let b = Matrix::<f32>::from([[10.0f32, 20.0], [30.0, 40.0]]);

        // add
        a.add(&b);
        assert!(approx_eq_slice_f32(
            &a.data,
            &[11.0, 22.0, 33.0, 44.0],
            1e-6
        ));

        // sub (back to original)
        a.sub(&b);
        assert!(approx_eq_slice_f32(&a.data, &[1.0, 2.0, 3.0, 4.0], 1e-6));

        // scl
        a.scl(0.5);
        assert!(approx_eq_slice_f32(&a.data, &[0.5, 1.0, 1.5, 2.0], 1e-6));
    }

    #[test]
    fn add_sub_scl_f64() {
        let mut a = Matrix::<f64>::from([[1.0, 2.0], [3.0, 4.0]]);
        let b = Matrix::<f64>::from([[10.0, 20.0], [30.0, 40.0]]);

        a.add(&b);
        assert!(approx_eq_slice_f64(
            &a.data,
            &[11.0, 22.0, 33.0, 44.0],
            1e-12
        ));

        a.sub(&b);
        assert!(approx_eq_slice_f64(&a.data, &[1.0, 2.0, 3.0, 4.0], 1e-12));

        a.scl(0.5);
        assert!(approx_eq_slice_f64(&a.data, &[0.5, 1.0, 1.5, 2.0], 1e-12));
    }

    #[test]
    #[should_panic(expected = "Size mismatch at matrix.add")]
    fn add_panics_on_shape_mismatch() {
        let mut a = Matrix::<f32>::from([[1.0f32, 2.0], [3.0, 4.0]]);
        let b = Matrix::<f32>::from([[5.0f32, 6.0, 7.0], [8.0, 9.0, 10.0]]);
        a.add(&b);
    }

    // ---------- formatting ----------
    #[test]
    fn display_alignment_default_precision() {
        // Your Display uses the formatter's precision (default 0)
        let m = Matrix::<f64>::from([[1.0, 22.0], [333.0, 4.0]]);
        let s = disp(&m);
        // width should match max of "1", "22", "333", "4" => 3
        let expected = "[   1.0  22.0 ]\n[ 333.0   4.0 ]\n";
        assert_eq!(s, expected);
    }

    #[test]
    fn display_alignment_with_precision_2() {
        let m = Matrix::<f64>::from([[1.0, 22.0], [333.0, 4.0]]);
        let s = disp_p(&m, 2);
        // width should match max of "333.00" => 6
        let expected = "[   1.00  22.00 ]\n[ 333.00   4.00 ]\n";
        assert_eq!(s, expected);
    }

    #[test]
    fn debug_matches_display_rules() {
        // If your Debug uses same algorithm as Display (you showed earlier), check both
        let m = Matrix::<f32>::from([[1.0f32, 22.0], [333.0, 4.0]]);
        let d0 = dbg_str(&m);
        let d2 = dbg_str_p(&m, 2);
        assert_eq!(d0, "[   1.0  22.0 ]\n[ 333.0   4.0 ]\n");
        assert_eq!(d2, "[   1.00  22.00 ]\n[ 333.00   4.00 ]\n");
    }

    // ---------- variable type / inference specifics ----------
    #[test]
    fn explicit_f32_literals_required_if_no_type_annotation() {
        // This compiles because we annotate Matrix::<f32>
        let m_ok = Matrix::<f32>::from([[1.0f32, 2.0f32], [3.0f32, 4.0f32]]);
        assert_is_matrix_f32(&m_ok);

        // This compiles as f64 without annotation (unsuffixed -> f64):
        let m64 = Matrix::<f64>::from([[1.0, 2.0], [3.0, 4.0]]);
        assert_is_matrix_f64(&m64);

        // If you tried: let m = Matrix::from([[1.0, 2.0], [3.0, 4.0]]);
        // the type parameter would be ambiguous unless you use a context type or suffix.
    }

    #[test]
    fn rename_helpers_demonstrate_actual_types() {
        // These helpers exist purely to ensure the compiler infers the right K.
        let a32 = Matrix::<f32>::from([[0.5f32, 1.5], [2.5, 3.5]]);
        let a64 = Matrix::<f64>::from([[0.5, 1.5], [2.5, 3.5]]);
        assert_is_matrix_f32(&a32);
        assert_is_matrix_f64(&a64);
    }

    // ---------- edge-ish numeric checks ----------
    #[test]
    fn scale_by_zero_and_one() {
        let mut m = Matrix::<f64>::from([[1.25, -2.5], [0.0, 4.0]]);
        let original = m.clone();

        m.scl(1.0); // identity
        assert!(approx_eq_slice_f64(&m.data, &original.data, 0.0));

        m.scl(0.0); // annihilate
        assert!(approx_eq_slice_f64(&m.data, &[0.0, 0.0, 0.0, 0.0], 0.0));
    }

    #[test]
    fn large_and_small_values_precision_smoke() {
        // Just a smoke check that formatting width logic still aligns for different magnitudes.
        let m = Matrix::<f32>::from([[1e-6f32, 2.0], [3.0, 4.0e6]]);
        let s0 = disp(&m);
        let s3 = disp_p(&m, 3);

        // Not asserting the exact strings here (since width depends on your precise rules)
        // but do ensure lines & brackets exist and no panic happened.
        assert!(s0.starts_with("[ "));
        assert!(s0.ends_with(" ]\n"));
        assert!(s3.contains("."));
    }

    // ---------- (optional) show how to test panics without crashing run ----------
    #[test]
    fn size_mismatch_panics_can_be_caught() {
        use std::panic;
        let mut a = Matrix::<f32>::from([[1.0f32, 2.0], [3.0, 4.0]]);
        let b = Matrix::<f32>::from([[5.0f32, 6.0, 7.0], [8.0, 9.0, 10.0]]);
        let result = panic::catch_unwind(move || {
            a.add(&b);
        });
        assert!(result.is_err());
    }

    // ---------- doc-style negative test (compile-fail) ----------
    // NOTE: This is just documentation; it will NOT run as part of cargo test.
    // Your Matrix methods are bounded by `num_traits::Float`, so Matrix<i32> is not allowed.
    //
    // ```compile_fail
    // let m_bad = Matrix::<i32>::from([[1, 2], [3, 4]]);
    // ```
}
