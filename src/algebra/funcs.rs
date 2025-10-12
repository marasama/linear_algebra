use super::matrix::Matrix;
use super::vector::Vector;
use num_traits::{Float, NumCast};

impl<K: Float> Vector<K> {
    pub fn norm_1(&mut self) -> f32 {
        if self.data.is_empty() {
            return 0.0;
        }
        let mut sum: K = K::zero();
        for &a in self.data.iter() {
            sum = sum + a.max(-a)
        }
        NumCast::from(sum).unwrap_or(0.0f32)
    }
    pub fn norm(&mut self) -> f32 {
        if self.data.is_empty() {
            return 0.0;
        }
        let mut sum: K = K::zero();
        for &a in self.data.iter() {
            sum = a.mul_add(a, sum);
        }
        let sqrt: K = NumCast::from(0.5f32).unwrap();
        let sum_sqrt = sum.powf(sqrt);
        NumCast::from(sum_sqrt).unwrap_or(0.0f32)
    }
    pub fn norm_inf(&mut self) -> f32 {
        if self.data.is_empty() {
            return 0.0;
        }
        let mut max_val: K = K::zero();
        for &a in self.data.iter() {
            let abs_a: K = a.max(-a);
            max_val = max_val.max(abs_a);
        }
        NumCast::from(max_val).unwrap_or(0.0f32)
    }
    pub fn dot(&self, v: Vector<K>) -> K {
        assert_eq!(self.size(), v.size(), "Size mismatch at Vector::dot()!");
        assert!(
            !(self.data.is_empty() || v.data.is_empty()),
            "Empty vector input at Vector::dot()!"
        );
        let mut sum: K = K::zero();
        for i in 0..self.size() {
            sum = self.data[i].mul_add(v.data[i], sum);
        }
        sum
    }
}

impl<K: Float> Matrix<K> {
    pub fn mul_vec(&mut self, vec: Vector<K>) -> Vector<K> {
        assert_eq!(
            self.cols,
            vec.size(),
            "MxN matrix need R^N vector to multiply in Matrix::mul_vec()!"
        );
        let mut new_vec = vec![K::zero(); self.rows];
        for i in 0..self.rows {
            for j in 0..self.cols {
                new_vec[i] = vec.data[j].mul_add(self.data[i * self.cols + j], new_vec[i]);
            }
        }
        Vector { data: new_vec }
    }
    pub fn mul_mat(&mut self, mat: Matrix<K>) -> Matrix<K> {
        assert_eq!(
            self.cols, mat.rows,
            "MxN matrix need NxP matrix to multiply in Matrix::mul_mat()!"
        );
        let M = self.rows;
        let N = self.cols;
        let P = mat.cols;
        let mut new_mat = vec![K::zero(); M * P];
        for i in 0..M {
            for j in 0..P {
                let mut acc = K::zero();
                for k in 0..N {
                    let a_ik: K = self.data[i * N + k];
                    let b_kj: K = mat.data[k * P + j];
                    acc = a_ik.mul_add(b_kj, acc);
                }
                new_mat[i * P + j] = acc;
            }
        }
        Matrix {
            data: new_mat,
            rows: M,
            cols: P,
        }
    }
    pub fn trace(&mut self) -> K {
        if self.data.is_empty() {
            return K::zero();
        }
        assert_eq!(
            self.cols, self.rows,
            "Must be NxN matrix to calculate the trace at trace()!"
        );
        let mut trace = K::zero();
        for a in 0..self.cols {
            trace = trace + self.data[a * self.cols + a];
        }
        trace
    }
    pub fn transpose(&mut self) -> Matrix<K> {
        if self.data.is_empty() {
            return Matrix::new(vec![], 0, 0);
        }
        let mut trans = Matrix::new(vec![K::zero(); self.cols * self.rows], self.cols, self.rows);
        for r in 0..trans.rows {
            for c in 0..trans.cols {
                trans.data[c + r * trans.cols] = self.data[r + c * trans.rows];
            }
        }
        trans
    }
}
impl<K: Float> Matrix<K> {
    pub fn swap_rows(&mut self, row1: usize, row2: usize) {
        assert!(
            row1 < self.rows && row2 < self.rows,
            "Wrong line input at Matrix::switch_rows()!"
        );
        if row1 == row2 {
            return;
        }
        for col in 0..self.cols {
            self.data
                .swap(row1 * self.cols + col, row2 * self.cols + col);

            //let temp: K = self.data[i + row1 * self.cols];
            //self.data[i + row1 * self.cols] = self.data[i + row2 * self.cols];
            //self.data[i + row2 * self.cols] = temp;

            //self.data[i + row1 * self.cols] =
            //    self.data[i + row1 * self.cols] + self.data[i + row2 * self.cols];
            //self.data[i + row2 * self.cols] =
            //    self.data[i + row1 * self.cols] - self.data[i + row2 * self.cols];
            //self.data[i + row1 * self.cols] =
            //    self.data[i + row1 * self.cols] - self.data[i + row2 * self.cols];
        }
    }
    pub fn row_echelon(&mut self) -> Matrix<K> {
        let mut pivot_row = 0;
        let mut pivot_col = 0;
        while pivot_row < self.rows && pivot_col < self.cols {
            if let Some(i) = (pivot_row..self.rows)
                .position(|r| self.data[r * self.cols + pivot_col] != K::zero())
            {
                self.swap_rows(pivot_row, pivot_row + i);
                let pivot_val = self.data[pivot_row * self.cols + pivot_col];
                for j in pivot_col..self.cols {
                    self.data[pivot_row * self.cols + j] =
                        self.data[pivot_row * self.cols + j] / pivot_val;
                }
                for l in (pivot_row + 1)..self.rows {
                    let factor = self.data[l * self.cols + pivot_col];
                    for m in pivot_col..self.cols {
                        let subtrahend = factor * self.data[pivot_row * self.cols + m];
                        //println!("subtrahend: {}", subtrahend);
                        self.data[l * self.cols + m] = self.data[l * self.cols + m] - subtrahend;
                    }
                }
                //println!(
                //    "pivot_row: {} pivot_col: {}\n{:.3}",
                //    pivot_row, pivot_col, self
                //);
                pivot_row += 1;
            }
            pivot_col += 1;
        }
        self.clone()
    }
    pub fn gaussian_elimination(&mut self) -> Self {
        let mut pivot_row = 0;
        let mut pivot_col = 0;
        while pivot_row < self.rows && pivot_col < self.cols {
            if let Some(i) = (pivot_row..self.rows)
                .position(|r| self.data[r * self.cols + pivot_col] != K::zero())
            {
                self.swap_rows(pivot_row, pivot_row + i);

                for l in (pivot_row + 1)..self.rows {
                    let factor = self.data[l * self.cols + pivot_col]
                        / self.data[pivot_row * self.cols + pivot_col];
                    for m in pivot_col..self.cols {
                        let subtrahend = factor * self.data[pivot_row * self.cols + m];
                        self.data[l * self.cols + m] = self.data[l * self.cols + m] - subtrahend;
                    }
                }
                pivot_row += 1;
            }
            pivot_col += 1;
        }
        self.clone()
    }
    pub fn determinant(&mut self) -> K {
        assert_eq!(
            self.cols, self.rows,
            "Matrix size must NxN at Matrix::determinant()!"
        );
        // Swap rows --> Multiply determinant with -1;
        // Multiply a row with k --> Multiply determinant with k;
        // Adding a multiple of one row to another --> Doesnt effect the determinant;
        let mut det: K = K::one();
        let mut mat = self.clone();
        let mut swap_count = 0;

        for pivot_row in 0..mat.rows {
            let mut pivot = pivot_row;
            while pivot < mat.rows && mat.data[pivot * mat.cols + pivot_row] == K::zero() {
                pivot += 1;
            }

            if pivot == mat.rows {
                return K::zero();
            }

            if pivot != pivot_row {
                mat.swap_rows(pivot_row, pivot);
                swap_count += 1;
            }

            let pivot_val = mat.data[pivot_row * mat.cols + pivot_row];
            for r in (pivot_row + 1)..mat.cols {
                let factor = mat.data[r * mat.cols + pivot_row] / pivot_val;
                for c in pivot_row..mat.cols {
                    let idx = r * mat.cols + c;
                    mat.data[idx] = mat.data[idx] - factor * mat.data[pivot_row * mat.cols + c];
                }
            }
        }
        for i in 0..mat.cols {
            det = det * mat.data[i * self.cols + i];
        }

        if swap_count % 2 == 1 {
            det = K::zero() - det;
        }

        det
    }
}

pub fn angle_cos<K: Float>(u: &Vector<K>, v: &Vector<K>) -> f32 {
    assert_eq!(u.size(), v.size(), "Size mismatch at angle_cos()!");
    let u_norm = u.clone().norm();
    let v_norm = v.clone().norm();
    assert!(
        u_norm != 0.0f32 && v_norm != 0.0f32,
        "Zero vector at angle_cos()!"
    );
    let dot_pro: f32 = NumCast::from(u.dot(v.clone())).unwrap_or(0.0f32);
    dot_pro / (u_norm * v_norm)
}

pub fn cross_product<K: Float>(u: &Vector<K>, v: &Vector<K>) -> Vector<K> {
    assert!(
        u.size() == 3 && v.size() == 3,
        "Vectors must be 3 dimensional at cross_product()!"
    );
    let mut cross = Vector::new(vec![K::zero(); 3]);
    cross.data[0] = u.data[1].mul_add(v.data[2], -(u.data[2] * v.data[1]));
    cross.data[1] = u.data[2].mul_add(v.data[0], -(u.data[0] * v.data[2]));
    cross.data[2] = u.data[0].mul_add(v.data[1], -(u.data[1] * v.data[0]));
    cross
}

pub fn linear_combination<K: Float>(u: &[Vector<K>], coefs: &[K]) -> Vector<K> {
    assert_eq!(
        u.len(),
        coefs.len(),
        "Size mismatch at Vector::linear_combination()!"
    );
    if u.is_empty() {
        return Vector { data: Vec::new() };
    }
    let n = u[0].size();
    for vec in u.iter() {
        assert_eq!(
            vec.size(),
            n,
            "Not all the vectors are in same size Vector::linear_combination()!"
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
        assert_eq!(u.size(), v.size(), "Size mismatch at lerp(Vector)!");
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
        assert!(
            u.rows == v.rows && u.cols == v.cols,
            "Shape mismatch at lerp(Matrix)!"
        );
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
        assert!((b - 27.3f64).abs() < 1e-6);
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

    // ---------------- norms ----------------

    // Small helper for approximate equality
    fn approx_eq(a: f32, b: f32, eps: f32) -> bool {
        (a - b).abs() <= eps
    }

    #[test]
    fn norm_empty_returns_zero() {
        let mut v = Vector::<f32>::new(vec![]);
        assert_eq!(v.norm_1(), 0.0);
        assert_eq!(v.norm(), 0.0);
        assert_eq!(v.norm_inf(), 0.0);
    }

    #[test]
    fn norm_examples_from_subject() {
        // v = [0, 0, 0]
        let mut v0 = Vector::from([0.0f32, 0.0, 0.0]);
        assert_eq!(v0.norm_1(), 0.0);
        assert_eq!(v0.norm(), 0.0);
        assert_eq!(v0.norm_inf(), 0.0);

        // v = [1, 2, 3]
        let mut v123 = Vector::from([1.0f32, 2.0, 3.0]);
        let n1 = v123.norm_1();
        let n2 = v123.norm();
        let ni = v123.norm_inf();

        assert!(approx_eq(n1, 6.0, 1e-6), "norm_1 expected 6.0, got {}", n1);
        assert!(
            approx_eq(n2, 3.7416575, 1e-5),
            "norm expected ~3.7416575, got {}",
            n2
        );
        assert!(
            approx_eq(ni, 3.0, 1e-6),
            "norm_inf expected 3.0, got {}",
            ni
        );

        // v = [-1, -2]
        let mut vm = Vector::from([-1.0f32, -2.0]);
        let n1m = vm.norm_1();
        let n2m = vm.norm();
        let nim = vm.norm_inf();

        assert!(
            approx_eq(n1m, 3.0, 1e-6),
            "norm_1 expected 3.0, got {}",
            n1m
        );
        assert!(
            approx_eq(n2m, 2.2360679, 1e-6),
            "norm expected ~2.2360679, got {}",
            n2m
        );
        assert!(
            approx_eq(nim, 2.0, 1e-6),
            "norm_inf expected 2.0, got {}",
            nim
        );
    }

    #[test]
    fn norm_mixed_signs_and_zeros() {
        let mut v = Vector::from([0.0f32, -4.0, 2.0, 0.0, 1.0]);
        assert!(approx_eq(v.norm_1(), 7.0, 1e-6));
        assert!(approx_eq(v.norm_inf(), 4.0, 1e-6));

        let n2 = v.norm();
        // sqrt( (-4)^2 + 2^2 + 1^2 ) = sqrt(16 + 4 + 1) = sqrt(21) ≈ 4.5825758
        assert!(
            approx_eq(n2, 4.5825758, 1e-5),
            "norm expected ~4.5825758, got {}",
            n2
        );
    }

    #[test]
    fn norm_homogeneity_scaling() {
        let mut v = Vector::from([1.0f32, -2.0, 3.0]);
        let a = 2.5f32;

        let n1 = v.norm_1();
        let n2 = v.norm();
        let ni = v.norm_inf();

        // Scale the vector manually: a * v
        let mut sv = v.clone();
        for x in sv.data.iter_mut() {
            *x *= a;
        }

        let eps = 1e-5;
        assert!(approx_eq(sv.norm_1(), a.abs() * n1, eps));
        assert!(approx_eq(sv.norm(), a.abs() * n2, eps));
        assert!(approx_eq(sv.norm_inf(), a.abs() * ni, eps));
    }

    #[test]
    fn norm_inequality_chain() {
        // For any v: ||v||_inf <= ||v||_2 <= ||v||_1
        let mut v = Vector::from([3.0f32, -1.0, 2.0, -4.0]);
        let n1 = v.norm_1();
        let n2 = v.norm();
        let ni = v.norm_inf();

        assert!(ni <= n2 + 1e-7, "Linf <= L2 violated: {} !<= {}", ni, n2);
        assert!(n2 <= n1 + 1e-7, "L2 <= L1 violated: {} !<= {}", n2, n1);
    }

    #[test]
    fn norm_triangle_inequality() {
        // ||u + v|| <= ||u|| + ||v||
        let mut u = Vector::from([1.0f32, -2.0, 3.0]);
        let mut v = Vector::from([-4.0f32, 2.0, 0.5]);

        // u + v
        let mut s = u.clone();
        assert_eq!(s.size(), v.size());
        for (x, &y) in s.data.iter_mut().zip(v.data.iter()) {
            *x += y;
        }

        let lhs1 = s.norm_1();
        let rhs1 = u.norm_1() + v.norm_1();
        assert!(
            lhs1 <= rhs1 + 1e-6,
            "Triangle (L1) failed: {} !<= {}",
            lhs1,
            rhs1
        );

        let lhs2 = s.norm();
        let rhs2 = u.norm() + v.norm();
        assert!(
            lhs2 <= rhs2 + 1e-6,
            "Triangle (L2) failed: {} !<= {}",
            lhs2,
            rhs2
        );

        let lhs_inf = s.norm_inf();
        let rhs_inf = u.norm_inf() + v.norm_inf();
        assert!(
            lhs_inf <= rhs_inf + 1e-6,
            "Triangle (Linf) failed: {} !<= {}",
            lhs_inf,
            rhs_inf
        );
    }

    #[test]
    fn norm_large_values_stability() {
        // Just ensure no NaN/Inf and monotone chain still holds
        let mut v = Vector::from([1.0e10f32, -2.0e10, 3.0e10, -4.0e10]);
        let n1 = v.norm_1();
        let n2 = v.norm();
        let ni = v.norm_inf();

        assert!(
            n1.is_finite() && n2.is_finite() && ni.is_finite(),
            "norm produced non-finite value(s)"
        );
        assert!(ni <= n2 + 1e-3, "Linf <= L2 violated at large scale");
        assert!(n2 <= n1 + 1e-3, "L2 <= L1 violated at large scale");
    }

    // ---------------- angle_cos ----------------

    #[test]
    fn angle_cos_parallel_gives_one() {
        let u = Vector::from([1.0f32, 0.0]);
        let v = Vector::from([2.0f32, 0.0]); // same direction
        let c = angle_cos(&u, &v);
        println!("angle_cos parallel = {}", c);
        assert!(approx_eq(c, 1.0, 1e-6));
    }

    #[test]
    fn angle_cos_perpendicular_gives_zero() {
        let u = Vector::from([1.0f32, 0.0]);
        let v = Vector::from([0.0f32, 1.0]);
        let c = angle_cos(&u, &v);
        println!("angle_cos perpendicular = {}", c);
        assert!(approx_eq(c, 0.0, 1e-6));
    }

    #[test]
    fn angle_cos_opposite_gives_minus_one() {
        let u = Vector::from([-1.0f32, 1.0]);
        let v = Vector::from([1.0f32, -1.0]);
        let c = angle_cos(&u, &v);
        println!("angle_cos opposite = {}", c);
        assert!(approx_eq(c, -1.0, 1e-6));
    }

    #[test]
    fn angle_cos_colinear_scaled_is_one() {
        let u = Vector::from([2.0f32, 1.0]);
        let v = Vector::from([4.0f32, 2.0]); // positive scalar multiple
        let c = angle_cos(&u, &v);
        println!("angle_cos colinear scaled = {}", c);
        assert!(approx_eq(c, 1.0, 1e-6));
    }

    #[test]
    fn angle_cos_arbitrary_matches_subject_value() {
        // Subject example: u=[1,2,3], v=[4,5,6] → ~0.974631846
        let u = Vector::from([1.0f32, 2.0, 3.0]);
        let v = Vector::from([4.0f32, 5.0, 6.0]);
        let c = angle_cos(&u, &v);
        println!("angle_cos arbitrary = {}", c);
        assert!(approx_eq(c, 0.974631846, 1e-6));
    }

    #[test]
    #[should_panic(expected = "Zero vector at angle_cos()!")]
    fn angle_cos_zero_vector_panics() {
        let u = Vector::from([0.0f32, 0.0]);
        let v = Vector::from([1.0f32, 0.0]);
        let _ = angle_cos(&u, &v);
    }

    #[test]
    #[should_panic(expected = "Size mismatch at angle_cos()!")]
    fn angle_cos_size_mismatch_panics() {
        let u = Vector::from([1.0f32, 0.0]);
        let v = Vector::from([1.0f32, 0.0, 2.0]); // different size
        let _ = angle_cos(&u, &v);
    }

    // ---------------- cross_product ----------------

    // Small helper for approximate equality on f32

    #[test]
    fn cross_axes_unit_example() {
        // [0,0,1] x [1,0,0] = [0,1,0]
        let u = Vector::from([0.0f32, 0.0, 1.0]);
        let v = Vector::from([1.0f32, 0.0, 0.0]);
        let r = cross_product(&u, &v);

        let exp = Vector::from([0.0f32, 1.0, 0.0]);
        assert_eq!(svec(&r), svec(&exp));
    }

    #[test]
    fn cross_arbitrary_from_subject_1() {
        // [1,2,3] x [4,5,6] = [-3,6,-3]
        let u = Vector::from([1.0f32, 2.0, 3.0]);
        let v = Vector::from([4.0f32, 5.0, 6.0]);
        let r = cross_product(&u, &v);

        let exp = Vector::from([-3.0f32, 6.0, -3.0]);
        assert_eq!(svec(&r), svec(&exp));
    }

    #[test]
    fn cross_arbitrary_from_subject_2() {
        // [4,2,-3] x [-2,-5,16] = [17,-58,-16]
        let u = Vector::from([4.0f32, 2.0, -3.0]);
        let v = Vector::from([-2.0f32, -5.0, 16.0]);
        let r = cross_product(&u, &v);

        let exp = Vector::from([17.0f32, -58.0, -16.0]);
        assert_eq!(svec(&r), svec(&exp));
    }

    #[test]
    fn cross_is_perpendicular_to_operands() {
        let u = Vector::from([1.0f32, 2.0, 3.0]);
        let v = Vector::from([4.0f32, -1.0, 2.0]);
        let r = cross_product(&u, &v);

        // u · (u × v) = 0 and v · (u × v) = 0
        let d1: f32 = num_traits::NumCast::from(u.dot(r.clone())).unwrap();
        let d2: f32 = num_traits::NumCast::from(v.dot(r.clone())).unwrap();
        assert!(approx_eq(d1, 0.0, 1e-5), "u·(u×v) expected 0, got {}", d1);
        assert!(approx_eq(d2, 0.0, 1e-5), "v·(u×v) expected 0, got {}", d2);
    }

    #[test]
    fn cross_is_anticommutative() {
        let u = Vector::from([2.0f32, -3.0, 4.0]);
        let v = Vector::from([1.0f32, 5.0, -2.0]);

        let uv = cross_product(&u, &v);
        let vu = cross_product(&v, &u);

        // vu should be -uv
        let neg_uv = Vector::from([-uv.data[0], -uv.data[1], -uv.data[2]]);
        assert_eq!(svec(&vu), svec(&neg_uv));
    }

    #[test]
    #[should_panic(expected = "Vectors must be 3 dimensional at cross_product()!")]
    fn cross_panics_for_2d_vectors() {
        let u = Vector::from([1.0f32, 2.0]); // not 3D
        let v = Vector::from([3.0f32, 4.0]); // not 3D
        let _ = cross_product(&u, &v);
    }

    #[test]
    #[should_panic(expected = "Vectors must be 3 dimensional at cross_product()!")]
    fn cross_panics_for_4d_vectors() {
        let u = Vector::from([1.0f32, 2.0, 3.0, 4.0]); // not 3D
        let v = Vector::from([0.0f32, 1.0, 0.0, 0.0]); // not 3D
        let _ = cross_product(&u, &v);
    }

    // --- tiny helpers --------------------------------------------------------
    type F = f32;
    fn feq(a: F, b: F, eps: F) -> bool {
        (a - b).abs() <= eps
    }
    fn assert_vec_eq_eps(got: &Vector<F>, want: &[F], eps: F) {
        assert_eq!(got.size(), want.len(), "vector length mismatch");
        for (i, (g, w)) in got.data.iter().zip(want.iter()).enumerate() {
            assert!(feq(*g, *w, eps), "vec[{}]: got {}, want {}", i, g, w);
        }
    }
    fn assert_mat_eq_eps(got: &Matrix<F>, want: &[&[F]], eps: F) {
        assert_eq!(got.rows, want.len(), "rows mismatch");
        assert_eq!(got.cols, want[0].len(), "cols mismatch");
        for i in 0..got.rows {
            for j in 0..got.cols {
                let g = got.data[i * got.cols + j];
                let w = want[i][j];
                assert!(feq(g, w, eps), "mat[{},{}]: got {}, want {}", i, j, g, w);
            }
        }
    }

    // shorthands to build vectors/matrices in row-major for tests
    fn v(xs: &[F]) -> Vector<F> {
        Vector::new(xs.to_vec())
    }
    fn m(rows: usize, cols: usize, data_row_major: &[F]) -> Matrix<F> {
        assert_eq!(rows * cols, data_row_major.len());
        Matrix {
            rows,
            cols,
            data: data_row_major.to_vec(),
        }
    }

    const EPS: F = 1e-5;

    // --- mul_vec -------------------------------------------------------------

    #[test]
    fn mul_vec_identity_2x2() {
        let mut a = m(2, 2, &[1., 0., 0., 1.]);
        let x = v(&[4., 2.]);
        let y = a.mul_vec(x.clone());
        assert_vec_eq_eps(&y, &[4., 2.], EPS);
    }

    #[test]
    fn mul_vec_diagonal_scales() {
        let mut a = m(3, 3, &[2., 0., 0., 0., 3., 0., 0., 0., -1.]);
        let x = v(&[1., 2., -5.]);
        let y = a.mul_vec(x.clone());
        assert_vec_eq_eps(&y, &[2., 6., 5.], EPS);
    }

    #[test]
    fn mul_vec_nonsquare_2x3_times_r3() {
        // A (2x3) * v (3) -> (2)
        let mut a = m(2, 3, &[1., 2., 3., 4., 5., 6.]);
        let x = v(&[7., 8., 9.]);
        let y = a.mul_vec(x.clone());
        assert_vec_eq_eps(&y, &[50., 122.], EPS);
    }

    #[test]
    #[should_panic(expected = "Matrix::mul_vec")]
    fn mul_vec_dimension_mismatch_panics() {
        let mut a = m(2, 3, &[1., 2., 3., 4., 5., 6.]);
        let x = v(&[1., 2.]); // len 2, needs 3
        let _ = a.mul_vec(x.clone());
    }

    // --- mul_mat -------------------------------------------------------------

    #[test]
    fn mul_mat_identity_right() {
        // A * I = A
        let mut a = m(2, 2, &[3., -5., 6., 8.]);
        let i = m(2, 2, &[1., 0., 0., 1.]);
        let c = a.mul_mat(i.clone());
        assert_mat_eq_eps(&c, &[&[3., -5.], &[6., 8.]], EPS);
    }

    #[test]
    fn mul_mat_identity_left() {
        // I * B = B
        let mut i = m(2, 2, &[1., 0., 0., 1.]);
        let b = m(2, 2, &[2., 1., 4., 2.]);
        let c = i.mul_mat(b.clone());
        assert_mat_eq_eps(&c, &[&[2., 1.], &[4., 2.]], EPS);
    }

    #[test]
    fn mul_mat_nonsquare_2x3_times_3x2() {
        // (2x3) * (3x2) -> (2x2)
        let mut a = m(2, 3, &[1., 2., 3., 4., 5., 6.]);
        let b = m(3, 2, &[7., 8., 9., 10., 11., 12.]);
        let c = a.mul_mat(b.clone());
        assert_mat_eq_eps(&c, &[&[58., 64.], &[139., 154.]], EPS);
    }

    #[test]
    fn mul_mat_nonsquare_3x2_times_2x4() {
        // (3x2) * (2x4) -> (3x4)
        let mut a = m(3, 2, &[1., 4., 2., 5., 3., 6.]);
        let b = m(2, 4, &[7., 8., 9., 10., 11., 12., 13., 14.]);
        let c = a.mul_mat(b.clone());
        assert_mat_eq_eps(
            &c,
            &[
                &[51., 56., 61., 66.],
                &[69., 76., 83., 90.],
                &[87., 96., 105., 114.],
            ],
            EPS,
        );
    }

    #[test]
    fn mul_mat_associativity_with_vec() {
        // (A * B) * v == A * (B * v)
        let mut a = m(2, 3, &[1., 2., 0., 0., 1., 1.]);
        let b = m(3, 2, &[2., 1., 3., 0., 4., -1.]);
        let v = v(&[1., 2.]);
        let left = a.mul_mat(b.clone()).mul_vec(v.clone());
        let right = a.mul_vec(b.clone().mul_vec(v.clone()));
        assert_vec_eq_eps(&left, &right.data, EPS);
    }

    #[test]
    fn mul_mat_with_zero_matrix() {
        let mut a = m(2, 3, &[1., 2., 3., 4., 5., 6.]);
        let z = m(3, 4, &[0.; 12]);
        let c = a.mul_mat(z.clone());
        assert_mat_eq_eps(&c, &[&[0., 0., 0., 0.], &[0., 0., 0., 0.]], EPS);
    }

    #[test]
    #[should_panic(expected = "Matrix::mul_mat")]
    fn mul_mat_dimension_mismatch_panics() {
        // cols(A)=3, rows(B)=4 -> should panic
        let mut a = m(2, 3, &[1., 2., 3., 4., 5., 6.]);
        let b = m(4, 2, &[1., 2., 3., 4., 5., 6., 7., 8.]);
        let _ = a.mul_mat(b.clone());
    }

    //----------- trace ----------------------
    // Helper function to create a matrix from a 2D vector (for testing purposes)
    fn create_matrix(data: Vec<Vec<f32>>) -> Matrix<f32> {
        let rows = data.len();
        let cols = data[0].len();
        let mut flattened = Vec::new();
        for row in data {
            flattened.extend(row);
        }
        Matrix {
            data: flattened,
            rows,
            cols,
        }
    }

    #[test]
    fn test_trace_square_matrix() {
        let mut mat = create_matrix(vec![
            vec![1.0, 2.0, 3.0],
            vec![4.0, 5.0, 6.0],
            vec![7.0, 8.0, 9.0],
        ]);
        assert_eq!(mat.trace(), 15.0); // Trace is 1 + 5 + 9 = 15
    }

    #[test]
    fn test_trace_empty_matrix() {
        let mut mat: Matrix<f32> = Matrix::new(vec![], 0, 0);
        assert_eq!(mat.trace(), 0.0); // Empty matrix should return trace 0
    }

    #[test]
    fn test_trace_identity_matrix() {
        let mut mat = create_matrix(vec![
            vec![1.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0],
            vec![0.0, 0.0, 1.0],
        ]);
        assert_eq!(mat.trace(), 3.0); // Trace of identity matrix is n (size of the matrix)
    }

    #[test]
    fn test_trace_negative_elements() {
        let mut mat = create_matrix(vec![
            vec![-1.0, -2.0, -3.0],
            vec![-4.0, -5.0, -6.0],
            vec![-7.0, -8.0, -9.0],
        ]);
        assert_eq!(mat.trace(), -15.0); // Trace is -1 + (-5) + (-9) = -15
    }

    #[test]
    #[should_panic(expected = "Must be NxN matrix to calculate the trace at trace()!")]
    fn test_trace_non_square_matrix() {
        let mut mat = create_matrix(vec![vec![1.0, 2.0], vec![3.0, 4.0], vec![5.0, 6.0]]);
        mat.trace(); // This should panic since it's not a square matrix
    }

    #[test]
    fn test_trace_single_element_matrix() {
        let mut mat = create_matrix(vec![vec![42.0]]);
        assert_eq!(mat.trace(), 42.0); // Trace of a 1x1 matrix is the only element
    }

    //--------- transpose -------------------------
    // Helper function to create a matrix

    // Test 1: Square matrix (3x3)
    #[test]
    fn test_transpose_square_3x3() {
        let mut mat = m(3, 3, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]);
        let transposed = mat.transpose();
        assert_eq!(
            transposed.data,
            vec![1.0, 4.0, 7.0, 2.0, 5.0, 8.0, 3.0, 6.0, 9.0]
        );
        assert_eq!(transposed.rows, 3);
        assert_eq!(transposed.cols, 3);
    }

    // Test 2: Single row matrix (1x5)
    #[test]
    fn test_transpose_single_row() {
        let mut mat = m(1, 5, &[1.0, 2.0, 3.0, 4.0, 5.0]);
        let transposed = mat.transpose();
        assert_eq!(transposed.data, vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        assert_eq!(transposed.rows, 5);
        assert_eq!(transposed.cols, 1);
    }

    // Test 3: Single column matrix (5x1)
    #[test]
    fn test_transpose_single_column() {
        let mut mat = m(5, 1, &[1.0, 2.0, 3.0, 4.0, 5.0]);
        let transposed = mat.transpose();
        assert_eq!(transposed.data, vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        assert_eq!(transposed.rows, 1);
        assert_eq!(transposed.cols, 5);
    }

    // Test 4: Square matrix (4x4)
    #[test]
    fn test_transpose_square_4x4() {
        let mut mat = m(
            4,
            4,
            &[
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0,
                16.0,
            ],
        );
        let transposed = mat.transpose();
        assert_eq!(
            transposed.data,
            vec![
                1.0, 5.0, 9.0, 13.0, 2.0, 6.0, 10.0, 14.0, 3.0, 7.0, 11.0, 15.0, 4.0, 8.0, 12.0,
                16.0
            ]
        );
        assert_eq!(transposed.rows, 4);
        assert_eq!(transposed.cols, 4);
    }

    // Test 5: Rectangular matrix (4x2)
    #[test]
    fn test_transpose_rectangular_4x2() {
        let mut mat = m(4, 2, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
        let transposed = mat.transpose();

        // This is the correct flattened data for the 2x4 transposed matrix
        let expected_data = vec![1.0, 3.0, 5.0, 7.0, 2.0, 4.0, 6.0, 8.0];

        assert_eq!(transposed.data, expected_data);
        assert_eq!(transposed.rows, 2);
        assert_eq!(transposed.cols, 4);
    }

    // Test 6: 1x1 matrix (smallest possible matrix)
    #[test]
    fn test_transpose_1x1() {
        let mut mat = m(1, 1, &[42.0]);
        let transposed = mat.transpose();
        assert_eq!(transposed.data, vec![42.0]);
        assert_eq!(transposed.rows, 1);
        assert_eq!(transposed.cols, 1);
    }

    // Test 7: Matrix with negative values
    #[test]
    fn test_transpose_with_negatives() {
        let mut mat = m(2, 3, &[-1.0, 2.0, -3.0, 4.0, -5.0, 6.0]);
        let transposed = mat.transpose();

        // Correct the expected vector to match the actual correct transpose
        let expected = vec![-1.0, 4.0, 2.0, -5.0, -3.0, 6.0];

        assert_eq!(transposed.data, expected);
        assert_eq!(transposed.rows, 3);
        assert_eq!(transposed.cols, 2);
    }

    // Test 8: Identity matrix (3x3)
    #[test]
    fn test_transpose_identity() {
        let mut mat = m(3, 3, &[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]);
        let transposed = mat.transpose();
        assert_eq!(
            transposed.data,
            vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
        );
        assert_eq!(transposed.rows, 3);
        assert_eq!(transposed.cols, 3);
    }

    // ---------- row-echelon --------------------
    #[test]
    fn test_identity_matrix() {
        let mut mat = Matrix::from([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]);
        let ref_mat = mat.row_echelon();
        assert_eq!(
            ref_mat.data,
            vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
        );
    }

    #[test]
    fn test_simple_2x2() {
        let mut mat = Matrix::from([[2.0, 4.0], [1.0, 3.0]]);
        let ref_mat = mat.row_echelon();
        assert_eq!(ref_mat.data, vec![1.0, 2.0, 0.0, 1.0]);
    }

    #[test]
    fn test_matrix_with_zero_row() {
        let mut mat = Matrix::from([[1.0, 2.0], [0.0, 0.0]]);
        let ref_mat = mat.row_echelon();
        assert_eq!(ref_mat.data, vec![1.0, 2.0, 0.0, 0.0]);
    }

    #[test]
    fn test_rectangular_matrix() {
        let mut mat = Matrix::from([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]);
        let ref_mat = mat.row_echelon();
        println!("Output: \n{}", ref_mat);
        assert_eq!(ref_mat.data, vec![1.0, 2.0, 3.0, 0.0, 1.0, 2.0]);
    }

    #[test]
    fn test_requires_row_swap() {
        let mut mat = Matrix::from([[0.0, 2.0, 3.0], [1.0, 4.0, 5.0]]);
        let ref_mat = mat.row_echelon();
        assert_eq!(ref_mat.data, vec![1.0, 4.0, 5.0, 0.0, 1.0, 1.5]);
    }

    #[test]
    fn test_3x3_example() {
        let mut mat = Matrix::from([[2.0, 1.0, -1.0], [-3.0, -1.0, 2.0], [-2.0, 1.0, 2.0]]);
        let ref_mat = mat.row_echelon();
        let expected = vec![1.0, 0.5, -0.5, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0];
        for (a, b) in ref_mat.data.iter().zip(expected.iter()) {
            assert!((a - b).abs() < 1e-6);
        }
    }

    // ---------- determinant -----------
    #[test]
    fn test_1x1_matrix() {
        let mut mat = Matrix::from([[5.0]]);
        assert_eq!(mat.determinant(), 5.0);

        let mut mat_zero = Matrix::from([[0.0]]);
        assert_eq!(mat_zero.determinant(), 0.0);
    }

    #[test]
    fn test_2x2_matrix() {
        let mut mat = Matrix::from([[1.0, 2.0], [3.0, 4.0]]);
        assert_eq!(mat.determinant(), -2.0);

        let mut mat_singular = Matrix::from([[2.0, 4.0], [1.0, 2.0]]);
        assert_eq!(mat_singular.determinant(), 0.0);
    }

    #[test]
    fn test_3x3_matrix() {
        let mut mat = Matrix::from([[1.0, 2.0, 3.0], [0.0, 1.0, 4.0], [5.0, 6.0, 0.0]]);
        assert_eq!(mat.determinant(), 1.0);

        let mut mat_singular = Matrix::from([[1.0, 2.0, 3.0], [2.0, 4.0, 6.0], [3.0, 6.0, 9.0]]);
        assert_eq!(mat_singular.determinant(), 0.0);
    }

    #[test]
    fn test_4x4_matrix() {
        let mut mat = Matrix::from([
            [1.0, 0.0, 2.0, -1.0],
            [3.0, 0.0, 0.0, 5.0],
            [2.0, 1.0, 4.0, -3.0],
            [1.0, 0.0, 5.0, 0.0],
        ]);
        assert_eq!(mat.determinant(), 30.0);

        let mut mat_singular = Matrix::from([
            [1.0, 2.0, 3.0, 4.0],
            [2.0, 4.0, 6.0, 8.0],
            [3.0, 6.0, 9.0, 12.0],
            [4.0, 8.0, 12.0, 16.0],
        ]);
        assert_eq!(mat_singular.determinant(), 0.0);
    }

    #[test]
    fn test_identity_matrix_2() {
        let mut mat = Matrix::from([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]);
        assert_eq!(mat.determinant(), 1.0);

        let mut mat4 = Matrix::from([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]);
        assert_eq!(mat4.determinant(), 1.0);
    }

    #[test]
    fn test_upper_triangular_matrix() {
        let mut mat = Matrix::from([[2.0, 3.0, 1.0], [0.0, 5.0, 4.0], [0.0, 0.0, 6.0]]);
        // Determinant is the product of diagonal elements: 2*5*6 = 60
        assert_eq!(mat.determinant(), 60.0);
    }

    #[test]
    fn test_lower_triangular_matrix() {
        let mut mat = Matrix::from([[3.0, 0.0, 0.0], [2.0, 1.0, 0.0], [4.0, 5.0, 2.0]]);
        // Determinant is the product of diagonal elements: 3*1*2 = 6
        assert_eq!(mat.determinant(), 6.0);
    }

    #[test]
    fn test_matrix_with_zero_determinant() {
        let mut mat = Matrix::from([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]);
        assert_eq!(mat.determinant(), 0.0); // Singular matrix
    }
}
