use crate::algebra::vector::Vector;

pub trait Lerp: Sized {
    fn lerp(&self, other: &Self, t: f32) -> Self;
}

impl Lerp for Vector {}
