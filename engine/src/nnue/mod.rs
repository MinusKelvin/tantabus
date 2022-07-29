use cozy_chess::*;

mod ops;
mod layers;

use self::layers::*;
use self::ops::*;

const FEATURES: usize = 768;
const FT_OUT: usize = 32;
const L1_OUT: usize = 16;

const ACTIVATION_RANGE: i8 = 127;
const WEIGHT_SCALE: i8 = 64;
const OUTPUT_SCALE: i32 = 203;

#[derive(Debug, Clone)]
pub struct Nnue {
    pub ft: BitLinear<i16, FEATURES, FT_OUT>,
    pub l1: Linear<i8, i32, {FT_OUT * Color::NUM}, L1_OUT>
}

impl Nnue {
    pub const DEFAULT: Self = include!("model.txt");

    pub fn new_state(&self) -> NnueState<'_> {
        let mut accumulator = [[0; FT_OUT]; Color::NUM];
        self.ft.empty(&mut accumulator[Color::White as usize]);
        self.ft.empty(&mut accumulator[Color::Black as usize]);
        NnueState {
            model: self,
            accumulator,
            material: 0,
        }
    }
}

#[derive(Debug, Clone)]
pub struct NnueState<'m> {
    model: &'m Nnue,
    accumulator: [[i16; FT_OUT]; Color::NUM],
    material: usize,
}

pub fn feature(perspective: Color, mut color: Color, piece: Piece, mut square: Square) -> usize {
    if perspective == Color::Black {
        square = square.flip_rank();
        color = !color;
    }
    macro_rules! index {
        ($([$index:expr; $count:expr])*) => {{
            let mut index = 0;
            $(index = index * $count + $index;)*
            index
        }}
    }
    index! {
        [color as usize; Color::NUM]
        [piece as usize; Piece::NUM]
        [square as usize; Square::NUM]
    }
}

const VALUES: [usize; Piece::NUM] = [1, 3, 3, 5, 8, 0];

impl<'s> NnueState<'s> {
    pub fn model(&self) -> &Nnue {
        &self.model
    }

    pub fn accumulator(&self) -> &[[i16; FT_OUT]; Color::NUM] {
        &self.accumulator
    }

    pub fn add(&mut self, color: Color, piece: Piece, square: Square) {
        self.material += VALUES[piece as usize];
        for &perspective in &Color::ALL {
            let feature = feature(perspective, color, piece, square);
            self.model.ft.add(feature, &mut self.accumulator[perspective as usize]);
        }
    }

    pub fn sub(&mut self, color: Color, piece: Piece, square: Square) {
        self.material -= VALUES[piece as usize];
        for &perspective in &Color::ALL {
            let feature = feature(perspective, color, piece, square);
            self.model.ft.sub(feature, &mut self.accumulator[perspective as usize]);
        }
    }

    pub fn evaluate(&self, side_to_move: Color) -> i32 {
        let mut inputs = [[0; FT_OUT]; Color::NUM];
        self.accumulator[side_to_move as usize]
            .clipped_relu(0, ACTIVATION_RANGE, &mut inputs[0]);
        self.accumulator[(!side_to_move) as usize]
            .clipped_relu(0, ACTIVATION_RANGE, &mut inputs[1]);
        let inputs = bytemuck::cast(inputs);
        let bucket = 15.min(self.material * 16 / 76);
        let output = self.model.l1.activate(&inputs, bucket);
        output * OUTPUT_SCALE / WEIGHT_SCALE as i32 / ACTIVATION_RANGE as i32
    }
}
