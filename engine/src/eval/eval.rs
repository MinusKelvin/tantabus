use cozy_chess::*;
use serde::{Serialize, Deserialize};

use super::Eval;
use super::pst::*;
use super::mob::*;
use super::trace::*;
use super::phased_eval::*;
use super::eval_consts::EVAL_WEIGHTS;

#[derive(Default, Debug, Clone, Serialize, Deserialize)]
pub struct EvalTerms<E> {
    pub piece_tables: PstEvalSet<E>,
    pub mobility: Mobility<E>,
    pub virtual_queen_mobility: [E; 28],
    pub passed_pawns: KingRelativePst<E>,
    pub bishop_pair: E,
    pub rook_on_open_file: E,
    pub rook_on_semiopen_file: E
}

pub type EvalTrace = EvalTerms<i16>;
pub type EvalWeights = EvalTerms<PhasedEval>;

pub const MAX_PHASE: u32 = 256;

// CITE: This way of calculating the game phase was apparently done in Fruit.
// https://www.chessprogramming.org/Tapered_Eval#Implementation_example
pub fn game_phase(board: &Board) -> u32 {
    macro_rules! game_phase_fn {
        ($($piece:ident=$weight:expr,$count:expr;)*) => {
            const INIT_PHASE: u32 = (0 $( + $count * $weight)*) * 2;
            let inv_phase = 0 $( + board.pieces(Piece::$piece).popcnt() * $weight)*;
            let phase = INIT_PHASE.saturating_sub(inv_phase); //Early promotions
            (phase * MAX_PHASE + (INIT_PHASE / 2)) / INIT_PHASE
        }
    }
    game_phase_fn! {
        Pawn   = 0, 8;
        Knight = 1, 2;
        Bishop = 1, 2;
        Rook   = 2, 2;
        Queen  = 4, 1;
    }
}

fn sign(color: Color) -> i16 {
    if color == Color::White { 1 } else { -1 }
}

pub fn evaluate(board: &Board) -> Eval {
    EvalContext {
        board,
        trace: &mut (),
        weights: &EVAL_WEIGHTS
    }.eval()
}

pub fn evaluate_with_weights_and_trace(board: &Board, weights: &EvalWeights) -> (Eval, EvalTrace) {
    let mut trace = EvalTrace::default();
    let eval = EvalContext {
        board,
        trace: &mut trace,
        weights
    }.eval();
    (eval, trace)
}
struct EvalContext<'c, T> {
    board: &'c Board,
    trace: &'c mut T,
    weights: &'c EvalTerms<PhasedEval>
}

impl<'c, T: TraceTarget> EvalContext<'c, T> {
    fn eval(&mut self) -> Eval {
        use Color::*;
        let eval =
            (self.psqt_terms(White) - self.psqt_terms(Black)) +
            (self.mobility_terms(White) - self.mobility_terms(Black)) +
            (self.virtual_queen_mobility(White) - self.virtual_queen_mobility(Black)) +
            (self.passed_pawn_terms(White) - self.passed_pawn_terms(Black)) +
            (self.rook_on_open_file_terms(White) - self.rook_on_open_file_terms(Black)) +
            (self.bishop_pair_terms(White) - self.bishop_pair_terms(Black));

        let phase = game_phase(self.board) as i32;
        let interpolated = (
            (eval.0 as i32 * (MAX_PHASE as i32 - phase)) +
            (eval.1 as i32 * phase)
        ) / MAX_PHASE as i32;
        Eval::cp(interpolated as i16 * sign(self.board.side_to_move()))
    }

    fn psqt_terms(&mut self, color: Color) -> PhasedEval {
        let mut eval = PhasedEval::ZERO;
        let our_pieces = self.board.colors(color);
        let our_king = self.board.king(color);
        for &piece in &Piece::ALL {
            let pieces = our_pieces & self.board.pieces(piece);
            for square in pieces {
                self.trace.trace(|terms| {
                    *terms.piece_tables.get_mut(piece, color, our_king, square) += sign(color);
                });
                eval += *self.weights.piece_tables.get(piece, color, our_king, square);
            }
        }
        eval
    }

    fn mobility_terms(&mut self, color: Color) -> PhasedEval {
        let mut eval = PhasedEval::ZERO;
        let our_pieces = self.board.colors(color);
        let occupied = self.board.occupied();
        for &piece in &Piece::ALL {
            let pieces = our_pieces & self.board.pieces(piece);
            let mobility_table = self.weights.mobility.get(piece);
            for square in pieces {
                let approx_moves = match piece {
                    Piece::Pawn => (
                        get_pawn_quiets(square, color, occupied) |
                        (get_pawn_attacks(square, color) & self.board.colors(!color))
                    ),
                    Piece::Knight => get_knight_moves(square) & !our_pieces,
                    Piece::Bishop => get_bishop_moves(square, occupied) & !our_pieces,
                    Piece::Rook => get_rook_moves(square, occupied) & !our_pieces,
                    Piece::Queen => (
                        get_bishop_moves(square, occupied) |
                        get_rook_moves(square, occupied)
                    ) & !our_pieces,
                    Piece::King => get_king_moves(square) & !our_pieces
                };
                let mobility = approx_moves.popcnt() as usize;
                self.trace.trace(|terms| {
                    terms.mobility.get_mut(piece)[mobility] += sign(color);
                });
                eval += mobility_table[mobility];
            }
        }
        eval
    }

    fn virtual_queen_mobility(&mut self, color: Color) -> PhasedEval {
        let occupied = self.board.occupied();
        let our_pieces = self.board.colors(color);
        let our_king = self.board.king(color);
        let approx_queen_moves = (
            get_bishop_moves(our_king, occupied) |
            get_rook_moves(our_king, occupied)
        ) & !our_pieces;
        let mobility = approx_queen_moves.popcnt() as usize;
        self.trace.trace(|terms| {
            terms.virtual_queen_mobility[mobility] += sign(color);
        });
        self.weights.virtual_queen_mobility[mobility]
    }

    fn passed_pawn_terms(&mut self, color: Color) -> PhasedEval {
        let our_pieces = self.board.colors(color);
        let pawns = self.board.pieces(Piece::Pawn);
        let our_pawns = our_pieces & pawns;
        let their_pawns = pawns ^ our_pawns;
        let our_king = self.board.king(color);
        let promotion_rank = Rank::Eighth.relative_to(color);

        let mut eval = PhasedEval::ZERO;
        for pawn in our_pawns {
            let promo_square = Square::new(pawn.file(), promotion_rank);
            let front_span = get_between_rays(pawn, promo_square);
            let mut blocker_mask = front_span;
            for attack in get_pawn_attacks(pawn, color) {
                let telestop = Square::new(attack.file(), promotion_rank);
                let front_span = get_between_rays(attack, telestop);
                blocker_mask |= front_span | attack.bitboard();
            }

            let passed = (their_pawns & blocker_mask).is_empty()
                && (our_pawns & front_span).is_empty();
            if passed {
                self.trace.trace(|terms| {
                    *terms.passed_pawns.get_mut(color, our_king, pawn) += sign(color);
                });
                eval += *self.weights.passed_pawns.get(color, our_king, pawn);
            }
        }
        eval
    }

    fn rook_on_open_file_terms(&mut self, color: Color) -> PhasedEval {
        let our_pieces = self.board.colors(color);
        let pawns = self.board.pieces(Piece::Pawn);
        let our_pawns = our_pieces & pawns;
        let our_rooks = our_pieces & self.board.pieces(Piece::Rook);
        
        let mut eval = PhasedEval::ZERO;
        for rook in our_rooks {
            let file = rook.file();
            let file_bb = file.bitboard();
            if (file_bb & pawns).is_empty() {
                self.trace.trace(|terms| {
                    terms.rook_on_open_file += sign(color);
                });
                eval += self.weights.rook_on_open_file;
            } else if (file_bb & our_pawns).is_empty() {
                self.trace.trace(|terms| {
                    terms.rook_on_semiopen_file += sign(color);
                });
                eval += self.weights.rook_on_semiopen_file;
            }
        }
        eval
    }

    fn bishop_pair_terms(&mut self, color: Color) -> PhasedEval {
        let mut eval = PhasedEval::ZERO;
        let our_pieces = self.board.colors(color);
        if (our_pieces & self.board.pieces(Piece::Bishop)).popcnt() >= 2 {
            self.trace.trace(|terms| {
                terms.bishop_pair += sign(color);
            });
            eval += self.weights.bishop_pair;
        }
        eval
    }
}
