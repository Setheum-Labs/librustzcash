//! The [Groth16] proving system.
//!
//! [Groth16]: https://eprint.iacr.org/2016/260

use group::{CurveAffine, CurveProjective, Group};
use pairing::{Engine, PairingCurveAffine};
use ff::{Field};

use crate::{SynthesisError, Circuit, ConstraintSystem, Index, Variable, LinearCombination};
use crate::domain::{EvaluationDomain, Point};
use crate::multiexp::{multiexp, FullDensity, SourceBuilder, DensityTracker};
use crate::multicore::Worker;

use byteorder::{BigEndian, ReadBytesExt, WriteBytesExt};
use std::io::{self, Read, Write};
use std::sync::Arc;
use futures::Future;

use rand_core::RngCore;
use std::ops::{AddAssign, Neg, MulAssign};

#[cfg(test)]
mod tests;

mod generator;
mod prover;
mod verifier;

pub use self::generator::*;
pub use self::prover::*;
pub use self::verifier::*;

#[derive(Clone)]
pub struct Proof<E: Engine> {
    pub a: E::G1Affine,
    pub b: E::G2Affine,
    pub c: E::G1Affine,
}

impl<E: Engine> PartialEq for Proof<E> {
    fn eq(&self, other: &Self) -> bool {
        self.a == other.a && self.b == other.b && self.c == other.c
    }
}

impl<E: Engine> Proof<E> {
    pub fn write<W: Write>(&self, mut writer: W) -> io::Result<()> {
        writer.write_all(self.a.to_compressed().as_ref())?;
        writer.write_all(self.b.to_compressed().as_ref())?;
        writer.write_all(self.c.to_compressed().as_ref())?;

        Ok(())
    }

    pub fn read<R: Read>(mut reader: R) -> io::Result<Self> {
        let read_g1 = |reader: &mut R| -> io::Result<E::G1Affine> {
            let mut g1_repr = <E::G1Affine as CurveAffine>::Compressed::default();
            reader.read_exact(g1_repr.as_mut())?;

            let affine = E::G1Affine::from_compressed(&g1_repr);
            let affine = if affine.is_some().into() {
                Ok(affine.unwrap())
            } else {
                Err(io::Error::new(io::ErrorKind::InvalidData, "invalid G1"))
            };

            affine.and_then(|e| {
                if e.is_identity().into() {
                    Err(io::Error::new(
                        io::ErrorKind::InvalidData,
                        "point at infinity",
                    ))
                } else {
                    Ok(e)
                }
            })
        };

        let read_g2 = |reader: &mut R| -> io::Result<E::G2Affine> {
            let mut g2_repr = <E::G2Affine as CurveAffine>::Compressed::default();
            reader.read_exact(g2_repr.as_mut())?;

            let affine = E::G2Affine::from_compressed(&g2_repr);
            let affine = if affine.is_some().into() {
                Ok(affine.unwrap())
            } else {
                Err(io::Error::new(io::ErrorKind::InvalidData, "invalid G2"))
            };

            affine.and_then(|e| {
                if e.is_identity().into() {
                    Err(io::Error::new(
                        io::ErrorKind::InvalidData,
                        "point at infinity",
                    ))
                } else {
                    Ok(e)
                }
            })
        };

        let a = read_g1(&mut reader)?;
        let b = read_g2(&mut reader)?;
        let c = read_g1(&mut reader)?;

        Ok(Proof { a, b, c })
    }
}

#[derive(Clone)]
pub struct VerifyingKey<E: Engine> {
    // alpha in g1 for verifying and for creating A/C elements of
    // proof. Never the point at infinity.
    pub alpha_g1: E::G1Affine,

    // beta in g1 and g2 for verifying and for creating B/C elements
    // of proof. Never the point at infinity.
    pub beta_g1: E::G1Affine,
    pub beta_g2: E::G2Affine,

    // gamma in g2 for verifying. Never the point at infinity.
    pub gamma_g2: E::G2Affine,

    // delta in g1/g2 for verifying and proving, essentially the magic
    // trapdoor that forces the prover to evaluate the C element of the
    // proof with only components from the CRS. Never the point at
    // infinity.
    pub delta_g1: E::G1Affine,
    pub delta_g2: E::G2Affine,

    // Elements of the form (beta * u_i(tau) + alpha v_i(tau) + w_i(tau)) / gamma
    // for all public inputs. Because all public inputs have a dummy constraint,
    // this is the same size as the number of inputs, and never contains points
    // at infinity.
    pub ic: Vec<E::G1Affine>,
}

impl<E: Engine> PartialEq for VerifyingKey<E> {
    fn eq(&self, other: &Self) -> bool {
        self.alpha_g1 == other.alpha_g1
            && self.beta_g1 == other.beta_g1
            && self.beta_g2 == other.beta_g2
            && self.gamma_g2 == other.gamma_g2
            && self.delta_g1 == other.delta_g1
            && self.delta_g2 == other.delta_g2
            && self.ic == other.ic
    }
}

impl<E: Engine> VerifyingKey<E> {
    pub fn write<W: Write>(&self, mut writer: W) -> io::Result<()> {
        writer.write_all(self.alpha_g1.to_uncompressed().as_ref())?;
        writer.write_all(self.beta_g1.to_uncompressed().as_ref())?;
        writer.write_all(self.beta_g2.to_uncompressed().as_ref())?;
        writer.write_all(self.gamma_g2.to_uncompressed().as_ref())?;
        writer.write_all(self.delta_g1.to_uncompressed().as_ref())?;
        writer.write_all(self.delta_g2.to_uncompressed().as_ref())?;
        writer.write_u32::<BigEndian>(self.ic.len() as u32)?;
        for ic in &self.ic {
            writer.write_all(ic.to_uncompressed().as_ref())?;
        }

        Ok(())
    }

    pub fn read<R: Read>(mut reader: R) -> io::Result<Self> {
        let read_g1 = |reader: &mut R| -> io::Result<E::G1Affine> {
            let mut g1_repr = <E::G1Affine as CurveAffine>::Uncompressed::default();
            reader.read_exact(g1_repr.as_mut())?;

            let affine = E::G1Affine::from_uncompressed(&g1_repr);
            if affine.is_some().into() {
                Ok(affine.unwrap())
            } else {
                Err(io::Error::new(io::ErrorKind::InvalidData, "invalid G1"))
            }
        };

        let read_g2 = |reader: &mut R| -> io::Result<E::G2Affine> {
            let mut g2_repr = <E::G2Affine as CurveAffine>::Uncompressed::default();
            reader.read_exact(g2_repr.as_mut())?;

            let affine = E::G2Affine::from_uncompressed(&g2_repr);
            if affine.is_some().into() {
                Ok(affine.unwrap())
            } else {
                Err(io::Error::new(io::ErrorKind::InvalidData, "invalid G2"))
            }
        };

        let alpha_g1 = read_g1(&mut reader)?;
        let beta_g1 = read_g1(&mut reader)?;
        let beta_g2 = read_g2(&mut reader)?;
        let gamma_g2 = read_g2(&mut reader)?;
        let delta_g1 = read_g1(&mut reader)?;
        let delta_g2 = read_g2(&mut reader)?;

        let ic_len = reader.read_u32::<BigEndian>()? as usize;

        let mut ic = vec![];

        for _ in 0..ic_len {
            let g1 = read_g1(&mut reader).and_then(|e| {
                if e.is_identity().into() {
                    Err(io::Error::new(
                        io::ErrorKind::InvalidData,
                        "point at infinity",
                    ))
                } else {
                    Ok(e)
                }
            })?;

            ic.push(g1);
        }

        Ok(VerifyingKey {
            alpha_g1,
            beta_g1,
            beta_g2,
            gamma_g2,
            delta_g1,
            delta_g2,
            ic,
        })
    }
}

#[derive(Clone)]
pub struct Parameters<E: Engine> {
    pub vk: VerifyingKey<E>,

    // Elements of the form ((tau^i * t(tau)) / delta) for i between 0 and
    // m-2 inclusive. Never contains points at infinity.
    pub h: Arc<Vec<E::G1Affine>>,

    // Elements of the form (beta * u_i(tau) + alpha v_i(tau) + w_i(tau)) / delta
    // for all auxiliary inputs. Variables can never be unconstrained, so this
    // never contains points at infinity.
    pub l: Arc<Vec<E::G1Affine>>,

    // QAP "A" polynomials evaluated at tau in the Lagrange basis. Never contains
    // points at infinity: polynomials that evaluate to zero are omitted from
    // the CRS and the prover can deterministically skip their evaluation.
    pub a: Arc<Vec<E::G1Affine>>,

    // QAP "B" polynomials evaluated at tau in the Lagrange basis. Needed in
    // G1 and G2 for C/B queries, respectively. Never contains points at
    // infinity for the same reason as the "A" polynomials.
    pub b_g1: Arc<Vec<E::G1Affine>>,
    pub b_g2: Arc<Vec<E::G2Affine>>,
}

impl<E: Engine> PartialEq for Parameters<E> {
    fn eq(&self, other: &Self) -> bool {
        self.vk == other.vk
            && self.h == other.h
            && self.l == other.l
            && self.a == other.a
            && self.b_g1 == other.b_g1
            && self.b_g2 == other.b_g2
    }
}

fn read_g1<R: Read, E: Engine>(reader: &mut R, checked: bool) -> io::Result<E::G1Affine> {
    let mut repr = <E::G1Affine as CurveAffine>::Uncompressed::default();
    reader.read_exact(repr.as_mut())?;

    let affine = if checked {
        E::G1Affine::from_uncompressed(&repr)
    } else {
        E::G1Affine::from_uncompressed_unchecked(&repr)
    };

    let affine = if affine.is_some().into() {
        Ok(affine.unwrap())
    } else {
        Err(io::Error::new(io::ErrorKind::InvalidData, "invalid G1"))
    };

    affine.and_then(|e| {
        if e.is_identity().into() {
            Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "point at infinity",
            ))
        } else {
            Ok(e)
        }
    })
}


fn read_g2<R: Read, E: Engine>(reader: &mut R, checked: bool) -> io::Result<E::G2Affine>  {
    let mut repr = <E::G2Affine as CurveAffine>::Uncompressed::default();
    reader.read_exact(repr.as_mut())?;

    let affine = if checked {
        E::G2Affine::from_uncompressed(&repr)
    } else {
        E::G2Affine::from_uncompressed_unchecked(&repr)
    };

    let affine = if affine.is_some().into() {
        Ok(affine.unwrap())
    } else {
        Err(io::Error::new(io::ErrorKind::InvalidData, "invalid G2"))
    };

    affine.and_then(|e| {
        if e.is_identity().into() {
            Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "point at infinity",
            ))
        } else {
            Ok(e)
        }
    })
}

impl<E: Engine> Parameters<E> {
    pub fn write<W: Write>(&self, mut writer: W) -> io::Result<()> {
        self.vk.write(&mut writer)?;

        writer.write_u32::<BigEndian>(self.h.len() as u32)?;
        for g in &self.h[..] {
            writer.write_all(g.to_uncompressed().as_ref())?;
        }

        writer.write_u32::<BigEndian>(self.l.len() as u32)?;
        for g in &self.l[..] {
            writer.write_all(g.to_uncompressed().as_ref())?;
        }

        writer.write_u32::<BigEndian>(self.a.len() as u32)?;
        for g in &self.a[..] {
            writer.write_all(g.to_uncompressed().as_ref())?;
        }

        writer.write_u32::<BigEndian>(self.b_g1.len() as u32)?;
        for g in &self.b_g1[..] {
            writer.write_all(g.to_uncompressed().as_ref())?;
        }

        writer.write_u32::<BigEndian>(self.b_g2.len() as u32)?;
        for g in &self.b_g2[..] {
            writer.write_all(g.to_uncompressed().as_ref())?;
        }

        Ok(())
    }

    pub fn read<R: Read>(mut reader: R, checked: bool) -> io::Result<Self> {
        let vk = VerifyingKey::<E>::read(&mut reader)?;

        let mut h = vec![];
        let mut l = vec![];
        let mut a = vec![];
        let mut b_g1 = vec![];
        let mut b_g2 = vec![];

        {
            let len = reader.read_u32::<BigEndian>()? as usize;
            for _ in 0..len {
                h.push(read_g1::<R, E>(&mut reader, checked)?);
            }
        }

        {
            let len = reader.read_u32::<BigEndian>()? as usize;
            for _ in 0..len {
                l.push(read_g1::<R, E>(&mut reader, checked)?);
            }
        }

        {
            let len = reader.read_u32::<BigEndian>()? as usize;
            for _ in 0..len {
                a.push(read_g1::<R, E>(&mut reader, checked)?);
            }
        }

        {
            let len = reader.read_u32::<BigEndian>()? as usize;
            for _ in 0..len {
                b_g1.push(read_g1::<R, E>(&mut reader, checked)?);
            }
        }

        {
            let len = reader.read_u32::<BigEndian>()? as usize;
            for _ in 0..len {
                b_g2.push(read_g2::<R, E>(&mut reader, checked)?);
            }
        }

        Ok(Parameters {
            vk,
            h: Arc::new(h),
            l: Arc::new(l),
            a: Arc::new(a),
            b_g1: Arc::new(b_g1),
            b_g2: Arc::new(b_g2),
        })
    }
}

pub struct PreparedVerifyingKey<E: Engine> {
    /// Pairing result of alpha*beta
    alpha_g1_beta_g2: E::Fqk,
    /// -gamma in G2
    neg_gamma_g2: <E::G2Affine as PairingCurveAffine>::Prepared,
    /// -delta in G2
    neg_delta_g2: <E::G2Affine as PairingCurveAffine>::Prepared,
    /// Copy of IC from `VerifiyingKey`.
    ic: Vec<E::G1Affine>,
}

pub trait ParameterSource<E: Engine> {
    type G1Builder: SourceBuilder<E::G1Affine>;
    type G2Builder: SourceBuilder<E::G2Affine>;

    fn get_vk(&mut self, num_ic: usize) -> Result<VerifyingKey<E>, SynthesisError>;
    fn get_h(&mut self, num_h: usize) -> Result<Self::G1Builder, SynthesisError>;
    fn get_l(&mut self, num_l: usize) -> Result<Self::G1Builder, SynthesisError>;
    fn get_a(
        &mut self,
        num_inputs: usize,
        num_aux: usize,
    ) -> Result<(Self::G1Builder, Self::G1Builder), SynthesisError>;
    fn get_b_g1(
        &mut self,
        num_inputs: usize,
        num_aux: usize,
    ) -> Result<(Self::G1Builder, Self::G1Builder), SynthesisError>;
    fn get_b_g2(
        &mut self,
        num_inputs: usize,
        num_aux: usize,
    ) -> Result<(Self::G2Builder, Self::G2Builder), SynthesisError>;
}

impl<'a, E: Engine> ParameterSource<E> for &'a Parameters<E> {
    type G1Builder = (Arc<Vec<E::G1Affine>>, usize);
    type G2Builder = (Arc<Vec<E::G2Affine>>, usize);

    fn get_vk(&mut self, _: usize) -> Result<VerifyingKey<E>, SynthesisError> {
        Ok(self.vk.clone())
    }

    fn get_h(&mut self, _: usize) -> Result<Self::G1Builder, SynthesisError> {
        Ok((self.h.clone(), 0))
    }

    fn get_l(&mut self, _: usize) -> Result<Self::G1Builder, SynthesisError> {
        Ok((self.l.clone(), 0))
    }

    fn get_a(
        &mut self,
        num_inputs: usize,
        _: usize,
    ) -> Result<(Self::G1Builder, Self::G1Builder), SynthesisError> {
        Ok(((self.a.clone(), 0), (self.a.clone(), num_inputs)))
    }

    fn get_b_g1(
        &mut self,
        num_inputs: usize,
        _: usize,
    ) -> Result<(Self::G1Builder, Self::G1Builder), SynthesisError> {
        Ok(((self.b_g1.clone(), 0), (self.b_g1.clone(), num_inputs)))
    }

    fn get_b_g2(
        &mut self,
        num_inputs: usize,
        _: usize,
    ) -> Result<(Self::G2Builder, Self::G2Builder), SynthesisError> {
        Ok(((self.b_g2.clone(), 0), (self.b_g2.clone(), num_inputs)))
    }
}

/// This is our assembly structure that we'll use to synthesize the
/// circuit into a QAP.
pub struct KeypairAssembly<E: Engine> {
    num_inputs: usize,
    num_aux: usize,
    num_constraints: usize,
    at_inputs: Vec<Vec<(E::Fr, usize)>>,
    bt_inputs: Vec<Vec<(E::Fr, usize)>>,
    ct_inputs: Vec<Vec<(E::Fr, usize)>>,
    at_aux: Vec<Vec<(E::Fr, usize)>>,
    bt_aux: Vec<Vec<(E::Fr, usize)>>,
    ct_aux: Vec<Vec<(E::Fr, usize)>>,
}

impl<E: Engine> ConstraintSystem<E> for KeypairAssembly<E> {
    type Root = Self;

    fn alloc<F, A, AR>(&mut self, _: A, _: F) -> Result<Variable, SynthesisError>
        where
            F: FnOnce() -> Result<E::Fr, SynthesisError>,
            A: FnOnce() -> AR,
            AR: Into<String>,
    {
        // There is no assignment, so we don't even invoke the
        // function for obtaining one.

        let index = self.num_aux;
        self.num_aux += 1;

        self.at_aux.push(vec![]);
        self.bt_aux.push(vec![]);
        self.ct_aux.push(vec![]);

        Ok(Variable(Index::Aux(index)))
    }

    fn alloc_input<F, A, AR>(&mut self, _: A, _: F) -> Result<Variable, SynthesisError>
        where
            F: FnOnce() -> Result<E::Fr, SynthesisError>,
            A: FnOnce() -> AR,
            AR: Into<String>,
    {
        // There is no assignment, so we don't even invoke the
        // function for obtaining one.

        let index = self.num_inputs;
        self.num_inputs += 1;

        self.at_inputs.push(vec![]);
        self.bt_inputs.push(vec![]);
        self.ct_inputs.push(vec![]);

        Ok(Variable(Index::Input(index)))
    }

    fn enforce<A, AR, LA, LB, LC>(&mut self, _: A, a: LA, b: LB, c: LC)
        where
            A: FnOnce() -> AR,
            AR: Into<String>,
            LA: FnOnce(LinearCombination<E>) -> LinearCombination<E>,
            LB: FnOnce(LinearCombination<E>) -> LinearCombination<E>,
            LC: FnOnce(LinearCombination<E>) -> LinearCombination<E>,
    {
        fn eval<E: Engine>(
            l: LinearCombination<E>,
            inputs: &mut [Vec<(E::Fr, usize)>],
            aux: &mut [Vec<(E::Fr, usize)>],
            this_constraint: usize,
        ) {
            for (index, coeff) in l.0 {
                match index {
                    Variable(Index::Input(id)) => inputs[id].push((coeff, this_constraint)),
                    Variable(Index::Aux(id)) => aux[id].push((coeff, this_constraint)),
                }
            }
        }

        eval(
            a(LinearCombination::zero()),
            &mut self.at_inputs,
            &mut self.at_aux,
            self.num_constraints,
        );
        eval(
            b(LinearCombination::zero()),
            &mut self.bt_inputs,
            &mut self.bt_aux,
            self.num_constraints,
        );
        eval(
            c(LinearCombination::zero()),
            &mut self.ct_inputs,
            &mut self.ct_aux,
            self.num_constraints,
        );

        self.num_constraints += 1;
    }

    fn push_namespace<NR, N>(&mut self, _: N)
        where
            NR: Into<String>,
            N: FnOnce() -> NR,
    {
        // Do nothing; we don't care about namespaces in this context.
    }

    fn pop_namespace(&mut self) {
        // Do nothing; we don't care about namespaces in this context.
    }

    fn get_root(&mut self) -> &mut Self::Root {
        self
    }
}

pub struct ExtendedParameters<E: Engine> {
    pub params: Parameters<E>,

    pub taus_g1: Vec<E::G1Affine>,
    pub taus_g2: Vec<E::G2Affine>,
}

impl<E: Engine> PartialEq for ExtendedParameters<E> {
    fn eq(&self, other: &Self) -> bool {
        self.params == other.params
            && self.taus_g1 == other.taus_g1
            && self.taus_g2 == other.taus_g2
    }
}

impl<E: Engine> ExtendedParameters<E> {

    // Checks the CRS for possible subversion by the malicious generator. It does not guarantee subversion soundness,
    // meaning that the generator can still use the trapdoor to produce valid proofs of false statements,
    // but does guarantee subversion zero-knowledgeness, so a proof generated by an honest prover will not reveal any information about the witness.
    // This is useful in the case, when the verifier plays the role of the generator and passes the CRS to the prover, who runs this check against it.
    // Then the verifier can be sure in the soundness as only it knows the trapdoor, and the prover is given it's privacy.
    // Follows the procedure from Georg Fuchsbauer, Subversion-zero-knowledge SNARKs (https://eprint.iacr.org/2017/587), p. 26
    pub fn verify<C: Circuit<E>, R: RngCore>(&self, circuit: C, rng: &mut R) -> Result<(), SynthesisError> {
        assert_eq!(self.taus_g1.len(), self.taus_g2.len());
        // generator points
        let g1 = self.taus_g1[0];
        let g2 = self.taus_g2[0];

        let d = self.taus_g1.len() - 1;
        let worker = Worker::new();

        let pvk = prepare_verifying_key(&self.params.vk);

        // https://eprint.iacr.org/2017/587, p. 26

        // 1
        // P1 != 0
        if g1.is_identity().into() {
            return Err(SynthesisError::MalformedCrs);
        }
        // P2 != 0
        if g2.is_identity().into() {
            return Err(SynthesisError::MalformedCrs);
        }

        // 2
        // pk_alpha != 0
        if self.params.vk.alpha_g1.is_identity().into() {
            return Err(SynthesisError::MalformedCrs);
        }
        // pk_beta != 0
        if self.params.vk.beta_g1.is_identity().into() {
            return Err(SynthesisError::MalformedCrs);
        }
        // vk'_gamma != 0
        if self.params.vk.gamma_g2.is_identity().into() {
            return Err(SynthesisError::MalformedCrs);
        }
        // pk_delta != 0
        if self.params.vk.delta_g1.is_identity().into() {
            return Err(SynthesisError::MalformedCrs);
        }
        // pk_{Z,0} = t(tau)/delta != 0
        if self.params.h[0].is_identity().into() {
            return Err(SynthesisError::MalformedCrs);
        }
        // btw, nondegeneracy of beta and delta in G2 follows from the checks in #4

        // 4
        // e(P1, pk'_beta) = e(pk_beta, P2)
        if E::pairing(g1, self.params.vk.beta_g2) != E::pairing(self.params.vk.beta_g1, g2) {
            return Err(SynthesisError::MalformedCrs);
        }
        // e(P1, pk'_delta) = e(pk_delta, P2)
        if E::pairing(g1, self.params.vk.delta_g2) != E::pairing(self.params.vk.delta_g1, g2) {
            return Err(SynthesisError::MalformedCrs);
        }

        // The following 2 blocks perform checks 3 (powers of tau) and 4 part 2 (H-query)
        // using small exponent test to batch pairing computations. The formulas are in
        // https://hackmd.io/OF8ERbVkSI6kh46WTXOlOw though the notation may not match
        // TODO: permalink

        {
            let h_query = start_timer!(|| "H-query validation");

            let mut r = vec![];
            r.resize_with(d, || { E::Fr::random(rng) });
            let r = Arc::new(r);

            let taus_g2 = Arc::new(self.taus_g2.clone().into_iter().take(d).collect::<Vec<_>>()); // tau^0, ..., tau^(d-1) in G2
            let taus_g2_shifted = Arc::new(self.taus_g2.clone().into_iter().skip(1).collect()); // tau^1, ..., tau^d in G2

            assert_eq!(self.params.h.len(), d);
            assert_eq!(taus_g2.len(), d);
            assert_eq!(r.len(), d);

            let acc_h_g1: E::G1 = multiexp(&worker, (self.params.h.clone(), 0), FullDensity, r.clone()).wait().unwrap();
            let acc_taus_g2: E::G2 = multiexp(&worker, (taus_g2, 0), FullDensity, r.clone()).wait().unwrap();
            let acc_taus_g2_shifted: E::G2 = multiexp(&worker, (taus_g2_shifted, 0), FullDensity, r).wait().unwrap();

            // The vanishing polynomial is z(X) = X^{d+1} - 1 for our domain, where d+1 is domain size
            let res = E::final_exponentiation(&E::miller_loop(
                [
                    (&acc_h_g1.to_affine().prepare(), &pvk.neg_delta_g2),
                    (&g1.neg().prepare(), &acc_taus_g2.to_affine().prepare()),
                    (&self.taus_g1[d].prepare(), &acc_taus_g2_shifted.to_affine().prepare())
                ].iter()
            )).unwrap();
            if res != E::Fqk::one() {
                return Err(SynthesisError::MalformedCrs);
            }
            end_timer!(h_query);
        }

        {
            let taus_validation = start_timer!(|| "Powers of tau validation");

            let mut p = vec![];
            let mut q = vec![];

            // TODO: 128-bit scalar multiexps
            p.resize_with(d, || { E::Fr::random(rng) });
            q.resize_with(d, || { E::Fr::random(rng) });

            let mut pq = p.clone();
            for (p, q) in pq.iter_mut().zip(q.iter()) {
                p.add_assign(q);
            }

            let p = Arc::new(p);
            let q = Arc::new(q);
            let pq = Arc::new(pq);

            let bases_pq = Arc::new(self.taus_g1.clone().into_iter().skip(1).collect()); // tau^1, ..., tau^d in G1
            let bases_p = Arc::new(self.taus_g1.clone().into_iter().take(d).collect()); // tau^0, ..., tau^(d-1) in G1
            let bases_q = Arc::new(self.taus_g2.clone().into_iter().skip(1).collect()); // tau^1, ..., tau^d in G2

            let pq_tau_g1: E::G1 = multiexp(&worker, (bases_pq, 0), FullDensity, pq).wait().unwrap();
            let p_tau_g1: E::G1 = multiexp(&worker, (bases_p, 0), FullDensity, p).wait().unwrap();
            let q_tau_g2: E::G2 = multiexp(&worker, (bases_q, 0), FullDensity, q).wait().unwrap();
            //TODO: i guess joining wouldn't help

            let g1 = self.taus_g1[0];
            let neg_g2 = self.taus_g2[0].neg();
            let tau_g2 = self.taus_g2[1];
            let res = E::final_exponentiation(&E::miller_loop(
                [
                    (&pq_tau_g1.to_affine().prepare(), &neg_g2.prepare()),
                    (&p_tau_g1.to_affine().prepare(), &tau_g2.prepare()),
                    (&g1.prepare(), &q_tau_g2.to_affine().prepare())
                ].iter()
            )).unwrap();
            if res != E::Fqk::one() {
                return Err(SynthesisError::MalformedCrs);
            }
            end_timer!(taus_validation);
        }

        // Convert the circuit in R1CS to the QAP in Lagrange base (QAP polynomials evaluations in the roots of unity)
        // The additional input and constraints are Groth16/bellman specific, see the code in generator or prover

        let qap_synthesis = start_timer!(|| "QAP synthesis");
        // TODO: we don't need to distinguish input and auxiliary wires here
        let mut assembly = KeypairAssembly {
            num_inputs: 0,
            num_aux: 0,
            num_constraints: 0,
            at_inputs: vec![],
            bt_inputs: vec![],
            ct_inputs: vec![],
            at_aux: vec![],
            bt_aux: vec![],
            ct_aux: vec![],
        };

        // Allocate the "one" input variable
        assembly.alloc_input(|| "", || Ok(E::Fr::one()))?;

        // Synthesize the circuit.
        circuit.synthesize(&mut assembly)?;

        // Input constraints to ensure full density of IC query
        // x * 0 = 0
        for i in 0..assembly.num_inputs {
            assembly.enforce(|| "", |lc| lc + Variable(Index::Input(i)), |lc| lc, |lc| lc);
        }

        // R1CS -> QAP in Lagrange base
        end_timer!(qap_synthesis);

        // Evaluate the QAP polynomials in point tau in the exponent

        let qap_evaluation = start_timer!(|| "QAP evaluation");
        // The code bellow is borrowed from https://github.com/ebfull/powersoftau/blob/5429415959175082207fd61c10319e47a6b56e87/src/bin/verify.rs#L162-L225
        let worker = Worker::new();

        let mut g1_coeffs = EvaluationDomain::<E, _>::from_coeffs(
            self.taus_g1.iter()
            .map(|e| Point(e.to_projective()))
            .collect()
        ).unwrap(); //TODO: remove Arc?

        let mut g2_coeffs = EvaluationDomain::<E, _>::from_coeffs(
            self.taus_g2.iter()
                .map(|e| Point(e.to_projective()))
                .collect()
        ).unwrap(); //TODO: remove Arc?

        // This converts all of the elements into Lagrange coefficients
        // for later construction of interpolation polynomials

        g1_coeffs.ifft(&worker);
        g2_coeffs.ifft(&worker);
        let g1_coeffs = g1_coeffs.into_coeffs();
        let g2_coeffs = g2_coeffs.into_coeffs();

        // Remove the Point() wrappers
        let mut g1_coeffs = g1_coeffs.into_iter()
            .map(|e| e.0)
            .collect::<Vec<_>>();
        let mut g2_coeffs = g2_coeffs.into_iter()
            .map(|e| e.0)
            .collect::<Vec<_>>();

        // Batch normalize
//        let mut g1_coeffs = vec![E::G1Affine::identity(); g1_coeffs_proj.len()];
//        let mut g2_coeffs = vec![E::G2Affine::identity(); g2_coeffs_proj.len()];
//        E::G1::batch_normalize(&g1_coeffs_proj, &mut g1_coeffs);
//        E::G2::batch_normalize(&g2_coeffs_proj, &mut g2_coeffs);

        // And the following code is adapted from https://github.com/ebfull/phase2/blob/58ebd37d9d25b6779320b0ca99b3c484b679b538/src/lib.rs#L503-L636

        // These are `Arc` so that later it'll be easier
        // to use multiexp during QAP evaluation (which
        // requires a futures-based API)
        let coeffs_g1 = Arc::new(g1_coeffs);
        let coeffs_g2 = Arc::new(g2_coeffs);

        // TODO: simplify KeypairAssembly
        let at  = assembly.at_inputs.into_iter().chain(assembly.at_aux.into_iter()).collect::<Vec<_>>();
        let bt  = assembly.bt_inputs.into_iter().chain(assembly.bt_aux.into_iter()).collect::<Vec<_>>();
        let ct  = assembly.ct_inputs.into_iter().chain(assembly.ct_aux.into_iter()).collect::<Vec<_>>();
        let num_wires = assembly.num_inputs + assembly.num_aux;

        // Sanity check
        assert_eq!(num_wires, at.len());
        assert_eq!(num_wires, bt.len());
        assert_eq!(num_wires, ct.len());

        let mut a_g1 = vec![E::G1Affine::identity(); num_wires];
        let mut b_g1 = vec![E::G1Affine::identity(); num_wires];
        let mut b_g2 = vec![E::G2Affine::identity(); num_wires];
        let mut c_g1 = vec![E::G1Affine::identity(); num_wires];

        // Evaluate polynomials in multiple threads
        worker.scope(a_g1.len(), |scope, chunk| {
            for ((((((a_g1, b_g1), b_g2), c_g1), at), bt), ct) in
            a_g1.chunks_mut(chunk)
                .zip(b_g1.chunks_mut(chunk))
                .zip(b_g2.chunks_mut(chunk))
                .zip(c_g1.chunks_mut(chunk))
                .zip(at.chunks(chunk))
                .zip(bt.chunks(chunk))
                .zip(ct.chunks(chunk))
            {
                let coeffs_g1 = coeffs_g1.clone();
                let coeffs_g2 = coeffs_g2.clone();

                scope.spawn(move |_| {
                    let mut a_g1_proj = vec![E::G1::identity(); a_g1.len()];
                    let mut b_g1_proj = vec![E::G1::identity(); b_g1.len()];
                    let mut b_g2_proj = vec![E::G2::identity(); b_g2.len()];
                    let mut c_g1_proj = vec![E::G1::identity(); c_g1.len()];

                    for ((((((a_g1, b_g1), b_g2), c_g1), at), bt), ct) in
                    a_g1_proj.iter_mut()
                        .zip(b_g1_proj.iter_mut())
                        .zip(b_g2_proj.iter_mut())
                        .zip(c_g1_proj.iter_mut())
                        .zip(at.iter())
                        .zip(bt.iter())
                        .zip(ct.iter())
                    {
                        for &(coeff, lag) in at {
                            let mut n = coeffs_g1[lag];
                            MulAssign::<E::Fr>::mul_assign(&mut n, coeff);
                            AddAssign::<&E::G1>::add_assign(a_g1, &n);
                        }

                        for &(coeff, lag) in bt {
                            let mut n = coeffs_g1[lag];
                            MulAssign::<E::Fr>::mul_assign(&mut n, coeff);
                            AddAssign::<&E::G1>::add_assign(b_g1, &n);

                            let mut n = coeffs_g2[lag];
                            MulAssign::<E::Fr>::mul_assign(&mut n, coeff);
                            AddAssign::<&E::G2>::add_assign(b_g2, &n);
                        }

                        for &(coeff, lag) in ct {
                            let mut n = coeffs_g1[lag];
                            MulAssign::<E::Fr>::mul_assign(&mut n, coeff);
                            AddAssign::<&E::G1>::add_assign(c_g1, &n);
                        }
                    }

                    // Batch normalize
                    E::G1::batch_normalize(&a_g1_proj, a_g1);
                    E::G1::batch_normalize(&b_g1_proj, b_g1);
                    E::G2::batch_normalize(&b_g2_proj, b_g2);
                    E::G1::batch_normalize(&c_g1_proj, c_g1);
                });
            }
        });

        end_timer!(qap_evaluation);

        let a_g1_affine = Arc::new(a_g1.into_iter().filter(|e| bool::from(!e.is_identity())).collect::<Vec<_>>());
        let b_g1_affine = Arc::new(b_g1.into_iter().filter(|e| bool::from(!e.is_identity())).collect::<Vec<_>>());
        let b_g2_affine = Arc::new(b_g2.into_iter().filter(|e| bool::from(!e.is_identity())).collect::<Vec<_>>());
        let c_g1_affine = Arc::new(c_g1.into_iter().filter(|e| bool::from(!e.is_identity())).collect::<Vec<_>>());

        //TODO: do something!
        fn get_density<T>(constraints: Vec<Vec<T>>) -> DensityTracker {
            let mut density = DensityTracker::new();
            for (i, ati) in constraints.iter().enumerate() {
                density.add_element();
                if !ati.is_empty() {
                    density.inc(i);
                }
            }
            density
        }

        //TODO: sizes
        assert_eq!(self.params.l.len(), assembly.num_aux);

        {
            let worker = Worker::new();

            let circuit_validation = start_timer!(|| "circuit validation");

            let mut z = vec![];
            z.resize_with(num_wires, || { E::Fr::random(rng) });
            let mut z_inp = z.clone();
            let z_aux = z_inp.split_off( assembly.num_inputs);

            let z = Arc::new(z);
            let z_inp = Arc::new(z_inp);
            let z_aux = Arc::new(z_aux);

            let acc_a_g1: E::G1 = multiexp(&worker, (a_g1_affine.clone(), 0), Arc::new(get_density(at)), z.clone()).wait().unwrap();
            let acc_b_g2: E::G2 = multiexp(&worker, (b_g2_affine.clone(), 0), Arc::new(get_density(bt)), z.clone()).wait().unwrap();
            let acc_c_g1: E::G1 = multiexp(&worker, (c_g1_affine, 0), Arc::new(get_density(ct)), z).wait().unwrap();
            let acc_l_g1: E::G1 = multiexp(&worker, (self.params.l.clone(), 0), FullDensity, z_aux).wait().unwrap();
            let acc_ic_g1: E::G1 = multiexp(&worker, (Arc::new(self.params.vk.ic.clone()), 0), FullDensity, z_inp).wait().unwrap();

            let res = E::final_exponentiation(&E::miller_loop(
                [
                    (&acc_a_g1.to_affine().prepare(), &self.params.vk.beta_g2.prepare()),
                    (&self.params.vk.alpha_g1.prepare(), &acc_b_g2.to_affine().prepare()),
                    (&acc_c_g1.to_affine().prepare(), &g2.prepare()),
                    (&acc_l_g1.to_affine().prepare(), &pvk.neg_delta_g2),
                    (&acc_ic_g1.to_affine().prepare(), &pvk.neg_gamma_g2)
                ].iter()
            )).unwrap();
            if res != E::Fqk::one() {
                return Err(SynthesisError::MalformedCrs);
            }

            end_timer!(circuit_validation);
        }

        // Check that QAP polynomial evaluations given in the CRS coincide with those computed above
        // TODO: return polys
        assert_eq!(a_g1_affine, self.params.a);
        assert_eq!(b_g1_affine, self.params.b_g1);
        assert_eq!(b_g2_affine, self.params.b_g2);

        Ok(())
    }

    pub fn write<W: Write>(&self, mut writer: W) -> io::Result<()> {
        self.params.write(&mut writer)?;

        writer.write_u32::<BigEndian>(self.taus_g1.len() as u32)?;
        for g in &self.taus_g1[..] {
            writer.write_all(g.to_uncompressed().as_ref())?;
        }

        writer.write_u32::<BigEndian>(self.taus_g2.len() as u32)?;
        for g in &self.taus_g2[..] {
            writer.write_all(g.to_uncompressed().as_ref())?;
        }

        Ok(())
    }

    pub fn read<R: Read>(mut reader: R, checked: bool) -> io::Result<Self> {
        let params = Parameters::<E>::read(&mut reader, checked)?;

        let mut taus_g1 = vec![];
        let mut taus_g2 = vec![];

        {
            let len = reader.read_u32::<BigEndian>()? as usize;
            for _ in 0..len {
                taus_g1.push(read_g1::<R, E>(&mut reader, checked)?);
            }
        }

        {
            let len = reader.read_u32::<BigEndian>()? as usize;
            for _ in 0..len {
                taus_g2.push(read_g2::<R, E>(&mut reader, checked)?);
            }
        }

        Ok(ExtendedParameters {
            params: params,
            taus_g1: taus_g1,
            taus_g2: taus_g2,
        })
    }
}

#[cfg(test)]
mod test_with_bls12_381 {
    use super::*;
    use crate::{Circuit, ConstraintSystem, SynthesisError};

    use ff::Field;
    use pairing::bls12_381::{Bls12, Fr};
    use rand::thread_rng;
    use std::ops::MulAssign;

    struct MySillyCircuit<E: Engine> {
        a: Option<E::Fr>,
        b: Option<E::Fr>,
    }

    impl<E: Engine> Circuit<E> for MySillyCircuit<E> {
        fn synthesize<CS: ConstraintSystem<E>>(
            self,
            cs: &mut CS,
        ) -> Result<(), SynthesisError> {
            for _ in 0..100 {
                let a = cs.alloc(|| "a", || self.a.ok_or(SynthesisError::AssignmentMissing))?;
                let b = cs.alloc(|| "b", || self.b.ok_or(SynthesisError::AssignmentMissing))?;
                let c = cs.alloc_input(
                    || "c",
                    || {
                        let mut a = self.a.ok_or(SynthesisError::AssignmentMissing)?;
                        let b = self.b.ok_or(SynthesisError::AssignmentMissing)?;

                        a.mul_assign(&b);
                        Ok(a)
                    },
                )?;

                cs.enforce(|| "a*b=c", |lc| lc + a, |lc| lc + b, |lc| lc + c);
            }
            Ok(())
        }
    }

    #[test]
    fn serialization() {
        let rng = &mut thread_rng();

        let params =
            generate_random_parameters::<Bls12, _, _>(MySillyCircuit { a: None, b: None }, rng)
                .unwrap();

        {
            let mut v = vec![];

            params.write(&mut v).unwrap();
            assert_eq!(v.len(), 2136);

            let de_params = Parameters::read(&v[..], true).unwrap();
            assert!(params == de_params);

            let de_params = Parameters::read(&v[..], false).unwrap();
            assert!(params == de_params);
        }

        let pvk = prepare_verifying_key::<Bls12>(&params.vk);

        for _ in 0..100 {
            let a = Fr::random(rng);
            let b = Fr::random(rng);
            let mut c = a;
            c.mul_assign(&b);

            let proof = create_random_proof(
                MySillyCircuit {
                    a: Some(a),
                    b: Some(b),
                },
                &params,
                rng,
            )
            .unwrap();

            let mut v = vec![];
            proof.write(&mut v).unwrap();

            assert_eq!(v.len(), 192);

            let de_proof = Proof::read(&v[..]).unwrap();
            assert!(proof == de_proof);

            assert!(verify_proof(&pvk, &proof, &[c]).unwrap());
            assert!(!verify_proof(&pvk, &proof, &[a]).unwrap());
        }
    }

    #[test]
    fn extended_serialization() {
        let rng = &mut thread_rng();
        let params = generate_extended_random_parameters::<Bls12, _, _>(MySillyCircuit { a: None, b: None }, rng).unwrap();

        let mut v = vec![];

        params.write(&mut v).unwrap();

        let de_params = ExtendedParameters::read(&v[..], true).unwrap();
        assert!(params == de_params);

        let de_params = ExtendedParameters::read(&v[..], false).unwrap();
        assert!(params == de_params);
    }

    #[test]
    fn subversion_check() {
        let rng = &mut thread_rng();
        let params = generate_extended_random_parameters::<Bls12, _, _>(MySillyCircuit { a: None, b: None }, rng).unwrap();
        assert!(params.verify(MySillyCircuit { a: None, b: None }, rng).is_ok());
    }
}
