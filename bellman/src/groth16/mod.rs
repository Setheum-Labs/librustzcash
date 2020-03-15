//! The [Groth16] proving system.
//!
//! [Groth16]: https://eprint.iacr.org/2016/260

use group::{CurveAffine, EncodedPoint, CurveProjective};
use pairing::{Engine, PairingCurveAffine};
use ff::{Field, PrimeField};

use crate::{SynthesisError, Circuit, ConstraintSystem, Index, Variable};
use crate::domain::{EvaluationDomain, Scalar, Point};
use crate::multiexp::{multiexp, FullDensity, SourceBuilder};
use crate::multicore::Worker;

use byteorder::{BigEndian, ReadBytesExt, WriteBytesExt};
use std::io::{self, Read, Write};
use std::sync::Arc;
use std::time::SystemTime;
use futures::Future;

use rand_core::RngCore;

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
        writer.write_all(self.a.into_compressed().as_ref())?;
        writer.write_all(self.b.into_compressed().as_ref())?;
        writer.write_all(self.c.into_compressed().as_ref())?;

        Ok(())
    }

    pub fn read<R: Read>(mut reader: R) -> io::Result<Self> {
        let mut g1_repr = <E::G1Affine as CurveAffine>::Compressed::empty();
        let mut g2_repr = <E::G2Affine as CurveAffine>::Compressed::empty();

        reader.read_exact(g1_repr.as_mut())?;
        let a = g1_repr
            .into_affine()
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))
            .and_then(|e| {
                if e.is_zero() {
                    Err(io::Error::new(
                        io::ErrorKind::InvalidData,
                        "point at infinity",
                    ))
                } else {
                    Ok(e)
                }
            })?;

        reader.read_exact(g2_repr.as_mut())?;
        let b = g2_repr
            .into_affine()
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))
            .and_then(|e| {
                if e.is_zero() {
                    Err(io::Error::new(
                        io::ErrorKind::InvalidData,
                        "point at infinity",
                    ))
                } else {
                    Ok(e)
                }
            })?;

        reader.read_exact(g1_repr.as_mut())?;
        let c = g1_repr
            .into_affine()
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))
            .and_then(|e| {
                if e.is_zero() {
                    Err(io::Error::new(
                        io::ErrorKind::InvalidData,
                        "point at infinity",
                    ))
                } else {
                    Ok(e)
                }
            })?;

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
        writer.write_all(self.alpha_g1.into_uncompressed().as_ref())?;
        writer.write_all(self.beta_g1.into_uncompressed().as_ref())?;
        writer.write_all(self.beta_g2.into_uncompressed().as_ref())?;
        writer.write_all(self.gamma_g2.into_uncompressed().as_ref())?;
        writer.write_all(self.delta_g1.into_uncompressed().as_ref())?;
        writer.write_all(self.delta_g2.into_uncompressed().as_ref())?;
        writer.write_u32::<BigEndian>(self.ic.len() as u32)?;
        for ic in &self.ic {
            writer.write_all(ic.into_uncompressed().as_ref())?;
        }

        Ok(())
    }

    pub fn read<R: Read>(mut reader: R) -> io::Result<Self> {
        let mut g1_repr = <E::G1Affine as CurveAffine>::Uncompressed::empty();
        let mut g2_repr = <E::G2Affine as CurveAffine>::Uncompressed::empty();

        reader.read_exact(g1_repr.as_mut())?;
        let alpha_g1 = g1_repr
            .into_affine()
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;

        reader.read_exact(g1_repr.as_mut())?;
        let beta_g1 = g1_repr
            .into_affine()
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;

        reader.read_exact(g2_repr.as_mut())?;
        let beta_g2 = g2_repr
            .into_affine()
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;

        reader.read_exact(g2_repr.as_mut())?;
        let gamma_g2 = g2_repr
            .into_affine()
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;

        reader.read_exact(g1_repr.as_mut())?;
        let delta_g1 = g1_repr
            .into_affine()
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;

        reader.read_exact(g2_repr.as_mut())?;
        let delta_g2 = g2_repr
            .into_affine()
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;

        let ic_len = reader.read_u32::<BigEndian>()? as usize;

        let mut ic = vec![];

        for _ in 0..ic_len {
            reader.read_exact(g1_repr.as_mut())?;
            let g1 = g1_repr
                .into_affine()
                .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))
                .and_then(|e| {
                    if e.is_zero() {
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

impl<E: Engine> Parameters<E> {
    pub fn write<W: Write>(&self, mut writer: W) -> io::Result<()> {
        self.vk.write(&mut writer)?;

        writer.write_u32::<BigEndian>(self.h.len() as u32)?;
        for g in &self.h[..] {
            writer.write_all(g.into_uncompressed().as_ref())?;
        }

        writer.write_u32::<BigEndian>(self.l.len() as u32)?;
        for g in &self.l[..] {
            writer.write_all(g.into_uncompressed().as_ref())?;
        }

        writer.write_u32::<BigEndian>(self.a.len() as u32)?;
        for g in &self.a[..] {
            writer.write_all(g.into_uncompressed().as_ref())?;
        }

        writer.write_u32::<BigEndian>(self.b_g1.len() as u32)?;
        for g in &self.b_g1[..] {
            writer.write_all(g.into_uncompressed().as_ref())?;
        }

        writer.write_u32::<BigEndian>(self.b_g2.len() as u32)?;
        for g in &self.b_g2[..] {
            writer.write_all(g.into_uncompressed().as_ref())?;
        }

        Ok(())
    }

    pub fn read<R: Read>(mut reader: R, checked: bool) -> io::Result<Self> {
        let read_g1 = |reader: &mut R| -> io::Result<E::G1Affine> {
            let mut repr = <E::G1Affine as CurveAffine>::Uncompressed::empty();
            reader.read_exact(repr.as_mut())?;

            if checked {
                repr.into_affine()
            } else {
                repr.into_affine_unchecked()
            }
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))
            .and_then(|e| {
                if e.is_zero() {
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
            let mut repr = <E::G2Affine as CurveAffine>::Uncompressed::empty();
            reader.read_exact(repr.as_mut())?;

            if checked {
                repr.into_affine()
            } else {
                repr.into_affine_unchecked()
            }
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))
            .and_then(|e| {
                if e.is_zero() {
                    Err(io::Error::new(
                        io::ErrorKind::InvalidData,
                        "point at infinity",
                    ))
                } else {
                    Ok(e)
                }
            })
        };

        let vk = VerifyingKey::<E>::read(&mut reader)?;

        let mut h = vec![];
        let mut l = vec![];
        let mut a = vec![];
        let mut b_g1 = vec![];
        let mut b_g2 = vec![];

        {
            let len = reader.read_u32::<BigEndian>()? as usize;
            for _ in 0..len {
                h.push(read_g1(&mut reader)?);
            }
        }

        {
            let len = reader.read_u32::<BigEndian>()? as usize;
            for _ in 0..len {
                l.push(read_g1(&mut reader)?);
            }
        }

        {
            let len = reader.read_u32::<BigEndian>()? as usize;
            for _ in 0..len {
                a.push(read_g1(&mut reader)?);
            }
        }

        {
            let len = reader.read_u32::<BigEndian>()? as usize;
            for _ in 0..len {
                b_g1.push(read_g1(&mut reader)?);
            }
        }

        {
            let len = reader.read_u32::<BigEndian>()? as usize;
            for _ in 0..len {
                b_g2.push(read_g2(&mut reader)?);
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

pub struct ExtendedParameters<E: Engine> {
    pub params: Parameters<E>,

    pub g1: E::G1Affine,
    pub g2: E::G2Affine,

    pub taus_g1: Vec<E::G1Affine>,
    pub taus_g2: Vec<E::G2Affine>,

    pub taum_g1: E::G1Affine,
}

impl<E: Engine> ExtendedParameters<E> {

    // Checks the CRS for possible subversion by the malicious generator. It does not guarantee subversion soundness,
    // meaning that the generator can still use the trapdoor to produce valid proofs of false statements,
    // but does guarantee subversion zero-knowledgeness, so a proof generated by an honest prover will not reveal any information about the witness.
    // This is useful in the case, when the verifier plays the role of the generator and passes the CRS to the prover, who runs this check against it.
    // Then the verifier can be sure in the soundness as only it knows the trapdoor, and the prover is given it's privacy.
    // Follows the procedure from Georg Fuchsbauer, Subversion-zero-knowledge SNARKs (https://eprint.iacr.org/2017/587), p. 26
    pub fn verify<C: Circuit<E>, R: RngCore>(&self, circuit: C, rng: &mut R) -> Result<(), SynthesisError> {

        // https://eprint.iacr.org/2017/587, p. 26
        let t = SystemTime::now();
        let pvk = prepare_verifying_key(&self.params.vk); //TODO: return it

        // 1
        // P1 != 0
        if self.g1.is_zero() {
            return Err(SynthesisError::MalformedCrs);
        }
        // P2 != 0
        if self.g2.is_zero() {
            return Err(SynthesisError::MalformedCrs);
        }

        // 2
        // pk_alpha != 0
        if self.params.vk.alpha_g1.is_zero() {
            return Err(SynthesisError::MalformedCrs);
        }
        // pk_beta != 0
        if self.params.vk.beta_g1.is_zero() {
            return Err(SynthesisError::MalformedCrs);
        }
        // vk'_gamma != 0
        if self.params.vk.gamma_g2.is_zero() {
            return Err(SynthesisError::MalformedCrs);
        }
        // pk_delta != 0
        if self.params.vk.delta_g1.is_zero() {
            return Err(SynthesisError::MalformedCrs);
        }
        // pk_{Z,0} = t(tau)/delta != 0
        if self.params.h[0].is_zero() {
            return Err(SynthesisError::MalformedCrs);
        }
        // btw, nondegeneracy of beta and delta in G2 follows from the checks in #4

        // 3
        // pk_{H,0} = P1
        if self.taus_g1[0] != self.g1 {
            return Err(SynthesisError::MalformedCrs);
        }
        // pk'_{H,0} = P2
        if self.taus_g2[0] != self.g2 {
            return Err(SynthesisError::MalformedCrs);
        }
        for (tau_i_g1, tau_j_g1) in self.taus_g1.iter().skip(1).zip(self.taus_g1.iter()) {
            // j = i - 1
            if E::pairing(tau_i_g1.clone(), self.g2) != E::pairing(tau_j_g1.clone(), self.taus_g2[1]) {
                return Err(SynthesisError::MalformedCrs);
            }
        }
        for (tau_i_g1, tau_i_g2) in self.taus_g1.iter().zip(self.taus_g2.iter()).skip(1) {
            if E::pairing(self.g1, tau_i_g2.clone()) != E::pairing(tau_i_g1.clone(), self.g2) {
                return Err(SynthesisError::MalformedCrs);
            }
        }

        // 4
        // e(P1, pk'_beta) = e(pk_beta, P2)
        if E::pairing(self.g1, self.params.vk.beta_g2) != E::pairing(self.params.vk.beta_g1, self.g2) {
            return Err(SynthesisError::MalformedCrs);
        }
        // e(P1, pk'_delta) = e(pk_delta, P2)
        if E::pairing(self.g1, self.params.vk.delta_g2) != E::pairing(self.params.vk.delta_g1, self.g2) {
            return Err(SynthesisError::MalformedCrs);
        }

        // Convert the circuit in R1CS to the QAP in Lagrange base (QAP polynomials evaluations in the roots of unity)
        // The additional input and constraints are Groth16/bellman specific, see the code in generator or prover
        let t = SystemTime::now();

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
        println!("QAP synthesis = {}", t.elapsed().unwrap().as_millis());

        // Evaluate the QAP polynomials in point tau in the exponent
        let t = SystemTime::now();

        // The code bellow is borrowed from https://github.com/ebfull/powersoftau/blob/5429415959175082207fd61c10319e47a6b56e87/src/bin/verify.rs#L162-L225
        let worker = Worker::new();

        let mut g1_coeffs = EvaluationDomain::from_coeffs(
            self.taus_g1.iter()
            .map(|e| Point(e.into_projective()))
            .collect()
        ).unwrap(); //TODO: remove Arc?

        let mut g2_coeffs = EvaluationDomain::from_coeffs(
            self.taus_g2.iter()
                .map(|e| Point(e.into_projective()))
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
        E::G1::batch_normalization(&mut g1_coeffs);
        E::G2::batch_normalization(&mut g2_coeffs);

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

        let mut a_g1 = vec![E::G1::zero(); num_wires];
        let mut b_g1 = vec![E::G1::zero(); num_wires];
        let mut b_g2 = vec![E::G2::zero(); num_wires];
        let mut c_g1 = vec![E::G1::zero(); num_wires];

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
                    for ((((((a_g1, b_g1), b_g2), c_g1), at), bt), ct) in
                    a_g1.iter_mut()
                        .zip(b_g1.iter_mut())
                        .zip(b_g2.iter_mut())
                        .zip(c_g1.iter_mut())
                        .zip(at.iter())
                        .zip(bt.iter())
                        .zip(ct.iter())
                    {
                        for &(coeff, lag) in at {
                            let mut n = coeffs_g1[lag];
                            n.mul_assign(coeff);
                            a_g1.add_assign(&n);
                        }

                        for &(coeff, lag) in bt {
                            let mut n = coeffs_g1[lag];
                            n.mul_assign(coeff);
                            b_g1.add_assign(&n);

                            let mut n = coeffs_g2[lag];
                            n.mul_assign(coeff);
                            b_g2.add_assign(&n);
                        }

                        for &(coeff, lag) in ct {
                            let mut n = coeffs_g1[lag];
                            n.mul_assign(coeff);
                            c_g1.add_assign(&n);
                        }
                    }

                    // Batch normalize
                    E::G1::batch_normalization(a_g1);
                    E::G1::batch_normalization(b_g1);
                    E::G2::batch_normalization(b_g2);
                    E::G1::batch_normalization(c_g1);
                });
            }
        });

        println!("QAP evaluation = {}", t.elapsed().unwrap().as_millis());

        //TODO: sizes
        assert_eq!(self.params.l.len(), assembly.num_aux);



        for (((li, ai_g1), bi_g2), ci_g1) in self.params.l.iter()
            .zip(a_g1.iter().skip(assembly.num_inputs))
            .zip(b_g2.iter().skip(assembly.num_inputs))
            .zip(c_g1.iter().skip(assembly.num_inputs))
        {
            let res = E::final_exponentiation(&E::miller_loop(
                [
                    // TODO: optimize conversions
                    (&li.prepare(), &pvk.neg_delta_g2),
                    (&ai_g1.into_affine().prepare(), &self.params.vk.beta_g2.prepare()),
                    (&self.params.vk.alpha_g1.prepare(), &bi_g2.into_affine().prepare()),
                    (&ci_g1.into_affine().prepare(), &self.g2.prepare())
                ].iter()
            )).unwrap();
            if res != E::Fqk::one() {
                return Err(SynthesisError::MalformedCrs);
            }
//            let lhs = E::pairing(li.clone(), self.params.vk.delta_g2);
//            let mut rhs = E::pairing(ai_g1.clone(), self.params.vk.beta_g2);
//            rhs.mul_assign(&E::pairing(self.params.vk.alpha_g1, bi_g2.clone()));
//            rhs.mul_assign(&E::pairing(ci_g1.clone(), self.g2));
//            if lhs != rhs {
//                return Err(SynthesisError::MalformedCrs);
//            }

        }

        // 5
        // z (aka t in Groth16/bellman) is the vanishing polynomial of the domain. In our case z = x^m - 1
        // btw, there's a typo un Fuc19, as z should have degree d-1 in his notation
        let mut z = self.taum_g1.into_projective();
        let g1 = self.taus_g1[0].into_projective();
        z.sub_assign(&g1);
        for (hi, tau_i_g2) in self.params.h.iter().zip(self.taus_g2.iter()) {
            if E::pairing(hi.clone(), self.params.vk.delta_g2) != E::pairing(z, tau_i_g2.clone()) {
                return Err(SynthesisError::MalformedCrs);
            }
        }

        for (((ici, ai_g1), bi_g2), ci_g1) in self.params.vk.ic.iter()
            .zip(a_g1.iter())
            .zip(b_g2.iter())
            .zip(c_g1.iter())
        {
            let res = E::final_exponentiation(&E::miller_loop(
                [
                    // TODO: optimize conversions
                    (&ici.prepare(), &pvk.neg_gamma_g2),
                    (&ai_g1.into_affine().prepare(), &self.params.vk.beta_g2.prepare()),
                    (&self.params.vk.alpha_g1.prepare(), &bi_g2.into_affine().prepare()),
                    (&ci_g1.into_affine().prepare(), &self.g2.prepare())
                ].iter()
            )).unwrap();
            if res != E::Fqk::one() {
                return Err(SynthesisError::MalformedCrs);
            }
//            let lhs = E::pairing(ici.clone(), self.params.vk.gamma_g2);
//            let mut rhs = E::pairing(ai_g1.clone(), self.params.vk.beta_g2);
//            rhs.mul_assign(&E::pairing(self.params.vk.alpha_g1, bi_g2.clone()));
//            rhs.mul_assign(&E::pairing(ci_g1.clone(), self.g2));
//            if lhs != rhs {
//                return Err(SynthesisError::MalformedCrs);
//            }
        }

        println!("checks 1-5 = {}", t.elapsed().unwrap().as_millis());

        // Check that QAP polynomial evaluations given in the CRS coincide with those computed above

//        assert_eq!(self.params.a.len(), assembly.num_inputs + assembly.num_aux);
//        assert_eq!(self.params.b_g1.l en(), assembly.num_inputs + assembly.num_aux);
//        assert_eq!(self.params.b_g2.len(), assembly.num_inputs + assembly.num_aux);

        // TODO: filter zero evaluations at the very beginning
        for (((((ai_g1, bi_g1), bi_g2), crs_ai_g1), crs_bi_g1), crs_bi_g2) in a_g1.iter().filter(|e| !e.is_zero())
            .zip(b_g1.iter().filter(|e| !e.is_zero()))
            .zip(b_g2.iter().filter(|e| !e.is_zero()))
            .zip(self.params.a.iter())
            .zip(self.params.b_g1.iter())
            .zip(self.params.b_g2.iter())
        {
            if ai_g1.into_affine() != *crs_ai_g1
                || bi_g1.into_affine() != *crs_bi_g1
                || bi_g2.into_affine() != *crs_bi_g2 {
                return Err(SynthesisError::MalformedCrs);
            }
        }

        Ok(())
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
            for _ in 0..10 {
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
    fn subversion_check() {
        let rng = &mut thread_rng();
        let params = generate_extended_random_parameters::<Bls12, _, _>(MySillyCircuit { a: None, b: None }, rng).unwrap();
        assert!(params.verify(MySillyCircuit { a: None, b: None }, rng).is_ok());
    }
}
