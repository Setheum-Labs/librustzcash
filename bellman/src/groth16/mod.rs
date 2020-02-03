//! The [Groth16] proving system.
//!
//! [Groth16]: https://eprint.iacr.org/2016/260

use group::{CurveAffine, EncodedPoint, CurveProjective};
use pairing::{Engine, PairingCurveAffine};
use ff::Field;

use crate::{SynthesisError, Circuit, ConstraintSystem, Index, Variable};
use crate::domain::{EvaluationDomain, Scalar};

use crate::multiexp::SourceBuilder;
use crate::multicore::Worker;
use byteorder::{BigEndian, ReadBytesExt, WriteBytesExt};
use std::io::{self, Read, Write};
use std::sync::Arc;

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

    pub taus_g1: Arc<Vec<E::G1Affine>>,
    pub taus_g2: Arc<Vec<E::G2Affine>>,

    pub taum_g1: E::G1Affine,
}

// Evaluate the polynomial in the exponent
fn eval1<E: Engine>(
    poly: &Vec<(E::Fr, usize)>, // polynomial represented as non-zero evaluations in roots of unity
    powers: &Vec<E::G1Affine>, // powers of the evaluation point in the exponent
    degree: usize // degree of the polynomial
) -> Result<E::G1, SynthesisError>
{
    // Lagrange coefficients of the polynomial
    let mut evals = vec![Scalar::<E>(E::Fr::zero()); degree];
    // Convert from compact representation
    for (value, i) in poly {
        evals[*i] = Scalar::<E>(*value);
    }
    let mut evals = EvaluationDomain::from_coeffs(evals)?;
    evals.ifft(&Worker::new());
    // Monomial coefficients of the polynomial
    let coeffs = evals.into_coeffs(); // TODO: size

    let mut result = E::G1Affine::zero().into_projective();
    for (coeff, power) in coeffs.iter().zip(powers.iter()) {
        result.add_assign(&power.mul(coeff.0));
    }

    Ok(result)
}

// TODO: merge with eval1
fn eval2<E: Engine>(
    poly: &Vec<(E::Fr, usize)>, // polynomial represented as non-zero evaluations in roots of unity
    powers: &Vec<E::G2Affine>, // powers of the evaluation point in the exponent
    degree: usize // degree of the polynomial
) -> Result<E::G2, SynthesisError>
{
    // Lagrange coefficients of the polynomial
    let mut evals = vec![Scalar::<E>(E::Fr::zero()); degree];
    // Convert from compact representation
    for (value, i) in poly {
        evals[*i] = Scalar::<E>(*value);
    }
    let mut evals = EvaluationDomain::from_coeffs(evals)?;
    evals.ifft(&Worker::new());
    // Monomial coefficients of the polynomial
    let coeffs = evals.into_coeffs(); // TODO: size

    let mut result = E::G2Affine::zero().into_projective();
    for (coeff, power) in coeffs.iter().zip(powers.iter()) {
        result.add_assign(&power.mul(coeff.0));
    }

    Ok(result)
}

impl<E: Engine> ExtendedParameters<E> {

    // Checks the CRS for possible subversion by the malicious generator. It does not guarantee subversion soundness,
    // meaning that the generator can still use the trapdoor to produce valid proofs of false statements,
    // but does guarantee subversion zero-knowledgeness, so a proof generated by an honest prover will not reveal any information about the witness.
    // This is useful in the case, when the verifier plays the role of the generator and passes the CRS to the prover, who runs this check against it.
    // Then the verifier can be sure in the soundness as only it knows the trapdoor, and the prover is given it's privacy.
    // Follows the procedure from Georg Fuchsbauer, Subversion-zero-knowledge SNARKs (https://eprint.iacr.org/2017/587), p. 26
    // with the following deviations:
    // - typos
    // - in the article the circuit is represented as the set of QAP polynomials,
    //   while the implementation accepts the circuit in R1CS and converts it to a QAP
    // - bases
    pub fn verify<C: Circuit<E>>(&self, circuit: C) -> Result<(), SynthesisError> {

        // Convert the circuit in R1CS to the QAP in Lagrange base in the roots of unity.
        // As the check is given in monomial base, later the polynomials are interpolated with iFFT.
        // The additional input and constraints are Groth16/bellman specific, see the code in generator or prover.

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

        //TODO: sizes
        assert_eq!(self.params.l.len(), assembly.num_aux);

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
        //
        for (((li, ai), bi), ci) in self.params.l.iter()
            .zip(assembly.at_aux.iter())
            .zip(assembly.bt_aux.iter())
            .zip(assembly.ct_aux.iter())
        {
            let lhs = E::pairing(li.clone(), self.params.vk.delta_g2);
            let eval_a = eval1::<E>(ai, &self.taus_g1, assembly.num_constraints)?;
            let eval_b = eval2::<E>(bi, &self.taus_g2, assembly.num_constraints)?;
            let eval_c = eval1::<E>(ci, &self.taus_g1, assembly.num_constraints)?;
            let mut rhs = E::pairing(eval_a, self.params.vk.beta_g2);
            rhs.mul_assign(&E::pairing(self.params.vk.alpha_g1, eval_b));
            rhs.mul_assign(&E::pairing(eval_c, self.g2));
            if lhs != rhs {
                return Err(SynthesisError::MalformedCrs);
            }
        }

        // 5
        // z (aka t in Groth16/bellman) is the vanishing polynomial of the domain. In our case z = x^m - 1
        let mut z = self.gm.into_projective();
        let p = self.taus_g1[0].into_projective();
        z.sub_assign(&p);
        for (hi, tau_i_g2) in self.params.h.iter().zip(self.taus_g2.iter()) {
            if E::pairing(hi.clone(), self.params.vk.delta_g2) != E::pairing(z, tau_i_g2.clone()) {
                return Err(SynthesisError::MalformedCrs);
            }
        }

        for (((ici, ai), bi), ci) in self.params.vk.ic.iter()
            .zip(assembly.at_inputs.iter())
            .zip(assembly.bt_inputs.iter())
            .zip(assembly.ct_inputs.iter())
        {
            let lhs = E::pairing(ici.clone(), self.params.vk.gamma_g2);
            let eval_a = eval1::<E>(ai, &self.taus_g1, assembly.num_constraints)?;
            let eval_b = eval2::<E>(bi, &self.taus_g2, assembly.num_constraints)?;
            let eval_c = eval1::<E>(ci, &self.taus_g1, assembly.num_constraints)?;
            let mut rhs = E::pairing(eval_a, self.params.vk.beta_g2);
            rhs.mul_assign(&E::pairing(self.params.vk.alpha_g1, eval_b));
            rhs.mul_assign(&E::pairing(eval_c, self.g2));
            if lhs != rhs {
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
        assert!(params.verify(MySillyCircuit { a: None, b: None }).is_ok());
    }
}
