use ff::PrimeField;
use group::{
    cofactor::{CofactorCurve, CofactorCurveAffine},
    GroupEncoding, UncompressedEncoding,
};

use super::*;
use crate::*;

#[test]
fn test_pairing_result_against_relic() {
    /*
    Sent to me from Diego Aranha (author of RELIC library):

    1250EBD871FC0A92 A7B2D83168D0D727 272D441BEFA15C50 3DD8E90CE98DB3E7 B6D194F60839C508 A84305AACA1789B6
    089A1C5B46E5110B 86750EC6A5323488 68A84045483C92B7 AF5AF689452EAFAB F1A8943E50439F1D 59882A98EAA0170F
    1368BB445C7C2D20 9703F239689CE34C 0378A68E72A6B3B2 16DA0E22A5031B54 DDFF57309396B38C 881C4C849EC23E87
    193502B86EDB8857 C273FA075A505129 37E0794E1E65A761 7C90D8BD66065B1F FFE51D7A579973B1 315021EC3C19934F
    01B2F522473D1713 91125BA84DC4007C FBF2F8DA752F7C74 185203FCCA589AC7 19C34DFFBBAAD843 1DAD1C1FB597AAA5
    018107154F25A764 BD3C79937A45B845 46DA634B8F6BE14A 8061E55CCEBA478B 23F7DACAA35C8CA7 8BEAE9624045B4B6
    19F26337D205FB46 9CD6BD15C3D5A04D C88784FBB3D0B2DB DEA54D43B2B73F2C BB12D58386A8703E 0F948226E47EE89D
    06FBA23EB7C5AF0D 9F80940CA771B6FF D5857BAAF222EB95 A7D2809D61BFE02E 1BFD1B68FF02F0B8 102AE1C2D5D5AB1A
    11B8B424CD48BF38 FCEF68083B0B0EC5 C81A93B330EE1A67 7D0D15FF7B984E89 78EF48881E32FAC9 1B93B47333E2BA57
    03350F55A7AEFCD3 C31B4FCB6CE5771C C6A0E9786AB59733 20C806AD36082910 7BA810C5A09FFDD9 BE2291A0C25A99A2
    04C581234D086A99 02249B64728FFD21 A189E87935A95405 1C7CDBA7B3872629 A4FAFC05066245CB 9108F0242D0FE3EF
    0F41E58663BF08CF 068672CBD01A7EC7 3BACA4D72CA93544 DEFF686BFD6DF543 D48EAA24AFE47E1E FDE449383B676631
    */

    assert_eq!(Bls12::pairing(&G1Affine::generator(), &G2Affine::generator()), Fq12 {
        c0: Fq6 {
            c0: Fq2 {
                c0: Fq::from_str("2819105605953691245277803056322684086884703000473961065716485506033588504203831029066448642358042597501014294104502").unwrap(),
                c1: Fq::from_str("1323968232986996742571315206151405965104242542339680722164220900812303524334628370163366153839984196298685227734799").unwrap()
            },
            c1: Fq2 {
                c0: Fq::from_str("2987335049721312504428602988447616328830341722376962214011674875969052835043875658579425548512925634040144704192135").unwrap(),
                c1: Fq::from_str("3879723582452552452538684314479081967502111497413076598816163759028842927668327542875108457755966417881797966271311").unwrap()
            },
            c2: Fq2 {
                c0: Fq::from_str("261508182517997003171385743374653339186059518494239543139839025878870012614975302676296704930880982238308326681253").unwrap(),
                c1: Fq::from_str("231488992246460459663813598342448669854473942105054381511346786719005883340876032043606739070883099647773793170614").unwrap()
            }
        },
        c1: Fq6 {
            c0: Fq2 {
                c0: Fq::from_str("3993582095516422658773669068931361134188738159766715576187490305611759126554796569868053818105850661142222948198557").unwrap(),
                c1: Fq::from_str("1074773511698422344502264006159859710502164045911412750831641680783012525555872467108249271286757399121183508900634").unwrap()
            },
            c1: Fq2 {
                c0: Fq::from_str("2727588299083545686739024317998512740561167011046940249988557419323068809019137624943703910267790601287073339193943").unwrap(),
                c1: Fq::from_str("493643299814437640914745677854369670041080344349607504656543355799077485536288866009245028091988146107059514546594").unwrap()
            },
            c2: Fq2 {
                c0: Fq::from_str("734401332196641441839439105942623141234148957972407782257355060229193854324927417865401895596108124443575283868655").unwrap(),
                c1: Fq::from_str("2348330098288556420918672502923664952620152483128593484301759394583320358354186482723629999370241674973832318248497").unwrap()
            }
        }
    });
}

fn uncompressed_test_vectors<G: CofactorCurve>(expected: &[u8])
where
    G::Affine: UncompressedEncoding,
{
    let mut e = G::identity();
    let encoded_len = <G::Affine as UncompressedEncoding>::Uncompressed::default()
        .as_ref()
        .len();

    let mut v = vec![];
    {
        let mut expected = expected;
        for _ in 0..1000 {
            let e_affine = e.to_affine();
            let encoded = e_affine.to_uncompressed();
            v.extend_from_slice(encoded.as_ref());

            let mut decoded = <G::Affine as UncompressedEncoding>::Uncompressed::default();
            decoded.as_mut().copy_from_slice(&expected[0..encoded_len]);
            expected = &expected[encoded_len..];
            let decoded = G::Affine::from_uncompressed(&decoded).unwrap();
            assert_eq!(e_affine, decoded);

            e.add_assign(&G::generator());
        }
    }

    assert_eq!(&v[..], expected);
}

fn compressed_test_vectors<G: CofactorCurve>(expected: &[u8]) {
    let mut e = G::identity();
    let encoded_len = <G::Affine as GroupEncoding>::Repr::default().as_ref().len();

    let mut v = vec![];
    {
        let mut expected = expected;
        for _ in 0..1000 {
            let e_affine = e.to_affine();
            let encoded = e_affine.to_bytes();
            v.extend_from_slice(encoded.as_ref());

            let mut decoded = <G::Affine as GroupEncoding>::Repr::default();
            decoded.as_mut().copy_from_slice(&expected[0..encoded_len]);
            expected = &expected[encoded_len..];
            let decoded = G::Affine::from_bytes(&decoded).unwrap();
            assert_eq!(e_affine, decoded);

            e.add_assign(&G::generator());
        }
    }

    assert_eq!(&v[..], expected);
}

#[test]
fn test_g1_uncompressed_valid_vectors() {
    uncompressed_test_vectors::<G1>(include_bytes!("g1_uncompressed_valid_test_vectors.dat"));
}

#[test]
fn test_g1_compressed_valid_vectors() {
    compressed_test_vectors::<G1>(include_bytes!("g1_compressed_valid_test_vectors.dat"));
}

#[test]
fn test_g2_uncompressed_valid_vectors() {
    uncompressed_test_vectors::<G2>(include_bytes!("g2_uncompressed_valid_test_vectors.dat"));
}

#[test]
fn test_g2_compressed_valid_vectors() {
    compressed_test_vectors::<G2>(include_bytes!("g2_compressed_valid_test_vectors.dat"));
}

#[test]
fn test_g1_uncompressed_invalid_vectors() {
    {
        let z = G1Affine::identity().to_uncompressed();

        {
            let mut z = z;
            z.as_mut()[0] |= 0b1000_0000;
            if G1Affine::from_uncompressed(&z).is_none().into() {
                // :)
            } else {
                panic!("should have rejected the point because we expected an uncompressed point");
            }
        }

        {
            let mut z = z;
            z.as_mut()[0] |= 0b0010_0000;
            if G1Affine::from_uncompressed(&z).is_none().into() {
                // :)
            } else {
                panic!("should have rejected the point because the parity bit should not be set if the point is at infinity");
            }
        }

        for i in 0..G1Uncompressed::size() {
            let mut z = z;
            z.as_mut()[i] |= 0b0000_0001;
            if G1Affine::from_uncompressed(&z).is_none().into() {
                // :)
            } else {
                panic!("should have rejected the point because the coordinates should be zeroes at the point at infinity");
            }
        }
    }

    let o = G1Affine::generator().to_uncompressed();

    {
        let mut o = o;
        o.as_mut()[0] |= 0b1000_0000;
        if G1Affine::from_uncompressed(&o).is_none().into() {
            // :)
        } else {
            panic!("should have rejected the point because we expected an uncompressed point");
        }
    }

    let m = Fq::char();

    {
        let mut o = o;
        o.as_mut()[..48].copy_from_slice(m.as_ref());

        if G1Affine::from_uncompressed(&o).is_none().into() {
            // x coordinate
        } else {
            panic!("should have rejected the point")
        }
    }

    {
        let mut o = o;
        o.as_mut()[48..].copy_from_slice(m.as_ref());

        if G1Affine::from_uncompressed(&o).is_none().into() {
            // y coordinate
        } else {
            panic!("should have rejected the point")
        }
    }

    {
        let m = Fq::zero().to_repr();

        let mut o = o;
        o.as_mut()[..48].copy_from_slice(m.as_ref());

        if G1Affine::from_uncompressed(&o).is_none().into() {
            // :)
        } else {
            panic!("should have rejected the point because it isn't on the curve")
        }
    }

    {
        let mut o = o;
        let mut x = Fq::one();

        loop {
            let mut x3b = x.square();
            x3b.mul_assign(&x);
            x3b.add_assign(&Fq::from(4)); // TODO: perhaps expose coeff_b through API?

            let y = x3b.sqrt();
            if y.is_some().into() {
                let y = y.unwrap();

                // We know this is on the curve, but it's likely not going to be in the correct subgroup.
                o.as_mut()[..48].copy_from_slice(x.to_repr().as_ref());
                o.as_mut()[48..].copy_from_slice(y.to_repr().as_ref());

                if G1Affine::from_uncompressed(&o).is_none().into() {
                    break;
                } else {
                    panic!(
                        "should have rejected the point because it isn't in the correct subgroup"
                    )
                }
            } else {
                x.add_assign(&Fq::one());
            }
        }
    }
}

#[test]
fn test_g2_uncompressed_invalid_vectors() {
    {
        let z = G2Affine::identity().to_uncompressed();

        {
            let mut z = z;
            z.as_mut()[0] |= 0b1000_0000;
            if G2Affine::from_uncompressed(&z).is_none().into() {
                // :)
            } else {
                panic!("should have rejected the point because we expected an uncompressed point");
            }
        }

        {
            let mut z = z;
            z.as_mut()[0] |= 0b0010_0000;
            if G2Affine::from_uncompressed(&z).is_none().into() {
                // :)
            } else {
                panic!("should have rejected the point because the parity bit should not be set if the point is at infinity");
            }
        }

        for i in 0..G2Uncompressed::size() {
            let mut z = z;
            z.as_mut()[i] |= 0b0000_0001;
            if G2Affine::from_uncompressed(&z).is_none().into() {
                // :)
            } else {
                panic!("should have rejected the point because the coordinates should be zeroes at the point at infinity");
            }
        }
    }

    let o = G2Affine::generator().to_uncompressed();

    {
        let mut o = o;
        o.as_mut()[0] |= 0b1000_0000;
        if G2Affine::from_uncompressed(&o).is_none().into() {
            // :)
        } else {
            panic!("should have rejected the point because we expected an uncompressed point");
        }
    }

    let m = Fq::char();

    {
        let mut o = o;
        o.as_mut()[..48].copy_from_slice(m.as_ref());

        if G2Affine::from_uncompressed(&o).is_none().into() {
            // x coordinate (c1)
        } else {
            panic!("should have rejected the point")
        }
    }

    {
        let mut o = o;
        o.as_mut()[48..96].copy_from_slice(m.as_ref());

        if G2Affine::from_uncompressed(&o).is_none().into() {
            // x coordinate (c0)
        } else {
            panic!("should have rejected the point")
        }
    }

    {
        let mut o = o;
        o.as_mut()[96..144].copy_from_slice(m.as_ref());

        if G2Affine::from_uncompressed(&o).is_none().into() {
            // y coordinate (c1)
        } else {
            panic!("should have rejected the point")
        }
    }

    {
        let mut o = o;
        o.as_mut()[144..].copy_from_slice(m.as_ref());

        if G2Affine::from_uncompressed(&o).is_none().into() {
            // y coordinate (c0)
        } else {
            panic!("should have rejected the point")
        }
    }

    {
        let m = Fq::zero().to_repr();

        let mut o = o;
        o.as_mut()[..48].copy_from_slice(m.as_ref());
        o.as_mut()[48..96].copy_from_slice(m.as_ref());

        if G2Affine::from_uncompressed(&o).is_none().into() {
            // :)
        } else {
            panic!("should have rejected the point because it isn't on the curve")
        }
    }

    {
        let mut o = o;
        let mut x = Fq2::one();

        loop {
            let mut x3b = x.square();
            x3b.mul_assign(&x);
            x3b.add_assign(&Fq2 {
                c0: Fq::from(4),
                c1: Fq::from(4),
            }); // TODO: perhaps expose coeff_b through API?

            let y = x3b.sqrt();
            if y.is_some().into() {
                let y = y.unwrap();

                // We know this is on the curve, but it's likely not going to be in the correct subgroup.
                o.as_mut()[..48].copy_from_slice(x.c1.to_repr().as_ref());
                o.as_mut()[48..96].copy_from_slice(x.c0.to_repr().as_ref());
                o.as_mut()[96..144].copy_from_slice(y.c1.to_repr().as_ref());
                o.as_mut()[144..].copy_from_slice(y.c0.to_repr().as_ref());

                if G2Affine::from_uncompressed(&o).is_none().into() {
                    break;
                } else {
                    panic!(
                        "should have rejected the point because it isn't in the correct subgroup"
                    )
                }
            } else {
                x.add_assign(&Fq2::one());
            }
        }
    }
}

#[test]
fn test_g1_compressed_invalid_vectors() {
    {
        let z = G1Affine::identity().to_bytes();

        {
            let mut z = z;
            z.as_mut()[0] &= 0b0111_1111;
            if G1Affine::from_bytes(&z).is_none().into() {
                // :)
            } else {
                panic!("should have rejected the point because we expected a compressed point");
            }
        }

        {
            let mut z = z;
            z.as_mut()[0] |= 0b0010_0000;
            if G1Affine::from_bytes(&z).is_none().into() {
                // :)
            } else {
                panic!("should have rejected the point because the parity bit should not be set if the point is at infinity");
            }
        }

        for i in 0..G1Compressed::size() {
            let mut z = z;
            z.as_mut()[i] |= 0b0000_0001;
            if G1Affine::from_bytes(&z).is_none().into() {
                // :)
            } else {
                panic!("should have rejected the point because the coordinates should be zeroes at the point at infinity");
            }
        }
    }

    let o = G1Affine::generator().to_bytes();

    {
        let mut o = o;
        o.as_mut()[0] &= 0b0111_1111;
        if G1Affine::from_bytes(&o).is_none().into() {
            // :)
        } else {
            panic!("should have rejected the point because we expected a compressed point");
        }
    }

    let m = Fq::char();

    {
        let mut o = o;
        o.as_mut()[..48].copy_from_slice(m.as_ref());
        o.as_mut()[0] |= 0b1000_0000;

        if G1Affine::from_bytes(&o).is_none().into() {
            // x coordinate
        } else {
            panic!("should have rejected the point")
        }
    }

    {
        let mut o = o;
        let mut x = Fq::one();

        loop {
            let mut x3b = x.square();
            x3b.mul_assign(&x);
            x3b.add_assign(&Fq::from(4)); // TODO: perhaps expose coeff_b through API?

            if x3b.sqrt().is_some().into() {
                x.add_assign(&Fq::one());
            } else {
                o.as_mut().copy_from_slice(x.to_repr().as_ref());
                o.as_mut()[0] |= 0b1000_0000;

                if G1Affine::from_bytes(&o).is_none().into() {
                    break;
                } else {
                    panic!("should have rejected the point because it isn't on the curve")
                }
            }
        }
    }

    {
        let mut o = o;
        let mut x = Fq::one();

        loop {
            let mut x3b = x.square();
            x3b.mul_assign(&x);
            x3b.add_assign(&Fq::from(4)); // TODO: perhaps expose coeff_b through API?

            if x3b.sqrt().is_some().into() {
                // We know this is on the curve, but it's likely not going to be in the correct subgroup.
                o.as_mut().copy_from_slice(x.to_repr().as_ref());
                o.as_mut()[0] |= 0b1000_0000;

                if G1Affine::from_bytes(&o).is_none().into() {
                    break;
                } else {
                    panic!(
                        "should have rejected the point because it isn't in the correct subgroup"
                    )
                }
            } else {
                x.add_assign(&Fq::one());
            }
        }
    }
}

#[test]
fn test_g2_compressed_invalid_vectors() {
    {
        let z = G2Affine::identity().to_bytes();

        {
            let mut z = z;
            z.as_mut()[0] &= 0b0111_1111;
            if G2Affine::from_bytes(&z).is_none().into() {
                // :)
            } else {
                panic!("should have rejected the point because we expected a compressed point");
            }
        }

        {
            let mut z = z;
            z.as_mut()[0] |= 0b0010_0000;
            if G2Affine::from_bytes(&z).is_none().into() {
                // :)
            } else {
                panic!("should have rejected the point because the parity bit should not be set if the point is at infinity");
            }
        }

        for i in 0..G2Compressed::size() {
            let mut z = z;
            z.as_mut()[i] |= 0b0000_0001;
            if G2Affine::from_bytes(&z).is_none().into() {
                // :)
            } else {
                panic!("should have rejected the point because the coordinates should be zeroes at the point at infinity");
            }
        }
    }

    let o = G2Affine::generator().to_bytes();

    {
        let mut o = o;
        o.as_mut()[0] &= 0b0111_1111;
        if G2Affine::from_bytes(&o).is_none().into() {
            // :)
        } else {
            panic!("should have rejected the point because we expected a compressed point");
        }
    }

    let m = Fq::char();

    {
        let mut o = o;
        o.as_mut()[..48].copy_from_slice(m.as_ref());
        o.as_mut()[0] |= 0b1000_0000;

        if G2Affine::from_bytes(&o).is_none().into() {
            // x coordinate (c1)
        } else {
            panic!("should have rejected the point")
        }
    }

    {
        let mut o = o;
        o.as_mut()[48..96].copy_from_slice(m.as_ref());
        o.as_mut()[0] |= 0b1000_0000;

        if G2Affine::from_bytes(&o).is_none().into() {
            // x coordinate (c0)
        } else {
            panic!("should have rejected the point")
        }
    }

    {
        let mut o = o;
        let mut x = Fq2 {
            c0: Fq::one(),
            c1: Fq::one(),
        };

        loop {
            let mut x3b = x.square();
            x3b.mul_assign(&x);
            x3b.add_assign(&Fq2 {
                c0: Fq::from(4),
                c1: Fq::from(4),
            }); // TODO: perhaps expose coeff_b through API?

            if x3b.sqrt().is_some().into() {
                x.add_assign(&Fq2::one());
            } else {
                o.as_mut()[..48].copy_from_slice(x.c1.to_repr().as_ref());
                o.as_mut()[48..].copy_from_slice(x.c0.to_repr().as_ref());
                o.as_mut()[0] |= 0b1000_0000;

                if G2Affine::from_bytes(&o).is_none().into() {
                    break;
                } else {
                    panic!("should have rejected the point because it isn't on the curve")
                }
            }
        }
    }

    {
        let mut o = o;
        let mut x = Fq2 {
            c0: Fq::one(),
            c1: Fq::one(),
        };

        loop {
            let mut x3b = x.square();
            x3b.mul_assign(&x);
            x3b.add_assign(&Fq2 {
                c0: Fq::from(4),
                c1: Fq::from(4),
            }); // TODO: perhaps expose coeff_b through API?

            if x3b.sqrt().is_some().into() {
                // We know this is on the curve, but it's likely not going to be in the correct subgroup.
                o.as_mut()[..48].copy_from_slice(x.c1.to_repr().as_ref());
                o.as_mut()[48..].copy_from_slice(x.c0.to_repr().as_ref());
                o.as_mut()[0] |= 0b1000_0000;

                if G2Affine::from_bytes(&o).is_none().into() {
                    break;
                } else {
                    panic!(
                        "should have rejected the point because it isn't in the correct subgroup"
                    )
                }
            } else {
                x.add_assign(&Fq2::one());
            }
        }
    }
}
