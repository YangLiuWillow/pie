//! A family's generative facts (design D12) are what `forward-pass.reading`
//! and `input` resolve against, so the runtime refuses at registration any
//! statement it could not resolve by name: readings must sit at their own
//! index, names and port names must be unique, a token-less reading must
//! carry a latents port (its lane's row count), positions carry at most
//! four axes, and one velocity width serves the whole eta profile — which
//! `velocity_facts` reads off the first velocity reading.
//!
//! `cargo test -p runtime --test a_reading_states_what_its_pass_binds`

use models::{Generative, PortFact, PortKind, ReadingFact, ReadoutKind, Stream};
use runtime::model::{validate_generative, velocity_facts};

fn text() -> ReadingFact {
    ReadingFact {
        name: "text",
        index: 0,
        has_kv: true,
        takes_tokens: true,
        streams: Vec::new(),
        ports: Vec::new(),
        readout: ReadoutKind::Hidden,
        readout_width: 512,
    }
}

fn denoise() -> ReadingFact {
    ReadingFact {
        name: "denoise",
        index: 1,
        has_kv: false,
        takes_tokens: false,
        streams: vec![Stream::Image],
        ports: vec![
            PortFact {
                name: "latents",
                kind: PortKind::Latents,
                width: 64,
            },
            PortFact {
                name: "timestep",
                kind: PortKind::LaneVector,
                width: 1,
            },
            PortFact {
                name: "positions",
                kind: PortKind::AxisPositions,
                width: 3,
            },
            PortFact {
                name: "context",
                kind: PortKind::Context,
                width: 512,
            },
        ],
        readout: ReadoutKind::Velocity,
        readout_width: 64,
    }
}

fn family(readings: Vec<ReadingFact>) -> Generative {
    Generative {
        readings,
        latent: None,
        schedule: None,
        max_rows: 4096,
    }
}

#[test]
fn a_well_formed_family_is_accepted_and_states_its_velocity() {
    let both = family(vec![text(), denoise()]);
    assert_eq!(validate_generative(&both), Ok(()));
    assert_eq!(velocity_facts(&both.readings), (true, 64));
    assert_eq!(velocity_facts(&[text()]), (false, 0));
    assert_eq!(velocity_facts(&[]), (false, 0));
}

#[test]
fn a_reading_resolves_its_ports_by_name_to_kind_relative_indices() {
    let reading = denoise();
    let (index, latents) = reading.port("latents").expect("declared");
    assert_eq!((index, latents.kind), (0, PortKind::Latents));
    let (index, _) = reading.port("timestep").expect("declared");
    assert_eq!(index, 0, "the first lane vector is lane-vector port 0");
    assert!(reading.port("guidance").is_none());

    let mut two = denoise();
    two.ports.push(PortFact {
        name: "guidance",
        kind: PortKind::LaneVector,
        width: 1,
    });
    let (index, _) = two.port("guidance").expect("declared");
    assert_eq!(index, 1, "the second lane vector is lane-vector port 1");
    let indexed: Vec<(u8, &str)> = two.ports_indexed().map(|(i, p)| (i, p.name)).collect();
    assert_eq!(
        indexed,
        vec![
            (0, "latents"),
            (0, "timestep"),
            (0, "positions"),
            (0, "context"),
            (1, "guidance")
        ]
    );
}

#[test]
fn a_family_out_of_index_order_is_refused() {
    let mut swapped = denoise();
    swapped.index = 0;
    let why = validate_generative(&family(vec![text(), swapped])).unwrap_err();
    assert!(why.contains("`denoise`") && why.contains("index"), "{why}");

    let mut text_at_one = text();
    text_at_one.index = 1;
    assert!(validate_generative(&family(vec![text_at_one])).is_err());
}

#[test]
fn duplicate_names_are_refused() {
    let mut second = text();
    second.index = 1;
    let why = validate_generative(&family(vec![text(), second])).unwrap_err();
    assert!(why.contains("declared twice"), "{why}");

    let mut twice = denoise();
    twice.ports.push(PortFact {
        name: "latents",
        kind: PortKind::Latents,
        width: 64,
    });
    let why = validate_generative(&family(vec![text(), twice])).unwrap_err();
    assert!(why.contains("`latents`") && why.contains("twice"), "{why}");
}

#[test]
fn a_token_less_reading_must_carry_a_latents_port() {
    let mut rowless = denoise();
    rowless.ports.retain(|port| port.kind != PortKind::Latents);
    let why = validate_generative(&family(vec![text(), rowless])).unwrap_err();
    assert!(why.contains("no latents port"), "{why}");
}

#[test]
fn a_positions_port_carries_at_most_four_axes() {
    let mut wide = denoise();
    wide.ports[2].width = 5;
    let why = validate_generative(&family(vec![text(), wide])).unwrap_err();
    assert!(why.contains("axes"), "{why}");
}

#[test]
fn readings_that_disagree_on_the_velocity_width_are_refused() {
    let mut low = denoise();
    low.name = "denoise.low";
    low.index = 2;
    low.readout_width = 128;
    let why = validate_generative(&family(vec![text(), denoise(), low])).unwrap_err();
    assert!(why.contains("velocity"), "{why}");
}
